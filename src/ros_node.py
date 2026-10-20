"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node)."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import rclpy
from langchain_core.messages import AIMessage, HumanMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String, UInt8MultiArray
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from . import local_rules
from .building_directory import directory_path, format_directory_block, load_directory
from .destination_loader import load_voice_destinations
from .emergency_filter import detect_emergency, stop_pause_intent_for
from .history import ConversationHistory
from .langchain_intent_parser import (
    get_backend_manager, is_instant_utterance, is_scripted_intent, is_scripted_reply,
    parse_intent, parse_intent_audio)
from .ledger_view import render_ledger
from .mission_phrases import WAIT_BEACON
from .mission_question import (quiet_for_mission, reask_held_resume,
                               should_reask_resume_on_silence)
from .llm_backend import BackendState, parse_goal_event
from .realtime_intent import (ab_shadow_enabled, audio_turn_applies, get_realtime_client,
                              pcm16_from_audio_msg)
from .situation_board import SituationBoard, parse_goal_event_name
from .replies import (COMMAND_DECLINED, LLM_UNAVAILABLE, RESUME_CONFIRM, RETRY_PROMPT,
                      WAKE_GREETING, expects_answer)
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import should_forward_intent, RobotState, VicaIntent
from .stt_guard import is_hallucination  # noqa: F401  (텍스트 경로 관문용, 소리 모드는 안 쓴다)
from .tts_queue import RESPONSE, build_request, request_for_intent


FOLLOWUP_CONTEXT_SEC = 40.0

ROBOT_ECHO_TTL_SEC = 12.0

AUDIO_CLIP_MAX_SEC = 12.0
REALTIME_RETRY_AFTER_SEC = float(os.environ.get("VICA_REALTIME_RETRY_AFTER", "30"))


class LlmIntentNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_llm_intent_node")
        self.declare_parameter(
            "destinations_yaml",
            str(
                Path.home()
                / "vica_data"
                / "destinations"
                / "vica_map_0630"
                / "destinations.yaml"
            ),
        )
        self._destinations_path = Path(
            str(self.get_parameter("destinations_yaml").value)
        ).expanduser()
        self._destinations_mtime_ns: int | None = None
        self._destinations = []
        self._reload_destinations_if_changed(force=True)
        self._directory = load_directory(directory_path())
        self.get_logger().info(f"건물 디렉터리 {len(self._directory)}곳: {directory_path()}")
        self._robot_state = RobotState()
        self._intent_input = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower()
        self._local = local_rules.enabled()
        self._spoken_history = self._intent_input == "audio" or self._local
        if self._spoken_history:
            self._history = ConversationHistory(
                max_messages=local_rules.HISTORY_MAX_MESSAGES, idle_reset_sec=float("inf"))
        else:
            self._history = ConversationHistory()
        self._prev_dialog_state = ""
        self._misheard = local_rules.MisheardGate()
        self._board = SituationBoard()

        self._intent_pub = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self._listen_pub = self.create_publisher(Bool, "/vica/listen_request", 10)
        self._thinking_pub = self.create_publisher(Bool, "/vica/thinking", 10)

        self._audio_turn: dict = {}
        self._ab_shadow = ab_shadow_enabled()
        self._last_text_publish_t = 0.0
        self._last_text_publish_intent = ""
        self._robot_recent: list = []
        self.create_subscription(String, "/vica/tts_done", self._on_tts_done_text, 10)
        self.create_subscription(UInt8MultiArray, "/vica/user_audio", self._on_user_audio, 10)
        self.get_logger().info(
            f"의도 입력 모드: {self._intent_input} | 로컬 규칙: {'켬' if self._local else '꺼짐'}"
            f" | 비교 기록(GPT): {'켬' if self._ab_shadow else '꺼짐'}")

        self.create_subscription(String, "/vica/user_text", self._on_user_text, 10)
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)
        self._followup_until = 0.0
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)
        self.create_subscription(String, "/vica/listen_state", self._on_listen_state_for_resume, 10)
        self.create_subscription(String, "/vica/wake", self._on_wake_signal, 10)

        self._backend = get_backend_manager()
        self._backend.set_logger(self._backend_log)
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)

        threading.Thread(target=self._backend_loop, daemon=True, name="llm-backend").start()

        self.get_logger().info(
            "VICA LLM intent node 시작 (구독: /vica/user_text, /vica/robot_state | 발행: /vica/intent)"
        )
        threading.Thread(target=self._warmup_llm, daemon=True).start()

    def _backend_log(self, level: str, msg: str) -> None:
        """폴백 관리자의 로그를 ROS 로거로. 심각도마다 **다른 줄**에서 부른다(위 주석)."""
        logger = self.get_logger()
        if level == "error":
            logger.error(msg)
        elif level == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)

    def _warmup_llm(self) -> None:
        started = time.monotonic()
        try:
            parse_intent("워밍업", [], history=[])
            self.get_logger().info(
                f"LLM 워밍업 완료 ({time.monotonic() - started:.1f}초)")
        except Exception as exc:
            self.get_logger().warning(f"LLM 워밍업 실패(무시 가능): {exc}")
        if self._backend.state is BackendState.LOCAL:
            self._backend.warm_local_async()
        if self._intent_input == "audio":
            try:
                rt_dt = get_realtime_client().warm()
                self.get_logger().info(f"Realtime 연결 예열 완료 ({rt_dt:.1f}초)")
            except Exception as exc:
                self.get_logger().warning(f"Realtime 연결 예열 실패(무시 가능): {exc}")

    def _on_listen_request(self, msg: Bool) -> None:
        if msg.data:
            self._followup_until = time.time() + FOLLOWUP_CONTEXT_SEC
        else:
            self._followup_until = 0.0

    def _on_wake_signal(self, _msg: String) -> None:
        self._followup_until = 0.0
        self._misheard.reset()

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """로봇 상태 메시지를 받아 최신값으로 보관한다. 전환 담당 모듈에도 넘긴다."""
        self._robot_state = msg_to_robot_state(msg)
        self._backend.on_robot_state(msg.is_moving, msg.is_paused)
        if self._local:
            new_state = self._robot_state.dialog_state
            if local_rules.should_clear_history(self._prev_dialog_state, new_state) and len(self._history):
                self._history.clear()
                self.get_logger().info("접근 질문 시작 — 대화 기록 비움(로컬 규칙)")
            self._prev_dialog_state = new_state

    def _on_goal_event(self, msg: String) -> None:
        """미션 매니저의 주행 사건 → 전환 담당 모듈(주행 중엔 클라우드로 안 돌아간다)."""
        event = parse_goal_event(msg.data)
        if event:
            self._backend.on_goal_event(event)
        ev, name = parse_goal_event_name(msg.data)
        self._board.on_goal_event(ev, name)
        if ev == "return_home_sent" and self._spoken_history and len(self._history):
            self._history.clear()
            self.get_logger().info("안내 종료(대기 없음) — 대화 기록 비움")

    def _backend_loop(self) -> None:
        """1 Hz: 클라우드 확인·복귀 판정."""
        while rclpy.ok():
            try:
                before = self._backend.state
                after = self._backend.tick()
                if before is not after:
                    self.get_logger().info(f"[LLM] 백엔드 {before.value} → {after.value}")
            except Exception as exc:
                self.get_logger().warning(f"[LLM] 백엔드 루프 오류(무시): {exc}")
            time.sleep(1.0)

    def _on_user_text(self, msg: String) -> None:
        """발화를 받아 VicaIntent 를 만들어 발행한다."""
        text = msg.data.strip()
        if not text:
            return
        self._reload_destinations_if_changed()

        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")

        pause = stop_pause_intent_for(text, self._robot_state.dialog_state)
        if pause is not None:
            turn, self._audio_turn = self._audio_turn, {}
            if (self._intent_input == "audio" and audio_turn_applies(turn, time.time())
                    and getattr(turn.get("intent"), "intent", "") == "pause"):
                self.get_logger().info(f"[멈춤] 소리 경로가 이미 일시정지 발행 — 받아쓰기 '{text}' 생략")
                return
            self.get_logger().warning(f"[멈춤] '{text}' — 주행 중이라 일시정지 요청")
            self._publish_intent(pause, text)
            self._last_text_publish_t = time.time()
            self._last_text_publish_intent = pause.intent
            return

        keyword = detect_emergency(text)
        if keyword:
            intent = VicaIntent(
                intent="unknown",
                reply="",
                need_confirm=False,
                safety_flag="emergency",
            )
            self.get_logger().warn(f"[긴급] '{keyword}' 감지 -> safety_flag=emergency")
        else:
            if self._intent_input == "audio" and audio_turn_applies(self._audio_turn, time.time()):
                turn, self._audio_turn = self._audio_turn, {}
                self._shadow_text(text, turn)
                return
            if self._intent_input == "audio" and self._audio_turn:
                if self._audio_turn.get("handled"):
                    self.get_logger().info("[A/B] 옛 소리 결과 버림")
                else:
                    self.get_logger().info("[A/B] 소리 경로 결과 없음 — 텍스트 경로가 처리")
                self._audio_turn = {}
            thinking = not is_instant_utterance(text)
            if thinking:
                self._thinking_pub.publish(Bool(data=True))

            try:
                intent = parse_intent(
                    text,
                    self._destinations,
                    history=self._history.messages,
                    robot_state=self._robot_state,
                    situation=self._situation_block(),
                )
            finally:
                if thinking:
                    self._thinking_pub.publish(Bool(data=False))
        self._publish_intent(intent, text)
        self._last_text_publish_t = time.time()
        self._last_text_publish_intent = intent.intent

    def _publish_intent(self, intent: VicaIntent, text: str, llm_first: bool = False) -> None:
        """intent 확정 뒤 공통 후처리 (재청취 기각 → 발행 → TTS → 재청취 준비 → 로그 → 히스토리)."""
        intent = reask_held_resume(intent, self._history.messages)
        if (not llm_first
                and time.time() < self._followup_until
                and intent.intent in ("unknown", "clarify")
                and not intent.need_confirm
                and not is_scripted_intent(intent, self._destinations)
                and intent.safety_flag != "emergency"):
            self.get_logger().info(
                f"재청취 기각(무의미): '{text}' intent={intent.intent}")
            return

        if self._local and not llm_first:
            intent = self._apply_misheard_rule(intent, text)

        last_t, last_text = self._robot_recent[-1] if self._robot_recent else (0.0, "")
        quieted = quiet_for_mission(intent, self._robot_state.dialog_state, last_text,
                                    time.time() - last_t, [d.name for d in self._destinations])
        silenced = quieted is not intent
        intent = quieted

        if should_forward_intent(intent):
            self._intent_pub.publish(intent_to_msg(intent))
            self._board.on_intent(intent)
        else:
            self.get_logger().info("resume 확언 대기 — 발행 보류 (질문만 나감)")

        request = request_for_intent(intent)
        if request:
            self._tts_pub.publish(String(data=request))

        if (intent.need_confirm or intent.intent == "clarify"
                or expects_answer(intent.reply)):
            self._listen_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )

        if self._spoken_history or silenced:
            self._history.extend([HumanMessage(text)])
        else:
            self._history.extend([HumanMessage(text), AIMessage(intent.reply)])

    def _on_listen_state_for_resume(self, msg: String) -> None:
        """듣기 창이 빈손으로 닫혔다. 마지막 로봇 말이 음성이 쥔 "다시 출발할까요?"이고 아직"""
        if not (msg.data or "").startswith("empty"):
            return
        last_t, last_text = self._robot_recent[-1] if self._robot_recent else (0.0, "")
        if not should_reask_resume_on_silence(self._history.messages, last_text,
                                              time.time() - last_t):
            return
        self.get_logger().info("다시 출발 확인 — 대답이 없어 한 번 다시 묻는다")
        self._tts_pub.publish(String(data=build_request(RESPONSE, RESUME_CONFIRM)))
        self._listen_pub.publish(Bool(data=True))
        if not self._spoken_history:
            self._history.extend([AIMessage(RESUME_CONFIRM)])

    def _apply_misheard_rule(self, intent: VicaIntent, text: str) -> VicaIntent:
        """로컬 규칙: 창 밖 못 알아들은 말의 대꾸를 고정 문구 한 번 → 침묵으로 바꾼다."""
        fixed = {WAKE_GREETING, COMMAND_DECLINED, LLM_UNAVAILABLE}
        misheard = (time.time() >= self._followup_until
                    and intent.intent in ("unknown", "question")
                    and intent.safety_flag != "emergency"
                    and intent.reply not in fixed
                    and not is_scripted_reply(intent.reply, self._destinations))
        if not misheard:
            if intent.intent not in ("unknown", "question"):
                self._misheard.reset()
            return intent
        reply = self._misheard.decide(RETRY_PROMPT)
        self.get_logger().info(
            f"못 알아들음(로컬 규칙): '{text}' intent={intent.intent} "
            f"LLM 대꾸={intent.reply!r} → {'고정 문구' if reply else '침묵'}")
        return intent.model_copy(update={"reply": reply or ""})

    def _on_tts_done_text(self, msg: String) -> None:
        """로봇이 방금 한 말을 기억한다(에코 대조용, 웨이크워드 노드와 같은 방어)."""
        now = time.time()
        self._robot_recent = [
            (t, s) for t, s in self._robot_recent if now - t < ROBOT_ECHO_TTL_SEC]
        self._robot_recent.append((now, msg.data))
        text = (msg.data or "").strip()
        if self._spoken_history and text and text != WAIT_BEACON:
            self._history.extend([AIMessage(text)])

    def _on_user_audio(self, msg: UInt8MultiArray) -> None:
        """audio 모드: 소리를 Realtime 에 보내 의도를 받는다. 관문에 걸리거나 실패하면"""
        if self._intent_input != "audio":
            return
        if time.time() - self._last_text_publish_t < 2.0:
            self.get_logger().info("[A/B] 텍스트가 먼저 처리됨 — 소리 경로 생략")
            return
        turn_started = time.time()
        self._audio_turn = {"handled": False, "t": turn_started}
        if get_realtime_client().recently_failed(REALTIME_RETRY_AFTER_SEC):
            self.get_logger().info(
                f"[A/B] Realtime 최근 실패 — {REALTIME_RETRY_AFTER_SEC:.0f}초간 소리 경로 생략, 텍스트 경로가 처리")
            return
        try:
            label = msg.layout.dim[0].label if msg.layout.dim else ""
            pcm = pcm16_from_audio_msg(msg.data, label)
        except ValueError as exc:
            self.get_logger().warning(f"[A/B] 소리 메시지 무시: {exc}")
            return
        clip_sec = len(pcm) / 2 / 16000
        if clip_sec > AUDIO_CLIP_MAX_SEC:
            self.get_logger().info(f"[A/B] 클립 {clip_sec:.1f}s — 너무 길어 소리 경로 생략")
            return
        self._reload_destinations_if_changed()
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")
        history_snapshot = self._history.messages
        robot_state = self._robot_state
        self._audio_turn.update(history=history_snapshot, robot_state=robot_state)
        self._thinking_pub.publish(Bool(data=True))
        call_started = time.monotonic()
        try:
            intent, heard, dt, info = parse_intent_audio(
                pcm, self._destinations, history=history_snapshot, robot_state=robot_state,
                situation=self._situation_block(),
                directory_block=format_directory_block(
                    self._directory, self._robot_state.current_building, self._robot_state.current_floor))
        except Exception as exc:
            self.get_logger().warning(
                f"[A/B] 소리 경로 실패({type(exc).__name__}: {exc}) "
                f"dt={time.monotonic() - call_started:.2f}s — 텍스트 경로가 처리")
            return
        finally:
            self._thinking_pub.publish(Bool(data=False))

        pause = stop_pause_intent_for(heard, self._robot_state.dialog_state)
        if pause is not None:
            if (self._last_text_publish_t > turn_started
                    and self._last_text_publish_intent == "pause"):
                self.get_logger().info(f"[멈춤] 텍스트가 이미 일시정지 발행 — 소리 결과 버림: heard='{heard}'")
                return
            self._audio_turn.update(handled=True, intent=pause, heard=heard, dt=dt)
            self.get_logger().warning(f"[멈춤] heard='{heard}' — 주행 중이라 일시정지 요청 (dt={dt:.2f}s)")
            self._publish_intent(pause, heard, llm_first=True)
            return

        if detect_emergency(heard):
            self.get_logger().warning(f"[A/B] 긴급어 — 텍스트 경로에 위임: heard='{heard}'")
            return

        if self._last_text_publish_t > turn_started:
            self.get_logger().info(f"[A/B] 텍스트가 먼저 발행 — 소리 결과 버림: heard='{heard}'")
            return

        self._audio_turn.update(handled=True, intent=intent, heard=heard, dt=dt)
        usage = info.get("usage") or {}
        audio_tok = usage.get("audio_tokens", 0)
        text_tok = usage.get("text_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cached_tok = usage.get("cached_tokens", 0)
        self.get_logger().info(
            f"[RT] heard='{heard}' intent={intent.intent} dest={intent.matched_destination_id or '-'} "
            f"nc={intent.need_confirm} conf={intent.confidence:.2f} reply='{intent.reply[:30]}' "
            f"src={info['src']} dt={dt:.2f}s tokens={audio_tok}/{text_tok}/{out_tok} "
            f"cached={cached_tok} clip={clip_sec:.2f}s")
        self._publish_intent(intent, heard, llm_first=True)

    def _situation_block(self) -> str:
        """지시문 맨 뒤 [지금 상황]: 미션 대장(P1)이 오면 그것, 옛 미션이면 goal-event 상황판."""
        awaiting = time.time() < self._followup_until
        text = render_ledger(self._robot_state, awaiting_answer=awaiting, now_text=time.strftime("%H:%M"))
        return text or self._board.render(awaiting_answer=awaiting)

    def _shadow_text(self, text: str, turn: dict) -> None:
        """audio 모드에서 같은 발화의 텍스트 경로 결과를 로그로만 남긴다(발행 안 함)."""
        a = turn.get("intent")
        audio_part = (f"{a.intent}/{a.matched_destination_id or '-'} {turn.get('dt', 0.0):.2f}s"
                      if a is not None else "?")
        if not self._ab_shadow:
            self.get_logger().info(
                f"[A/B] audio={audio_part} | text=꺼짐 | heard='{turn.get('heard', '')}' | whisper='{text}'")
            return
        history = turn.get("history", self._history.messages)
        robot_state = turn.get("robot_state", self._robot_state)
        destinations = self._destinations

        def work():
            started = time.monotonic()
            try:
                shadow = parse_intent(text, destinations, history=history, robot_state=robot_state)
                text_part = f"{shadow.intent}/{shadow.matched_destination_id or '-'} {time.monotonic() - started:.2f}s"
            except Exception as exc:
                text_part = f"실패({type(exc).__name__}) {time.monotonic() - started:.2f}s"
            self.get_logger().info(
                f"[A/B] audio={audio_part} | text={text_part} | heard='{turn.get('heard', '')}' | whisper='{text}'")

        threading.Thread(target=work, daemon=True, name="ab-shadow").start()

    def _reload_destinations_if_changed(self, force: bool = False) -> None:
        """저장 노드가 YAML을 교체하면 다음 발화 전에 catalog를 갱신한다."""
        try:
            mtime_ns = self._destinations_path.stat().st_mtime_ns
        except FileNotFoundError:
            if force or self._destinations:
                self._destinations = []
                self._destinations_mtime_ns = None
                self.get_logger().warn(
                    f"목적지 catalog가 없어 빈 목록을 사용합니다: "
                    f"{self._destinations_path}"
                )
            return
        if not force and mtime_ns == self._destinations_mtime_ns:
            return
        try:
            loaded = load_voice_destinations(self._destinations_path)
        except Exception as exc:
            self.get_logger().error(
                f"목적지 catalog reload 실패, 이전 목록 유지: {exc}"
            )
            return
        self._destinations = loaded
        self._destinations_mtime_ns = mtime_ns
        self._directory = load_directory(directory_path())
        private = sum(1 for d in loaded if d.authorization != "public")
        self.get_logger().info(
            f"목적지 {len(loaded)}개 로드(비공개 {private}개는 접근 불가로): "
            f"{self._destinations_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LlmIntentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
