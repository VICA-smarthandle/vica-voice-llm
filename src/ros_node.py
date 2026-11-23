"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node)."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from langchain_core.messages import AIMessage, HumanMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String, UInt8MultiArray
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg
try:
    from vica_interfaces.msg import VicaQuestion as VicaQuestionMsg
except ImportError:
    VicaQuestionMsg = None

from . import answer_field, fixed_answers, intent_recut, local_rules, question_record
from .building_directory import directory_path, format_directory_block, load_directory
from .destination_loader import load_voice_destinations
from .emergency_filter import detect_emergency, stop_pause_intent_for
from .history import ConversationHistory
from .speaker_link import SpeakerLink, verdict_from_labels
from .langchain_intent_parser import (
    get_backend_manager, is_instant_utterance, is_relisten_noise, is_scripted_reply,
    parse_intent, parse_intent_audio, text_tools_active)
from .ledger_view import render_ledger
from .mission_phrases import WAIT_BEACON
from .mission_question import (identity_answer, quiet_for_mission, reask_held_resume,
                               should_reask_resume_on_silence, spk_gate_question)
from .llm_backend import BackendState, parse_goal_event
from .realtime_intent import (RealtimeConnectionError, ab_shadow_enabled, audio_tools_active, audio_turn_applies,
                              get_realtime_client, pcm16_from_audio_msg)
from .situation_board import SituationBoard, parse_goal_event_name
from .replies import (COMMAND_DECLINED, LLM_UNAVAILABLE, RESUME_CONFIRM, RETRY_PROMPT,
                      WAKE_GREETING, expects_answer)
from .one_voice import (demote_unasked_confirm, hand_over_question, held_resume_carrier, mission_asks_resume,
                        voice_may_speak)
from .wake_echo import strip_wake_echo
from .ros_convert import intent_to_msg, msg_to_robot_state, supports_ask_fields
from .schema import should_forward_intent, RobotState, VicaIntent
from .stt_guard import is_hallucination  # noqa: F401  (텍스트 경로 관문용, 소리 모드는 안 쓴다)
from .tts_queue import RESPONSE, build_request, request_for_intent
from .reservation_flow import (FlowOutput, ReservationFlow, ReserveCall, created_event, handle_result,
                               join_situation, start_gate, to_intent)
from .reservation_store import ReservationStore, default_facilities, reservation_dir
from .tool_calls import audio_tools_on, text_tools_on


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
        self._hand_over = supports_ask_fields(VicaIntentMsg())
        self.get_logger().info("질문 넘김: 켬" if self._hand_over else "질문 넘김: 꺼짐(옛 vica_interfaces)")
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self._listen_pub = self.create_publisher(Bool, "/vica/listen_request", 10)
        self._thinking_pub = self.create_publisher(Bool, "/vica/thinking", 10)

        self._audio_turn: dict = {}
        self._ab_shadow = ab_shadow_enabled()
        self._last_text_publish_t = 0.0
        self._last_text_publish_intent = ""
        self._robot_recent: list = []
        self._last_robot_line = (0.0, "")
        self.create_subscription(String, "/vica/tts_done", self._on_tts_done_text, 10)
        self._question = None
        if VicaQuestionMsg is not None:
            self.create_subscription(
                VicaQuestionMsg, "/vica/question", self._on_question,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        else:
            self.get_logger().info("질문 기록 없음(옛 vica_interfaces) — 구독 안 함")
        self.create_subscription(UInt8MultiArray, "/vica/user_audio", self._on_user_audio, 10)
        self.get_logger().info(
            f"의도 입력 모드: {self._intent_input} | 로컬 규칙: {'켬' if self._local else '꺼짐'}"
            f" | 비교 기록(GPT): {'켬' if self._ab_shadow else '꺼짐'}")
        switch_warn = fixed_answers.switch_warning()
        if switch_warn:
            self.get_logger().warning(switch_warn)

        self._spk_link = SpeakerLink()
        self._audio_spk = None
        self._spk_on = os.environ.get("VICA_SPK_FOLLOW", "off").strip().lower() == "on"
        self.create_subscription(String, "/vica/user_speaker",
                                 lambda m: self._spk_link.put(m.data, time.time()), 10)
        self.create_subscription(String, "/vica/user_text", self._on_user_text, 10)
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)
        self._followup_until = 0.0
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)
        self.create_subscription(String, "/vica/listen_state", self._on_listen_state_for_resume, 10)
        self.create_subscription(String, "/vica/wake", self._on_wake_signal, 10)

        self._reservation: ReservationFlow | None = None
        self._reservation_pub = None
        if (text_tools_on() or audio_tools_on()) and not (VicaQuestionMsg is not None and self._hand_over):
            self.get_logger().error("시설 예약 끔 — 질문 넘김·질문 기록이 없는 옛 vica_interfaces")
        if (text_tools_on() or audio_tools_on()) and VicaQuestionMsg is not None and self._hand_over:
            try:
                facilities = default_facilities()
                if not facilities:
                    raise ValueError("예약할 시설 목록이 비어 있다(facilities.yaml)")
                self._reservation = ReservationFlow(
                    ReservationStore(reservation_dir() / "reservations.json"), facilities)
                self._reservation_pub = self.create_publisher(String, "/vica/reservation/event", 10)
                self.get_logger().info(
                    f"시설 예약: 켬(글자 {'on' if text_tools_on() else 'off'} · 소리 {'on' if audio_tools_on() else 'off'})"
                    f" | 시설 {len(facilities)}곳 | 저장 {self._reservation.store.path}")
            except Exception as exc:
                self._reservation = None
                self.get_logger().error(f"시설 예약 준비 실패({exc}) — 예약 없이 동작한다")
            if self._intent_input == "audio" and not (text_tools_on() and audio_tools_on()):
                self.get_logger().warning(
                    "시설 예약: 소리 모드인데 VICA_TOOLS_TEXT·VICA_TOOLS_AUDIO 가 하나만 켜졌다 — 꺼진 경로의 예약 말은 받지 않는다")

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
        raw = msg.data.strip()
        if not raw:
            return
        text = strip_wake_echo(raw, self._question, self._robot_state.dialog_state)
        if text != raw:
            self.get_logger().info(f"앞 '네' 뗌: '{raw}' → '{text}'")
        self._reload_destinations_if_changed()

        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")
            if self._reservation is not None:
                self._reservation.drop()

        pause = stop_pause_intent_for(text, self._robot_state.dialog_state)
        if pause is not None:
            turn, self._audio_turn = self._audio_turn, {}
            if (self._intent_input == "audio" and audio_turn_applies(turn, time.time())
                    and getattr(turn.get("intent"), "intent", "") == "pause"):
                self.get_logger().info(f"[멈춤] 소리 경로가 이미 일시정지 발행 — 받아쓰기 '{text}' 생략")
                return
            self.get_logger().warning(f"[멈춤] '{text}' — 주행 중이라 일시정지 요청")
            self._publish_intent(pause, text, link_text=raw)
            self._last_text_publish_t = time.time()
            self._last_text_publish_intent = pause.intent
            return

        keyword = detect_emergency(text)
        flow_out: FlowOutput | None = None
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
                self._log_transcript_rule(text, turn)
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
                    question=self._question,
                )
            finally:
                if thinking:
                    self._thinking_pub.publish(Bool(data=False))
            if self._reservation is not None or isinstance(intent, ReserveCall):
                flow = self._reservation
                gate = (start_gate(self._robot_state.dialog_state, self._question, flow.asked_text)
                        if flow is not None else "busy")
                tools_used = text_tools_active() and self._backend.state is not BackendState.LOCAL
                flow_out, intent = handle_result(flow, intent, text, gate=gate, tools_used=tools_used)
        if flow_out is not None:
            self._reservation_say(flow_out, text, link_text=raw)
        elif intent is not None:
            self._publish_intent(intent, text, link_text=raw)
        else:
            self.get_logger().info(f"[예약] 말하지 않는 상태 — 아무것도 내지 않음: '{text}'")
        self._last_text_publish_t = time.time()
        self._last_text_publish_intent = intent.intent if intent is not None else ""

    def _on_question(self, msg) -> None:
        """미션의 질문 기록. id 가 빈 기록은 열린 질문 없음."""
        self._question = question_record.from_fields(
            msg.id, msg.kind, msg.text, msg.option_ids, msg.opened_at.sec + msg.opened_at.nanosec / 1e9)
        if self._reservation is not None:
            self._reservation.on_question_record(msg.text if msg.id else "")

    def _reservation_say(self, out: FlowOutput, text: str, link_text: str = "") -> None:
        """예약 흐름이 정한 문장을 내보낸다. 저장됐으면 앱 알림 사건을 먼저 낸다(설계 5.5)."""
        if out.saved is not None:
            self._reservation_pub.publish(String(data=created_event(out.saved, datetime.now())))
            self.get_logger().info(
                f"[예약] 저장 id={out.saved.id} {out.saved.facility} {out.saved.date} {out.saved.start}~{out.saved.end}")
        self.get_logger().info(f"[예약] kind={out.kind or '안내'} 문장='{out.say}'")
        self._publish_intent(to_intent(out), text, scripted=True, link_text=link_text)

    def _publish_intent(self, intent: VicaIntent, text: str, llm_first: bool = False,
                        link_text: str = "", scripted: bool = False) -> None:
        """intent 확정 뒤 공통 후처리 (재청취 기각 → 발행 → TTS → 재청취 준비 → 로그 → 히스토리)."""
        if not answer_field.on():
            intent = reask_held_resume(intent, self._history.messages)
        question = self._question
        spk = self._spk_link.take(link_text or text, time.time())
        if spk is None and self._audio_spk is not None:
            spk, self._audio_spk = self._audio_spk, None
        if spk is not None:
            intent = intent.model_copy(update={
                "speaker_verdict": str(spk.get("verdict", "")),
                "speaker_score": float(spk.get("score", -1.0)),
                "speaker_joined": float(spk.get("joined", -1.0))})
        if intent.intent == intent_recut.NOT_FOR_ROBOT:
            self.get_logger().info(
                f"[끼어들지 않음] 나에게 한 말이 아님: '{text}' 화자={intent.speaker_verdict or '-'}")
            return
        if intent.intent == fixed_answers.REPEAT:
            settled = fixed_answers.settle_repeat(question, self._robot_recent)
            self.get_logger().info(
                f"[다시 말함] 질문={question.kind if question is not None else '-'} → intent={settled.intent} "
                f"'{settled.reply}'")
            intent = settled.model_copy(update={
                "confidence": intent.confidence, "speaker_verdict": intent.speaker_verdict,
                "speaker_score": intent.speaker_score, "speaker_joined": intent.speaker_joined})
        gate_t, gate_text = self._last_robot_line
        spk_gate = spk_gate_question(gate_text, time.time() - gate_t,
                                     self._robot_state.dialog_state, self._spk_on)
        intent = identity_answer(intent, text, spk_gate)
        last_spoken = self._robot_recent[-1][1] if self._robot_recent else ""
        noise = False if scripted else (
            not llm_first and is_relisten_noise(intent, self._destinations,
                                                in_window=time.time() < self._followup_until,
                                                last_robot_text=last_spoken))
        if noise:
            self.get_logger().info(
                f"재청취 기각(무의미): '{text}' intent={intent.intent}")
            return

        if self._local and not llm_first and not scripted:
            intent = self._apply_misheard_rule(intent, text)

        last_t, last_text = self._robot_recent[-1] if self._robot_recent else (0.0, "")
        quieted = intent if scripted else quiet_for_mission(
            intent, self._robot_state.dialog_state, last_text,
            time.time() - last_t, [d.name for d in self._destinations],
            spk_on=self._spk_on, spk_gate=spk_gate)
        silenced = quieted is not intent
        intent = quieted

        if self._hand_over:
            asked = demote_unasked_confirm(intent, question, self._destinations)
            if asked is not intent:
                self.get_logger().info(
                    f"[확정 막음] 열린 확인 질문이 없어 제안으로 낮춤: dest={intent.matched_destination_id}")
            intent = asked
        history_line = (intent.ask_text or intent.reply) if scripted else intent.reply
        if mission_asks_resume(intent):
            history_line = ""
        if self._hand_over:
            handed = hand_over_question(intent, self._destinations)
            if handed is not intent:
                self.get_logger().info(
                    f"[한목소리] 질문을 미션으로 kind={handed.ask_kind or 'confirm'} 글자='{history_line}'")
            intent = handed

        if should_forward_intent(intent):
            self._intent_pub.publish(intent_to_msg(intent))
            self._board.on_intent(intent)
        elif intent.ask_text:
            self._intent_pub.publish(intent_to_msg(held_resume_carrier(intent)))
            self.get_logger().info("resume 확언 대기 — 발행 보류 (질문은 미션이 말함)")
        else:
            self.get_logger().info("resume 확언 대기 — 발행 보류 (질문만 나감)")

        if not self._hand_over or voice_may_speak(intent, question):
            request = request_for_intent(intent)
        else:
            request = None
            if intent.reply:
                self.get_logger().info(f"[한목소리] 질문 열림 — 음성 말 안 함: '{intent.reply}'")
        if request:
            self._tts_pub.publish(String(data=request))

        if not self._hand_over and (intent.need_confirm or intent.intent == "clarify"
                                    or expects_answer(intent.reply)):
            self._listen_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )
        self.get_logger().info(question_record.log_line(question, time.time()))
        if answer_field.on():
            self.get_logger().info(answer_field.log_line(intent, question))

        human = [] if scripted and not text else [HumanMessage(text)]
        if (self._spoken_history or silenced or intent.intent == fixed_answers.REPEAT
                or mission_asks_resume(intent)):
            self._history.extend(human)
        else:
            self._history.extend([*human, AIMessage(history_line)])

    def _on_listen_state_for_resume(self, msg: String) -> None:
        """듣기 창이 빈손으로 닫혔다. 마지막 로봇 말이 음성이 쥔 "다시 출발할까요?"이고 아직"""
        state = msg.data or ""
        if not state.startswith("empty"):
            return
        if (self._reservation is not None and self._reservation.active
                and not state.startswith("empty:wake-")):
            gate = start_gate(self._robot_state.dialog_state, self._question, self._reservation.asked_text)
            out = self._reservation.on_silence(gate)
            if out is not None:
                self.get_logger().info(f"[예약] 대답 없음({state[:24]}) — {'다시 묻는다' if out.kind else '취소'}")
                self._reservation_say(out, "")
                return
        if answer_field.on():
            return
        last_t, last_text = self._robot_recent[-1] if self._robot_recent else (0.0, "")
        if not should_reask_resume_on_silence(self._history.messages, last_text,
                                              time.time() - last_t):
            return
        self.get_logger().info("다시 출발 확인 — 대답이 없어 한 번 다시 묻는다")
        if self._hand_over:
            ask = VicaIntent(intent="resume", need_confirm=True, ask_text=RESUME_CONFIRM, ask_kind="yesno")
            self._intent_pub.publish(intent_to_msg(held_resume_carrier(ask)))
        else:
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
        if text and text != WAIT_BEACON:
            self._last_robot_line = (now, text)
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
        self._audio_spk = verdict_from_labels([d.label for d in msg.layout.dim])
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
            if self._reservation is not None:
                self._reservation.drop()
        history_snapshot = self._history.messages
        robot_state = self._robot_state
        question = self._question
        self._audio_turn.update(history=history_snapshot, robot_state=robot_state, question=question)
        situation = intent_recut.with_speaker(self._situation_block(), (self._audio_spk or {}).get("verdict"))
        self._thinking_pub.publish(Bool(data=True))
        call_started = time.monotonic()
        try:
            intent, heard, dt, info = parse_intent_audio(
                pcm, self._destinations, history=history_snapshot, robot_state=robot_state,
                situation=situation,
                directory_block=format_directory_block(
                    self._directory, self._robot_state.current_building, self._robot_state.current_floor),
                question=question,
                tool_trail=self._reservation.tool_trail() if self._reservation is not None else None)
        except Exception as exc:
            to_local = isinstance(exc, RealtimeConnectionError) and self._backend.go_local("Realtime 연결 오류")
            if isinstance(exc, RealtimeConnectionError) and not self._backend.has_local:
                where = "로컬 LLM 없음(VICA_LLM_FALLBACK_MODEL 비어 있음) — 클라우드 글자 경로"
            elif self._backend.state is BackendState.LOCAL:
                where = "로컬 LLM" + (" 으로 바로 전환" if to_local else "")
            else:
                where = "클라우드 글자 LLM"
            self.get_logger().warning(
                f"[A/B] 소리 경로 실패({type(exc).__name__}: {exc}) "
                f"dt={time.monotonic() - call_started:.2f}s — 텍스트 경로가 처리({where})")
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

        usage = info.get("usage") or {}
        audio_tok = usage.get("audio_tokens", 0)
        text_tok = usage.get("text_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cached_tok = usage.get("cached_tokens", 0)
        if isinstance(intent, ReserveCall):
            given = " ".join(f"{k}={v}" for k, v in vars(intent).items() if v and k != "name") or "-"
            self.get_logger().info(
                f"[RT] heard='{heard}' intent=reserve_facility {given} name={'있음' if intent.name else '-'} "
                f"src={info['src']} dt={dt:.2f}s tokens={audio_tok}/{text_tok}/{out_tok} "
                f"cached={cached_tok} clip={clip_sec:.2f}s")
        else:
            self.get_logger().info(
                f"[RT] heard='{heard}' intent={intent.intent} dest={intent.matched_destination_id or '-'} "
                f"nc={intent.need_confirm} conf={intent.confidence:.2f} reply='{intent.reply[:30]}' "
                f"src={info['src']} dt={dt:.2f}s tokens={audio_tok}/{text_tok}/{out_tok} "
                f"cached={cached_tok} clip={clip_sec:.2f}s")
        flow_out: FlowOutput | None = None
        if self._reservation is not None or isinstance(intent, ReserveCall):
            flow = self._reservation
            gate = (start_gate(self._robot_state.dialog_state, self._question, flow.asked_text)
                    if flow is not None else "busy")
            flow_out, intent = handle_result(flow, intent, heard, gate=gate, tools_used=audio_tools_active())
        shown = to_intent(flow_out) if flow_out is not None else intent
        self._audio_turn.update(handled=True, intent=shown, heard=heard, dt=dt)
        if flow_out is not None:
            self._reservation_say(flow_out, heard)
        elif intent is not None:
            self._publish_intent(intent, heard, llm_first=True)
        else:
            self.get_logger().info(f"[예약] 말하지 않는 상태 — 아무것도 내지 않음: heard='{heard}'")

    def _situation_block(self) -> str:
        """지시문 맨 뒤 [지금 상황]: 미션 대장(P1)이 오면 그것, 옛 미션이면 goal-event 상황판."""
        awaiting = time.time() < self._followup_until and not answer_field.on()
        text = render_ledger(self._robot_state, awaiting_answer=awaiting, now_text=time.strftime("%H:%M"))
        text = text or self._board.render(awaiting_answer=awaiting)
        return join_situation(text, self._reservation.situation_line() if self._reservation is not None else "")

    def _log_transcript_rule(self, text: str, turn: dict) -> None:
        """소리 모드: 받아쓴 글자가 통째로 짧은 대답(네/아니요 · N분)이면 그 규칙의 대답과 Realtime 이 낸 대답을"""
        question = turn.get("question")
        ruled = answer_field.short_answer(text, question)
        if ruled is None:
            return
        heard = turn.get("intent")
        self.get_logger().info(
            f"[대답] 받아쓴 글자 규칙={ruled[0]}/{ruled[1] or '-'} · "
            f"Realtime={getattr(heard, 'answer', '') or '-'}/{getattr(heard, 'answer_value', '') or '-'} · "
            f"kind={question.kind} · 글자='{text}'")

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
                if isinstance(shadow, ReserveCall):
                    text_part = f"reserve_facility/- {time.monotonic() - started:.2f}s"
                else:
                    text_part = (f"{shadow.intent}/{shadow.matched_destination_id or '-'} "
                                 f"{time.monotonic() - started:.2f}s")
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
