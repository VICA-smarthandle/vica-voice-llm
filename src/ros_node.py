"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node).

구독: /vica/user_text   (std_msgs/String)        - 사용자 발화 텍스트(STT 결과 등)
구독: /vica/robot_state (vica_interfaces/RobotState) - 로봇 현재 상태
발행: /vica/intent      (vica_interfaces/VicaIntent) - intent 해석 결과('제안')
발행: /vica/tts_request (std_msgs/String)        - 사용자에게 들려줄 응답

발화 주체를 나누는 이유는 tts_queue.request_for_intent 주석 참고. 요약하면,
navigate 확정 요청의 결과는 Mission Manager 만 알 수 있으므로 그쪽이 말한다.

안전 원칙 (CLAUDE.md):
- 이 노드는 /cmd_vel 이나 Nav2 goal 을 직접 보내지 않는다.
- VicaIntent 는 state machine 에 전달되는 '제안'일 뿐이다.
- 긴급어는 parse_intent 이전에 emergency_filter 가 처리한다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_node
"""
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

from .destination_loader import load_destinations
from .emergency_filter import detect_emergency
from .history import ConversationHistory
from .langchain_intent_parser import (
    SHORTCUT_REPLIES, get_backend_manager, is_instant_utterance, parse_intent,
    parse_intent_audio)
from .llm_backend import BackendState, parse_goal_event
from .realtime_intent import audio_turn_applies, get_realtime_client, pcm16_from_audio_msg
from .situation_board import SituationBoard, parse_goal_event_name
from .replies import expects_answer
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import should_forward_intent, RobotState, VicaIntent
from .stt_guard import is_hallucination  # noqa: F401  (텍스트 경로 관문용, 소리 모드는 안 쓴다)
from .tts_queue import request_for_intent


# 재청취 창 문맥으로 보는 시간. 질문 발화(수 초) + 청취 창 + 답 처리까지
# 덮는 넉넉한 값이다. "비카야" 직접 호출이 오면 즉시 해제된다.
FOLLOWUP_CONTEXT_SEC = 40.0

# 에코 대조 창(audio 모드). ros_wakeword_node.ROBOT_ECHO_TTL_SEC 과 같은 값(12초) —
# 둘 다 "로봇 자신의 최근 발화를 기억해 전사에서 걷어낸다"는 같은 방어라서
# 값이 어긋나면 안 된다. 바꿀 때는 두 파일을 함께 고친다.
ROBOT_ECHO_TTL_SEC = 12.0

# 이 길이를 넘는 클립은 소리 경로를 생략한다(항목 H) — 정상 발화 범위를
# 크게 벗어나면 Realtime 왕복 비용·지연만 늘고, 텍스트 경로가 어차피 처리한다.
AUDIO_CLIP_MAX_SEC = 12.0
# Realtime 이 방금 실패했으면(와이파이 없는 복도 등) 이 시간 동안은 소리 경로를 건너뛴다 —
# 발화마다 timeout(8초)을 기다리지 않게. 그동안은 whisper+로컬 텍스트 경로가 답한다.
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
        self._robot_state = RobotState()  # robot_state 토픽이 오기 전 기본값
        # 멀티턴 기억. 공용 로봇이라 한동안 발화가 없으면 새 대화로 보고 비운다.
        # 소리 모드(모델 전결)는 로봇이 실제로 한 말(미션 질문 포함)까지 이력에
        # 넣으므로 두 배로 둔다 — 8이면 왕복 4번이 안 된다.
        self._intent_input = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower()
        self._history = ConversationHistory(max_messages=16 if self._intent_input == "audio" else 8)
        # 상황판(소리 모드): 이동 중·마지막 도착·대기 요청 — 이력이 비어도 남는 사실.
        self._board = SituationBoard()

        self._intent_pub = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        # 질문("~할까요?", 되묻기)을 말할 때 true — 웨이크워드 노드가 그 질문 TTS 가
        # 끝나는 순간 재청취 창을 열어, 사용자가 "비카야" 재호출 없이 "응"으로
        # 답할 수 있게 한다.
        self._listen_pub = self.create_publisher(Bool, "/vica/listen_request", 10)
        # LLM 해석 중 "생각 중" 배경 운율 스위치 — TTS 노드가 반복 재생한다
        # (2026-09-01 사용자 결정: "확인할게요" 말 대신 운율 루프).
        self._thinking_pub = self.create_publisher(Bool, "/vica/thinking", 10)

        # ----- 소리→의도 직행 (audio 모드, 2026-09-20 실험) ------------------
        # text(기본): /vica/user_text 로 지금처럼. audio: /vica/user_audio 를 Realtime 에
        # 보내 의도를 받고, 같은 발화의 텍스트 경로 결과는 로그([A/B])로만 남긴다.
        #
        # 이 구독은 반드시 /vica/user_text 보다 먼저 만든다 — rclpy 단일
        # 실행기는 같은 주기에 준비된 구독을 생성 순서로 부른다. 소리가
        # 텍스트보다 먼저 처리돼야 이중 발행이 없다(항목 C).
        self._audio_turn: dict = {}   # 직전 소리 발화의 결과(그림자 비교·실패 시 텍스트 인계용)
        self._last_text_publish_t = 0.0   # 텍스트 경로가 방금 발행했으면 소리 경로를 생략한다
        # 에코 대조용 최근 로봇 발화(웨이크워드 노드와 같은 방어, stt_guard.strip_robot_echo).
        self._robot_recent: list = []
        self.create_subscription(String, "/vica/tts_done", self._on_tts_done_text, 10)
        self.create_subscription(UInt8MultiArray, "/vica/user_audio", self._on_user_audio, 10)
        self.get_logger().info(f"의도 입력 모드: {self._intent_input}")

        self.create_subscription(String, "/vica/user_text", self._on_user_text, 10)
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)
        # 재청취 창 문맥 추적 (2026-09-01): 질문 뒤 자동으로 열린 창에서 온
        # 무의미 발화(unknown·clarify)는 침묵으로 버린다 — 잡담·오전사에
        # 일일이 대꾸하면 그 대답이 또 창을 열어 온갖 문장이 연쇄로 나온다
        # (2026-08-31 야간 실기). "비카야" 직접 호출은 대답할 자격을 되살린다.
        self._followup_until = 0.0
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)
        self.create_subscription(String, "/vica/wake", self._on_wake_signal, 10)

        # ----- 클라우드→로컬 자동 전환 (2026-09-19 설계) ------------------------
        # 관리자는 파서와 같은 싱글턴이다. 주행 사건·로봇 상태를 흘려 넣고,
        # 별도 스레드가 1초마다 tick(클라우드 확인·복귀 판정)을 맡는다. 앱 진단
        # 표시는 두지 않는다(2026-09-19 실기 결정: 통신이 끊기면 앱도 끊긴다,
        # 상태는 로그로만).
        self._backend = get_backend_manager()
        # 전환 담당 모듈의 [LLM] 로그를 노드 로거로 보낸다 — 화면뿐 아니라
        # ~/.ros/log 파일에도 남아 실기 뒤에 대피·복귀 시각을 되짚을 수 있다(09-19 실기 교훈).
        # rclpy 로거는 **호출한 코드 줄마다 심각도 하나**만 허용한다(rcutils_logger:
        # "Logger severity cannot be changed between calls"). 한 줄 lambda 로 info/
        # warning/error 를 다 보내면 두 번째 다른 심각도에서 ValueError 가 나고, 그게
        # 구독 콜백(도착 사건 → 클라우드 복귀 로그) 안이면 노드가 통째로 죽는다
        # (2026-09-20 18:55 실기: "그래"·"여기서 대기해"에 무응답 → 미션이 귀가).
        self._backend.set_logger(self._backend_log)
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)

        threading.Thread(target=self._backend_loop, daemon=True, name="llm-backend").start()

        self.get_logger().info(
            "VICA LLM intent node 시작 (구독: /vica/user_text, /vica/robot_state | 발행: /vica/intent)"
        )
        # 첫 호출의 콜드스타트(연결 준비 4~6초 실측, 2026-08-28)를 사용자 대신
        # 여기서 치른다. 실패해도(네트워크 없음 등) 노드는 그대로 간다.
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
        # 워밍업 중 클라우드가 실패했으면 관리자는 이미 LOCAL 이다. 그 경우 로컬
        # 모델을 지금 미리 올려 첫 발화가 적재 7초를 기다리지 않게 한다.
        if self._backend.state is BackendState.LOCAL:
            self._backend.warm_local_async()
        # audio 모드는 Realtime 접속도 미리 맺는다(항목 E) — 첫 발화가 연결까지
        # 기다리지 않게 한다. 실패해도(네트워크 없음 등) 노드는 그대로 간다.
        if self._intent_input == "audio":
            try:
                rt_dt = get_realtime_client().warm()
                self.get_logger().info(f"Realtime 연결 예열 완료 ({rt_dt:.1f}초)")
            except Exception as exc:
                self.get_logger().warning(f"Realtime 연결 예열 실패(무시 가능): {exc}")

    def _on_listen_request(self, msg: Bool) -> None:
        # 이 노드 자신이 낸 요청도 같은 토픽으로 돌아온다 — 효과는 같다.
        if msg.data:
            self._followup_until = time.time() + FOLLOWUP_CONTEXT_SEC
        else:
            self._followup_until = 0.0

    def _on_wake_signal(self, _msg: String) -> None:
        self._followup_until = 0.0

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """로봇 상태 메시지를 받아 최신값으로 보관한다. 전환 담당 모듈에도 넘긴다."""
        self._robot_state = msg_to_robot_state(msg)
        self._backend.on_robot_state(msg.is_moving, msg.is_paused)

    def _on_goal_event(self, msg: String) -> None:
        """미션 매니저의 주행 사건 → 전환 담당 모듈(주행 중엔 클라우드로 안 돌아간다)."""
        event = parse_goal_event(msg.data)
        if event:
            self._backend.on_goal_event(event)
        ev, name = parse_goal_event_name(msg.data)
        self._board.on_goal_event(ev, name)

    def _backend_loop(self) -> None:
        """1 Hz: 클라우드 확인·복귀 판정."""
        while rclpy.ok():
            try:
                before = self._backend.state
                after = self._backend.tick()
                if before is not after:
                    self.get_logger().info(f"[LLM] 백엔드 {before.value} → {after.value}")
            except Exception as exc:  # 루프는 죽지 않는다
                self.get_logger().warning(f"[LLM] 백엔드 루프 오류(무시): {exc}")
            time.sleep(1.0)

    def _on_user_text(self, msg: String) -> None:
        """발화를 받아 VicaIntent 를 만들어 발행한다."""
        text = msg.data.strip()
        if not text:
            return
        self._reload_destinations_if_changed()

        # 0) 한동안 발화가 없었으면 다른 사용자로 보고 이전 맥락을 버린다.
        #    안 그러면 다음 사람의 "거기로 가줘"가 앞사람 목적지로 해석된다.
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")

        # 1) 긴급어는 LLM 이전에 처리한다 (안전 경로).
        keyword = detect_emergency(text)
        if keyword:
            # reply 는 비운다 — 걸림 멘트("안전을 위해 멈추겠습니다. 관리자를
            # 호출했습니다")는 래치를 아는 미션이 말한다. 여기서도 말하면
            # "멈춰" 한 번에 비슷한 말 2연발이었다 (2026-08-31 감량).
            intent = VicaIntent(
                intent="unknown",
                reply="",
                need_confirm=False,
                safety_flag="emergency",
            )
            self.get_logger().warn(f"[긴급] '{keyword}' 감지 -> safety_flag=emergency")
        else:
            if self._intent_input == "audio" and audio_turn_applies(self._audio_turn, time.time()):
                # 이 발화는 소리 경로가 이미 처리했다. 텍스트 경로 결과는 비교 로그로만.
                turn, self._audio_turn = self._audio_turn, {}   # 1회 소비 — 다음 발화에 새지 않게
                self._shadow_text(text, turn)
                return
            if self._intent_input == "audio" and self._audio_turn:
                # 긴급 검증 구제 경로 등 on_user_audio 를 거치지 않고 들어온 텍스트가
                # 옛 소리 결과를 주워 먹지 않도록 버린다(2026-09-20 리뷰, 발화 소실 방지).
                # handled 였으면 소리 경로가 결과는 냈지만 너무 오래돼(15초) 못
                # 붙인 것이고, 아니면 소리 경로가 애초에 관문(B·C·D·H)에서
                # 기각했다는 뜻이다 — 원인이 달라 로그를 나눈다(항목 F).
                if self._audio_turn.get("handled"):
                    self.get_logger().info("[A/B] 옛 소리 결과 버림")
                else:
                    self.get_logger().info("[A/B] 소리 경로 결과 없음 — 텍스트 경로가 처리")
                self._audio_turn = {}
            # (audio 모드인데 소리 경로가 실패했거나 소리가 오지 않았으면 여기로 내려와 지금처럼 처리한다)
            # 1-1) LLM 응답까지는 수 초가 걸린다. 그동안 침묵하면 눈으로 확인할 수
            #      없는 사용자는 로봇이 들었는지 알 수 없다. "확인할게요" 같은
            #      말 대신 배경 운율을 응답이 나올 때까지 반복한다
            #      (2026-09-01 사용자 결정 — 말은 잡담 유입마다 나가 너무 잦았다).
            #      지름길 즉답("그래" 등)은 진짜 답이 0초에 뒤따르므로 생략.
            thinking = not is_instant_utterance(text)
            if thinking:
                self._thinking_pub.publish(Bool(data=True))

            # 2) 일반 발화는 LLM intent 파서로 해석한다 (대화 히스토리 포함 = 멀티턴).
            try:
                intent = parse_intent(
                    text,
                    self._destinations,
                    history=self._history.messages,
                    robot_state=self._robot_state,
                )
            finally:
                # 실패해도 반드시 끈다 — 운율이 혼자 도는 것이 최악이다.
                if thinking:
                    self._thinking_pub.publish(Bool(data=False))
        self._publish_intent(intent, text)
        # C) 소리 경로 이중 처리 방지 — 텍스트가 방금 이 발화를 발행했다는
        # 표식. _on_user_audio 는 이 시각에서 2초 안이면 소리 경로를 생략한다.
        self._last_text_publish_t = time.time()

    def _publish_intent(self, intent: VicaIntent, text: str, llm_first: bool = False) -> None:
        """intent 확정 뒤 공통 후처리 (재청취 기각 → 발행 → TTS → 재청취 준비 → 로그 → 히스토리).

        llm_first(소리 모드 모델 전결): 재청취 기각을 하지 않는다 — 대꾸할지 침묵할지
        (reply 가 빈 문자열)는 모델이 정했다.
        """
        # 2-1) 재청취 창의 무의미 발화는 침묵으로 버린다 — 대꾸도, 기록도
        #      하지 않는다 (멘트 최소주의: 실패·경계는 로그. 못 들은 질문의
        #      재질문은 미션이 유일한 목소리다). 히스토리에 안 남기는 것이
        #      특히 중요하다 — 잡담이 기록에 쌓이면 다음 "그래"가 엉뚱한
        #      목적지로 붙는다 (2026-08-31 야간).
        #      단, **할 말을 들고 온 판정은 통과**시킨다 (2026-09-02). 지름길이
        #      만든 호출 응답("네?")이 intent 이름만 보고 함께 삼켜져 '피카야'가
        #      무응답이 됐다(실기). 판별은 SHORTCUT_REPLIES 로 한다 — LLM 이
        #      지어낸 잡담 대꾸는 이 목록에 없으므로 종전대로 버려진다.
        #      긴급(safety_flag)도 절대 삼키지 않는다 — fail-closed.
        if (not llm_first
                and time.time() < self._followup_until
                and intent.intent in ("unknown", "clarify")
                and not intent.need_confirm
                and intent.reply not in SHORTCUT_REPLIES
                and intent.safety_flag != "emergency"):
            self.get_logger().info(
                f"재청취 기각(무의미): '{text}' intent={intent.intent}")
            return

        # 3) VicaIntent 를 커스텀 메시지로 발행한다 (이동 명령이 아니라 '제안').
        #    resume 제안만 확인 응답("네")까지 보류한다 — should_forward_intent.
        if should_forward_intent(intent):
            self._intent_pub.publish(intent_to_msg(intent))
            self._board.on_intent(intent)
        else:
            self.get_logger().info("resume 확언 대기 — 발행 보류 (질문만 나감)")

        # 3-1) 이 노드가 말해야 하는 응답만 TTS 로 보낸다.
        #      navigate 확정 요청은 Mission Manager 가 게이트 판단 뒤에 말한다.
        request = request_for_intent(intent)
        if request:
            self._tts_pub.publish(String(data=request))

        # 3-2) 지금 한 말이 질문이면 답을 들을 준비를 시킨다. 의도 종류만
        #      보면 LLM 이 자유 생성한 질문(unknown 등)을 놓친다 — 문장 꼴
        #      판정(expects_answer)을 함께 쓴다 (2026-08-28).
        if (intent.need_confirm or intent.intent == "clarify"
                or expects_answer(intent.reply)):
            self._listen_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )

        # 4) 대화 히스토리를 갱신한다 (다음 발화가 맥락을 기억하도록).
        #    소리 모드에서는 로봇 줄(AI)을 여기서 넣지 않는다 — 실제로 소리 난 말이
        #    /vica/tts_done 으로 들어와 _on_tts_done_text 가 넣는다(미션의 질문 포함).
        if self._intent_input == "audio":
            self._history.extend([HumanMessage(text)])
        else:
            self._history.extend([HumanMessage(text), AIMessage(intent.reply)])

    def _on_tts_done_text(self, msg: String) -> None:
        """로봇이 방금 한 말을 기억한다(에코 대조용, 웨이크워드 노드와 같은 방어).

        소리 모드(모델 전결)에서는 이력에도 AI 줄로 넣는다 — 미션이 말한 질문
        ("몇 분쯤 걸리실까요?")을 모델이 봐야 "오 분"을 그 답으로 읽는다
        (09-20 실기 #7: 두 경로 모두 이 질문을 못 본 채 해석해 실패).
        """
        now = time.time()
        self._robot_recent = [
            (t, s) for t, s in self._robot_recent if now - t < ROBOT_ECHO_TTL_SEC]
        self._robot_recent.append((now, msg.data))
        text = (msg.data or "").strip()
        if self._intent_input == "audio" and text:
            self._history.extend([AIMessage(text)])

    def _on_user_audio(self, msg: UInt8MultiArray) -> None:
        """audio 모드: 소리를 Realtime 에 보내 의도를 받는다. 관문에 걸리거나 실패하면
        뒤따라 오는 텍스트가 처리한다(handled 는 self._audio_turn 에 그대로 False 로 남는다)."""
        if self._intent_input != "audio":
            return
        # C) 이중 처리 방지 — 텍스트 경로가 이 발화를 방금(2초 안) 이미
        #    발행했으면 소리 경로는 아무것도 남기지 않고 그냥 생략한다.
        if time.time() - self._last_text_publish_t < 2.0:
            self.get_logger().info("[A/B] 텍스트가 먼저 처리됨 — 소리 경로 생략")
            return
        turn_started = time.time()
        self._audio_turn = {"handled": False, "t": turn_started}
        # 텍스트 백엔드가 LOCAL(클라우드 대피 중)이어도 소리 경로는 따로 시도한다 —
        # 18:54 실기: 그림자 텍스트 경로의 gpt 호출이 한 번 연결 오류를 내자 LOCAL 로
        # 넘어가며 멀쩡한 Realtime 까지 그 주행 내내 꺼졌다. Realtime 이 실제로 실패하면
        # 예외로 돌아와 아래에서 텍스트 경로에 넘긴다(그 발화는 whisper+로컬이 처리).
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
        # H) 비정상적으로 긴 클립은 소리 경로를 생략한다 — Realtime 왕복
        #    비용·지연만 늘고, 텍스트 경로가 어차피 처리한다.
        clip_sec = len(pcm) / 2 / 16000
        if clip_sec > AUDIO_CLIP_MAX_SEC:
            self.get_logger().info(f"[A/B] 클립 {clip_sec:.1f}s — 너무 길어 소리 경로 생략")
            return
        self._reload_destinations_if_changed()
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")
        # A) 이력 스냅샷은 parse_intent_audio 호출 '직전'에 찍어 turn 에 담는다.
        #    audio 경로가 whisper 텍스트 경로보다 느리면, 그 사이 텍스트 경로가
        #    history 에 [Human(heard), AI(reply)] 를 먼저 쌓을 수 있다 — 그 뒤의
        #    self._history.messages 로 그림자 비교를 하면 지금 발화를 '한 턴
        #    미래'의 문맥으로 다시 해석하게 된다. _shadow_text 는 이 스냅샷만 쓴다.
        history_snapshot = self._history.messages
        robot_state = self._robot_state
        self._audio_turn.update(history=history_snapshot, robot_state=robot_state)
        self._thinking_pub.publish(Bool(data=True))
        call_started = time.monotonic()
        try:
            intent, heard, dt, info = parse_intent_audio(
                pcm, self._destinations, history=history_snapshot, robot_state=robot_state,
                situation=self._board.render())
        except Exception as exc:
            self.get_logger().warning(
                f"[A/B] 소리 경로 실패({type(exc).__name__}: {exc}) "
                f"dt={time.monotonic() - call_started:.2f}s — 텍스트 경로가 처리")
            return
        finally:
            self._thinking_pub.publish(Bool(data=False))

        # D) 긴급어는 텍스트 경로(_on_user_text -> detect_emergency)에 맡긴다 —
        #    검증된 안전 경로 하나만 쓴다(fail-closed, 우회 금지).
        if detect_emergency(heard):
            self.get_logger().warning(f"[A/B] 긴급어 — 텍스트 경로에 위임: heard='{heard}'")
            return

        # Realtime 이 느린 사이(예: 8초 대기) 텍스트 경로가 같은 발화를 먼저 발행했으면
        # 여기서 접는다 — 같은 말에 의도가 두 번 나가면 미션이 두 번 움직인다.
        if self._last_text_publish_t > turn_started:
            self.get_logger().info(f"[A/B] 텍스트가 먼저 발행 — 소리 결과 버림: heard='{heard}'")
            return

        # 모델 전결(2026-09-20): 빈 말·환각·에코 판정도 모델 몫이다 — 사람 말이
        # 아니면 모델이 unknown + 빈 reply 로 답하고, 그건 침묵으로 발행된다.
        # 텍스트 경로가 이 발화를 다시 처리하지 않도록 handled 로 표시한다.
        self._audio_turn.update(handled=True, intent=intent, heard=heard, dt=dt)
        usage = info.get("usage") or {}
        audio_tok = usage.get("audio_tokens", 0)
        text_tok = usage.get("text_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        self.get_logger().info(
            f"[RT] heard='{heard}' intent={intent.intent} dest={intent.matched_destination_id or '-'} "
            f"nc={intent.need_confirm} conf={intent.confidence:.2f} reply='{intent.reply[:30]}' "
            f"src={info['src']} dt={dt:.2f}s tokens={audio_tok}/{text_tok}/{out_tok} clip={clip_sec:.2f}s")
        self._publish_intent(intent, heard, llm_first=True)

    def _shadow_text(self, text: str, turn: dict) -> None:
        """audio 모드에서 같은 발화의 텍스트 경로 결과를 로그로만 남긴다(발행 안 함).

        history/robot_state 는 turn 이 소리 경로 호출 '직전'에 찍어 둔 스냅샷을
        쓴다(항목 A) — turn 에 없을 때만(구조상 거의 없다) 지금 값으로 대신한다.
        """
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
            a = turn.get("intent")
            audio_part = (f"{a.intent}/{a.matched_destination_id or '-'} {turn.get('dt', 0.0):.2f}s"
                          if a is not None else "?")
            self.get_logger().info(
                f"[A/B] audio={audio_part} | text={text_part} | heard='{turn.get('heard', '')}' | whisper='{text}'")

        threading.Thread(target=work, daemon=True, name="ab-shadow").start()

    def _reload_destinations_if_changed(self, force: bool = False) -> None:
        """저장 노드가 YAML을 교체하면 다음 발화 전에 public catalog를 갱신한다."""
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
            loaded = load_destinations(self._destinations_path)
        except Exception as exc:
            self.get_logger().error(
                f"목적지 catalog reload 실패, 이전 목록 유지: {exc}"
            )
            return
        self._destinations = [
            destination
            for destination in loaded
            if destination.authorization == "public"
        ]
        self._destinations_mtime_ns = mtime_ns
        self.get_logger().info(
            f"public 목적지 {len(self._destinations)}개 로드: "
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
        if rclpy.ok():  # 이미 종료된 컨텍스트에 shutdown 을 또 부르지 않는다.
            rclpy.shutdown()


if __name__ == "__main__":
    main()
