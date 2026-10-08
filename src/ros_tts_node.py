"""VICA TTS ROS2 노드.

구독: /vica/tts_request (std_msgs/String, "{priority}:{text}")
        Mission Manager 의 안내 멘트(안내 시작·도착·거부 사유 등)와 LLM 응답이
        모두 이 하나의 입구로 들어온다. 이 노드는 무엇을 말할지 판단하지 않고,
        들어온 순서와 우선순위대로 재생만 한다.
      /vica/tts_stop    (std_msgs/Empty) - barge-in: 하던 말 즉시 중단 +
        대기 중 비긴급 발화 폐기. 웨이크워드 노드가 재생 중 호출·긴급·질문
        답변을 감지했을 때 보낸다. 긴급 발화는 큐에 남는다.
발행: /vica/tts_state   (std_msgs/Bool) - 재생 중 여부
      /vica/tts_now     (std_msgs/String) - 지금 재생을 시작하는 문장(2026-10-08). 귀 노드가
        '비카야'가 든 로봇 말에 스스로 호출되지 않게 막는 데 쓴다.
      /vica/tts_done    (std_msgs/String) - 한 발화가 **끝난 시점**(완주든
        중단이든)에 그 문장을 발행한다. Mission 이 질문의 응답 대기(8초)를
        "발화 종료 시점"부터 세는 근거다. 예전엔 완주만 알렸는데, 그러면
        barge-in·긴급 선점으로 끊긴 질문은 시계가 영영 시작되지 않아
        ASKING 상태에 영구 정지했다(2026-08-31 근본 수리). 끊김 = 사용자가
        끼어들었거나(답하는 중 — 귀 홀드가 시계를 잡아줌) 긴급 선점(상태가
        어차피 떠남)이라, 종료를 알리는 것이 항상 옳다.

/vica/tts_state 를 두는 이유:
    긴급어 상시 감시(ros_emergency_node)는 마이크를 계속 열어 두므로 스피커로 나간
    로봇 자기 목소리도 듣는다. 멘트에 "멈춰"/"정지" 가 없어도 목적지 이름 같은 데서
    걸릴 수 있어(예: "행정지원실" 안의 "정지"), 재생 중에는 감시를 쉬게 한다.

    재생 구간을 짧게 유지하려고 문장 단위로 끊어 재생하고, 문장 사이마다 감시를
    다시 연다. 사용자의 진짜 긴급 발화를 놓치는 창을 줄이기 위해서다.

재생은 별도 스레드에서 돈다. 구독 콜백에서 직접 재생하면 재생이 끝날 때까지
다음 요청을 받지 못해, 긴급 발화가 앞선 안내 뒤에 줄을 서게 된다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_tts_node
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from vica_interfaces.msg import RobotState as RobotStateMsg

from . import audio_cue
from .approach_chime import ApproachChime
from .destination_loader import load_destinations
from .ment_cache import MentCache
from .mission_phrases import ARRIVAL_QUESTIONS, merged_prewarm, standalone_prewarm
from .synth_cache import SynthCache
from .tts import VicaTTS
from .tts_queue import AMBIENT, TtsQueue, parse_request
from .tts_text import split_sentences

# 재생이 끝난 뒤 감시를 다시 열기까지의 여유. 스피커 잔향과 마이크 입력 지연 때문에
# 0 으로 두면 방금 끝난 로봇 목소리를 그대로 다시 듣는다.
TAIL_SEC = 0.4

# 큐가 비었을 때 재생 스레드가 쉬는 간격.
IDLE_POLL_SEC = 0.05

# "생각 중" 배경 운율 (2026-09-01 — "확인할게요" 발화 대체). 정지 신호를
# 놓쳐도 영원히 돌지 않게 시한을 두고, 짧은 조각으로 잘라 재생해 응답
# 발화가 오면 한 조각(0.2초) 안에 자리를 내준다. 스피커는 이 노드의
# persistent stream 단일 출구라 루프도 반드시 여기서 튼다 — 다른 노드가
# 틀면 응답 발화가 장치를 기다리게 된다.
THINKING_MAX_SEC = 20.0
THINKING_SLICE_SEC = 0.2


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        # 고정 멘트는 합성 대신 구워 둔 녹음을 즉시 재생한다 (ment_cache 정본).
        self._ments = MentCache()
        if self._ments.missing:
            self.get_logger().warn(
                f"녹음 없는 고정 멘트(합성 폴백): {', '.join(self._ments.missing)}"
                " — scripts/make_cue_wavs.py 로 굽는다")
        self._queue = TtsQueue()
        self._preempt = threading.Event()
        self._running = True
        # 지금 재생 중인 말의 등급(없으면 None). 배경 알림을 버리거나 끊는 판단에 쓴다.
        self._playing_priority = None
        # 합성 결과 캐시 + 합성 직렬화 잠금 (워밍업 스레드와 재생 스레드가
        # 동시에 모델을 부르지 않게). 자주 나오는 고정 문장은 기동 시 미리
        # 합성해 첫 사용부터 0초로 만든다.
        self._synth_cache = SynthCache()
        self._synth_lock = threading.Lock()
        # 미리 합성할 목적지 파일. launch 가 LLM·웨이크워드 노드와 같은 값을 넘긴다
        # (2026-10-07). 예전엔 환경변수가 없으면 vica_map_0630 고정이라 다른 지도를 띄워도
        # 옛 지도 문장을 데웠다 — 도착 M1 을 목적지마다 미리 만들기로 해 맞아야 한다.
        self.declare_parameter(
            "destinations_yaml",
            str(Path.home() / "vica_data" / "destinations" / "vica_map_0630"
                / "destinations.yaml"),
        )
        threading.Thread(target=self._prewarm_synth, daemon=True).start()

        self._state_pub = self.create_publisher(Bool, "/vica/tts_state", 10)
        self._done_pub = self.create_publisher(String, "/vica/tts_done", 10)
        self._now_pub = self.create_publisher(String, "/vica/tts_now", 10)
        self._publish_state(False)  # 시작 상태를 명시적으로 알린다

        self.create_subscription(String, "/vica/tts_request", self._on_request, 10)
        self.create_subscription(Empty, "/vica/tts_stop", self._on_stop, 10)
        # "생각 중" 운율 스위치 — LLM 노드가 해석 시작/끝에 켜고 끈다.
        # 재생 중에도 tts_state 는 켜지 않는다(효과음과 같은 원칙 — 낮은
        # 음량의 순음이라 긴급 감시를 쉬게 할 이유가 없다).
        self._thinking_until = 0.0
        self._thinking_pos = 0
        self._thinking_wave = audio_cue.thinking_loop()
        self.create_subscription(Bool, "/vica/thinking", self._on_thinking, 10)
        # 사람 접근 차임 (2026-10-07 사용자 결정): 미션이 시각장애인에게 다가가는
        # 중(dialog_state "approaching")이라 알리는 동안 2초마다 종 두 음 "딩—동↗"
        # 으로 위치를 알린다. 첫 인사("동행로봇 비카가 다가가고 있어요")는 미션이
        # 말한다. 말이 우선이고, 생각 중 운율처럼 tts_state 는 켜지 않는다(감시 유지).
        chime_on = os.environ.get("VICA_APPROACH_CHIME", "on").strip().lower() not in (
            "off", "0", "false")
        self._chime = ApproachChime(enabled=chime_on)
        self._chime_wave = audio_cue.approach_chime()
        try:
            # 순음·종소리는 같은 최고점이라도 말보다 크게 들린다 — 말(-3)보다 낮춘다.
            self._chime_dbfs = float(os.environ.get("VICA_APPROACH_CHIME_DBFS", "-12"))
        except ValueError:
            self._chime_dbfs = -12.0
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/tts_request | 발행: /vica/tts_state)"
        )

    # -- 입력 ----------------------------------------------------------------

    def _on_request(self, msg: String) -> None:
        if msg.data == "control:stop":
            # 같은 토픽의 청소 명령 (2026-09-01): 청소와 뒤이어 시킬 새 발화
            # ("네?"·"안내를 시작합니다")의 처리 순서를 보장한다 — 별도
            # 토픽(tts_stop)으로 보내면 도착 순서가 이따금 뒤집혀 청소가
            # 자기가 시킨 말을 지웠다 (실기: "비카야" 후 "네?" 증발 3회).
            self._do_stop()
            return
        priority, text = parse_request(msg.data)
        self._enqueue(priority, text)

    def _enqueue(self, priority: str, text: str) -> None:
        if not text:
            return
        now = time.time()
        # 배경 알림(ambient, 대기 중 10초 알림)은 다른 말이 나가는 중이면 버린다 — 생각 중
        # 운율도 대화 중이라는 뜻이라 같이 본다(2026-10-07, 작업 계획 탭 M3).
        busy = self._playing_priority is not None or now < self._thinking_until
        result = self._queue.push(priority, text, now=now, busy=busy)
        if (result.accepted and priority != AMBIENT
                and self._playing_priority == AMBIENT):
            # 재생 중인 배경 알림은 할 말에 바로 비킨다.
            self._preempt.set()
            self._tts.stop()
            self.get_logger().info(f"배경 알림 중단 — 할 말 우선: {text}")

        # 사라진 말은 반드시 남긴다. 조용히 버리면 "왜 그 안내가 안 나왔는지"를
        # 사후에 추적할 수 없다 (docs/voice-improvement-backlog.md 3절).
        if not result.accepted:
            self.get_logger().warn(f"발화 무시({result.reason}): {text}")
            return
        if result.dropped:
            why = "긴급 선점" if result.preempt else "큐 정원 초과·배경 알림 비킴"
            for lost in result.dropped:
                self.get_logger().warn(f"발화 폐기({why}): {lost}")

        if result.preempt:
            self._preempt.set()
            self._tts.stop()
            self.get_logger().warn(f"긴급 발화로 선점: {text}")
        else:
            self.get_logger().info(f"발화 대기[{priority}]: {text}")

    def _on_thinking(self, msg: Bool) -> None:
        if msg.data:
            self._thinking_pos = 0
            self._thinking_until = time.time() + THINKING_MAX_SEC
        else:
            # 끄기만 한다 — stop() 은 부르지 않는다. 응답 발화가 이미
            # 시작됐을 수 있고, 그걸 끊으면 안 된다. 루프는 다음 조각
            # 경계(0.2초)에서 스스로 멈춘다.
            self._thinking_until = 0.0

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        self._chime.on_robot_state(msg.dialog_state, time.time())

    def _play_chime(self) -> None:
        """사람 접근 차임 한 번. 할 말이 들어오거나 선점되면 그 자리에서 비킨다."""
        self._chime.played(time.time())
        self._tts.play_audio(
            self._chime_wave, audio_cue.SAMPLE_RATE,
            should_stop=lambda: self._preempt.is_set() or len(self._queue) > 0,
            peak_dbfs=self._chime_dbfs)

    def _play_thinking_slice(self) -> None:
        """운율 한 조각(0.2초)을 재생한다. 조각 사이마다 큐를 다시 본다."""
        wave = self._thinking_wave
        n = int(THINKING_SLICE_SEC * audio_cue.SAMPLE_RATE)
        i = self._thinking_pos
        chunk = wave[i:i + n]
        if len(chunk) < n:
            import numpy as np
            chunk = np.concatenate([chunk, wave[:n - len(chunk)]])
        self._thinking_pos = (i + n) % len(wave)
        self._tts.play_audio(chunk, audio_cue.SAMPLE_RATE)

    def _on_stop(self, _msg: Empty) -> None:
        self._do_stop()

    def _do_stop(self) -> None:
        """barge-in·청소 — 하던 말을 즉시 끊고 대기 발화를 버린다.

        일반 대기 발화도 함께 버린다: 대화가 시작된 뒤에 옛 안내가 이어지면
        사용자가 현재 상태를 오해한다 (큐의 신선도 원칙과 동일). 긴급은 남긴다.
        """
        dropped = self._queue.drop_pending(keep_emergency=True)
        self._preempt.set()
        self._tts.stop()
        for lost in dropped:
            self.get_logger().warn(f"발화 폐기(barge-in): {lost}")
        self.get_logger().info("barge-in — 재생 중단")

    # -- 재생 ----------------------------------------------------------------

    def _publish_state(self, speaking: bool) -> None:
        msg = Bool()
        msg.data = speaking
        self._state_pub.publish(msg)

    def _playback_loop(self) -> None:
        while self._running:
            # 이전 선점 신호는 다음 말을 꺼내기 **전에** 소비한다 — 꺼낸 뒤에
            # 지우면 그 사이에 온 barge-in("비카야")이 지워져 말이 끝까지
            # 나간다(2026-10-06 실기: 예고 4.8초 완주, "네?" 4.7초 지연).
            # 꺼낸 뒤에 온 선점은 _speak → play(should_stop) 가 본다.
            self._preempt.clear()
            item = self._queue.pop()
            for stale in self._queue.take_expired():
                # 낡아서 버린 말도 반드시 남긴다 — 조용히 사라지면 추적 불가.
                self.get_logger().info(f"발화 만료 폐기: {stale}")
            if item is None:
                if time.time() < self._thinking_until:
                    self._play_thinking_slice()
                elif self._chime.due(time.time()):
                    self._play_chime()
                else:
                    time.sleep(IDLE_POLL_SEC)
                continue

            self.get_logger().info(f"재생[{item.priority}]: {item.text}")
            self._playing_priority = item.priority
            try:
                completed = self._speak(item.text)
            finally:
                self._playing_priority = None
            self._chime.on_speech_end(time.time())   # 말 직후 한숨 쉬고 차임
            # 완주·중단 불문 종료를 알린다 — 응답 시계의 기점 (docstring).
            done = String()
            done.data = item.text
            self._done_pub.publish(done)
            if not completed:
                self.get_logger().info("발화 중단 — 종료 신호는 발행 (시계 기점)")

    def _speak(self, text: str) -> bool:
        """재생하고, 끊기지 않고 끝까지 갔으면 True 를 돌려준다.

        고정 멘트(캐시 적중)는 합성 없이 통짜로 재생한다 — 문장 사이 감시
        열기가 없어지지만, 표준 배선(AEC 감시 유지)에서는 재생 중에도 감시가
        계속되므로 공백이 아니다. 캐시에 없으면 기존 문장 단위 합성 경로다.
        """
        cached = self._ments.lookup(text)
        if cached is not None:
            wav, rate = cached
            self._now_pub.publish(String(data=text))
            self._publish_state(True)
            try:
                self._tts.play_audio(wav, rate, should_stop=self._preempt.is_set)
            finally:
                time.sleep(TAIL_SEC)
                self._publish_state(False)
            return not self._preempt.is_set()

        for chunk in split_sentences(text):
            if self._preempt.is_set():
                return False
            self._now_pub.publish(String(data=chunk))
            self._publish_state(True)
            try:
                hit = self._synth_cache.get(chunk)
                if hit is None:
                    with self._synth_lock:
                        wav, rate = self._tts.synthesize(chunk)
                    self._synth_cache.put(chunk, wav, rate)
                    hit = (wav, rate)
                self._tts.play_audio(*hit, should_stop=self._preempt.is_set)
            finally:
                # 재생이 실패해도 감시는 반드시 다시 열어야 한다.
                time.sleep(TAIL_SEC)
                self._publish_state(False)
        return not self._preempt.is_set()

    def _prewarm_synth(self) -> None:
        """자주 나오는 고정 문장을 미리 합성한다 (접수 멘트·확인 질문·도착 멘트).

        확인 질문은 목적지별 고정 문장인데 매번 합성해 첫 응답이 늦었다
        (2026-08-28 실측 0.9초대). 실패해도 재생 경로가 그때그때 합성한다.
        """
        # 접수 신호("확인할게요" 풀)는 2026-09-01 배경 운율로 대체돼 뺐다.
        # 도착 발화(도착 멘트 + M1 입구 방향 + 질문)는 미션이 한 발화로 보내고 _speak 는
        # 통문장 녹음이 없으면 문장마다 합성 보관함만 본다 — 그래서 그 문장들은 따로 구워져
        # 있어도 문장 단위로 데운다(2026-10-07). 데운 문장은 고정해 LRU 에 밀리지 않게 한다.
        standalone: list[str] = []
        merged: list[str] = []
        yaml_path = ""
        try:
            # 로봇 지도의 목적지 파일 — launch 가 LLM·웨이크워드 노드와 같은 값을 넘긴다.
            yaml_path = os.environ.get(
                "VICA_DESTINATIONS_YAML",
                str(self.get_parameter("destinations_yaml").value))
            dests = load_destinations(yaml_path)
            standalone = standalone_prewarm(dests)
            merged = merged_prewarm(dests)
        except Exception as exc:
            self.get_logger().warn(f"목적지 멘트 워밍업 생략: {exc}")
            merged = list(ARRIVAL_QUESTIONS)
        # 혼자 말해지는 문장은 통문장 녹음이 있으면 건너뛴다(_speak 가 녹음을 튼다).
        chunks: list[str] = []
        for phrase in standalone:
            if not self._ments.lookup(phrase):
                chunks += split_sentences(phrase)
        for phrase in merged:
            chunks += split_sentences(phrase)
        started = time.monotonic()
        count = 0
        for chunk in chunks:
            if not self._running:
                return
            if self._synth_cache.get(chunk):
                continue
            try:
                with self._synth_lock:
                    wav, rate = self._tts.synthesize(chunk)
                self._synth_cache.put(chunk, wav, rate, pinned=True)
                count += 1
            except Exception as exc:
                self.get_logger().warn(f"워밍업 합성 실패({chunk!r}): {exc}")
                return
        self.get_logger().info(
            f"고정 문장 워밍업 완료({yaml_path}): {count}건, "
            f"{time.monotonic() - started:.1f}초")

    def shutdown(self) -> None:
        self._running = False
        self._tts.stop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TtsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
