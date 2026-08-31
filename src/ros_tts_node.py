"""VICA TTS ROS2 노드.

구독: /vica/tts_request (std_msgs/String, "{priority}:{text}")
        Mission Manager 의 안내 멘트(안내 시작·도착·거부 사유 등)와 LLM 응답이
        모두 이 하나의 입구로 들어온다. 이 노드는 무엇을 말할지 판단하지 않고,
        들어온 순서와 우선순위대로 재생만 한다.
      /vica/tts_stop    (std_msgs/Empty) - barge-in: 하던 말 즉시 중단 +
        대기 중 비긴급 발화 폐기. 웨이크워드 노드가 재생 중 호출·긴급·질문
        답변을 감지했을 때 보낸다. 긴급 발화는 큐에 남는다.
발행: /vica/tts_state   (std_msgs/Bool) - 재생 중 여부
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

from .destination_loader import load_destinations
from .ment_cache import MentCache
from .replies import ACK_LISTENING_POOL
from .synth_cache import SynthCache
from .tts import VicaTTS
from .tts_queue import TtsQueue, parse_request
from .tts_text import split_sentences

# 재생이 끝난 뒤 감시를 다시 열기까지의 여유. 스피커 잔향과 마이크 입력 지연 때문에
# 0 으로 두면 방금 끝난 로봇 목소리를 그대로 다시 듣는다.
TAIL_SEC = 0.4

# 큐가 비었을 때 재생 스레드가 쉬는 간격.
IDLE_POLL_SEC = 0.05


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
        # 합성 결과 캐시 + 합성 직렬화 잠금 (워밍업 스레드와 재생 스레드가
        # 동시에 모델을 부르지 않게). 자주 나오는 고정 문장은 기동 시 미리
        # 합성해 첫 사용부터 0초로 만든다.
        self._synth_cache = SynthCache()
        self._synth_lock = threading.Lock()
        threading.Thread(target=self._prewarm_synth, daemon=True).start()

        self._state_pub = self.create_publisher(Bool, "/vica/tts_state", 10)
        self._done_pub = self.create_publisher(String, "/vica/tts_done", 10)
        self._publish_state(False)  # 시작 상태를 명시적으로 알린다

        self.create_subscription(String, "/vica/tts_request", self._on_request, 10)
        self.create_subscription(Empty, "/vica/tts_stop", self._on_stop, 10)

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/tts_request | 발행: /vica/tts_state)"
        )

    # -- 입력 ----------------------------------------------------------------

    def _on_request(self, msg: String) -> None:
        priority, text = parse_request(msg.data)
        self._enqueue(priority, text)

    def _enqueue(self, priority: str, text: str) -> None:
        if not text:
            return
        result = self._queue.push(priority, text, now=time.time())

        # 사라진 말은 반드시 남긴다. 조용히 버리면 "왜 그 안내가 안 나왔는지"를
        # 사후에 추적할 수 없다 (docs/voice-improvement-backlog.md 3절).
        if not result.accepted:
            self.get_logger().warn(f"발화 무시({result.reason}): {text}")
            return
        if result.dropped:
            why = "긴급 선점" if result.preempt else "큐 정원 초과"
            for lost in result.dropped:
                self.get_logger().warn(f"발화 폐기({why}): {lost}")

        if result.preempt:
            self._preempt.set()
            self._tts.stop()
            self.get_logger().warn(f"긴급 발화로 선점: {text}")
        else:
            self.get_logger().info(f"발화 대기[{priority}]: {text}")

    def _on_stop(self, _msg: Empty) -> None:
        """barge-in — 사용자가 말을 시작했으니 하던 말을 즉시 끊는다.

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
            item = self._queue.pop()
            for stale in self._queue.take_expired():
                # 낡아서 버린 말도 반드시 남긴다 — 조용히 사라지면 추적 불가.
                self.get_logger().info(f"발화 만료 폐기: {stale}")
            if item is None:
                time.sleep(IDLE_POLL_SEC)
                continue

            # 이전 선점 신호는 여기서 소비한다 — 새 발화까지 끊기면 안 된다.
            self._preempt.clear()
            self.get_logger().info(f"재생[{item.priority}]: {item.text}")
            completed = self._speak(item.text)
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
            self._publish_state(True)
            try:
                self._tts.play_audio(wav, rate)
            finally:
                time.sleep(TAIL_SEC)
                self._publish_state(False)
            return not self._preempt.is_set()

        for chunk in split_sentences(text):
            if self._preempt.is_set():
                return False
            self._publish_state(True)
            try:
                hit = self._synth_cache.get(chunk)
                if hit is None:
                    with self._synth_lock:
                        wav, rate = self._tts.synthesize(chunk)
                    self._synth_cache.put(chunk, wav, rate)
                    hit = (wav, rate)
                self._tts.play_audio(*hit)
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
        phrases: list[str] = list(ACK_LISTENING_POOL)
        try:
            # 로봇 지도의 목적지 파일 — wakeword 노드의 장소 귀띔과 같은 경로
            yaml_path = os.environ.get(
                "VICA_DESTINATIONS_YAML",
                str(Path.home() / "vica_data" / "destinations" / "vica_map_0630"
                    / "destinations.yaml"))
            for dest in load_destinations(yaml_path):
                phrases += [dest.confirm_prompt, dest.arrival_message]
        except Exception as exc:
            self.get_logger().warn(f"목적지 멘트 워밍업 생략: {exc}")
        started = time.monotonic()
        count = 0
        for phrase in phrases:
            for chunk in split_sentences(phrase):
                if not self._running:
                    return
                if self._ments.lookup(chunk) or self._synth_cache.get(chunk):
                    continue
                try:
                    with self._synth_lock:
                        wav, rate = self._tts.synthesize(chunk)
                    self._synth_cache.put(chunk, wav, rate)
                    count += 1
                except Exception as exc:
                    self.get_logger().warn(f"워밍업 합성 실패({chunk!r}): {exc}")
                    return
        self.get_logger().info(
            f"고정 문장 워밍업 완료: {count}건, {time.monotonic() - started:.1f}초")

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
