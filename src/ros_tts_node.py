"""VICA TTS ROS2 노드."""
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

TAIL_SEC = 0.4

IDLE_POLL_SEC = 0.05

THINKING_MAX_SEC = 20.0
THINKING_SLICE_SEC = 0.2


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        self._ments = MentCache()
        if self._ments.missing:
            self.get_logger().warn(
                f"녹음 없는 고정 멘트(합성 폴백): {', '.join(self._ments.missing)}"
                " — scripts/make_cue_wavs.py 로 굽는다")
        self._queue = TtsQueue()
        self._preempt = threading.Event()
        self._running = True
        self._playing_priority = None
        self._synth_cache = SynthCache()
        self._synth_lock = threading.Lock()
        self.declare_parameter(
            "destinations_yaml",
            str(Path.home() / "vica_data" / "destinations" / "vica_map_0630"
                / "destinations.yaml"),
        )
        threading.Thread(target=self._prewarm_synth, daemon=True).start()

        self._state_pub = self.create_publisher(Bool, "/vica/tts_state", 10)
        self._done_pub = self.create_publisher(String, "/vica/tts_done", 10)
        self._now_pub = self.create_publisher(String, "/vica/tts_now", 10)
        self._publish_state(False)

        self.create_subscription(String, "/vica/tts_request", self._on_request, 10)
        self.create_subscription(Empty, "/vica/tts_stop", self._on_stop, 10)
        self._thinking_until = 0.0
        self._thinking_pos = 0
        self._thinking_wave = audio_cue.thinking_loop()
        self.create_subscription(Bool, "/vica/thinking", self._on_thinking, 10)
        chime_on = os.environ.get("VICA_APPROACH_CHIME", "on").strip().lower() not in (
            "off", "0", "false")
        self._chime = ApproachChime(enabled=chime_on)
        self._chime_wave = audio_cue.approach_chime()
        try:
            self._chime_dbfs = float(os.environ.get("VICA_APPROACH_CHIME_DBFS", "-12"))
        except ValueError:
            self._chime_dbfs = -12.0
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/tts_request | 발행: /vica/tts_state)"
        )

    def _on_request(self, msg: String) -> None:
        if msg.data == "control:stop":
            self._do_stop()
            return
        priority, text = parse_request(msg.data)
        self._enqueue(priority, text)

    def _enqueue(self, priority: str, text: str) -> None:
        if not text:
            return
        now = time.time()
        busy = self._playing_priority is not None or now < self._thinking_until
        result = self._queue.push(priority, text, now=now, busy=busy)
        if (result.accepted and priority != AMBIENT
                and self._playing_priority == AMBIENT):
            self._preempt.set()
            self._tts.stop()
            self.get_logger().info(f"배경 알림 중단 — 할 말 우선: {text}")

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
        """barge-in·청소 — 하던 말을 즉시 끊고 대기 발화를 버린다."""
        dropped = self._queue.drop_pending(keep_emergency=True)
        self._preempt.set()
        self._tts.stop()
        for lost in dropped:
            self.get_logger().warn(f"발화 폐기(barge-in): {lost}")
        self.get_logger().info("barge-in — 재생 중단")

    def _publish_state(self, speaking: bool) -> None:
        msg = Bool()
        msg.data = speaking
        self._state_pub.publish(msg)

    def _playback_loop(self) -> None:
        while self._running:
            self._preempt.clear()
            item = self._queue.pop()
            for stale in self._queue.take_expired():
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
            self._chime.on_speech_end(time.time())
            done = String()
            done.data = item.text
            self._done_pub.publish(done)
            if not completed:
                self.get_logger().info("발화 중단 — 종료 신호는 발행 (시계 기점)")

    def _speak(self, text: str) -> bool:
        """재생하고, 끊기지 않고 끝까지 갔으면 True 를 돌려준다."""
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
                time.sleep(TAIL_SEC)
                self._publish_state(False)
        return not self._preempt.is_set()

    def _prewarm_synth(self) -> None:
        """자주 나오는 고정 문장을 미리 합성한다 (접수 멘트·확인 질문·도착 멘트)."""
        standalone: list[str] = []
        merged: list[str] = []
        yaml_path = ""
        try:
            yaml_path = os.environ.get(
                "VICA_DESTINATIONS_YAML",
                str(self.get_parameter("destinations_yaml").value))
            dests = load_destinations(yaml_path)
            standalone = standalone_prewarm(dests)
            merged = merged_prewarm(dests)
        except Exception as exc:
            self.get_logger().warn(f"목적지 멘트 워밍업 생략: {exc}")
            merged = list(ARRIVAL_QUESTIONS)
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
