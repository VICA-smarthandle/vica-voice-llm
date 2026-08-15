"""웨이크워드 상시 감시 엔진 (P1-b) — 호출(모델 A) + 긴급(모델 B → STT 검증).

기존 EmergencyMonitor(whisper 상시, RTF 0.59)를 대체한다. openWakeWord 두 모델이
전처리를 공유하며 상시로 돌고(RTF 0.12), whisper 는 필요할 때만 부른다:

  80ms 프레임 → 모델 A·B 점수 (전처리 공유)
    ├─ B 관문(0.5×2프레임): 0.3초 더 듣고 → whisper → 정확 매칭
    │    → 통과 시 on_emergency(EmergencyEvent)  [긴급이 항상 우선]
    └─ A 관문(0.6×2프레임): on_wake(응답음) → 청취 창(발화 끝 감지, 최대 6초)
         → whisper → on_user_text(문장)          [이후는 기존 LLM 흐름]

EmergencyMonitor 의 검증된 운영 장치를 계승한다: TTS 재생 중 감시 억제(set_muted),
해제 시 버퍼 비우기, mute fail-safe 타임아웃, 이벤트 쿨다운.

predict / transcribe 를 주입식으로 받아 마이크·모델 없이 단위 테스트할 수 있다
(EmergencyMonitor.process_window 와 같은 패턴).

실측 근거·임계값 출처: vica-wakeword/docs/stt-gate-findings.md (잠정값).

안전 원칙: 이 모듈은 감지까지만 한다. 정지의 결정·실행은 Safety Supervisor /
State Machine 이 한다. /cmd_vel*, Nav2 goal, CAN 은 어디에도 없다.

CLI 데모:
    python -m src.wakeword_monitor        # Ctrl+C 종료
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from .schema import EmergencyEvent
from .wakeword_gate import FrameGate, match_emergency_transcript

SAMPLE_RATE = 16000
FRAME = 1280                    # 80ms — openWakeWord 계약
RING_FRAMES = 31                # 검증에 쓰는 직전 오디오 ≈ 2.5초
POST_ROLL_FRAMES = 4            # 긴급 발동 후 말끝 보존 0.32초
LISTEN_MAX_SEC = 6.0            # 호출 후 청취 창 상한
LISTEN_SILENCE_END_SEC = 0.8    # 발화 시작 후 이만큼 조용하면 청취 종료
SPEECH_RMS = 0.01               # 발화 판정 RMS (EmergencyMonitor 게이트와 동일)
# 재청취(arm_followup) 예약의 유효 시간. 질문 TTS 가 유실돼 mute 해제가 안 오면
# 예약이 남아, 한참 뒤 무관한 안내가 끝난 순간 마이크가 열리는 오동작을 막는다.
FOLLOWUP_ARM_TIMEOUT_SEC = 20.0


class WakewordMonitor:
    """호출·긴급 웨이크워드 상시 감시. 상태: idle / postroll / listen."""

    def __init__(
        self,
        on_emergency: Callable[[EmergencyEvent], None],
        on_user_text: Callable[[str], None],
        on_wake: Optional[Callable[[], None]] = None,
        predict: Optional[Callable[[np.ndarray], dict]] = None,
        transcribe: Optional[Callable[[np.ndarray], str]] = None,
        gate_a: float = 0.6,
        gate_b: float = 0.5,
        cooldown_a: float = 1.5,
        cooldown_b: float = 2.0,
    ):
        self._on_emergency = on_emergency
        self._on_user_text = on_user_text
        self._on_wake = on_wake or (lambda: None)
        self._predict = predict          # frame(int16 1280) -> {"a": 점수, "b": 점수}
        self._transcribe = transcribe    # int16 오디오 -> 한국어 텍스트

        self.gate_a = FrameGate(gate_a, persist=2, cooldown_sec=cooldown_a)
        self.gate_b = FrameGate(gate_b, persist=2, cooldown_sec=cooldown_b)

        self._ring: deque[np.ndarray] = deque(maxlen=RING_FRAMES)
        self._state = "idle"
        self._collect: list[np.ndarray] = []   # postroll·listen 수집분
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_is_followup = False       # 이 청취 창이 재청취로 열렸는가
        self._muted_until = 0.0
        self._muted = False
        # 재청취 예약: 로봇이 질문을 말하는 중("~할까요?")이면 노드가 걸어 두고,
        # TTS 가 끝나(mute 해제) 이 예약이 살아 있으면 웨이크워드 없이 청취를 연다.
        # 사용자가 "응" 한마디를 하려고 "비카야"를 다시 부를 필요가 없게 한다.
        self._followup_armed = False
        self._followup_armed_at = 0.0

    # ---------------------------------------------------------------- 재청취
    def arm_followup(self, now: Optional[float] = None) -> None:
        """"방금 질문을 말했다"는 예약. 다음 TTS 종료(mute 해제) 때 청취를 연다.

        질문을 하는 노드(LLM node, Mission Manager)가 /vica/listen_request 로
        알리고, 웨이크워드 노드가 이 메서드를 부른다.
        """
        self._followup_armed = True
        self._followup_armed_at = time.time() if now is None else now

    def disarm_followup(self) -> None:
        self._followup_armed = False

    # ---------------------------------------------------------------- mute
    def set_muted(self, muted: bool, now: Optional[float] = None,
                  failsafe_sec: float = 10.0) -> None:
        """TTS 재생 중 자기 목소리 억제. fail-safe: 해제 신호를 놓쳐도
        failsafe_sec 뒤 자동 해제된다. 해제 시 버퍼를 비운다(잔향 오인 방지)."""
        now = time.time() if now is None else now
        if muted:
            self._muted = True
            self._muted_until = now + failsafe_sec
            # TTS 는 문장마다 상태를 깜빡인다(문장 사이 감시 공백을 줄이는 설계).
            # 여러 문장짜리 질문이면 첫 문장 끝에 열린 재청취가 다음 문장 재생과
            # 겹친다. 그 창은 접고 예약을 되살려, "마지막 문장 끝"에 다시 열리게
            # 한다. (질문 시각 기준의 타임아웃은 유지 — armed_at 은 갱신 안 함)
            if self._state == "listen" and self._listen_is_followup:
                self._state = "idle"
                self._collect = []
                self._followup_armed = True
            return

        self._muted = False
        self._ring.clear()
        self.gate_a.reset()
        self.gate_b.reset()
        if self._followup_armed:
            self._followup_armed = False
            if now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC:
                self._open_listen(followup=True)

    def _is_muted(self, now: float) -> bool:
        if self._muted and now >= self._muted_until:   # fail-safe 타임아웃
            self.set_muted(False, now)
        return self._muted

    # ---------------------------------------------------------------- 핵심 로직
    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> Optional[str]:
        """int16 80ms 프레임 하나를 처리한다. 일어난 일을 문자열로 돌려준다
        (emergency / reject / wake / user_text / wake_silent / None) — 시험용.
        """
        now = time.time() if now is None else now
        self._ring.append(frame)

        if self._is_muted(now):
            self.gate_a.reset()
            self.gate_b.reset()
            return None

        scores = self._predict(frame)
        fire_b = self.gate_b.feed(float(scores["b"]), now)

        if self._state == "postroll":
            self._collect.append(frame)
            if len(self._collect) >= POST_ROLL_FRAMES:
                return self._verify_emergency(now)
            return None

        if self._state == "listen":
            # 청취 중에도 긴급이 절대 우선 (명세 11절)
            if fire_b:
                self._enter_postroll()
                return None
            return self._listen_step(frame, now)

        # idle
        if fire_b:
            self._enter_postroll()
            return None
        if self.gate_a.feed(float(scores["a"]), now):
            self.gate_b.reset()
            self._on_wake()
            self._open_listen(followup=False)
            return "wake"
        return None

    # ---------------------------------------------------------------- 내부
    def _open_listen(self, followup: bool) -> None:
        """청취 창을 연다. followup 이면 웨이크워드 없이(질문 답변용) 연 것이라
        인사(on_wake)를 하지 않는다 — 로봇이 방금 질문을 마쳤기 때문이다."""
        self._state = "listen"
        self._collect = []
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_is_followup = followup

    def _enter_postroll(self) -> None:
        self._state = "postroll"
        self._collect = []
        self.gate_a.reset()

    def _verify_emergency(self, now: float) -> str:
        audio = np.concatenate([*self._ring])   # 직전 ~2.5초 + 말끝
        text = self._transcribe(audio)
        self._state = "idle"
        self._collect = []
        keyword = match_emergency_transcript(text)
        if keyword is None:
            return "reject"
        event = EmergencyEvent(keyword=keyword, source_text=text, detected_at=now)
        self._on_emergency(event)
        return "emergency"

    def _listen_step(self, frame: np.ndarray, now: float) -> Optional[str]:
        self._collect.append(frame)
        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
        if rms >= SPEECH_RMS:
            self._listen_started_speech = True
            self._listen_silence = 0.0
        elif self._listen_started_speech:
            self._listen_silence += FRAME / SAMPLE_RATE

        done = (
            len(self._collect) * FRAME / SAMPLE_RATE >= LISTEN_MAX_SEC
            or (self._listen_started_speech
                and self._listen_silence >= LISTEN_SILENCE_END_SEC)
        )
        if not done:
            return None

        audio = np.concatenate(self._collect)
        self._state = "idle"
        self._collect = []
        if not self._listen_started_speech:
            return "wake_silent"    # 오탐이었음 — 조용히 복귀 (기록은 노드 몫)
        text = self._transcribe(audio).strip()
        if not text:
            return "wake_silent"
        self._on_user_text(text)
        return "user_text"

    # ---------------------------------------------------------------- 실행 (실기)
    def _load_real(self) -> None:
        """실전용 모델 로드 (주입이 없을 때만). 마이크 스레드 전에 1회."""
        if self._predict is None:
            from openwakeword.model import Model

            model_a = os.environ.get(
                "VICA_WAKE_MODEL_A", os.path.join("models", "vica_bikaya_v1.onnx"))
            model_b = os.environ.get(
                "VICA_WAKE_MODEL_B", os.path.join("models", "vica_modelb_v2.onnx"))
            m = Model(wakeword_models=[model_a, model_b], inference_framework="onnx")
            keys = list(m.models.keys())
            key_a = next(k for k in keys if "bikaya" in k)
            key_b = next(k for k in keys if k != key_a)

            def _predict(frame: np.ndarray) -> dict:
                s = m.predict(frame)
                return {"a": s[key_a], "b": s[key_b]}

            self._predict = _predict
        if self._transcribe is None:
            from faster_whisper import WhisperModel

            size = os.environ.get("VICA_VERIFY_STT_MODEL", "medium")
            device = os.environ.get("VICA_STT_DEVICE", "cpu")
            compute = os.environ.get("VICA_STT_COMPUTE",
                                     "float16" if device == "cuda" else "int8")
            wm = WhisperModel(size, device=device, compute_type=compute)

            def _transcribe(audio: np.ndarray) -> str:
                segs, _ = wm.transcribe(audio.astype(np.float32) / 32768.0,
                                        language="ko", beam_size=5)
                return "".join(s.text for s in segs).strip()

            self._transcribe = _transcribe

    def run(self) -> None:
        """reSpeaker ch0 상시 감시 루프 (blocking). Ctrl+C 로 종료."""
        import queue

        import sounddevice as sd

        self._load_real()
        device = next((i for i, d in enumerate(sd.query_devices())
                       if "respeaker" in d["name"].lower()
                       and d["max_input_channels"] >= 6), None)
        channels = 6 if device is not None else 1   # reSpeaker 없으면 기본 마이크

        q: queue.Queue[np.ndarray] = queue.Queue()

        def cb(indata, frames, t, status):  # noqa: ANN001
            block = np.frombuffer(indata, dtype=np.int16).reshape(-1, channels)
            q.put(block[:, 0].copy())

        print(f"웨이크워드 상시 감시 시작 (장치 {device}, {channels}ch — Ctrl+C 종료)")
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME,
                               channels=channels, dtype="int16",
                               device=device, callback=cb):
            while True:
                self.process_frame(q.get())


def _demo() -> None:
    monitor = WakewordMonitor(
        on_emergency=lambda e: print(f"\n🚨 긴급 '{e.keyword}' (인식: {e.source_text!r})"),
        on_user_text=lambda t: print(f"\n🗣️ 사용자: {t!r}"),
        on_wake=lambda: print("\n🙋 부르셨어요? (청취 중...)"),
    )
    monitor.run()


if __name__ == "__main__":
    _demo()
