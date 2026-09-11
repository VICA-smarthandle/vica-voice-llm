"""웨이크워드 상시 감시 엔진 (P1-b) — 호출(모델 A) + 긴급(모델 B → STT 검증)."""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from .handle_mode import (
    AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply)
from .schema import EmergencyEvent
from .stt_guard import accept_segments, is_hallucination
from .wakeword_gate import FrameGate, match_emergency_transcript

SAMPLE_RATE = 16000
FRAME = 1280
RING_FRAMES = 31
POST_ROLL_FRAMES = 4

CONFIRM_HINT = (
    "네. 그래. 응. 좋아요. 아니요. 아니. 싫어요. 취소. "
    "기다려. 기다려줘. 여기서 기다려. 대기. 대기해. "
    "오 분. 십 분. 십오 분. 이십 분. 삼십 분. 반시간. 한 시간."
)

LISTEN_MIN_SPEECH_SEC = 0.16
LISTEN_MIN_RMS = 0.008
SHORT_ANSWER_MIN_RMS = 0.02

LISTEN_WAKE_RESCUE_TOKENS = 2
LISTEN_WAKE_RESCUE_MIN_CHARS = 2
LISTEN_WAKE_RESCUE_MAX_CHARS = 5
WAKE_WORD_TEXT = "비카야"
_SHORT_ANSWER_WORDS = AFFIRMATIVES | SOFT_AFFIRMATIVES | NEGATIVES
LISTEN_MAX_SEC = float(os.environ.get("VICA_LISTEN_MAX_SEC", "15.0"))
LISTEN_SILENCE_END_SEC = float(os.environ.get("VICA_LISTEN_END_SEC", "0.8"))
LISTEN_MIN_OPEN_SEC = float(os.environ.get("VICA_LISTEN_MIN_OPEN_SEC", "2.5"))
LISTEN_BLIP_VOID_SEC = float(os.environ.get("VICA_LISTEN_BLIP_VOID_SEC", "0.32"))
CONFIRM_WINDOW_SEC = float(os.environ.get("VICA_CONFIRM_WINDOW_SEC", "30.0"))
PREROLL_FRAMES = 6
SPEECH_RMS = 0.01
HEAD_PRE_ROLL_FRAMES = 8


def _frame_rms(frame) -> float:
    return float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))


def _looks_like_short_call(text: str) -> bool:
    """호출 구제 대상인가 — 한 덩어리 짧은 말인지만 본다."""
    tokens = text.split()
    if not tokens or len(tokens) > LISTEN_WAKE_RESCUE_TOKENS:
        return False
    chars = normalize_short_reply(text)
    return (LISTEN_WAKE_RESCUE_MIN_CHARS
            <= len(chars) <= LISTEN_WAKE_RESCUE_MAX_CHARS)
FOLLOWUP_ARM_TIMEOUT_SEC = 20.0
BARGE_VAD_WINDOW = 10
BARGE_REQUIRE_BOTH = os.environ.get(
    "VICA_BARGE_REQUIRE_BOTH", "0").strip().lower() not in ("0", "false", "")
BARGE_VAD_MIN_HITS = int(os.environ.get(
    "VICA_BARGE_MIN_HITS", "3" if BARGE_REQUIRE_BOTH else "5"))
USER_DOA_LOCK_WIDTH = 15.0
USER_DOA_LOCK_TTL_SEC = 120.0


def _angle_diff(a: float, b: float) -> float:
    """두 방위각의 최단 차이 (0~180°). 0/359 경계를 올바로 다룬다."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
GATE_B_SPEAKING = 0.35


def capture_stats(audio: np.ndarray) -> dict:
    """청취 창 수음 품질 요약 — "수음이 나쁘다"를 감이 아니라 숫자로 만든다."""
    x = audio.astype(np.float32) / 32768.0
    if x.size == 0:
        return {"rms": 0.0, "peak": 0.0, "clip_ratio": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "peak": float(np.max(np.abs(x))),
        "clip_ratio": float(np.mean(np.abs(x) >= 0.99)),
    }


class WakewordMonitor:
    """호출·긴급 웨이크워드 상시 감시. 상태: idle / postroll / listen."""

    def __init__(
        self,
        on_emergency: Callable[[EmergencyEvent], None],
        on_user_text: Callable[[str], None],
        on_wake: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        on_reject: Optional[Callable[[str], None]] = None,
        on_listen_empty: Optional[Callable[[], None]] = None,
        on_listen_state: Optional[Callable[[str], None]] = None,
        on_wake_doa: Optional[Callable[[float], None]] = None,
        voice_barge_in: bool = True,
        doa_gate: bool = True,
        user_doa_center: Optional[float] = None,
        user_doa_width: float = 45.0,
        predict: Optional[Callable[[np.ndarray], dict]] = None,
        transcribe: Optional[Callable[[np.ndarray], str]] = None,
        listen_hint: Optional[str] = None,
        gate_a: float = 0.6,
        gate_b: float = 0.5,
        cooldown_a: float = 1.5,
        cooldown_b: float = 2.0,
    ):
        self._on_emergency = on_emergency
        self._on_user_text = on_user_text
        self._on_wake = on_wake or (lambda: None)
        self._on_barge_in = on_barge_in or (lambda: None)
        self._on_reject = on_reject or (lambda text: None)
        self._on_listen_empty = on_listen_empty or (lambda: None)
        self._on_listen_state = on_listen_state or (lambda s: None)
        self._on_wake_doa = on_wake_doa or (lambda doa: None)
        self._voice_barge_in = voice_barge_in
        self._doa_gate = doa_gate
        self._user_doa_center = user_doa_center
        self._user_doa_width = user_doa_width
        self._locked_doa: Optional[float] = None
        self._locked_doa_at = 0.0
        self._speaking = False
        self._vad_window: deque[bool] = deque(maxlen=BARGE_VAD_WINDOW)
        self._predict = predict
        self._transcribe = transcribe
        self._listen_hint = (listen_hint or "").strip() or None
        self._transcribe_listen: Optional[Callable[[np.ndarray], str]] = None
        self._transcribe_confirm: Optional[Callable[[np.ndarray], str]] = None

        self.gate_a = FrameGate(gate_a, persist=2, cooldown_sec=cooldown_a)
        self.gate_b = FrameGate(gate_b, persist=2, cooldown_sec=cooldown_b)
        self.gate_a_listen = FrameGate(gate_a, persist=2, cooldown_sec=0.0)
        self._gate_b_base = gate_b

        self._ring: deque[np.ndarray] = deque(maxlen=RING_FRAMES)
        self._state = "idle"
        self._postroll_from_listen = False
        self._collect: list[np.ndarray] = []
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_voiced_sec = 0.0
        self._listen_is_followup = False
        self._listen_heard_wake = False
        self._listen_opened_at = 0.0
        self._muted_until = 0.0
        self._muted = False
        self._followup_armed = False
        self._followup_armed_at = 0.0
        self._barge_tally = {"프레임": 0, "S": 0, "AND": 0, "방향": 0, "소리": 0}
        self.last_listen_stats: Optional[dict] = None
        self.last_listen_timing: Optional[dict] = None
        self._listen_speech_started_at = 0.0

    def lock_user_direction(self, doa: Optional[float],
                            now: Optional[float] = None) -> None:
        """"비카야"가 들린 방향을 이번 대화의 사용자 방향으로 잠근다."""
        if doa is None:
            return
        self._locked_doa = float(doa)
        self._locked_doa_at = time.time() if now is None else now

    def note_wake_direction(self, doa: Optional[float],
                            now: Optional[float] = None) -> None:
        """호출 순간의 방향을 잠그고(barge-in) 밖에 알린다(고개 돌리기)."""
        self.lock_user_direction(doa, now=now)
        if doa is not None:
            self._on_wake_doa(float(doa))

    def arm_followup(self, now: Optional[float] = None) -> None:
        """"방금 질문을 말했다"는 예약. 다음 TTS 종료(mute 해제) 때 청취를 연다."""
        self._followup_armed = True
        self._followup_armed_at = time.time() if now is None else now

    def disarm_followup(self) -> None:
        self._followup_armed = False

    def set_muted(self, muted: bool, now: Optional[float] = None,
                  failsafe_sec: float = 10.0) -> None:
        """TTS 재생 중 자기 목소리 억제. fail-safe: 해제 신호를 놓쳐도"""
        now = time.time() if now is None else now
        if muted:
            self._muted = True
            self._muted_until = now + failsafe_sec
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
            self._report_barge_miss()
            if now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC:
                self._open_listen(followup=True, now=now)

    def _is_muted(self, now: float) -> bool:
        if self._muted and now >= self._muted_until:
            self.set_muted(False, now)
        return self._muted

    def set_speaking(self, speaking: bool, now: Optional[float] = None) -> None:
        """AEC 배선 후의 TTS 경계 알림 — set_muted 의 대체이며 귀를 닫지 않는다."""
        now = time.time() if now is None else now
        self._speaking = speaking
        self._vad_window.clear()
        self.gate_b.threshold = GATE_B_SPEAKING if speaking else self._gate_b_base
        if speaking:
            self._reset_barge_tally()
            if self._state == "listen" and self._listen_is_followup:
                self._state = "idle"
                self._collect = []
                self._followup_armed = True
            return
        if self._followup_armed:
            self._followup_armed = False
            self._report_barge_miss()
            if now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC:
                self._open_listen(followup=True, now=now)

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None,
                      vad: Optional[bool] = None,
                      doa: Optional[float] = None,
                      vad2: Optional[bool] = None) -> Optional[str]:
        """int16 80ms 프레임 하나를 처리한다. 일어난 일을 문자열로 돌려준다"""
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
            if fire_b:
                self._enter_postroll(from_listen=True)
                return None
            if self.gate_a_listen.feed(float(scores["a"]), now):
                self._listen_heard_wake = True
            return self._listen_step(frame, now, vad)

        if fire_b:
            self._enter_postroll()
            return None
        if self.gate_a.feed(float(scores["a"]), now):
            self.gate_b.reset()
            self._on_wake()
            self._open_listen(followup=False, now=now)
            return "wake"

        if (
            self._voice_barge_in
            and self._speaking
            and self._followup_armed
            and vad is not None
            and now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC
        ):
            speech = bool(vad) and (not BARGE_REQUIRE_BOTH or bool(vad2))
            if not self._doa_gate:
                hit = speech
            else:
                sectors = []
                if (self._locked_doa is not None
                        and now - self._locked_doa_at <= USER_DOA_LOCK_TTL_SEC):
                    sectors.append((self._locked_doa, USER_DOA_LOCK_WIDTH))
                if self._user_doa_center is not None:
                    sectors.append((self._user_doa_center, self._user_doa_width))
                hit = (
                    speech
                    and doa is not None
                    and any(_angle_diff(float(doa), center) <= width
                            for center, width in sectors)
                )
            self._vad_window.append(hit)
            rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
            t = self._barge_tally
            t["프레임"] += 1
            t["S"] += bool(vad)
            t["AND"] += speech
            t["방향"] += hit
            t["소리"] += rms >= SPEECH_RMS
            if (
                len(self._vad_window) == BARGE_VAD_WINDOW
                and sum(self._vad_window) >= BARGE_VAD_MIN_HITS
                and rms >= SPEECH_RMS
            ):
                self._vad_window.clear()
                self._followup_armed = False
                self._reset_barge_tally()
                self._on_barge_in()
                self._open_listen(followup=True, now=now)
                self._collect = list(self._ring)[-BARGE_VAD_WINDOW:]
                self._listen_started_speech = True
                return "barge_in"
        return None

    def _reset_barge_tally(self) -> None:
        for k in self._barge_tally:
            self._barge_tally[k] = 0

    def _report_barge_miss(self) -> None:
        """질문이 끝나도록 끼어들기가 안 걸렸을 때, 어느 관문에서 막혔는지 남긴다."""
        t = self._barge_tally
        if t["S"] == 0:
            self._reset_barge_tally()
            return
        self._on_listen_state(
            f"barge-miss S={t['S']} AND={t['AND']} 방향={t['방향']} "
            f"소리={t['소리']} / 문턱 {BARGE_VAD_MIN_HITS} of {BARGE_VAD_WINDOW}")
        self._reset_barge_tally()

    def _open_listen(self, followup: bool, now: float) -> None:
        """청취 창을 연다. followup 이면 웨이크워드 없이(질문 답변용) 연 것이라"""
        self._state = "listen"
        self._collect = []
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_voiced_sec = 0.0
        self._listen_is_followup = followup
        self._listen_opened_at = now
        self._listen_speech_started_at = 0.0
        self._listen_heard_wake = False
        self.gate_a_listen.reset()
        self.last_listen_timing = None
        if followup:
            lookback = list(self._ring)[-HEAD_PRE_ROLL_FRAMES:]
            tail: list = []
            for f in reversed(lookback):
                if _frame_rms(f) < SPEECH_RMS:
                    break
                tail.append(f)
            if tail and len(tail) < len(lookback):
                tail.reverse()
                self._collect = list(tail)
                self._listen_started_speech = True
                self._listen_voiced_sec = len(tail) * (FRAME / SAMPLE_RATE)
                self._listen_speech_started_at = (
                    now - len(tail) * (FRAME / SAMPLE_RATE))
        self._on_listen_state("open")
        if self._listen_started_speech:
            self._on_listen_state("speech")

    def _enter_postroll(self, from_listen: bool = False) -> None:
        self._state = "postroll"
        self._saved_listen = self._collect if from_listen else None
        self._collect = []
        self.gate_a.reset()
        self._postroll_from_listen = from_listen

    def _verify_emergency(self, now: float) -> str:
        audio = np.concatenate([*self._ring])
        text = self._transcribe(audio)
        from_listen = self._postroll_from_listen
        self._postroll_from_listen = False
        self._state = "idle"
        self._collect = []
        keyword = match_emergency_transcript(text)
        if keyword is None:
            if from_listen:
                self._collect = (self._saved_listen or []) + self._collect
                self._saved_listen = None
                self._state = "listen"
                return "listen_resumed"
            text = text.strip()
            if (self._followup_armed and text and not is_hallucination(text)
                    and now - self._followup_armed_at
                    <= FOLLOWUP_ARM_TIMEOUT_SEC):
                self._followup_armed = False
                self.last_listen_stats = capture_stats(audio)
                self._on_listen_state("closed")
                self._on_user_text(text)
                return "user_text"
            self._on_reject(text)
            return "reject"
        if from_listen:
            self._saved_listen = None
            self._on_listen_state("empty")
        event = EmergencyEvent(keyword=keyword, source_text=text, detected_at=now)
        self._on_emergency(event)
        return "emergency"

    def _listen_step(self, frame: np.ndarray, now: float,
                     vad: Optional[bool]) -> Optional[str]:
        """청취 창 한 프레임. 발화 시작·끝은 칩의 발화 판정(vad)으로 잰다."""
        self._collect.append(frame)
        loud = _frame_rms(frame) >= SPEECH_RMS
        if vad:
            if not self._listen_started_speech:
                self._listen_speech_started_at = now
                self._on_listen_state("speech")
            self._listen_started_speech = True
            self._listen_silence = 0.0
            self._listen_voiced_sec += FRAME / SAMPLE_RATE
        elif vad is None and loud:
            if not self._listen_started_speech:
                self._listen_speech_started_at = now
                self._on_listen_state("speech")
            self._listen_started_speech = True
            self._listen_silence = 0.0
            self._listen_voiced_sec += FRAME / SAMPLE_RATE
        elif not self._listen_started_speech and loud:
            self._listen_speech_started_at = now
            self._listen_started_speech = True
            self._on_listen_state("speech")
        elif self._listen_started_speech:
            self._listen_silence += FRAME / SAMPLE_RATE
        else:
            del self._collect[:-PREROLL_FRAMES]

        max_sec = CONFIRM_WINDOW_SEC if self._listen_is_followup else LISTEN_MAX_SEC
        min_open = 0.0 if self._listen_is_followup else LISTEN_MIN_OPEN_SEC
        silence_done = (self._listen_started_speech
                        and self._listen_silence >= LISTEN_SILENCE_END_SEC)
        if (silence_done and not self._listen_is_followup
                and self._listen_voiced_sec < LISTEN_BLIP_VOID_SEC):
            self._listen_started_speech = False
            self._listen_silence = 0.0
            self._listen_voiced_sec = 0.0
            self._listen_speech_started_at = 0.0
            silence_done = False
        done = (
            now - self._listen_opened_at >= max_sec
            or (silence_done and now - self._listen_opened_at >= min_open)
        )
        if not done:
            return None

        audio = np.concatenate(self._collect)
        self.last_listen_stats = capture_stats(audio)
        self._state = "idle"
        self._collect = []
        if not self._listen_started_speech:
            if not self._listen_is_followup:
                self._on_listen_empty()
            self._on_listen_state("empty")
            return "wake_silent"
        speech_end = now - self._listen_silence
        speech_sec = speech_end - self._listen_speech_started_at
        rms = self.last_listen_stats["rms"]
        short = speech_sec < LISTEN_MIN_SPEECH_SEC
        short_rescue = (short and self._listen_is_followup
                        and rms >= SHORT_ANSWER_MIN_RMS)
        if rms < LISTEN_MIN_RMS or (short and not short_rescue):
            if not self._listen_is_followup:
                self._on_listen_empty()
            self._on_listen_state(
                f"empty:ghost speech={speech_sec:.2f}s "
                f"rms={rms:.4f}")
            return "wake_silent"
        if self._listen_is_followup and self._transcribe_confirm is not None:
            transcribe = self._transcribe_confirm
        else:
            transcribe = self._transcribe_listen or self._transcribe
        stt_started = time.monotonic()
        text = transcribe(audio).strip()
        self.last_listen_timing = {
            "wait": self._listen_speech_started_at - self._listen_opened_at,
            "speech": speech_end - self._listen_speech_started_at,
            "tail": now - speech_end,
            "stt": time.monotonic() - stt_started,
        }
        if self._listen_heard_wake and _looks_like_short_call(text):
            if (self._listen_is_followup
                    and normalize_short_reply(text) in _SHORT_ANSWER_WORDS):
                self._on_listen_state(f"answer-beats-rescue {text[:20]!r}")
            else:
                self._on_listen_state(f"wake-rescue {text[:20]!r}")
                text = WAKE_WORD_TEXT
        if not text or is_hallucination(text):
            if not self._listen_is_followup:
                self._on_listen_empty()
            self._on_listen_state(f"empty:reject {text[:30]!r}")
            return "wake_silent"
        if short_rescue and normalize_short_reply(text) not in _SHORT_ANSWER_WORDS:
            self._on_listen_state(f"empty:short-reject {text[:30]!r}")
            return "wake_silent"
        self._on_listen_state("closed")
        self._on_user_text(text)
        return "user_text"

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
            from . import stt  # noqa: F401
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

            def _transcribe_listen(audio: np.ndarray) -> str:
                segs, _ = wm.transcribe(audio.astype(np.float32) / 32768.0,
                                        language="ko", beam_size=5,
                                        initial_prompt=self._listen_hint,
                                        temperature=0.0,
                                        condition_on_previous_text=False)
                return accept_segments(segs)

            def _transcribe_confirm(audio: np.ndarray) -> str:
                segs, _ = wm.transcribe(audio.astype(np.float32) / 32768.0,
                                        language="ko", beam_size=5,
                                        initial_prompt=CONFIRM_HINT,
                                        temperature=0.0,
                                        condition_on_previous_text=False)
                return accept_segments(segs)

            self._transcribe = _transcribe
            self._transcribe_listen = _transcribe_listen
            self._transcribe_confirm = _transcribe_confirm

    def run(self) -> None:
        """reSpeaker ch0 상시 감시 루프 (blocking). Ctrl+C 로 종료."""
        import queue

        import sounddevice as sd

        self._load_real()
        from .dsp_state import DspState

        dsp = DspState()
        if not dsp.available:
            raise SystemExit(
                "reSpeaker 상태 레지스터(VAD·DOA)를 읽을 수 없다 — udev 규칙을 "
                "확인하라. 다른 방식으로 폴백하지 않는다 "
                "(정책: docs/respeaker-v3-capabilities.md)")
        device = next((i for i, d in enumerate(sd.query_devices())
                       if "respeaker" in d["name"].lower()
                       and d["max_input_channels"] >= 6), None)
        if device is None:
            raise SystemExit(
                "reSpeaker 6채널 입력을 찾을 수 없다 — 다른 마이크로 폴백하지 "
                "않는다 (정책: docs/respeaker-v3-capabilities.md)")
        channels = 6

        q: queue.Queue[np.ndarray] = queue.Queue()

        def cb(indata, frames, t, status):  # noqa: ANN001
            block = np.frombuffer(indata, dtype=np.int16).reshape(-1, channels)
            q.put(block[:, 0].copy())

        print(f"웨이크워드 상시 감시 시작 (장치 {device}, {channels}ch — Ctrl+C 종료)")
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME,
                               channels=channels, dtype="int16",
                               device=device, callback=cb):
            while True:
                frame = q.get()
                vad = doa = vad2 = None
                if self._state == "listen":
                    vad = dsp.speech_detected()
                elif (self._voice_barge_in and self._speaking
                        and self._followup_armed):
                    vad = dsp.speech_detected()
                    if vad:
                        doa = dsp.doa_angle()
                        if BARGE_REQUIRE_BOTH:
                            vad2 = dsp.voice_activity()
                r = self.process_frame(frame, vad=vad, doa=doa, vad2=vad2)
                if r == "wake":
                    self.note_wake_direction(dsp.doa_angle())


def _demo() -> None:
    monitor = WakewordMonitor(
        on_emergency=lambda e: print(f"\n🚨 긴급 '{e.keyword}' (인식: {e.source_text!r})"),
        on_user_text=lambda t: print(f"\n🗣️ 사용자: {t!r}"),
        on_wake=lambda: print("\n🙋 부르셨어요? (청취 중...)"),
    )
    monitor.run()


if __name__ == "__main__":
    _demo()
