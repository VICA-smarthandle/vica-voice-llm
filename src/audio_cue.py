"""청각 안내음 생성·재생.

파형을 만드는 부분은 순수 함수라 소리 장치 없이 시험된다. 재생만 sounddevice 를
쓰고, 실패해도 예외를 밖으로 내지 않는다 — 소리가 안 나는 것이 파이프라인을
멈출 이유는 되지 않는다.

TTS 를 거치지 않는 이유:
    큐에 줄을 서지 않아 즉시 난다. 그리고 /vica/tts_state 를 켜지 않으므로
    긴급어 상시 감시가 쉬지 않는다. 0.15초짜리 순음이 "멈춰"·"비카야" 로
    오인될 여지는 사실상 없다.

소리 설계 (2026-08-05):
    호출 응답   880Hz 단음      — 기존 _ack_beep 과 같은 소리
    좌회전      660Hz 단음      — 낮은 음
    우회전      990Hz 단음      — 높은 음
    도착        784→1047Hz 2음  — 상행. 회전음과 뚜렷이 구분된다

좌우를 음높이로 나눈 것은 1차 안이다. 스피커가 스테레오면 좌/우 채널로 나누는
편이 직관적이다(소리 나는 쪽으로 몸이 반응한다). AEC 배선 후 출력 특성을 확인하고
정한다.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
DEFAULT_VOLUME = 0.4



def tone(freq_hz: float, duration_sec: float, volume: float = DEFAULT_VOLUME) -> np.ndarray:
    """한 음의 파형. 창(Hanning)을 씌워 시작·끝의 딱 소리를 없앤다."""
    if duration_sec <= 0 or freq_hz <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = int(duration_sec * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    wave = volume * np.sin(2 * np.pi * freq_hz * t) * np.hanning(samples)
    return wave.astype(np.float32)


def sequence(freqs_hz, duration_sec: float, volume: float = DEFAULT_VOLUME) -> np.ndarray:
    """여러 음을 이어 붙인다 (도착음처럼 두 음 이상인 경우)."""
    parts = [tone(f, duration_sec, volume) for f in freqs_hz]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


# "생각 중" 배경 운율 (2026-09-01 사용자 결정 — "확인할게요" 말 대체).
# LLM 응답을 만드는 수 초 동안 이 한 바퀴(~2.1초)를 반복 재생한다. 말보다
# 조용한 부드러운 아르페지오 — 배경으로 깔려 "듣고 생각 중"을 알린다.
# 끝이 무음(gap)이라 반복 이음새에 딱 소리가 없다.
THINKING_NOTES_HZ = (523.25, 659.25, 783.99, 659.25)   # C5 E5 G5 E5
THINKING_NOTE_SEC = 0.42
THINKING_GAP_SEC = 0.10
THINKING_VOLUME = 0.16


def thinking_loop() -> np.ndarray:
    parts: list[np.ndarray] = []
    gap = np.zeros(int(THINKING_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
    for f in THINKING_NOTES_HZ:
        parts.append(tone(f, THINKING_NOTE_SEC, THINKING_VOLUME))
        parts.append(gap)
    return np.concatenate(parts)


# 사람 접근 차임 (2026-10-07 사용자 결정 — 시청 페이지 차임 1번 "딩—동↗").
# 후진음("삐—")은 "피하라"로 들려 쓰지 않는다. 아래에서 위로 오르는 종 두 음이
# "반갑게 다가온다"를 알리고, 배음이 섞인 종소리라 순음보다 방향을 잡기 쉽다.
# '생각 중' 운율의 도·미·솔과 겹치지 않는 라(880)→레(1175)다.
APPROACH_CHIME_NOTES_HZ = (880.0, 1174.66)
APPROACH_CHIME_GAP_SEC = 0.24          # 둘째 음을 치는 시각
APPROACH_CHIME_NOTE_SEC = 1.0          # 음 하나의 여운 길이
# (배수, 세기, 줄어드는 시간): 배음마다 따로 잦아든다 — 종소리 결.
APPROACH_CHIME_PARTIALS = ((1, 1.0, 0.45), (2, 0.28, 0.25), (3, 0.10, 0.14))
APPROACH_CHIME_ATTACK_SEC = 0.004      # 딸깍 없이 치는 짧은 시작


def _bell_note(freq_hz: float) -> np.ndarray:
    samples = int(APPROACH_CHIME_NOTE_SEC * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    wave = np.zeros(samples, dtype=np.float64)
    for multiple, amp, fade_sec in APPROACH_CHIME_PARTIALS:
        # fade_sec 동안 약 5 % 로 잦아든다(시간 상수 = fade_sec / 3).
        wave += amp * np.exp(-t * 3.0 / fade_sec) * np.sin(2 * np.pi * freq_hz * multiple * t)
    attack = max(1, int(APPROACH_CHIME_ATTACK_SEC * SAMPLE_RATE))
    wave[:attack] *= np.linspace(0.0, 1.0, attack)
    return wave


def approach_chime() -> np.ndarray:
    """종 두 음 "딩—동↗" 한 번. 재생 크기는 부르는 쪽이 정한다(audio_out peak_dbfs)."""
    gap = int(APPROACH_CHIME_GAP_SEC * SAMPLE_RATE)
    first, second = (_bell_note(f) for f in APPROACH_CHIME_NOTES_HZ)
    wave = np.zeros(gap + len(second), dtype=np.float64)
    wave[: len(first)] += first
    wave[gap:] += second
    peak = float(np.max(np.abs(wave)))
    return (wave / peak).astype(np.float32) if peak > 0 else wave.astype(np.float32)
