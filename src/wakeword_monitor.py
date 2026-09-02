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

from .handle_mode import (
    AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply)
from .schema import EmergencyEvent
from .stt_guard import accept_segments, is_hallucination
from .wakeword_gate import FrameGate, match_emergency_transcript

SAMPLE_RATE = 16000
FRAME = 1280                    # 80ms — openWakeWord 계약
RING_FRAMES = 31                # 검증에 쓰는 직전 오디오 ≈ 2.5초
POST_ROLL_FRAMES = 4            # 긴급 발동 후 말끝 보존 0.32초

# 질문 답변 창의 정답 후보 귀띔 (whisper initial_prompt). 이 창은 확인
# 질문뿐 아니라 도착 후 대화("다녀오시는 동안 기다릴까요?"·"몇 분쯤?")의
# 답도 받으므로, 대기 시간 표현·종료어를 못 알아들으면 wait/finish 자체가
# 안 잡힌다 (2026-08-30). 너무 길거나 세게 기울이면 딴말이 후보로 둔갑하니
# 실제로 나올 답만 짧게 유지할 것.
CONFIRM_HINT = (
    "네. 그래. 응. 좋아요. 아니요. 아니. 싫어요. 취소. "
    "기다려. 기다려줘. 여기서 기다려. 대기. "
    "오 분. 십 분. 십오 분. 이십 분. 삼십 분. 반시간. 한 시간. "
    "이제 됐어. 그만할래. 안내 끝. 다 됐어. 고마워. "
    "대기해. 필요 없어요."
)

# 무음 문턱 — 이보다 짧거나 작은 수음은 STT 로 보내지 않는다. 귀띔이 무음
# 환각을 진짜 장소 이름('방2')으로 둔갑시킨 실측(2026-08-28, rms 0.0034·
# 발화 0.00초)이 근거. 진짜 조용한 답("응" rms 0.0185·0.24초)은 통과한다.
LISTEN_MIN_SPEECH_SEC = 0.16    # 발화 최소 길이 (2프레임)
LISTEN_MIN_RMS = 0.008          # 수음 최소 크기
# 짧은 답 구제 (2026-09-01, 질문 창 한정): "네" 한 글자는 0.16초 문턱 아래로
# 잘리기도 한다(8/31 야간 실측 — 0.08초·rms 0.0289 가 기각됨). 질문(재청취)
# 창에서는 짧아도 또렷하면(rms 이상) STT 재판을 받되, 통과는 정답 어휘
# (긍/부정)로만 제한해 유령 전사의 부활을 막는다. 자유 창(호출 직후)은
# 목적지 같은 긴 말을 기다리는 자리라 기존 문턱 그대로다 (사용자 결정).
SHORT_ANSWER_MIN_RMS = 0.02
_SHORT_ANSWER_WORDS = AFFIRMATIVES | SOFT_AFFIRMATIVES | NEGATIVES
# 청취 창 시간값 — 사용감을 정하는 파라미터라 환경변수로 조정하고, 확정은
# 실사용 측정으로 한다 [TARGET] (시나리오 2-1.4절과 같은 취급).
LISTEN_MAX_SEC = float(os.environ.get("VICA_LISTEN_MAX_SEC", "6.0"))
LISTEN_SILENCE_END_SEC = float(os.environ.get("VICA_LISTEN_END_SEC", "0.8"))
# 자유 창 최소 개방 시간 [TARGET] (2026-09-01): "네?" 에코·비카야 꼬리에
# VAD 가 반짝하면 말끝 시계(0.8초)가 조기 가동돼 창이 1~1.5초 만에 닫히고,
# 그 뒤에 말한 명령이 통째로 무시됐다(어제 10회+·오늘 2회 실측). 이 시간
# 안에는 침묵-마감을 무시한다. 산정: 관측된 정상 대기 최대 1.94초 + 여유
# 0.5초. 질문(재청취) 창은 30초짜리라 적용하지 않는다.
LISTEN_MIN_OPEN_SEC = float(os.environ.get("VICA_LISTEN_MIN_OPEN_SEC", "2.5"))
# 반짝 무효화 문턱 [실측 2026-09-01]: 에코 반짝의 VAD 연속 구간은 전부
# ≤0.14초(mic_probe, 로봇 단독 발화 조건), 진짜 발화의 최단은 0.48초
# (실기 계측 15표본). 그 한가운데 — 이보다 짧은 "발화"는 자유 창에서
# 없던 일로 되돌리고 6초 상한까지 계속 기다린다.
LISTEN_BLIP_VOID_SEC = float(os.environ.get("VICA_LISTEN_BLIP_VOID_SEC", "0.32"))
# 질문(재청취) 창은 시나리오 6.4의 확인 대기 30초와 일치시킨다 — 미션이
# 30초를 기다린다고 약속하는데 귀가 6초만 열려 있으면 안 된다.
CONFIRM_WINDOW_SEC = float(os.environ.get("VICA_CONFIRM_WINDOW_SEC", "30.0"))
# 발화 시작 전 보존할 말머리 여유(0.48초) — 긴 확인 창이 침묵 덩어리로
# whisper 에 통째로 가는 것을 막는다.
PREROLL_FRAMES = 6
SPEECH_RMS = 0.01               # barge-in 의 "소리는 나야 한다" 건전성 바닥
# 말머리 소급의 되돌아보기 상한 (0.64초). 링버퍼(2.5초) 안에서만 줍는다.
HEAD_PRE_ROLL_FRAMES = 8


def _frame_rms(frame) -> float:
    return float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
# 재청취(arm_followup) 예약의 유효 시간. 질문 TTS 가 유실돼 mute 해제가 안 오면
# 예약이 남아, 한참 뒤 무관한 안내가 끝난 순간 마이크가 열리는 오동작을 막는다.
FOLLOWUP_ARM_TIMEOUT_SEC = 20.0
# 음성 barge-in 판정 (재설계 2026-08-24): RMS 는 자기 잔여 에코와 사람을 못
# 가린다는 것이 두 번 실측돼 폐기했다. 대신 칩(XVF-3000)의 SPEECHDETECTED 를
# 쓴다 — AEC 후단에서 계산되므로 로봇 자신의 재생음에는 반응하지 않는다
# (tools/vad_probe 실측: 로봇 단독 재생 중 0.0%, 사용자 발화 중 47.6%).
# 발화 리듬상 절반쯤 켜지므로 "최근 창의 과반"으로 판정한다. 값은 노드가
# 프레임마다 vad 인자로 넣어 주고, 없으면(None) 음성 barge-in 은 잠든다.
BARGE_VAD_WINDOW = 10      # 최근 프레임 수 (0.8초)
BARGE_VAD_MIN_HITS = 5     # 그중 발화 판정 최소 개수
# "비카야" 순간에 잠근 사용자 방향의 허용 폭과 유효 시간 (사용자 제안
# 2026-08-24): 웨이크워드 순간은 로봇이 대개 조용해 방향이 깨끗하고(단일
# 화자 퍼짐 ±2~4° 실측), 사용자가 어디 서 있든 보정 없이 맞는다. 고정
# 부채꼴은 이중 발화 때 방향이 로봇 스피커 쪽으로 섞여 뚫렸다(217° 실측)
# — 잠금 폭을 그 섞임(중심에서 19°)보다 좁게 둔다.
USER_DOA_LOCK_WIDTH = 15.0
USER_DOA_LOCK_TTL_SEC = 120.0


def _angle_diff(a: float, b: float) -> float:
    """두 방위각의 최단 차이 (0~180°). 0/359 경계를 올바로 다룬다."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
# TTS 재생 중(AEC 모드)의 긴급 관문. 실측(2026-08-24, 외침 10회 프로토콜):
# 실패 주원인은 STT 기각이 아니라 관문 미달(4/10, 근접 0.27 포함)이었다.
# 로봇 자기 목소리의 모델 B 점수는 0.00 수준(함정 시험)이라 완화해도 자가
# 발동 위험이 낮고, whisper 정확 매칭이 2차 방어로 그대로 있다.
GATE_B_SPEAKING = 0.35


def capture_stats(audio: np.ndarray) -> dict:
    """청취 창 수음 품질 요약 — "수음이 나쁘다"를 감이 아니라 숫자로 만든다.

    rms(평균 음량), peak(최고점), clip_ratio(포화 샘플 비율, |x|≥0.99).
    개선(마이크 위치·거리 안내 등)의 전후 비교 기준이 된다. DSP 파라미터는
    동결(D7)이므로 여기 숫자가 나빠도 AGC/NS 를 바꾸는 게 아니라 DSP 밖에서
    해결한다 (vica-wakeword/docs/respeaker-dsp-config.md).
    """
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
        # 질문 재생 중 사용자가 답을 시작했을 때 (TTS 를 끊으라는 신호용)
        self._on_barge_in = on_barge_in or (lambda: None)
        # 긴급 관문이 발동했으나 STT 기각된 사건. 반드시 기록돼야 한다 —
        # 이 로그가 없어서 "멈춰가 씹혔는데 흔적이 없는" 관측 공백이 생겼다.
        self._on_reject = on_reject or (lambda text: None)
        # "비카야" 창이 빈손으로 닫힌 사건(발화 없음·빈 전사). 침묵하면
        # 사용자는 로봇이 죽었는지 못 들었는지 알 수 없다(2026-08-28 실측).
        # followup(질문 답변) 창의 무응답은 Mission 몫이라 알리지 않는다.
        self._on_listen_empty = on_listen_empty or (lambda: None)
        # 청취 상태 중계 (open/speech/closed/empty). 미션이 무응답 시계를
        # "귀가 바쁜 동안" 멈추는 데 쓴다 (2026-08-30 — 답이 STT·LLM 을
        # 통과하는 동안 8초 시계가 먼저 울려 홈으로 떠나던 결함).
        self._on_listen_state = on_listen_state or (lambda s: None)
        self._voice_barge_in = voice_barge_in
        # 방향 관문 스위치. 꺼지면 barge-in 은 방향을 안 보고 칩 VAD 만 본다
        # (2026-08-30 사용자 결정 — 장착 상태 DOA 실측이 아직 없어 해제.
        # 마이크 DOA 점검 후 center 측정과 함께 다시 켠다).
        self._doa_gate = doa_gate
        # 사용자 방향 부채꼴 (도). 설정되면 그 방향의 발화만 대답으로 인정한다 —
        # 칩 VAD 는 화자를 못 가려 옆사람 대화가 질문을 끊었다(2026-08-24 실측).
        # DOA 실측: 나란히 앉은 두 화자도 평균 23° 차·퍼짐 ±2~7° 로 갈라졌다.
        # ⚠️ 긴급어에는 방향 조건을 절대 쓰지 않는다 — 행인의 "멈춰"도 정지 대상.
        self._user_doa_center = user_doa_center
        self._user_doa_width = user_doa_width
        # "비카야" 순간에 잠근 이번 대화의 사용자 방향 (고정 부채꼴보다 우선)
        self._locked_doa: Optional[float] = None
        self._locked_doa_at = 0.0
        self._speaking = False
        # 칩 발화 판정의 최근 창 (질문 재생 중에만 쌓인다)
        self._vad_window: deque[bool] = deque(maxlen=BARGE_VAD_WINDOW)
        self._predict = predict          # frame(int16 1280) -> {"a": 점수, "b": 점수}
        self._transcribe = transcribe    # int16 오디오 -> 한국어 텍스트 (긴급 검증용)
        # 자유 명령 창의 장소 귀띔 (destination_loader.build_place_hint 산출물).
        # 긴급 검증 전사에는 절대 걸지 않는다 — 긴급어가 장소로 둔갑하면 안 된다.
        self._listen_hint = (listen_hint or "").strip() or None
        # 대화 청취용 전사 — 신뢰도 필터(stt_guard)가 걸린 판. 긴급 검증에는
        # 필터를 걸지 않는다: 작은 외침의 조각을 지울 위험이 실측됐고(외침 10회
        # 프로토콜에서 빈 전사 기각 1건), 유령은 정확 매칭이 이미 막는다.
        self._transcribe_listen: Optional[Callable[[np.ndarray], str]] = None
        # 질문 답변(followup) 창 전용 — 정답 후보를 귀띔(initial_prompt)한 판.
        # "그래"→'굿에이' 같은 짧은 답 오전사 대책 (2026-08-28 실측).
        self._transcribe_confirm: Optional[Callable[[np.ndarray], str]] = None

        self.gate_a = FrameGate(gate_a, persist=2, cooldown_sec=cooldown_a)
        self.gate_b = FrameGate(gate_b, persist=2, cooldown_sec=cooldown_b)
        self._gate_b_base = gate_b   # set_speaking 이 재생 중 완화/복원한다

        self._ring: deque[np.ndarray] = deque(maxlen=RING_FRAMES)
        self._state = "idle"
        self._postroll_from_listen = False   # 긴급 검증이 청취 창을 가로챘는가
        self._collect: list[np.ndarray] = []   # postroll·listen 수집분
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_voiced_sec = 0.0
        self._listen_is_followup = False       # 이 청취 창이 재청취로 열렸는가
        self._listen_opened_at = 0.0
        self._muted_until = 0.0
        self._muted = False
        # 재청취 예약: 로봇이 질문을 말하는 중("~할까요?")이면 노드가 걸어 두고,
        # TTS 가 끝나(mute 해제) 이 예약이 살아 있으면 웨이크워드 없이 청취를 연다.
        # 사용자가 "응" 한마디를 하려고 "비카야"를 다시 부를 필요가 없게 한다.
        self._followup_armed = False
        self._followup_armed_at = 0.0
        # 마지막 청취 창의 수음 품질 (노드가 로그로 남긴다)
        self.last_listen_stats: Optional[dict] = None
        # 마지막 청취 창의 구간 계측 — 대기·발화·말끝판정·STT (초).
        # "왜 오래 걸리나"를 구간별 숫자로 만든다 (2026-08-28 주행에서
        # 청취+STT 묶음 6.1초의 내역을 몰랐던 것이 도입 계기).
        self.last_listen_timing: Optional[dict] = None
        self._listen_speech_started_at = 0.0

    # ---------------------------------------------------------------- 방향 잠금
    def lock_user_direction(self, doa: Optional[float],
                            now: Optional[float] = None) -> None:
        """"비카야"가 들린 방향을 이번 대화의 사용자 방향으로 잠근다.

        노드가 wake 직후 칩의 DOAANGLE 을 읽어 넣는다. 잠금이 살아 있는 동안
        음성 barge-in 은 이 방향 ±USER_DOA_LOCK_WIDTH 의 발화만 대답으로
        인정한다. 긴급어에는 방향 조건이 없다 — 불변.
        """
        if doa is None:
            return
        self._locked_doa = float(doa)
        self._locked_doa_at = time.time() if now is None else now

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
                self._open_listen(followup=True, now=now)

    def _is_muted(self, now: float) -> bool:
        if self._muted and now >= self._muted_until:   # fail-safe 타임아웃
            self.set_muted(False, now)
        return self._muted

    def set_speaking(self, speaking: bool, now: Optional[float] = None) -> None:
        """AEC 배선 후의 TTS 경계 알림 — set_muted 의 대체이며 귀를 닫지 않는다.

        AEC 가 자기 목소리를 마이크 입력(ch0)에서 빼 주므로 재생 중에도 감시를
        계속한다. 로봇이 말하는 도중의 "멈춰"·"비카야"가 그대로 들린다.
        남는 일은 재청취 예약 관리뿐이다: 문장 재생이 시작되면 열려 있던 재청취
        창을 접고 예약을 되살리고(여러 문장짜리 질문), 재생이 끝나면 예약된
        창을 연다. 버퍼·관문은 비우지 않는다 — 잔향은 AEC 몫이고, 비우면 긴급
        검증이 참조할 직전 오디오가 사라진다.
        """
        now = time.time() if now is None else now
        self._speaking = speaking
        self._vad_window.clear()
        # 재생 중에는 긴급 관문을 완화한다 — 로봇 목소리·AEC 잔여에 섞인
        # 외침은 점수가 깎인다(실측 근접 미달 0.27). 검증은 그대로 거친다.
        self.gate_b.threshold = GATE_B_SPEAKING if speaking else self._gate_b_base
        if speaking:
            if self._state == "listen" and self._listen_is_followup:
                self._state = "idle"
                self._collect = []
                self._followup_armed = True
            return
        if self._followup_armed:
            self._followup_armed = False
            if now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC:
                self._open_listen(followup=True, now=now)

    # ---------------------------------------------------------------- 핵심 로직
    def process_frame(self, frame: np.ndarray, now: Optional[float] = None,
                      vad: Optional[bool] = None,
                      doa: Optional[float] = None) -> Optional[str]:
        """int16 80ms 프레임 하나를 처리한다. 일어난 일을 문자열로 돌려준다
        (emergency / reject / wake / user_text / wake_silent / barge_in / None).

        vad: 칩(XVF-3000)의 발화 판정(SPEECHDETECTED). 질문 재생 중 음성
        barge-in 판정에만 쓴다. None = 하드웨어 없음/미조회 — 판정 안 함.
        doa: 칩의 소리 방향(도). 사용자 부채꼴이 설정된 경우 대답 인정 조건.
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
                self._enter_postroll(from_listen=True)
                return None
            return self._listen_step(frame, now, vad)

        # idle
        if fire_b:
            self._enter_postroll()
            return None
        if self.gate_a.feed(float(scores["a"]), now):
            self.gate_b.reset()
            self._on_wake()
            self._open_listen(followup=False, now=now)
            return "wake"

        # 질문 재생 중 barge-in: 로봇이 질문을 말하는 도중(재청취 예약 상태)
        # 사용자가 답을 시작하면, TTS 를 끊고(콜백) 즉시 듣는다. 긴급(모델 B)과
        # 호출(모델 A)이 위에서 항상 먼저다. 예약 없는 일반 안내에는 끼어들기가
        # 없다. 판정은 칩의 발화 판정(vad) 창 과반 — RMS 는 자기 에코와 사람을
        # 못 가려 폐기했다(2026-08-24 자책골 2회 실측 + vad_probe 근거).
        if (
            self._voice_barge_in
            and self._speaking
            and self._followup_armed
            and vad is not None
            and now - self._followup_armed_at <= FOLLOWUP_ARM_TIMEOUT_SEC
        ):
            # 방향 관문: 사용자 방향 밖(옆사람·행인)의 발화는 대답이 아니다.
            # 잠금(비카야 순간 방향 ±15°, 신선할 때)과 장착 부채꼴(핸들 방향)의
            # **합집합** — 어느 쪽이든 들어오면 통과 (2026-08-30 사용자 결정:
            # 좁아서 못 듣는 쪽이 더 나쁘다. 이전엔 잠금이 신선하면 ±15°만 봐서
            # 반 발짝 옆의 사용자를 놓쳤다). 자책골 방어는 칩 VAD(에코 면역)와
            # 과반 창이 맡는다 — 실기에서 로봇이 말하다 스스로 끊기면 이 완화가
            # 1번 용의자다. 방향을 모르면 증거 부족 — 발동하지 않는다.
            if not self._doa_gate:
                # 방향 관문 해제 — 어느 방향의 발화든 대답으로 본다.
                hit = bool(vad)
            else:
                sectors = []
                if (self._locked_doa is not None
                        and now - self._locked_doa_at <= USER_DOA_LOCK_TTL_SEC):
                    sectors.append((self._locked_doa, USER_DOA_LOCK_WIDTH))
                if self._user_doa_center is not None:
                    sectors.append((self._user_doa_center, self._user_doa_width))
                hit = (
                    bool(vad)
                    and doa is not None
                    and any(_angle_diff(float(doa), center) <= width
                            for center, width in sectors)
                )
            self._vad_window.append(hit)
            rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
            if (
                len(self._vad_window) == BARGE_VAD_WINDOW
                and sum(self._vad_window) >= BARGE_VAD_MIN_HITS
                and rms >= SPEECH_RMS      # 최소한 소리는 나야 한다 (건전성 바닥)
            ):
                self._vad_window.clear()
                self._followup_armed = False
                self._on_barge_in()
                self._open_listen(followup=True, now=now)
                # 말머리를 버리지 않는다 — 판정 창만큼 직전 프레임부터 수집한다.
                self._collect = list(self._ring)[-BARGE_VAD_WINDOW:]
                self._listen_started_speech = True
                return "barge_in"
        return None

    # ---------------------------------------------------------------- 내부
    def _open_listen(self, followup: bool, now: float) -> None:
        """청취 창을 연다. followup 이면 웨이크워드 없이(질문 답변용) 연 것이라
        인사(on_wake)를 하지 않고, 창 길이도 확인 대기(30초)를 따른다."""
        self._state = "listen"
        self._collect = []
        self._listen_started_speech = False
        self._listen_silence = 0.0
        self._listen_voiced_sec = 0.0
        self._listen_is_followup = followup
        self._listen_opened_at = now
        self._listen_speech_started_at = 0.0
        self.last_listen_timing = None
        # 말머리 소급 (2026-09-01, 블랙박스 방식): 창이 열리기 직전에 이미
        # 말이 시작됐다면 링버퍼에서 그 머리를 줍는다 — "정확한 타이밍에
        # 말하지 않아도" 들리게. **질문(재청취) 창에만** 적용한다: 자유
        # 창(비카야 직후) 직전의 소리는 정의상 사용자의 "비카야" 자신이라,
        # 소급하면 그 꼬리가 "말 시작"이 되어 사용자가 "네?"를 기다리는
        # 침묵에 창이 1.4초 만에 닫혔다(실기 2회 — '안내소로 가줘' 통째
        # 무시). 되돌아본 구간이 전부 시끄러우면(로봇 목소리 연속) 역시
        # 줍지 않는다 — 경계의 조용한 프레임이 새 발화의 증거다.
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
            self._on_listen_state("speech")   # 귀 홀드가 볼 발화 표시

    def _enter_postroll(self, from_listen: bool = False) -> None:
        self._state = "postroll"
        # 청취를 가로챈 경우 수집분을 보존한다 — 오발동으로 판명되면 청취를
        # 그대로 복원해 이어 듣는다 (2026-08-31 근본 수리: 이전엔 여기서
        # 수집분을 파괴해, 검증 조각 전사가 빈손이면 발화가 통째로 죽었다.
        # "필요없다구" 기각 실측).
        self._saved_listen = self._collect if from_listen else None
        self._collect = []
        self.gate_a.reset()
        self._postroll_from_listen = from_listen

    def _verify_emergency(self, now: float) -> str:
        audio = np.concatenate([*self._ring])   # 직전 ~2.5초 + 말끝
        text = self._transcribe(audio)
        from_listen = self._postroll_from_listen
        self._postroll_from_listen = False
        self._state = "idle"
        self._collect = []
        keyword = match_emergency_transcript(text)
        if keyword is None:
            if from_listen:
                # 청취를 가로챈 오발동 — 청취를 그대로 복원해 이어 듣는다.
                # 조각(링 2.5초)만 건지던 옛 방식은 조각 전사가 빈손이면
                # 발화 전체를 잃었다(2026-08-31). 복원하면 발화가 자연스러운
                # 말끝까지 수집돼 정상 STT(귀띔 포함)를 탄다. 검증 중 흘러간
                # postroll 프레임도 발화의 일부이므로 이어 붙인다.
                self._collect = (self._saved_listen or []) + self._collect
                self._saved_listen = None
                self._state = "listen"
                return "listen_resumed"
            text = text.strip()
            # 창 밖 오발동이라도 질문의 답을 기다리는 중(followup 예약)이면
            # 그 전사는 십중팔구 답이다 — 버리면 "정확히 알아듣고도 기각"이
            # 된다("필요없다구" 실측 2026-08-31). 예약이 없을 때(행인 대화
            # 등)는 전처럼 기각 — 웨이크워드 규약은 유지된다.
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
            self._on_listen_state("empty")   # 청취는 긴급에 선점돼 폐기됐다
        event = EmergencyEvent(keyword=keyword, source_text=text, detected_at=now)
        self._on_emergency(event)
        return "emergency"

    def _listen_step(self, frame: np.ndarray, now: float,
                     vad: Optional[bool]) -> Optional[str]:
        """청취 창 한 프레임. 발화 시작·끝은 칩의 발화 판정(vad)으로 잰다.

        소리 크기(RMS) 판정은 폐기했다 — 약한 어미("...주세요")를 침묵으로
        오인해 자르고, 배경 소음에는 반대로 안 닫혔다. vad 는 노드가 청취 중
        매 프레임 칩(SPEECHDETECTED)에서 읽어 넣는다. None = 이번 프레임 발화
        증거 없음 (장치가 없으면 애초에 기동이 실패한다 — 폴백 없음).
        """
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
            # 칩 판정 부재 (2026-09-01): 제어 읽기가 죽으면 vad 가 None 으로
            # 오는데, 예전엔 그 동안의 진짜 말이 트림으로 통째 증발하거나
            # 침묵으로 세져 조기 마감됐다 — 실기 유령 "소리 큼 + 발화
            # 0.00초"의 정체. 판정이 없을 때만 소리 크기로 대체한다.
            if not self._listen_started_speech:
                self._listen_speech_started_at = now
                self._on_listen_state("speech")
            self._listen_started_speech = True
            self._listen_silence = 0.0
            self._listen_voiced_sec += FRAME / SAMPLE_RATE
        elif not self._listen_started_speech and loud:
            # 칩은 살아 있는데(False) 시작을 놓친 경우 — 시작만 소리로
            # 보강한다. 말끝은 기존대로 칩 기준: RMS 말끝은 약한 어미를
            # 자르고 배경 소음에 안 닫히던 전력으로 폐기됐다.
            self._listen_speech_started_at = now
            self._listen_started_speech = True
            self._on_listen_state("speech")
        elif self._listen_started_speech:
            self._listen_silence += FRAME / SAMPLE_RATE
        else:
            # 발화 시작 전에는 말머리 여유분만 남긴다 — 30초 확인 창이
            # 침묵 덩어리로 whisper 에 통째로 가는 것을 막는다.
            del self._collect[:-PREROLL_FRAMES]

        max_sec = CONFIRM_WINDOW_SEC if self._listen_is_followup else LISTEN_MAX_SEC
        # 자유 창은 최소 개방 시간 전에는 침묵으로 닫지 않는다 — 반짝 VAD
        # (에코·호출 꼬리)가 말끝 시계를 조기 가동시키는 것을 무력화한다.
        min_open = 0.0 if self._listen_is_followup else LISTEN_MIN_OPEN_SEC
        silence_done = (self._listen_started_speech
                        and self._listen_silence >= LISTEN_SILENCE_END_SEC)
        if (silence_done and not self._listen_is_followup
                and self._listen_voiced_sec < LISTEN_BLIP_VOID_SEC):
            # 반짝 무효화 (실측 근거는 상수 주석): 쌓인 말이 반짝 수준이면
            # 발화 도장을 취소하고 창을 계속 연다 — 원래 시나리오(6초
            # 대기)의 복원. 질문 창은 초단 답("네")이 합법이라 제외.
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
        # 무음 문턱: 스친 잡음·거의 무음은 STT 로 보내지 않는다 — whisper 는
        # 무음에서 아무 말이나 지어내고, 귀띔은 그 환각을 진짜 장소 이름으로
        # 만들어 유령 주행 명령이 된다 ('방2' 실측 2026-08-28).
        speech_end = now - self._listen_silence
        speech_sec = speech_end - self._listen_speech_started_at
        rms = self.last_listen_stats["rms"]
        short = speech_sec < LISTEN_MIN_SPEECH_SEC
        # 질문 창의 짧고 또렷한 소리는 버리지 않고 STT 재판을 받는다 —
        # 단 통과는 정답 어휘로만 (아래 short_rescue 판정).
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
        # 유령 방어: 무음 환각 단골 문구 전체 일치면 발화가 없었던 것으로 본다
        # (stt_guard 3겹 중 수배 전단. 신뢰도 필터는 transcribe_listen 안에 있다).
        if not text or is_hallucination(text):
            if not self._listen_is_followup:
                self._on_listen_empty()
            self._on_listen_state(f"empty:reject {text[:30]!r}")
            return "wake_silent"
        if short_rescue and normalize_short_reply(text) not in _SHORT_ANSWER_WORDS:
            # 구제 재판의 판정 제한: 짧은 소리의 전사가 정답 어휘가 아니면
            # 유령으로 본다 — 잡음 딸깍이 '방2' 같은 장소로 둔갑하는 것 방지.
            self._on_listen_state(f"empty:short-reject {text[:30]!r}")
            return "wake_silent"
        self._on_listen_state("closed")   # 발화가 STT 를 통과 — LLM 처리 예정
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
            # stt 모듈을 먼저 거친다 — import 부수효과 두 가지가 필요하다:
            # .env 로드(VICA_STT_* 반영)와 Jetson CUDA libctranslate2 선적재.
            # 이것 없이 faster_whisper 를 직접 import 하면 Jetson 에서
            # so 미발견·int8 거부로 감시 스레드가 죽는다 (2026-08-16 실기).
            from . import stt  # noqa: F401
            from faster_whisper import WhisperModel

            size = os.environ.get("VICA_VERIFY_STT_MODEL", "medium")
            device = os.environ.get("VICA_STT_DEVICE", "cpu")
            compute = os.environ.get("VICA_STT_COMPUTE",
                                     "float16" if device == "cuda" else "int8")
            wm = WhisperModel(size, device=device, compute_type=compute)

            def _transcribe(audio: np.ndarray) -> str:
                # 긴급 검증용 — 필터 없음. 작은 외침을 지우지 않는 것이 우선이고
                # (빈 전사 기각 실측), 유령은 정확 매칭이 막는다.
                segs, _ = wm.transcribe(audio.astype(np.float32) / 32768.0,
                                        language="ko", beam_size=5)
                return "".join(s.text for s in segs).strip()

            # 환각 억제 2종 (2026-08-28): temperature=0 은 "확신 없으면 온도를
            # 올려 아무 말이나 시도"하는 기본 사다리를 끈다. condition_...=False
            # 는 앞 조각의 오류가 뒤 조각으로 번지는 것을 끊는다. 긴급 검증
            # 전사(_transcribe)에는 걸지 않는다 — 그쪽은 실기 검증된 안전 축.
            def _transcribe_listen(audio: np.ndarray) -> str:
                # 대화 청취용 — 신뢰도 필터(stt_guard 2겹)로 유령 전사를 버린다.
                # 장소 귀띔(initial_prompt)으로 목적지 오전사를 줄인다.
                segs, _ = wm.transcribe(audio.astype(np.float32) / 32768.0,
                                        language="ko", beam_size=5,
                                        initial_prompt=self._listen_hint,
                                        temperature=0.0,
                                        condition_on_previous_text=False)
                return accept_segments(segs)

            def _transcribe_confirm(audio: np.ndarray) -> str:
                # 질문 답변 창 전용 — 나올 답의 후보를 귀띔해 짧은 답 인식을
                # 살린다. 귀띔은 기울이기일 뿐 강제(화이트리스트)가 아니므로
                # 자유 발화도 나올 수 있고, 신뢰도 필터는 동일하게 건다.
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
        # 칩 발화 판정(SPEECHDETECTED) 리더. 질문 재생 중에만 조회한다 —
        # USB 제어 채널 트래픽을 최소화하기 위해서다 (스트림 열기와 겹치면
        # 장치가 열기를 거부하는 것이 실측됨, tools/vad_probe 참조).
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
                vad = doa = None
                if self._state == "listen":
                    # 청취 중: 발화 시작/끝 판정용
                    vad = dsp.speech_detected()
                elif (self._voice_barge_in and self._speaking
                        and self._followup_armed):
                    # 질문 재생 중: barge-in 판정용 (+방향)
                    vad = dsp.speech_detected()
                    if vad:
                        doa = dsp.doa_angle()
                r = self.process_frame(frame, vad=vad, doa=doa)
                if r == "wake":
                    # "비카야"가 온 방향을 이번 대화의 사용자 방향으로 잠근다
                    self.lock_user_direction(dsp.doa_angle())


def _demo() -> None:
    monitor = WakewordMonitor(
        on_emergency=lambda e: print(f"\n🚨 긴급 '{e.keyword}' (인식: {e.source_text!r})"),
        on_user_text=lambda t: print(f"\n🗣️ 사용자: {t!r}"),
        on_wake=lambda: print("\n🙋 부르셨어요? (청취 중...)"),
    )
    monitor.run()


if __name__ == "__main__":
    _demo()
