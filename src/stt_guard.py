"""whisper 환각(유령 전사) 방어 — 무음·잡음에서 지어낸 문장을 걸러낸다.

whisper 는 입력에 말이 없으면 학습 데이터에서 흔했던 문장(방송 맺음말 등)을
지어내는 성질이 있다. 세 겹으로 막는다:

  1. 입구 (기존): 청취 창의 RMS 발화 판정 — 소리 자체가 없으면 STT 를 안 부른다
     (wakeword_monitor._listen_step 의 wake_silent).
  2. 출구 (accept_segments): segment 신뢰도 — no_speech_prob 이 높고
     avg_logprob 이 낮은 조각은 버린다. openai-whisper 기본 휴리스틱과 같은
     조건·같은 임계값이다.
  3. 수배 전단 (is_hallucination): 무음 환각 단골 문구 목록. **전사 전체가
     그 문구일 때만** 유령으로 본다 — 진짜 발화에 섞인 부분 일치는 지우지
     않는다("감사합니다"는 합법 발화다).

전부 순수 함수라 장치 없이 시험된다. 목록은 실측 로그에서 발견되는 대로
추가한다 (근거 없는 선제 추가는 하지 않는다).
"""
from __future__ import annotations

import re

# openai-whisper 의 기본값과 동일 (no_speech_threshold / logprob_threshold).
# 두 조건이 동시에 참일 때만 버린다 — 하나만으로 버리면 작게 말한 진짜
# 발화(낮은 확신)까지 지운다.
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0

# 한국어 whisper 무음 환각 단골 (유튜브 자막 말투). 전체 일치로만 쓴다.
HALLUCINATION_PHRASES = (
    "시청해주셔서감사합니다",
    "오늘도시청해주셔서감사합니다",
    "끝까지시청해주셔서감사합니다",
    "구독과좋아요부탁드립니다",
    "구독좋아요알림설정까지부탁드립니다",
    "다음영상에서만나요",
    "자막제공배달의민족",
)

# 뉴스 맺음말 계열("MBC 뉴스 ○○○입니다")은 이름이 매번 달라 목록으로 못
# 잡는다 — 방송사 접두로 판정한다. 실내 안내 로봇에게 올 일 없는 문장이다.
_NEWS_PREFIX = re.compile(r"^(MBC|KBS|SBS|YTN|JTBC)뉴스")

_STRIP = re.compile(r"[\s.,!?~♪…'\"”“]+")


def _normalize(text: str) -> str:
    return _STRIP.sub("", text)


def is_hallucination(text: str) -> bool:
    """전사 전체가 무음 환각 단골 문구인가. 부분 포함은 유령으로 보지 않는다."""
    norm = _normalize(text)
    if not norm:
        return False
    if norm in HALLUCINATION_PHRASES:
        return True
    return bool(_NEWS_PREFIX.match(norm))


def accept_segments(segments) -> str:
    """whisper segment 들에서 신뢰할 수 있는 조각만 이어 붙인다.

    faster-whisper 의 Segment(no_speech_prob, avg_logprob)를 받지만, 속성이
    없는 객체(시험용 스텁 등)는 통과시킨다 — 필터가 전사를 막는 쪽으로
    실패하면 안 된다.
    """
    kept = []
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > NO_SPEECH_THRESHOLD and logprob < LOGPROB_THRESHOLD:
            continue
        kept.append(seg.text)
    return "".join(kept).strip()
