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
    # 2026-09-01 실기: 사용자가 말한 적 없는 "고마워"가 반복 등장 —
    # '그래' 오인식·에코성 환각으로 확인(사용자 증언 + 로그). 전체 일치만
    # 기각이므로 "고마워. 대기해." 같은 혼합 발화는 여전히 통과한다.
    "고마워",
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


# ---------------------------------------------------------------- 에코 방어
# AEC 가 자기 목소리를 다 못 지우면 로봇의 질문·안내가 사용자 답으로 전사된다
# (2026-08-31 야간 실기: '화장실로 안내해 드릴까요? 어.' — 자기 질문 + 진짜 답,
# '무슨 뜻인지 잘 모르겠어요.' — 통째로 자기 멘트). 최근 로봇 발화와 대조해
# 문장 단위로 걷어낸다. 4번째 겹이다.

_ECHO_MIN_CHARS = 4  # 이보다 짧은 조각('어'·'네'·'그래')은 우연 일치가 잦아
                     # 에코로 보지 않는다 — 짧은 답이야말로 지키려는 대상이다.


def _norm_echo(text: str) -> str:
    return re.sub(r"[\s.,!?~…\"'·\-]+", "", text)


def strip_robot_echo(text: str, robot_texts) -> str:
    """전사에서 로봇 자신의 최근 발화(에코)를 걷어내고 나머지를 돌려준다.

    문장별 판정 — ① 문장이 로봇 발화의 조각이면 버린다(부분 전사:
    '다시 말씀해주세요' ← '잘 듣지 못했습니다. 다시 말씀해 주세요.').
    ② 문장 안에 로봇 발화가 통째로 들어 있으면 그 부분만 떼고 남는
    말('어')을 살린다 — 정규화형이지만 짧은 답은 그대로 통한다.
    전부 에코면 빈 문자열. 순수 함수 — 장치 없이 시험된다.
    """
    # 짧은 로봇 발화("네?")는 대조에서 뺀다 — 사용자의 "네"까지 지운다.
    robots = [n for n in (_norm_echo(r) for r in robot_texts if r)
              if len(n) >= _ECHO_MIN_CHARS]
    if not robots:
        return text.strip()
    kept: list[str] = []
    for sent in re.split(r"(?<=[.?!])\s*", text):
        s = sent.strip()
        if not s:
            continue
        n = _norm_echo(s)
        if len(n) >= _ECHO_MIN_CHARS and any(n in r for r in robots):
            continue
        hit = next((r for r in robots if r in n), None)
        if hit is not None:
            rest = n.replace(hit, "", 1)
            if rest:
                kept.append(rest)
            continue
        kept.append(s)
    return " ".join(kept).strip()
