"""whisper 환각(유령 전사) 방어 — 무음·잡음에서 지어낸 문장을 걸러낸다.

whisper 는 입력에 말이 없으면 학습 데이터에서 흔했던 문장(방송 맺음말 등)을
지어내는 성질이 있다. 세 겹으로 막는다:

  1. 입구 (기존): 청취 창의 RMS 발화 판정 — 소리 자체가 없으면 STT 를 안 부른다
     (wakeword_monitor._listen_step 의 wake_silent).
  2. 출구 (accept_segments): segment 신뢰도 — no_speech_prob 이 높고
     avg_logprob 이 낮은 조각은 버린다. openai-whisper 기본 휴리스틱과 같은
     조건·같은 임계값이다.
  3. 수배 전단 (is_hallucination): 무음 환각 단골 문구 목록. **전사가 그
     문구들만으로 이루어졌을 때만** 유령으로 본다 — 진짜 발화에 섞인 부분
     일치는 지우지 않는다("네, 감사합니다"는 통과). 문장이 여럿이면 조각을
     쪼개 전부 환각일 때만 기각한다 ("다 됐어. 고마워." 실측 2026-09-02).

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
    # '그래' 오인식·에코성 환각으로 확인(사용자 증언 + 로그).
    "고마워",
    # 2026-09-02 짧은답 30회: 사용자가 "네"·"대기해줘"라고 말한 자리에서
    # 아래 세 문구가 나왔다(사용자 증언 — 한 적 없는 말). 길이·음량으로는
    # 못 가린다 — 환각 발화 0.38~0.80초 vs 진짜 짧은 답 0.40~0.64초로
    # 완전히 겹치고 rms 도 비슷하다(실측). 그래서 어구로만 막는다.
    # 대가: 진짜 "다 됐어"로는 안내를 끝낼 수 없다. "이제 됐어"·"그만"·
    # "끝났어"는 살아 있다. 이 대가를 감수하는 이유는 오전사가 하필
    # finish(안내 전체 종료)로 몰려 되돌리기가 가장 어렵기 때문이다.
    "다됐어",
    "고맙습니다",
    # "감사합니다" 단독도 환각으로 본다 (2026-09-02 사용자 결정 — 종전의
    # "합법 발화라 살린다"를 뒤집는다). 실기에서 사용자가 말한 적 없이
    # 나왔고, 인사말 단독은 로봇이 할 일이 없어 기각해도 잃는 것이 없다.
    # 섞인 발화("안내해 주셔서 감사합니다")는 여전히 통과한다.
    "감사합니다",
)

# 뉴스 맺음말 계열("MBC 뉴스 ○○○입니다")은 이름이 매번 달라 목록으로 못
# 잡는다 — 방송사 접두로 판정한다. 실내 안내 로봇에게 올 일 없는 문장이다.
_NEWS_PREFIX = re.compile(r"^(MBC|KBS|SBS|YTN|JTBC)뉴스")

_STRIP = re.compile(r"[\s.,!?~♪…'\"”“]+")

# 문장 경계. 환각이 여러 문장으로 붙어 나오는 경우를 조각내 판정한다.
_SENTENCE_SPLIT = re.compile(r"[.!?…]+")


def _normalize(text: str) -> str:
    return _STRIP.sub("", text)


def _is_ghost_piece(norm: str) -> bool:
    return norm in HALLUCINATION_PHRASES or bool(_NEWS_PREFIX.match(norm))


def is_hallucination(text: str) -> bool:
    """전사가 무음 환각 단골 문구들만으로 이루어졌는가.

    문장이 여럿이면 조각마다 판정해 **전부** 환각일 때만 기각한다. 하나라도
    진짜 말이 섞여 있으면 통과시킨다 — 기각은 사용자의 말을 통째로 버리는
    일이라 확실할 때만 한다 (fail-open).
    """
    norm = _normalize(text)
    if not norm:
        return False
    if _is_ghost_piece(norm):
        return True
    pieces = [_normalize(p) for p in _SENTENCE_SPLIT.split(text)]
    pieces = [p for p in pieces if p]
    # 조각이 하나뿐이면 위에서 이미 판정했다 — 재판할 것이 없다.
    return len(pieces) > 1 and all(_is_ghost_piece(p) for p in pieces)


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
