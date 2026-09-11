"""whisper 환각(유령 전사) 방어 — 무음·잡음에서 지어낸 문장을 걸러낸다."""
from __future__ import annotations

import re

NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0

HALLUCINATION_PHRASES = (
    "시청해주셔서감사합니다",
    "오늘도시청해주셔서감사합니다",
    "끝까지시청해주셔서감사합니다",
    "구독과좋아요부탁드립니다",
    "구독좋아요알림설정까지부탁드립니다",
    "다음영상에서만나요",
    "자막제공배달의민족",
    "고마워",
    "다됐어",
    "고맙습니다",
    "감사합니다",
)

_NEWS_PREFIX = re.compile(r"^(MBC|KBS|SBS|YTN|JTBC)뉴스")

_STRIP = re.compile(r"[\s.,!?~♪…'\"”“]+")

_SENTENCE_SPLIT = re.compile(r"[.!?…]+")


def _normalize(text: str) -> str:
    return _STRIP.sub("", text)


def _is_ghost_piece(norm: str) -> bool:
    return norm in HALLUCINATION_PHRASES or bool(_NEWS_PREFIX.match(norm))


def is_hallucination(text: str) -> bool:
    """전사가 무음 환각 단골 문구들만으로 이루어졌는가."""
    norm = _normalize(text)
    if not norm:
        return False
    if _is_ghost_piece(norm):
        return True
    pieces = [_normalize(p) for p in _SENTENCE_SPLIT.split(text)]
    pieces = [p for p in pieces if p]
    return len(pieces) > 1 and all(_is_ghost_piece(p) for p in pieces)


def accept_segments(segments) -> str:
    """whisper segment 들에서 신뢰할 수 있는 조각만 이어 붙인다."""
    kept = []
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > NO_SPEECH_THRESHOLD and logprob < LOGPROB_THRESHOLD:
            continue
        kept.append(seg.text)
    return "".join(kept).strip()


_ECHO_MIN_CHARS = 4


def _norm_echo(text: str) -> str:
    return re.sub(r"[\s.,!?~…\"'·\-]+", "", text)


def strip_robot_echo(text: str, robot_texts) -> str:
    """전사에서 로봇 자신의 최근 발화(에코)를 걷어내고 나머지를 돌려준다."""
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
