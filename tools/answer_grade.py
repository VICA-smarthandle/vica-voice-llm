"""지금 구조의 의도를 설계 4.2 대답으로 옮기고, 정답과 비교해 채점한다 (설계 2026-11-14 5절).

answer_of 는 기준 점수(지금 구조) 전용 어댑터다. 새 구조는 대답을 직접 내므로 쓰지 않는다.
채점: ok = 맞음 · reask = 되묻기(잘못 아님) · wrong = 정반대·엉뚱한 출발·조용히 버림·딴말에 대꾸.
"""
from __future__ import annotations

from typing import Sequence

VALUED = ("destination", "choice", "minutes")   # 값까지 같아야 맞는 대답


def _squash(text: str) -> str:
    return "".join((text or "").split())


def _question_destination_id(question: str, destinations: Sequence) -> str:
    """질문 글자에 든 가장 긴 목적지 이름·별칭의 id(띄어쓰기 무시). 없으면 ""."""
    q = _squash(question)
    best = ("", "")
    for d in destinations:
        for key in (_squash(k) for k in [d.name, *(getattr(d, "aliases", []) or [])]):
            if key and key in q and len(key) > len(best[0]):
                best = (key, d.id)
    return best[1]


def answer_of(intent, kind: str, question: str, destinations: Sequence) -> tuple[str, str]:  # noqa: ANN001
    """지금 VicaIntent → (answer, value)."""
    i = intent.intent
    if i == "affirm":
        return "yes", ""
    if i == "deny":
        return "no", ""
    if i == "navigate":
        dest = intent.matched_destination_id or ""
        # other = LLM 이 지은 확인 질문("…로 안내하면 될까요?")도 같은 꼴
        if (kind in ("confirm", "other") and not intent.need_confirm and dest
                and dest == _question_destination_id(question, destinations)):
            return "yes", ""
        return ("choice", dest) if kind == "choice" else ("destination", dest)
    if i == "wait":
        if kind == "minutes" and (intent.wait_minutes or -1) > 0:
            return "minutes", str(intent.wait_minutes)
        return ("yes", "") if kind == "yesno" else ("other", "")
    if i in ("cancel", "finish"):
        if kind == "yesno_inverted" or "마칠까요" in (question or ""):
            return "yes", ""
        return "other", ""
    if i in ("unknown", "clarify"):
        return "unclear", ""
    return "other", ""


def grade(label: str, label_value: str, pred: str, pred_value: str) -> str:
    """정답(label)과 예측(pred) → ok · reask · wrong."""
    if pred == "unclear":
        return "ok" if label == "unclear" else "reask"
    if pred == label and (label not in VALUED or str(pred_value) == str(label_value)):
        return "ok"
    return "wrong"
