"""지금 구조의 의도를 설계 4.2 대답으로 옮기고, 정답과 비교해 채점한다 (설계 2026-11-14 5절)."""
from __future__ import annotations

from typing import Sequence

VALUED = ("destination", "choice", "minutes")
ACTIONS = ("destination", "choice", "minutes")


def _squash(text: str) -> str:
    return "".join((text or "").split())


def question_destination_id(question: str, destinations: Sequence) -> str:
    """질문이 묻는 목적지 id(이름·별칭, 띄어쓰기 무시). 없으면 ""."""
    q = _squash(question)
    best = (-1, 0, "")
    for d in destinations:
        for key in (_squash(k) for k in [d.name, *(getattr(d, "aliases", []) or [])]):
            if key and key in q:
                end = q.rindex(key) + len(key)
                best = max(best, (end, len(key), d.id))
    return best[2]


def answer_of(intent, kind: str, question: str, destinations: Sequence) -> tuple[str, str]:  # noqa: ANN001
    """지금 VicaIntent → (answer, value)."""
    i = intent.intent
    if i == "affirm":
        return "yes", ""
    if i == "deny":
        return "no", ""
    if i == "navigate":
        dest = intent.matched_destination_id or ""
        if kind in ("confirm", "other") and dest and dest == question_destination_id(question, destinations):
            return "yes", ""
        return ("choice", dest) if kind == "choice" else ("destination", dest)
    if i == "wait":
        if kind == "minutes" and (intent.wait_minutes or -1) > 0:
            return "minutes", str(intent.wait_minutes)
        if kind == "yesno":
            return ("no", "") if "마칠까요" in (question or "") else ("yes", "")
        return "other", ""
    if i in ("cancel", "finish"):
        if kind == "yesno_inverted" or "마칠까요" in (question or ""):
            return "yes", ""
        return "other", ""
    if i in ("unknown", "clarify"):
        return "unclear", ""
    return "other", ""


def pred_of(intent, kind: str, question: str, destinations: Sequence) -> tuple[str, str]:  # noqa: ANN001
    """채점에 쓸 (answer, value). 대답 칸이 채워졌으면 그 값(3-나-2 — 새 구조는 대답을 직접 낸다), 비었으면 지금"""
    answer = getattr(intent, "answer", "") or ""
    if answer and answer != "other":
        return answer, getattr(intent, "answer_value", "") or ""
    return answer_of(intent, kind, question, destinations)


def grade(label: str, label_value: str, pred: str, pred_value: str, kind: str = "") -> str:
    """정답(label)과 예측(pred) → ok · reask · wrong · mismatch."""
    if pred == "unclear":
        return "ok" if label == "unclear" else "reask"
    if pred == label and (label not in VALUED or str(pred_value) == str(label_value)):
        return "ok"
    if {label, pred} == {"yes", "no"}:
        return "wrong"
    if pred == "not_for_robot" and label not in ("not_for_robot", "unclear"):
        return "wrong"
    if pred in ACTIONS or (pred == "yes" and kind != "open"):
        return "wrong"
    return "mismatch"
