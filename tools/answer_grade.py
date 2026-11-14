"""지금 구조의 의도를 설계 4.2 대답으로 옮기고, 정답과 비교해 채점한다 (설계 2026-11-14 5절).

answer_of 는 기준 점수(지금 구조) 전용 어댑터다. 새 구조는 대답을 직접 내므로 쓰지 않는다.
채점: ok = 맞음 · reask = 되묻기 · wrong = 설계 5절의 잘못된 행동(정반대·엉뚱한 동작·조용히 버림)
· mismatch = 그 밖의 어긋남(딴말에 대꾸, 거절로 읽음 등 — 합격선에 넣지 않고 따로 센다).
"""
from __future__ import annotations

from typing import Sequence

VALUED = ("destination", "choice", "minutes")   # 값까지 같아야 맞는 대답
ACTIONS = ("destination", "choice", "minutes")  # 로봇이 움직이거나 정하는 대답


def _squash(text: str) -> str:
    return "".join((text or "").split())


def question_destination_id(question: str, destinations: Sequence) -> str:
    """질문이 묻는 목적지 id(이름·별칭, 띄어쓰기 무시). 없으면 "".

    여럿이면 가장 뒤에 나온 것 — 바꾸기 질문("A로 가는 중인데, B로 바꿀까요?")은 끝의 B 를 묻는다.
    끝 위치가 같으면 긴 이름("화장실"·"남자 화장실").
    """
    q = _squash(question)
    best = (-1, 0, "")   # (끝 위치, 길이, id)
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
        # 확인 질문(LLM 이 지은 것 포함)이 묻는 곳으로 가자 = 승낙. need_confirm 이어도 미션이 승낙으로 받는다.
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


def grade(label: str, label_value: str, pred: str, pred_value: str, kind: str = "") -> str:
    """정답(label)과 예측(pred) → ok · reask · wrong · mismatch."""
    if pred == "unclear":
        return "ok" if label == "unclear" else "reask"
    if pred == label and (label not in VALUED or str(pred_value) == str(label_value)):
        return "ok"
    if {label, pred} == {"yes", "no"}:
        return "wrong"                       # 정반대
    if pred == "not_for_robot" and label not in ("not_for_robot", "unclear"):
        return "wrong"                       # 조용히 버림
    if pred in ACTIONS or (pred == "yes" and kind != "open"):
        return "wrong"                       # 묻지 않은 동작(엉뚱한 출발·대기 시간·승낙)
    return "mismatch"
