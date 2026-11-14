"""정답 묶음 초안 (설계 2026-11-14 5절, 계획 Task 3)."""
from __future__ import annotations

import pytest

from src.schema import DestinationData
from tools.answer_label import draft_label, select_rows

KINDS = ["open", "confirm", "choice", "yesno", "yesno_inverted", "minutes", "other"]


@pytest.fixture
def dests():
    return [DestinationData(id="d1", name="사회복지"), DestinationData(id="d2", name="화장실")]


@pytest.fixture
def rows_200():
    return [{"ts": 1000.0 + i, "date": "10-10", "time": f"{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}",
             "kind": KINDS[i % len(KINDS)], "answer_text": f"말 {i}"} for i in range(200)]


@pytest.mark.parametrize("text,kind,expect", [
    ("비카야", "confirm", ("repeat", "", "rule")), ("피카야.", "yesno", ("repeat", "", "rule")),
    ("네. 네.", "yesno", ("yes", "", "rule")), ("그래 맞아", "yesno", ("yes", "", "rule")),
    ("아뇨.", "yesno", ("no", "", "rule")), ("아니 사회복지로 가자", "yesno", ("destination", "d1", "rule")),
    ("십 분.", "minutes", ("minutes", "10", "rule")), ("한 오 분.", "minutes", ("minutes", "5", "rule")),
    ("", "yesno", ("unclear", "", "rule")),
    ("급정지 되면은 그 사용감이 안 좋아서요.", "yesno", ("", "", "todo")),
])
def test_draft_label(text, kind, expect, dests):
    assert draft_label({"answer_text": text, "kind": kind}, dests) == expect


@pytest.mark.parametrize("text", ["네?", "네", "아니"])
def test_open_question_yes_no_is_unclear(text, dests):
    """호출 응답 '네?' 같은 열린 질문에 네/아니요는 답이 아니다 — 되묻기가 맞음."""
    assert draft_label({"answer_text": text, "kind": "open"}, dests) == ("unclear", "", "rule")


def test_draft_label_choice_question_gives_choice(dests):
    assert draft_label({"answer_text": "화장실로 가자", "kind": "choice"}, dests) == ("choice", "d2", "rule")


def test_draft_label_without_destination_file_skips_destination_rule():
    assert draft_label({"answer_text": "사회복지로 가자", "kind": "open"}, []) == ("", "", "todo")


def test_select_rows_is_stable_and_keeps_all_flagged(rows_200):
    flagged = {(r["date"], r["time"]) for r in rows_200[:20]}
    a, b = select_rows(rows_200, flagged, per_kind=5), select_rows(rows_200, flagged, per_kind=5)
    assert a == b and flagged <= {(r["date"], r["time"]) for r in a}
    assert len(a) == 20 + 5 * len(KINDS)
