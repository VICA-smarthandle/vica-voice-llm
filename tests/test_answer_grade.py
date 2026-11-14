"""지금 의도 → 대답 옮기기 · 잘못된 행동 채점 (설계 2026-11-14 5절, 계획 Task 2)."""
from __future__ import annotations

import pytest

from src.schema import DestinationData, VicaIntent
from tools.answer_grade import answer_of, grade


@pytest.fixture
def dests():
    return [DestinationData(id="d1", name="사회복지"), DestinationData(id="d2", name="화장실")]


@pytest.mark.parametrize("intent,kind,question,expect", [
    (VicaIntent(intent="affirm"), "yesno", "여기서 대기할까요?", ("yes", "")),
    (VicaIntent(intent="deny"), "yesno", "여기까지 안내를 마칠까요?", ("no", "")),
    (VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False), "confirm",
     "사회복지로 안내해드릴까요?", ("yes", "")),
    (VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=True), "confirm",
     "사회복지로 안내해드릴까요?", ("destination", "d2")),
    (VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False), "yesno",
     "아까 사회복지에 가자고 하신 사용자님이신가요?", ("destination", "d1")),
    (VicaIntent(intent="wait", wait_minutes=10), "minutes", "몇 분쯤 걸리실까요?", ("minutes", "10")),
    (VicaIntent(intent="cancel"), "yesno_inverted", "안내가 필요 없으신가요?", ("yes", "")),
    (VicaIntent(intent="cancel"), "yesno", "여기까지 안내를 마칠까요?", ("yes", "")),
    (VicaIntent(intent="cancel"), "yesno", "여기서 대기할까요?", ("other", "")),
    (VicaIntent(intent="unknown"), "yesno", "여기서 대기할까요?", ("unclear", "")),
    (VicaIntent(intent="clarify"), "open", "네?", ("unclear", "")),
    (VicaIntent(intent="question"), "yesno", "안내를 받으시겠어요?", ("other", "")),
    (VicaIntent(intent="pause"), "minutes", "몇 분쯤 걸리실까요?", ("other", "")),
])
def test_answer_of(intent, kind, question, expect, dests):
    assert answer_of(intent, kind, question, dests) == expect


def test_navigate_on_choice_question_is_choice(dests):
    intent = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=False)
    assert answer_of(intent, "choice", "사회복지와 화장실 중 어디로 갈까요?", dests) == ("choice", "d2")


def test_wait_without_minutes_on_arrival_question_is_yes(dests):
    assert answer_of(VicaIntent(intent="wait"), "yesno", "여기서 대기할까요?", dests) == ("yes", "")


def test_confirm_question_picks_longest_destination_name():
    dests = [DestinationData(id="a", name="화장실"), DestinationData(id="b", name="남자 화장실")]
    intent = VicaIntent(intent="navigate", matched_destination_id="b", need_confirm=False)
    assert answer_of(intent, "confirm", "남자 화장실로 안내해드릴까요?", dests) == ("yes", "")


@pytest.mark.parametrize("label,lv,pred,pv,out", [
    ("yes", "", "yes", "", "ok"), ("minutes", "10", "minutes", "10", "ok"),
    ("minutes", "10", "minutes", "15", "wrong"),
    ("no", "", "yes", "", "wrong"),                 # 정반대 (11-06 11:32 "아니" → 종료)
    ("yes", "", "destination", "d1", "wrong"),      # 엉뚱한 출발
    ("yes", "", "unclear", "", "reask"),
    ("no", "", "not_for_robot", "", "wrong"),       # 조용히 버림
    ("not_for_robot", "", "other", "", "wrong"),    # 딴말에 대꾸
    ("not_for_robot", "", "unclear", "", "reask"),
    ("repeat", "", "unclear", "", "reask"),
    ("unclear", "", "unclear", "", "ok"),
])
def test_grade(label, lv, pred, pv, out):
    assert grade(label, lv, pred, pv) == out
