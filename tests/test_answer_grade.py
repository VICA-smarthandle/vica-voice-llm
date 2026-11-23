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


def test_confirm_question_matches_alias_and_spaces():
    """10-24 Realtime 재생: 이름 '사회 복지창구'·질문 '사회복지로…' — 맞는 '그래'가 wrong 으로 세어졌다."""
    dests = [DestinationData(id="s", name="사회 복지창구", aliases=["사회복지", "장애인"]),
             DestinationData(id="t", name="화장실")]
    intent = VicaIntent(intent="navigate", matched_destination_id="s", need_confirm=False)
    assert answer_of(intent, "confirm", "사회복지로 안내해드릴까요?", dests) == ("yes", "")
    assert answer_of(intent, "confirm", "사회복지창구로 안내해드릴까요?", dests) == ("yes", "")


def test_confirm_like_free_question_is_yes(dests):
    """LLM 이 지은 확인 질문(kind other)에 확정 navigate 도 yes."""
    intent = VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False)
    assert answer_of(intent, "other", "장애인 관련 서류는 사회복지로 안내하면 될까요?", dests) == ("yes", "")


def test_navigate_on_choice_question_is_choice(dests):
    intent = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=False)
    assert answer_of(intent, "choice", "사회복지와 화장실 중 어디로 갈까요?", dests) == ("choice", "d2")


def test_wait_without_minutes_on_arrival_question_is_yes(dests):
    assert answer_of(VicaIntent(intent="wait"), "yesno", "여기서 대기할까요?", dests) == ("yes", "")


def test_confirm_question_picks_longest_destination_name():
    dests = [DestinationData(id="a", name="화장실"), DestinationData(id="b", name="남자 화장실")]
    intent = VicaIntent(intent="navigate", matched_destination_id="b", need_confirm=False)
    assert answer_of(intent, "confirm", "남자 화장실로 안내해드릴까요?", dests) == ("yes", "")


@pytest.mark.parametrize("label,lv,pred,pv,kind,out", [
    ("yes", "", "yes", "", "", "ok"), ("minutes", "10", "minutes", "10", "", "ok"),
    ("minutes", "10", "minutes", "15", "", "wrong"),
    ("no", "", "yes", "", "yesno", "wrong"),          # 정반대 (11-06 11:32 "아니" → 종료)
    ("yes", "", "no", "", "yesno", "wrong"),          # 정반대
    ("yes", "", "destination", "d1", "", "wrong"),    # 엉뚱한 출발
    ("not_for_robot", "", "yes", "", "confirm", "wrong"),   # 딴말에 출발
    ("yes", "", "unclear", "", "", "reask"),
    ("no", "", "not_for_robot", "", "", "wrong"),     # 조용히 버림
    ("not_for_robot", "", "other", "", "", "mismatch"),     # 딴말에 대꾸 — 설계 5절 세 가지 밖
    ("unclear", "", "yes", "", "open", "mismatch"),   # 열린 질문의 affirm 은 미션이 무시
    ("destination", "d1", "no", "", "yesno", "mismatch"),
    ("no", "", "other", "", "yesno", "mismatch"),
    ("not_for_robot", "", "unclear", "", "", "reask"),
    ("repeat", "", "unclear", "", "", "reask"),
    ("unclear", "", "unclear", "", "", "ok"),
])
def test_grade(label, lv, pred, pv, kind, out):
    assert grade(label, lv, pred, pv, kind) == out


def test_change_question_uses_the_new_destination():
    """검토 C2a: 바꾸기 질문엔 목적지가 둘 — 끝의 '…로 바꿀까요?' 쪽이 질문의 목적지."""
    dests = [DestinationData(id="s", name="사회 복지창구"), DestinationData(id="w", name="여자 화장실")]
    intent = VicaIntent(intent="navigate", matched_destination_id="w", need_confirm=False)
    q = "지금 사회 복지창구로 가는 중인데, 여자 화장실로 바꿀까요?"
    assert answer_of(intent, "other", q, dests) == ("yes", "")


def test_confirm_answer_with_need_confirm_still_yes(dests):
    """검토 C2b: 확인 중 같은 곳 재제안은 미션이 승낙으로 받는다(mission_logic on_confirm_answer)."""
    intent = VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=True)
    assert answer_of(intent, "confirm", "사회복지로 안내해드릴까요?", dests) == ("yes", "")


def test_wait_on_end_question_is_no(dests):
    """검토 I4: '여기까지 안내를 마칠까요?' + '기다려줘' = 안 마침."""
    assert answer_of(VicaIntent(intent="wait"), "yesno", "여기까지 안내를 마칠까요?", dests) == ("no", "")


def test_pred_of_prefers_the_answer_field(dests):
    from tools.answer_grade import pred_of
    said = VicaIntent(intent="navigate", matched_destination_id="d1", answer="yes")
    assert pred_of(said, "yesno", "여기서 대기할까요?", dests) == ("yes", "")
    plain = VicaIntent(intent="navigate", matched_destination_id="d1")
    assert pred_of(plain, "yesno", "여기서 대기할까요?", dests) == answer_of(plain, "yesno", "여기서 대기할까요?", dests)


def test_pred_of_reads_other_by_the_intent(dests):
    """other = 질문과 상관없는 말 — 로봇이 하는 동작은 의도 칸이 정한다(설계 4.5). 글자 그대로 채점하면 엉뚱한 출발이 가려진다(검토 I1)."""
    from tools.answer_grade import pred_of
    nav = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=True, answer="other")
    assert pred_of(nav, "yesno", "여기서 대기할까요?", dests) == ("destination", "d2")
    asked = VicaIntent(intent="question", reply="화장실은 오른쪽에 있어요.", answer="other")
    assert pred_of(asked, "confirm", "사회복지로 안내해드릴까요?", dests) == answer_of(asked, "confirm", "사회복지로 안내해드릴까요?", dests)
