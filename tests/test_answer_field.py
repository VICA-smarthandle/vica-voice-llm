"""대답 칸(3-나-2 1단계, 설계 2026-11-18 4.1~4.3)."""
from typing import get_args

import pytest
from pydantic import BaseModel

from src import answer_field as af
from src.question_record import OpenQuestion
from src.schema import VicaIntent, VicaIntentType

BASE = ("no", "destination", "other", "unclear")


def q(kind="confirm", text="사회복지로 안내해드릴까요?"):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")


def test_switch_defaults_to_off(monkeypatch):
    assert af.on() is False
    monkeypatch.setenv("VICA_ANSWER", "ON")
    assert af.on() is True
    monkeypatch.setenv("VICA_ANSWER", "1")
    assert af.on() is False


def test_local_schema_has_no_answer_field_unless_asked(monkeypatch):
    assert af.local_on() is False
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    assert af.local_on() is False
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert af.local_on() is True
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "1")
    assert af.local_on() is False
    monkeypatch.delenv("VICA_ANSWER_LOCAL")
    assert af.local_on() is False


def test_allowed_by_kind():
    for kind in ("open", "confirm", "yesno", "yesno_inverted"):
        assert set(af.allowed(kind)) == {"yes", *BASE}
    assert set(af.allowed("choice")) == {"choice", *BASE}
    assert set(af.allowed("minutes")) == {"minutes", *BASE}
    assert af.allowed("") == af.ANSWERS and af.allowed("새종류") == af.ANSWERS


def test_question_line(on):
    assert af.question_line(q()) == (
        '- 로봇이 물은 질문: "사회복지로 안내해드릴까요?" (확인 질문 — answer 는 yes·no·destination·other·unclear 중 하나: '
        "yes 승낙 · no 거절 · destination 다른 곳으로 가자(destination_candidate 에 그곳) · "
        "other 질문과 상관없는 명령이나 질문 · unclear 못 알아들음)")
    minutes = af.question_line(q("minutes", "몇 분쯤 걸리실까요?"))
    assert "minutes 시간을 말함(wait_minutes 에 분)" in minutes and "yes 승낙" not in minutes
    assert "choice 보기 중 하나를 고름(destination_candidate 에 그곳)" in af.question_line(q("choice", "A와 B 중 어디로 갈까요?"))
    inverted = af.question_line(q("yesno_inverted", "안내가 필요 없으신가요?"))
    assert "필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no" in inverted
    assert af.question_line(None) == ""


def test_with_question(on):
    line = af.question_line(q())
    assert af.with_question("", q()) == "\n[지금 상황]\n" + line + "\n"
    assert af.with_question("\n[지금 상황] (미션…)\n- 건물/층: 모름\n", q()) == (
        "\n[지금 상황] (미션…)\n- 건물/층: 모름\n" + line + "\n")
    assert af.with_question("블록", None) == "블록"


def test_off_is_inert():
    intent = VicaIntent(intent="affirm")
    assert af.with_question("블록", q()) == "블록"
    assert af.short_answer("네", q("yesno")) is None
    assert af.fill(intent, q(), ("yes", ""), "yes") is intent


@pytest.mark.parametrize("text,kind,out", [
    ("네.", "confirm", ("yes", "")), ("네. 네.", "yesno", ("yes", "")), ("아니요", "yesno", ("no", "")),
    ("그래.", "yesno_inverted", ("yes", "")),
])
def test_short_answer_whole_yes_no(on, text, kind, out):
    assert af.short_answer(text, q(kind)) == out


@pytest.mark.parametrize("text,kind", [
    ("네 화장실로 가자", "confirm"), ("그래 맞아.", "yesno"), ("부요.", "yesno"),
    ("음", "confirm"), ("음", "yesno"), ("어어", "yesno_inverted"),
    ("네", "open"), ("네", "choice"), ("네", "minutes"), ("삼십분.", "confirm"),
])
def test_short_answer_only_whole_utterance(on, text, kind):
    assert af.short_answer(text, q(kind)) is None


@pytest.mark.parametrize("kind", ["confirm", "yesno", "yesno_inverted"])
@pytest.mark.parametrize("text", ["취소", "취소.", "취소해줘", "안내 취소"])
def test_p6_cancel_word_is_not_a_yes_no_rule_answer(on, kind, text):
    """"취소"는 NEGATIVES 에도 있지만 뜻이 질문에 따라 갈린다 — "화장실로 안내해드릴까요?"에는 거절, "안내를 취소할까요?"에는"""
    assert af.short_answer(text, q(kind, "안내를 취소할까요?")) is None


@pytest.mark.parametrize("text", ["아니", "아니요", "아뇨", "아니야", "아니에요", "싫어", "싫어요", "아니요."])
def test_p6_other_negatives_stay_no(on, text):
    for kind in ("confirm", "yesno", "yesno_inverted"):
        assert af.short_answer(text, q(kind)) == ("no", "")


@pytest.mark.parametrize("text", ["가자", "가줘", "가자.", "가 줘"])
def test_go_words_are_yes_only_for_destination_confirm(on, text):
    """사용자 결정(2026-11-22): 통째 "가자"·"가줘"는 목적지 확인 질문(kind confirm)에서만 "네"다. 그 밖의 네/아니요 질문"""
    assert af.short_answer(text, q("confirm")) == ("yes", "")
    for kind in ("yesno", "yesno_inverted"):
        assert af.short_answer(text, q(kind, "안내를 취소할까요?")) is None


def test_go_word_keeps_the_llm_no_outside_destination_confirm(on):
    """"안내를 취소할까요?"에 "가자" — Realtime 이 거절(deny·no)로 읽으면 대답 칸은 no 그대로다(전에는 규칙 yes 와 반대라"""
    asked = q("yesno", "안내를 취소할까요?")
    out = af.fill(VicaIntent(intent="deny", reply=""), asked, af.short_answer("가자", asked), "no")
    assert (out.intent, out.answer) == ("deny", "no")
    confirm = q("confirm")
    go = af.fill(VicaIntent(intent="affirm", reply=""), confirm, af.short_answer("가자", confirm), "")
    assert (go.intent, go.answer) == ("affirm", "yes")


def test_short_answer_minutes(on):
    m = q("minutes", "몇 분쯤 걸리실까요?")
    assert af.short_answer("삼십분.", m) == ("minutes", "30")
    assert af.short_answer("한 십 분 정도.", m) == ("minutes", "10")
    assert af.short_answer("한 시간.", m) == ("minutes", "60")
    assert af.short_answer("십 분에서 십오 분", m, local=True) == ("minutes", "15")
    assert af.short_answer("십 분에서 십오 분", m) == ("minutes", "23")
    assert af.short_answer("삼십분.", None) is None


INTENTS = get_args(VicaIntentType)
LLM_ANSWERS = ("", *af.ANSWERS, " minutes ")
MINUTES_READ_INTENTS = {"wait"}
MINUTES_READ_ANSWERS = {"", "minutes", "unclear"}


YES_NO_INTENT = {"yes": "affirm", "no": "deny"}


def test_p6_standing_yes_no_only_when_the_llm_read_the_same():
    """수정 2차(검토 Critical): 규칙은 LLM 의 읽기를 뒤집지 않는다 — (가) LLM 대답이 규칙 대답과 같거나 (나) LLM 대답이 빈 값·"""
    for answer in ("yes", "no"):
        for name in INTENTS:
            for llm in LLM_ANSWERS:
                want = llm.strip() == answer or (llm.strip() in ("", "unclear") and name == YES_NO_INTENT[answer])
                assert af.standing((answer, ""), name, llm) is want, (answer, name, llm)


@pytest.mark.parametrize("ruled,name,llm,want", [
    (("yes", ""), "deny", "no", False),
    (("yes", ""), "resume", "no", False),
    (("yes", ""), "resume", "other", False),
    (("yes", ""), "navigate", "destination", False),
    (("yes", ""), "affirm", "", True), (("yes", ""), "affirm", "yes", True), (("yes", ""), "question", "yes", True),
    (("no", ""), "deny", "", True), (("no", ""), "cancel", "no", True),
    (("no", ""), "affirm", "", False), (("yes", ""), "resume", "", False), (("yes", ""), "not_for_robot", "", False),
    (("yes", ""), "affirm", "unclear", True),
    (("no", ""), "deny", "unclear", True),
    (("yes", ""), "unknown", "unclear", False), (("yes", ""), "deny", "unclear", False), (("yes", ""), "affirm", "other", False),
])
def test_p6_standing_yes_no_cases(ruled, name, llm, want):
    assert af.standing(ruled, name, llm) is want


def test_p6_standing_minutes_only_when_the_llm_did_not_read_it_otherwise():
    for name in INTENTS:
        for llm in LLM_ANSWERS:
            want = name in MINUTES_READ_INTENTS and llm.strip() in MINUTES_READ_ANSWERS
            assert af.standing(("minutes", "10"), name, llm) is want, (name, llm)


@pytest.mark.parametrize("name,llm,want", [
    ("finish", "no", False),
    ("finish", "", False),
    ("pause", "", False), ("cancel", "", False), ("deny", "", False), ("navigate", "", False),
    ("question", "", False), ("dismiss", "", False), ("affirm", "", False),
    ("wait", "no", False), ("wait", "destination", False), ("wait", "other", False),
    ("wait", "minutes", True), ("wait", "", True), ("wait", "unclear", True),
    ("unknown", "unclear", False), ("unknown", "", False), ("clarify", "", False), ("not_for_robot", "", False),
])
def test_p6_standing_minutes_cases(name, llm, want):
    assert af.standing(("minutes", "10"), name, llm) is want


def test_p6_standing_needs_a_rule_answer():
    for name in INTENTS:
        assert af.standing(None, name, "yes") is False
    assert not hasattr(af, "transcript_wins")


def test_p6_minutes_rule_does_not_override_another_reading(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    said_no = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "10"), "no")
    assert (said_no.intent, said_no.answer, said_no.answer_value) == ("finish", "no", "")
    no_field = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "10"), "")
    assert (no_field.intent, no_field.answer) == ("finish", "")


def test_p6_minutes_rule_corrects_the_value_when_the_llm_also_read_minutes(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    out = af.fill(VicaIntent(intent="wait", wait_minutes=13), q("minutes"), ("minutes", "15"), "minutes")
    assert (out.intent, out.answer, out.answer_value) == ("wait", "minutes", "15")
    assert out.wait_minutes == 15
    empty = af.fill(VicaIntent(intent="wait", wait_minutes=-1), q("minutes"), ("minutes", "15"), "")
    assert (empty.intent, empty.wait_minutes, empty.answer, empty.answer_value) == ("wait", 15, "minutes", "15")


def test_p7_an_answer_to_the_robots_question_is_never_dropped(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    yes = af.fill(VicaIntent(intent="not_for_robot"), q("confirm"), ("yes", ""), "")
    assert (yes.intent, yes.answer, yes.reply) == ("unknown", "unclear", "")
    mins = af.fill(VicaIntent(intent="not_for_robot"), q("minutes"), ("minutes", "10"), "")
    assert (mins.intent, mins.wait_minutes, mins.answer, mins.answer_value) == ("unknown", -1, "unclear", "")


def test_p7_a_rule_answer_never_leaves_as_not_for_robot(on):
    """not_for_robot + 규칙 대답(네/아니요든 분이든) → 1 에서 서든 안 서든 unknown·unclear(수정 2·3차 규칙 3)."""
    for kind, ruled in (("confirm", ("yes", "")), ("yesno", ("no", "")), ("yesno_inverted", ("yes", "")),
                        ("minutes", ("minutes", "5"))):
        for llm in ("", "unclear", ruled[0], "other"):
            out = af.fill(VicaIntent(intent="not_for_robot", reply="x", ask_text="x", ask_kind="open", need_confirm=True,
                                     destination_candidate="화장실", matched_destination_id="wc", confidence=0.7),
                          q(kind), ruled, llm)
            assert (out.intent, out.answer, out.answer_value, out.question_id) == ("unknown", "unclear", "", "q1"), (kind, llm)
            assert (out.reply, out.ask_text, out.ask_kind, out.need_confirm) == ("", "", "", False)
            assert (out.destination_candidate, out.matched_destination_id, out.confidence) == (None, "", 0.7)
            assert out.wait_minutes == -1
    dropped = af.fill(VicaIntent(intent="not_for_robot"), q("confirm"), None, "")
    assert (dropped.intent, dropped.answer) == ("not_for_robot", "")


def test_p6_whole_yes_no_does_not_overturn_the_llm_reading(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    out = af.fill(VicaIntent(intent="navigate", matched_destination_id="t", need_confirm=True, reply="화장실로 안내해드릴까요?"),
                  q("confirm"), ("no", ""), "destination")
    assert (out.intent, out.answer, out.answer_value, out.matched_destination_id, out.reply, out.need_confirm) == (
        "navigate", "destination", "t", "t", "화장실로 안내해드릴까요?", True)
    refused = af.fill(VicaIntent(intent="deny", reply=""), q("yesno", "안내를 취소할까요?"), ("yes", ""), "no")
    assert (refused.intent, refused.answer) == ("deny", "unclear")


def test_p6_whole_yes_no_only_fills_the_answer_when_the_reading_matches(on):
    asked = VicaIntent(intent="question", destination_candidate="화장실", matched_destination_id="wc", need_confirm=True,
                       reply="무엇을 도와드릴까요?", ask_text="무엇을 도와드릴까요?", ask_kind="open", confidence=0.4,
                       speaker_verdict="pass", speaker_score=0.8, speaker_joined=0.7)
    out = af.fill(asked, q("yesno"), ("yes", ""), "other")
    assert (out.intent, out.answer, out.answer_value, out.question_id) == ("question", "other", "", "q1")
    agreed = af.fill(asked, q("yesno"), ("yes", ""), "yes")
    assert (agreed.intent, agreed.answer, agreed.reply, agreed.ask_text, agreed.ask_kind, agreed.need_confirm) == (
        "question", "yes", "무엇을 도와드릴까요?", "무엇을 도와드릴까요?", "open", True)
    assert (agreed.matched_destination_id, agreed.confidence, agreed.speaker_verdict) == ("wc", 0.4, "pass")
    assert asked.answer == ""
    already = af.fill(VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False), q("confirm"), ("no", ""), "")
    assert (already.intent, already.answer, already.confidence) == ("deny", "no", 1.0)
    inverted = af.fill(VicaIntent(intent="unknown"), q("yesno_inverted", "안내가 필요 없으신가요?"), ("yes", ""), "")
    assert (inverted.intent, inverted.answer) == ("unknown", "")


@pytest.mark.parametrize("name", ["unknown", "clarify"])
@pytest.mark.parametrize("llm", ["unclear", ""])
def test_p6_minutes_rule_does_not_turn_a_non_reading_into_wait(on, name, llm):
    """수정 3차: 못 알아들은 말(unknown·clarify)을 규칙이 wait 로 바꾸지 않는다 — "아까 한 시간 기다렸잖아"가 60분 대기가 되면 안 된다."""
    said = VicaIntent(intent=name, reply="다시 말씀해 주시겠어요?", ask_text="다시 말씀해 주시겠어요?", ask_kind="open",
                      need_confirm=True, confidence=0.3, destination_candidate="화장실", matched_destination_id="wc")
    out = af.fill(said, q("minutes"), ("minutes", "60"), llm)
    assert (out.intent, out.wait_minutes, out.answer, out.answer_value) == (name, -1, llm, "")
    assert (out.reply, out.ask_text, out.ask_kind, out.need_confirm, out.matched_destination_id) == (
        said.reply, said.ask_text, "open", True, "wc")


def test_p6_minutes_rule_that_does_not_stand_keeps_the_llm_answer(on):
    paused = af.fill(VicaIntent(intent="pause", reply="네, 잠시 멈출게요."), q("minutes"), ("minutes", "5"), "")
    assert (paused.intent, paused.answer, paused.answer_value, paused.reply) == ("pause", "", "", "네, 잠시 멈출게요.")
    asked = af.fill(VicaIntent(intent="question", reply="지금 3시예요."), q("minutes"), ("minutes", "5"), "other")
    assert (asked.intent, asked.answer) == ("question", "other")
    odd = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "5"), "banana")
    assert odd.answer == "unclear"


def test_off_or_no_question_changes_nothing(monkeypatch):
    stray = VicaIntent(intent="unknown", reply="x")
    assert af.fill(stray, q("confirm"), ("yes", ""), "") is stray
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert af.fill(stray, None, ("yes", ""), "") is stray
    urgent = VicaIntent(intent="unknown", safety_flag="emergency")
    assert af.fill(urgent, q("minutes"), ("minutes", "10"), "") is urgent
    assert af.fill(urgent, q("confirm"), ("no", ""), "") is urgent


def test_fill_rule_first_then_llm(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="info", need_confirm=True)
    ruled = af.fill(nav, q("yesno"), ("yes", ""), "destination")
    assert (ruled.question_id, ruled.answer, ruled.answer_value) == ("q1", "destination", "info")
    assert (ruled.intent, ruled.matched_destination_id, ruled.need_confirm) == ("navigate", "info", True)
    same = af.fill(VicaIntent(intent="affirm"), q("yesno"), ("yes", ""), "")
    assert (same.intent, same.answer, same.answer_value) == ("affirm", "yes", "")
    by_llm = af.fill(nav, q(), None, "destination")
    assert (by_llm.answer, by_llm.answer_value) == ("destination", "info")
    waited = af.fill(VicaIntent(intent="wait", wait_minutes=10), q("minutes"), None, "minutes")
    assert (waited.answer, waited.answer_value) == ("minutes", "10")
    empty = af.fill(VicaIntent(intent="pause"), q(), None, "")
    assert (empty.question_id, empty.answer) == ("q1", "")


def test_fill_clamps(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="", need_confirm=True)
    assert af.fill(nav, q(), None, "destination").answer == "unclear"
    assert af.fill(VicaIntent(intent="wait"), q("minutes"), None, "minutes").answer == "unclear"
    assert af.fill(nav, q(), None, "choice").answer == "unclear"
    assert af.fill(nav, q(), None, "banana").answer == "unclear"
    urgent = VicaIntent(intent="unknown", safety_flag="emergency")
    assert af.fill(urgent, q(), ("yes", ""), "yes") is urgent
    assert af.fill(nav, None, None, "yes") is nav


def test_narrowed():
    class Draft(BaseModel):
        intent: str = ""

    assert af.narrowed(Draft, "") is Draft
    model = af.narrowed(Draft, "minutes")
    schema = model.model_json_schema()
    assert set(schema["properties"]["answer"]["enum"]) == {"minutes", *BASE}
    assert "answer" in schema["required"]
    assert af.narrowed(Draft, "minutes") is model and af.narrowed(Draft, "confirm") is not model


def test_log_line(on):
    filled = af.fill(VicaIntent(intent="affirm"), q("yesno"), ("yes", ""))
    assert af.log_line(filled, q("yesno")) == "[대답] answer=yes value=- 질문=q1 kind=yesno"
    assert af.log_line(VicaIntent(intent="affirm"), None) == "[대답] 없음"


def test_p7_question_line_says_not_for_robot_comes_first_when_recut_is_on(on, monkeypatch):
    """재단이 켜지면 질문 줄에 'not_for_robot 으로 판단한 말이면 answer 는 빈 문자열'을 덧붙인다(3단계 측정: 줄에 그 말이"""
    plain = af.question_line(q())
    assert "not_for_robot" not in plain
    monkeypatch.setenv("VICA_RECUT", "on")
    line = af.question_line(q())
    assert line.startswith(plain) and line != plain
    assert "not_for_robot" in line and "answer 는 빈 문자열" in line and "먼저" not in line
    assert af.question_line(None) == ""


def test_p6_unclear_from_the_llm_yields_to_an_understood_intent(on):
    """LLM 이 뜻은 읽고(의도) 대답 칸만 unclear 로 냈으면 other 로 바꾼다 — 미션이 대답 칸의 unclear 를 먼저 읽어 "잘 듣지"""
    asked = q()
    side = af.fill(VicaIntent(intent="question", reply="화장실은 4층에 있어요."), asked, None, "unclear")
    assert (side.intent, side.answer) == ("question", "other")
    stop = af.fill(VicaIntent(intent="pause"), asked, None, "unclear")
    assert (stop.intent, stop.answer) == ("pause", "other")
    for name in ("unknown", "clarify"):
        assert af.fill(VicaIntent(intent=name), asked, None, "unclear").answer == "unclear"
    no_id = af.fill(VicaIntent(intent="navigate"), asked, None, "destination")
    assert no_id.answer == "unclear"
    assert af.fill(VicaIntent(intent="question"), asked, None, "other").answer == "other"


CONFIRM_WC = OpenQuestion(id="q1", kind="confirm", text="화장실로 안내해드릴까요?", option_ids=("wc",), opened_at=0.0)


@pytest.mark.parametrize("kind", ["confirm", "yesno", "yesno_inverted"])
def test_p3_i1_llm_answer_against_its_own_intent_is_unclear(on, kind):
    """원칙 3·6: LLM 의 의도와 대답 칸이 정면으로 어긋나면(deny + yes · affirm + no) unclear 로 낮춘다 — 미션이 대답 칸을 먼저 읽어"""
    said_no = af.fill(VicaIntent(intent="deny", reply=""), q(kind), None, "yes")
    assert (said_no.intent, said_no.answer, said_no.answer_value, said_no.question_id) == ("deny", "unclear", "", "q1")
    said_yes = af.fill(VicaIntent(intent="affirm", reply=""), q(kind), None, "no")
    assert (said_yes.intent, said_yes.answer, said_yes.answer_value) == ("affirm", "unclear", "")
    assert af.fill(VicaIntent(intent="deny"), q(kind), None, "no").answer == "no"
    assert af.fill(VicaIntent(intent="affirm"), q(kind), None, "yes").answer == "yes"


@pytest.mark.parametrize("kind", ["confirm", "yesno", "yesno_inverted"])
@pytest.mark.parametrize("name", ["affirm", "deny", "resume", "question", "cancel", "finish", "navigate"])
def test_p6_i1_rule_answer_opposite_to_the_llm_answer_is_unclear(on, kind, name):
    """원칙 6: 들은 글자가 통째로 네/아니요(규칙 대답)인데 LLM 대답이 그 반대(yes↔no)면 어느 쪽도 따르지 않는다 — 대답 칸 unclear,"""
    said = VicaIntent(intent=name, reply="", need_confirm=False, confidence=0.6,
                      matched_destination_id="wc" if name == "navigate" else "")
    for ruled, llm in ((("no", ""), "yes"), (("yes", ""), "no")):
        out = af.fill(said, q(kind), ruled, llm)
        assert (out.intent, out.answer, out.answer_value, out.question_id) == (name, "unclear", "", "q1"), (ruled, llm)
        assert (out.need_confirm, out.matched_destination_id, out.confidence) == (False, said.matched_destination_id, 0.6)
    assert af.fill(said, q(kind), ("no", ""), "no").answer == "no"
    assert af.fill(said, q(kind), ("yes", ""), "yes").answer == "yes"


def test_p3_i1_yes_to_the_confirm_question_for_another_place_is_destination(on):
    """원칙 3: 확인 질문("화장실로 안내해드릴까요?") 중 LLM 이 그 질문의 후보가 아닌 곳(409호)으로 가는 navigate 를 내고 대답 칸만"""
    for nc in (True, False):
        out = af.fill(VicaIntent(intent="navigate", matched_destination_id="r409", destination_candidate="409호",
                                 need_confirm=nc), CONFIRM_WC, None, "yes")
        assert (out.intent, out.answer, out.answer_value, out.question_id) == ("navigate", "destination", "r409", "q1")
        assert (out.need_confirm, out.matched_destination_id, out.destination_candidate) == (nc, "r409", "409호")
    same = af.fill(VicaIntent(intent="navigate", matched_destination_id="wc"), CONFIRM_WC, None, "yes")
    assert (same.answer, same.answer_value) == ("yes", "")
    unknown_options = af.fill(VicaIntent(intent="navigate", matched_destination_id="r409"), q("confirm"), None, "yes")
    assert unknown_options.answer == "yes"
    resume_q = OpenQuestion(id="q2", kind="yesno", text="다시 출발할까요?", option_ids=("wc",), opened_at=0.0)
    assert af.fill(VicaIntent(intent="navigate", matched_destination_id="r409"), resume_q, None, "yes").answer == "yes"
    ruled_no = af.fill(VicaIntent(intent="navigate", matched_destination_id="r409"), CONFIRM_WC, ("no", ""), "yes")
    assert ruled_no.answer == "unclear"


@pytest.mark.parametrize("name,nc", [("affirm", False), ("deny", False), ("navigate", False), ("resume", False)])
def test_p3_i2_unclear_stays_for_a_yes_no_or_settled_intent(on, name, nc):
    """원칙 3·6: 승낙·거절·확정(affirm · deny · 확정 navigate · 확정 resume)의 unclear 는 other 로 바꾸지 않는다 — other 면 미션이"""
    said = VicaIntent(intent=name, need_confirm=nc, matched_destination_id="wc" if name == "navigate" else "")
    out = af.fill(said, CONFIRM_WC, None, "unclear")
    assert (out.intent, out.need_confirm, out.answer, out.answer_value) == (name, nc, "unclear", "")


@pytest.mark.parametrize("name,nc", [("question", False), ("pause", False), ("cancel", False), ("finish", False),
                                     ("wait", False), ("help", False), ("dismiss", False),
                                     ("navigate", True), ("resume", True)])
def test_p6_i2_unclear_still_yields_to_other_readings(on, name, nc):
    """그 밖의 의도(질문·멈춤·취소·끝·대기·제안 …)는 지금처럼 other — 미션이 의도대로 받는다(ca81b88, 확인 질문 중 "거기 어디야")."""
    said = VicaIntent(intent=name, need_confirm=nc, matched_destination_id="r409" if name == "navigate" else "")
    assert af.fill(said, CONFIRM_WC, None, "unclear").answer == "other"


def test_p6_i2_whole_yes_still_fills_over_unclear(on):
    """d956801 그대로 — "어."(규칙 yes) + affirm + unclear 는 규칙 대답 yes 다(standing). I2 는 규칙 대답이 없는 말('음')의 이야기다."""
    assert af.fill(VicaIntent(intent="affirm"), CONFIRM_WC, ("yes", ""), "unclear").answer == "yes"
    assert af.fill(VicaIntent(intent="deny"), CONFIRM_WC, ("no", ""), "unclear").answer == "no"


def test_i1_i2_off_returns_the_same_intent(monkeypatch):
    """꺼짐(VICA_ANSWER 없음)은 받은 의도 그대로 — I1·I2 의 갈래는 모두 이른 반환 뒤에 있다."""
    monkeypatch.delenv("VICA_ANSWER", raising=False)
    for said, ruled, llm in (
            (VicaIntent(intent="deny"), None, "yes"), (VicaIntent(intent="affirm"), ("no", ""), "yes"),
            (VicaIntent(intent="navigate", matched_destination_id="r409"), None, "yes"),
            (VicaIntent(intent="affirm"), None, "unclear")):
        assert af.fill(said, CONFIRM_WC, ruled, llm) is said


def test_p6_confirmed_navigate_with_unclear_is_settled_only_at_a_confirm_question(on):
    """확정 navigate + LLM 의 unclear: 확인 질문 중이면 그대로 unclear(그 확정이 곧 출발이라 다시 묻게 한다 — 최종 검토 I2)."""
    picked = VicaIntent(intent="navigate", matched_destination_id="m", need_confirm=False)
    assert af.fill(picked, q("choice", "남자 화장실과 여자 화장실 중 어디로 갈까요?"), None, "unclear").answer == "other"
    assert af.fill(picked, q("yesno", "여기서 기다릴까요?"), None, "unclear").answer == "other"
    assert af.fill(picked, q("confirm"), None, "unclear").answer == "unclear"
    assert af.fill(VicaIntent(intent="affirm"), q("choice", "A와 B 중 어디로 갈까요?"), None, "unclear").answer == "unclear"
