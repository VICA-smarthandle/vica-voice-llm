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
    monkeypatch.setenv("VICA_ANSWER", "1")      # on 이라는 글자만 켠다(VICA_TOOLS_* 와 같은 규칙)
    assert af.on() is False


def test_local_schema_has_no_answer_field_unless_asked(monkeypatch):
    assert af.local_on() is False                      # 대답 칸이 꺼져 있으면 VICA_ANSWER_LOCAL 만으로는 안 켜진다
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    assert af.local_on() is False
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert af.local_on() is True
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "1")       # on 이라는 글자만 켠다
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
    # 값의 뜻까지 줄에 적는다 — 로컬(Ollama)은 스키마를 출력 문법으로만 쓰고 칸 설명을 모델에 보여 주지 않는다(검토 I2).
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
    ("그래.", "yesno_inverted", ("yes", "")),      # 들은 대로 — 뜻(필요 없음)은 미션이 질문 종류로 읽는다
])
def test_short_answer_whole_yes_no(on, text, kind, out):
    assert af.short_answer(text, q(kind)) == out


@pytest.mark.parametrize("text,kind", [
    ("네 화장실로 가자", "confirm"), ("그래 맞아.", "yesno"), ("부요.", "yesno"),
    # 혼잣소리('음')는 규칙이 정하지 않는다 — yesno 에는 "다시 출발할까요?"처럼 로봇이 움직이는 질문도 있다(09-28, 검토 I4).
    ("음", "confirm"), ("음", "yesno"), ("어어", "yesno_inverted"),
    ("네", "open"), ("네", "choice"), ("네", "minutes"), ("삼십분.", "confirm"),
])
def test_short_answer_only_whole_utterance(on, text, kind):
    assert af.short_answer(text, q(kind)) is None


@pytest.mark.parametrize("kind", ["confirm", "yesno", "yesno_inverted"])
@pytest.mark.parametrize("text", ["취소", "취소.", "취소해줘", "안내 취소"])
def test_p6_cancel_word_is_not_a_yes_no_rule_answer(on, kind, text):
    """"취소"는 NEGATIVES 에도 있지만 뜻이 질문에 따라 갈린다 — "화장실로 안내해드릴까요?"에는 거절, "안내를 취소할까요?"에는
    승낙. 네/아니요 규칙이 정하지 않는다(수정 1차) — 의도는 지금 길(취소 지름길·LLM)이 정한다."""
    assert af.short_answer(text, q(kind, "안내를 취소할까요?")) is None


@pytest.mark.parametrize("text", ["아니", "아니요", "아뇨", "아니야", "아니에요", "싫어", "싫어요", "아니요."])
def test_p6_other_negatives_stay_no(on, text):
    for kind in ("confirm", "yesno", "yesno_inverted"):
        assert af.short_answer(text, q(kind)) == ("no", "")


def test_short_answer_minutes(on):
    m = q("minutes", "몇 분쯤 걸리실까요?")
    assert af.short_answer("삼십분.", m) == ("minutes", "30")
    assert af.short_answer("한 십 분 정도.", m) == ("minutes", "10")
    assert af.short_answer("한 시간.", m) == ("minutes", "60")
    assert af.short_answer("십 분에서 십오 분", m, local=True) == ("minutes", "15")
    assert af.short_answer("십 분에서 십오 분", m) == ("minutes", "23")   # 로컬 규칙 밖은 큰 쪽 × 1.5(지금 _finalize 와 같다)
    assert af.short_answer("삼십분.", None) is None


# ----- 규칙 대답이 서는 조건(Task 3.5b, 원칙 6·7) — transcript_wins 를 대신한다 -----------------------------------------
INTENTS = get_args(VicaIntentType)
LLM_ANSWERS = ("", *af.ANSWERS, " minutes ")
MINUTES_READ_INTENTS = {"wait", "unknown", "clarify", "not_for_robot"}     # 설명서 규칙 1 의 글자 그대로
MINUTES_READ_ANSWERS = {"", "minutes", "unclear"}


def test_p6_standing_whole_yes_no_always_stands():
    for ruled in (("yes", ""), ("no", "")):
        for name in INTENTS:
            for llm in LLM_ANSWERS:
                assert af.standing(ruled, name, llm) is True, (ruled, name, llm)


def test_p6_standing_minutes_only_when_the_llm_did_not_read_it_otherwise():
    for name in INTENTS:
        for llm in LLM_ANSWERS:
            want = name in MINUTES_READ_INTENTS and llm.strip() in MINUTES_READ_ANSWERS
            assert af.standing(("minutes", "10"), name, llm) is want, (name, llm)


@pytest.mark.parametrize("name,llm,want", [
    ("finish", "no", False),          # "십 분 말고 그냥 끝내 줘" — LLM 의 거절을 덮지 않는다(검토 I3)
    ("finish", "", False),            # 로컬 — 대답 칸이 없어도 의도가 다른 뜻이면 서지 않는다
    ("pause", "", False), ("cancel", "", False), ("deny", "", False), ("navigate", "", False),
    ("question", "", False), ("dismiss", "", False), ("affirm", "", False),
    ("wait", "no", False), ("wait", "destination", False), ("wait", "other", False),
    ("wait", "minutes", True), ("wait", "", True), ("unknown", "unclear", True),
    ("clarify", "", True), ("not_for_robot", "", True),
])
def test_p6_standing_minutes_cases(name, llm, want):
    assert af.standing(("minutes", "10"), name, llm) is want


def test_p6_standing_needs_a_rule_answer():
    for name in INTENTS:
        assert af.standing(None, name, "yes") is False
    assert not hasattr(af, "transcript_wins")          # 두 벌로 남기지 않는다


def test_p6_minutes_rule_does_not_override_another_reading(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    said_no = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "10"), "no")      # "십 분 말고 그냥 끝내 줘"
    assert (said_no.intent, said_no.answer, said_no.answer_value) == ("finish", "no", "")
    no_field = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "10"), "")       # 로컬 — 대답 칸 없음
    assert (no_field.intent, no_field.answer) == ("finish", "")


def test_p6_minutes_rule_corrects_the_value_when_the_llm_also_read_minutes(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    out = af.fill(VicaIntent(intent="wait", wait_minutes=13), q("minutes"), ("minutes", "15"), "minutes")
    assert (out.intent, out.answer, out.answer_value) == ("wait", "minutes", "15")
    # 수정 1차: 의도는 그대로, wait_minutes 도 규칙의 분 — 두 칸이 갈리면 미션이 읽는 칸에 따라 대기 시간이 달라진다.
    assert out.wait_minutes == 15


def test_p7_an_answer_to_the_robots_question_is_never_dropped(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    # not_for_robot 은 VicaIntent 가 받는 값이다(schema.VicaIntentType — 스위치와 무관하게 스키마에 있다)
    yes = af.fill(VicaIntent(intent="not_for_robot"), q("confirm"), ("yes", ""), "")
    assert (yes.intent, yes.answer, yes.reply) == ("affirm", "yes", "")
    mins = af.fill(VicaIntent(intent="not_for_robot"), q("minutes"), ("minutes", "10"), "")
    assert (mins.intent, mins.wait_minutes, mins.answer, mins.answer_value) == ("wait", 10, "minutes", "10")


def test_p7_a_standing_rule_answer_never_leaves_as_not_for_robot(on):
    for kind, ruled in (("confirm", ("yes", "")), ("yesno", ("no", "")), ("yesno_inverted", ("yes", "")),
                        ("minutes", ("minutes", "5"))):
        for llm in ("", "unclear"):
            out = af.fill(VicaIntent(intent="not_for_robot"), q(kind), ruled, llm)
            assert out.intent != "not_for_robot" and (out.answer, out.answer_value) == ruled, (kind, llm)


def test_p6_whole_yes_no_replaces_what_the_llm_made_of_it(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    out = af.fill(VicaIntent(intent="navigate", matched_destination_id="t", need_confirm=True, reply="화장실로 안내해드릴까요?"),
                  q("confirm"), ("no", ""), "destination")
    assert (out.intent, out.answer, out.matched_destination_id, out.reply, out.need_confirm) == ("deny", "no", "", "", False)


def test_p6_whole_yes_no_clears_the_sentence_fields_and_keeps_the_rest(on):
    asked = VicaIntent(intent="question", destination_candidate="화장실", matched_destination_id="wc", need_confirm=True,
                       reply="무엇을 도와드릴까요?", ask_text="무엇을 도와드릴까요?", ask_kind="open", confidence=0.4,
                       speaker_verdict="pass", speaker_score=0.8, speaker_joined=0.7)
    out = af.fill(asked, q("yesno"), ("yes", ""), "other")
    assert (out.intent, out.answer, out.answer_value, out.question_id) == ("affirm", "yes", "", "q1")
    assert (out.reply, out.ask_text, out.ask_kind, out.need_confirm) == ("", "", "", False)
    assert (out.destination_candidate, out.matched_destination_id) == (None, "")
    assert (out.confidence, out.safety_flag) == (0.4, "normal")                       # 다른 칸은 그대로
    assert (out.speaker_verdict, out.speaker_score, out.speaker_joined) == ("pass", 0.8, 0.7)
    assert asked.intent == "question" and asked.reply == "무엇을 도와드릴까요?"       # 받은 의도는 바꾸지 않는다(사본)
    already = af.fill(VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False), q("confirm"), ("no", ""), "")
    assert (already.intent, already.answer, already.confidence) == ("deny", "no", 1.0)
    inverted = af.fill(VicaIntent(intent="unknown"), q("yesno_inverted", "안내가 필요 없으신가요?"), ("yes", ""), "")
    assert (inverted.intent, inverted.answer) == ("affirm", "yes")    # 들은 대로 — 뜻은 미션이 질문 종류로 읽는다


@pytest.mark.parametrize("name", ["unknown", "clarify", "not_for_robot"])
def test_p6_minutes_rule_turns_a_non_reading_into_wait(on, name):
    out = af.fill(VicaIntent(intent=name, reply="다시 말씀해 주시겠어요?", ask_text="다시 말씀해 주시겠어요?",
                             need_confirm=True, confidence=0.3, destination_candidate="화장실", matched_destination_id="wc"),
                  q("minutes"), ("minutes", "10"), "unclear")
    assert (out.intent, out.wait_minutes, out.answer, out.answer_value) == ("wait", 10, "minutes", "10")
    assert (out.reply, out.ask_text, out.need_confirm, out.confidence) == ("", "", False, 0.3)
    assert (out.destination_candidate, out.matched_destination_id) == (None, "")     # 수정 1차 — 네/아니요와 같은 꼴


def test_p6_minutes_rule_that_does_not_stand_keeps_the_llm_answer(on):
    paused = af.fill(VicaIntent(intent="pause", reply="네, 잠시 멈출게요."), q("minutes"), ("minutes", "5"), "")
    assert (paused.intent, paused.answer, paused.answer_value, paused.reply) == ("pause", "", "", "네, 잠시 멈출게요.")
    asked = af.fill(VicaIntent(intent="question", reply="지금 3시예요."), q("minutes"), ("minutes", "5"), "other")
    assert (asked.intent, asked.answer) == ("question", "other")
    odd = af.fill(VicaIntent(intent="finish"), q("minutes"), ("minutes", "5"), "banana")
    assert odd.answer == "unclear"                                   # 서지 않으면 지금처럼 LLM 값을 다듬는다


def test_off_or_no_question_changes_nothing(monkeypatch):
    stray = VicaIntent(intent="unknown", reply="x")
    assert af.fill(stray, q("confirm"), ("yes", ""), "") is stray               # 꺼짐
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert af.fill(stray, None, ("yes", ""), "") is stray                       # 열린 질문 없음
    urgent = VicaIntent(intent="unknown", safety_flag="emergency")
    assert af.fill(urgent, q("minutes"), ("minutes", "10"), "") is urgent      # 긴급
    assert af.fill(urgent, q("confirm"), ("no", ""), "") is urgent


def test_fill_rule_first_then_llm(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="info", need_confirm=True)
    ruled = af.fill(nav, q("yesno"), ("yes", ""), "destination")
    assert (ruled.question_id, ruled.answer, ruled.answer_value) == ("q1", "yes", "")
    # Task 3.5b(규칙 2): 통째 네/아니요가 서면 의도도 affirm 이다 — 예전에는 navigate 를 그대로 두었다.
    assert (ruled.intent, ruled.matched_destination_id, ruled.need_confirm) == ("affirm", "", False)
    by_llm = af.fill(nav, q(), None, "destination")
    assert (by_llm.answer, by_llm.answer_value) == ("destination", "info")
    waited = af.fill(VicaIntent(intent="wait", wait_minutes=10), q("minutes"), None, "minutes")
    assert (waited.answer, waited.answer_value) == ("minutes", "10")
    empty = af.fill(VicaIntent(intent="pause"), q(), None, "")
    assert (empty.question_id, empty.answer) == ("q1", "")


def test_fill_clamps(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="", need_confirm=True)
    assert af.fill(nav, q(), None, "destination").answer == "unclear"          # 목적지 id 없음
    assert af.fill(VicaIntent(intent="wait"), q("minutes"), None, "minutes").answer == "unclear"   # 분 없음(-1)
    assert af.fill(nav, q(), None, "choice").answer == "unclear"               # 확인 질문에 허용 밖
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
