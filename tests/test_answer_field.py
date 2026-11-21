"""대답 칸(3-나-2 1단계, 설계 2026-11-18 4.1~4.3)."""
import pytest
from pydantic import BaseModel

from src import answer_field as af
from src.question_record import OpenQuestion
from src.schema import VicaIntent

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


def test_short_answer_minutes(on):
    m = q("minutes", "몇 분쯤 걸리실까요?")
    assert af.short_answer("삼십분.", m) == ("minutes", "30")
    assert af.short_answer("한 십 분 정도.", m) == ("minutes", "10")
    assert af.short_answer("한 시간.", m) == ("minutes", "60")
    assert af.short_answer("십 분에서 십오 분", m, local=True) == ("minutes", "15")
    assert af.short_answer("십 분에서 십오 분", m) == ("minutes", "23")   # 로컬 규칙 밖은 큰 쪽 × 1.5(지금 _finalize 와 같다)
    assert af.short_answer("삼십분.", None) is None


def test_transcript_rule_only_corrects_minutes():
    assert af.transcript_wins(("yes", ""), "destination") is True       # 통째 네/아니요는 받아쓴 글자가 이긴다(D11)
    assert af.transcript_wins(("no", ""), "yes") is True
    for heard in ("minutes", "", "unclear"):
        assert af.transcript_wins(("minutes", "30"), heard) is True      # 분은 값 바로잡기
    for heard in ("no", "other", "destination", "yes"):
        assert af.transcript_wins(("minutes", "10"), heard) is False     # "십 분 말고 그냥 끝내 줘" — Realtime 의 거절을 덮지 않는다
    assert af.transcript_wins(None, "yes") is False


def test_fill_rule_first_then_llm(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="info", need_confirm=True)
    ruled = af.fill(nav, q("yesno"), ("yes", ""), "destination")
    assert (ruled.question_id, ruled.answer, ruled.answer_value) == ("q1", "yes", "")
    assert (ruled.intent, ruled.matched_destination_id, ruled.need_confirm) == ("navigate", "info", True)
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
