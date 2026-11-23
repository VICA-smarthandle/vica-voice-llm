"""재단(의도 분류 다시 자르기) 문장 틀·스위치 시험 — 계획 2026-11-20 Task 2.1."""
from src.intent_recut import clarify_reply, clue_line, speaker_line, with_clue, with_speaker
from src.question_record import OpenQuestion
from src.replies import ASK_DESTINATION, not_available


def test_p11_not_available_sentence():
    assert not_available("스무디 파는 곳") == "제가 아는 목적지 중에서 스무디 파는 곳은 없는데, 어디로 안내해드릴까요?"
    assert not_available("카페") == "제가 아는 목적지 중에서 카페는 없는데, 어디로 안내해드릴까요?"
    assert not_available("") == "제가 아는 목적지 중에서 말씀하신 곳은 없는데, 어디로 안내해드릴까요?"


def test_p11_not_available_never_speaks_emergency_words():
    for wanted in ("멈춰 버튼", "정지하는 곳", "가" * 21, "스무디?\n파는 곳"):
        assert not_available(wanted).startswith("제가 아는 목적지 중에서 말씀하신 곳은")


def test_p4_clarify_reply_by_kind():
    assert clarify_reply("several", "", ["남자 화장실", "여자 화장실"]) == "남자 화장실과 여자 화장실 중 어디로 갈까요?"
    assert clarify_reply("several", "", ["남자 화장실"]) == ASK_DESTINATION
    assert clarify_reply("not_available", "스무디", []) == not_available("스무디")
    assert clarify_reply("no_destination", "", []) == ASK_DESTINATION
    assert clarify_reply("", "", []) == ASK_DESTINATION and clarify_reply("엉뚱", "", []) == ASK_DESTINATION


def test_p7_clue_line(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    wake = OpenQuestion("q1", "open", "네?", (), 0.0)
    asked = OpenQuestion("q2", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert clue_line(None) == ""
    assert clue_line(wake) == '- 로봇이 방금 한 말: "네?" (사용자가 "비카야"라고 불러서 대답했다)'
    assert clue_line(asked) == '- 로봇이 방금 한 말: "화장실로 안내해드릴까요?" (확인 질문 — 대답을 기다리는 중)'
    assert with_clue("\n[지금 상황]\n- 층: 4층\n", asked).endswith(clue_line(asked) + "\n")
    assert with_clue("", asked).startswith("\n[지금 상황]\n")


def test_clue_is_off_by_default_and_yields_to_the_answer_line(monkeypatch):
    asked = OpenQuestion("q2", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert with_clue("x", asked) == "x"
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_ANSWER", "on")
    assert with_clue("x", asked) == "x"


def test_p19_speaker_line():
    assert speaker_line("pass") == "- 이 말의 목소리: 안내를 요청한 사용자와 같다"
    assert speaker_line("fail") == "- 이 말의 목소리: 안내를 요청한 사용자와 다르다"
    assert speaker_line("short") == "" and speaker_line("no_target") == "" and speaker_line("") == ""


def test_switch_reads_env_each_call(monkeypatch):
    from src import intent_recut
    assert intent_recut.on() is False
    monkeypatch.setenv("VICA_RECUT", "on")
    assert intent_recut.on() is True


def test_p19_with_speaker_appends_one_line_only_when_on_and_known(monkeypatch):
    base = "\n[지금 상황]\n- 층: 4층\n"
    assert with_speaker(base, "pass") == base
    monkeypatch.setenv("VICA_RECUT", "on")
    assert with_speaker(base, "pass") == base + speaker_line("pass") + "\n"
    assert with_speaker(base.rstrip("\n"), "fail") == base + speaker_line("fail") + "\n"
    assert with_speaker("", "pass").startswith("\n[지금 상황]\n- 이 말의 목소리")
    for v in ("short", "no_target", "", None):
        assert with_speaker(base, v) == base


def test_p11_not_available_does_not_read_out_words_without_hangul():
    for wanted in ("I need some snack", "我证件在哪", "B1", "123"):
        assert not_available(wanted) == "제가 아는 목적지 중에서 말씀하신 곳은 없는데, 어디로 안내해드릴까요?"
    assert not_available("407호") == "제가 아는 목적지 중에서 407호는 없는데, 어디로 안내해드릴까요?"
    assert not_available("AI 체험관") == "제가 아는 목적지 중에서 AI 체험관은 없는데, 어디로 안내해드릴까요?"


def test_p13_after_a_call_the_clue_line_stays_even_with_answers_on(monkeypatch):
    """대답 칸이 켜져도 "네?"(열린 질문) 뒤에는 질문 줄 대신 단서 줄을 싣는다 — 미션은 열린 질문의 대답 칸을 읽지 않고,"""
    from src import answer_field
    wake = OpenQuestion("q1", "open", "네?", (), 0.0)
    asked = OpenQuestion("q2", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_ANSWER", "on")
    assert with_clue("x", wake) == "x\n" + clue_line(wake) + "\n"
    assert answer_field.with_question("x", wake) == "x"
    assert with_clue("x", asked) == "x"
    assert "로봇이 물은 질문" in answer_field.with_question("x", asked)
    assert "로봇이 물은 질문" in answer_field.with_question("x", wake, recut=False)
    monkeypatch.setenv("VICA_RECUT", "off")
    assert "로봇이 물은 질문" in answer_field.with_question("x", wake)


def test_p13_prompt_after_a_call_is_the_stage2_prompt_when_answers_are_on(monkeypatch):
    """재단 + 대답 칸에서 "네?" 뒤의 지시문은 재단만 켠 지시문(2단계에서 잰 것)과 글자까지 같다."""
    from src import langchain_intent_parser as lip
    wake = OpenQuestion("q1", "open", "네?", (), 0.0)
    monkeypatch.setenv("VICA_RECUT", "on")
    stage2 = lip._build_system_prompt([], None, question=wake)
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert lip._build_system_prompt([], None, question=wake) == stage2
