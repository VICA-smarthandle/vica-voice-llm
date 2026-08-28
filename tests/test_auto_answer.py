"""접근 질문 자동 답변 창 판정 시험."""
from src.auto_answer import should_auto_listen


def test_approach_question_triggers():
    assert should_auto_listen("안내가 필요하신가요?")


def test_other_utterances_do_not_trigger():
    # mission 의 다른 접근 멘트들 — 이 문장 뒤에 마이크가 열리면 안 된다.
    assert not should_auto_listen("네, 제가 도와드리겠습니다.")
    assert not should_auto_listen("실례했습니다. 필요하시면 언제든 불러 주세요.")
    assert not should_auto_listen("알겠습니다. 필요하시면 언제든 불러 주세요.")
    assert not should_auto_listen("안내소에 도착했습니다.")
