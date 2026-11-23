"""미션이 낸 질문 기록을 받아 로그로 남긴다(설계 2026-11-14 4.1, 2단계). 해석에는 쓰지 않는다."""
from __future__ import annotations

from pathlib import Path

from src.question_record import OpenQuestion, from_fields, log_line

NODE = (Path(__file__).resolve().parents[1] / "src" / "ros_node.py").read_text(encoding="utf-8")


def _body(name: str) -> str:
    start = NODE.index(f"    def {name}(")
    end = NODE.find("\n    def ", start + 1)
    return NODE[start: end if end != -1 else len(NODE)]


def test_empty_id_means_no_question():
    assert from_fields("", "", "", [], 0.0) is None


def test_fields_become_an_open_question():
    q = from_fields("q7-3", "confirm", "화장실로 안내해드릴까요?", ["d1"], 100.0)
    assert q == OpenQuestion(id="q7-3", kind="confirm", text="화장실로 안내해드릴까요?", option_ids=("d1",),
                             opened_at=100.0)


def test_log_line_without_question():
    assert log_line(None, 50.0) == "[질문] 없음"


def test_log_line_with_question():
    q = from_fields("q7-3", "yesno", "여기서 대기할까요?", [], 100.0)
    assert log_line(q, 104.2) == "[질문] id=q7-3 kind=yesno 나이=4.2s"


def test_age_is_never_negative():
    q = from_fields("q7-3", "open", "네?", [], 100.0)
    assert log_line(q, 99.7) == "[질문] id=q7-3 kind=open 나이=0.0s"


def test_node_guards_a_missing_interface():
    at = NODE.index("import VicaQuestion")
    assert "except ImportError" in NODE[at: at + 200]
    assert "if VicaQuestionMsg is not None" in NODE


def test_node_subscribes_latched_like_the_publisher():
    at = NODE.index('"/vica/question"')
    assert "TRANSIENT_LOCAL" in NODE[at: at + 300] and "depth=1" in NODE[at: at + 300]


def test_node_takes_the_question_before_publishing_the_intent():
    body = _body("_publish_intent")
    assert body.index("question = self._question") < body.index("self._intent_pub.publish(")


def test_node_logs_the_question_right_after_the_input_line():
    body = _body("_publish_intent")
    at = body.index("입력='{text}'")
    assert "log_line(question, time.time())" in body[at: at + 400]
