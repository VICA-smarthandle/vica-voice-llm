"""대화 맥락 보관 검증 (길이 제한 + 대화 경계). LLM/ROS 불필요."""
from __future__ import annotations

from src.history import DEFAULT_IDLE_RESET_SEC, ConversationHistory


def test_keeps_recent_messages_only():
    history = ConversationHistory(max_messages=4)
    history.extend([f"메시지 {i}" for i in range(6)])
    assert history.messages == ["메시지 2", "메시지 3", "메시지 4", "메시지 5"]


def test_first_turn_is_not_a_reset():
    history = ConversationHistory()
    assert history.begin_turn(now=100.0) is False


def test_continuous_conversation_keeps_context():
    history = ConversationHistory(idle_reset_sec=180.0)
    history.begin_turn(now=100.0)
    history.extend(["앞선 대화"])

    assert history.begin_turn(now=150.0) is False
    assert history.messages == ["앞선 대화"]


def test_idle_gap_clears_context():
    """다음 사용자의 "거기로 가줘"가 앞사람 목적지로 해석되면 안 된다."""
    history = ConversationHistory(idle_reset_sec=180.0)
    history.begin_turn(now=100.0)
    history.extend(["앞사람이 말한 목적지"])

    assert history.begin_turn(now=300.0) is True
    assert history.messages == []


def test_boundary_is_exclusive():
    history = ConversationHistory(idle_reset_sec=180.0)
    history.begin_turn(now=0.0)
    history.extend(["맥락"])
    assert history.begin_turn(now=179.9) is False
    assert history.messages == ["맥락"]


def test_gap_measured_from_last_turn_not_first():
    """짧은 간격으로 계속 말하면 총 시간이 길어도 같은 대화다."""
    history = ConversationHistory(idle_reset_sec=180.0)
    history.begin_turn(now=0.0)
    history.extend(["맥락"])
    for now in (100.0, 200.0, 300.0, 400.0):
        assert history.begin_turn(now=now) is False
    assert history.messages == ["맥락"]


def test_messages_property_is_a_copy():
    """밖에서 고쳐도 내부 상태가 바뀌면 안 된다."""
    history = ConversationHistory()
    history.extend(["원본"])
    history.messages.append("바깥에서 추가")
    assert history.messages == ["원본"]


def test_clear_resets_turn_tracking():
    history = ConversationHistory()
    history.begin_turn(now=100.0)
    history.extend(["맥락"])
    history.clear()

    assert history.messages == []
    assert history.begin_turn(now=100_000.0) is False  # 첫 발화로 취급


def test_len():
    history = ConversationHistory()
    history.extend(["하나", "둘"])
    assert len(history) == 2


def test_default_idle_reset_is_minutes_not_seconds():
    """사용자가 잠깐 생각하는 사이에 맥락이 사라지면 안 된다."""
    assert DEFAULT_IDLE_RESET_SEC >= 60.0
