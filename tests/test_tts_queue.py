"""발화 큐(우선순위·선점·중복 억제) 검증. 음성 모델/ROS 불필요."""
from __future__ import annotations

from src.tts_queue import (
    EMERGENCY,
    NARRATION,
    RESPONSE,
    TtsQueue,
    build_request,
    parse_request,
    request_for_intent,
)


# ---- 요청 파싱 ---------------------------------------------------------------


def test_parse_request_with_prefix():
    assert parse_request("narration:안내를 시작합니다.") == (
        NARRATION,
        "안내를 시작합니다.",
    )
    assert parse_request("response:네, 알겠습니다.") == (RESPONSE, "네, 알겠습니다.")


def test_parse_request_without_prefix_is_narration():
    assert parse_request("접두어가 없는 문장") == (NARRATION, "접두어가 없는 문장")


def test_parse_request_keeps_colon_in_body():
    """본문에 콜론이 있어도 잘리면 안 된다."""
    assert parse_request("response:도착 예정: 3분 뒤") == (RESPONSE, "도착 예정: 3분 뒤")


def test_parse_request_unknown_prefix_is_not_treated_as_priority():
    assert parse_request("주의:계단이 있습니다") == (NARRATION, "주의:계단이 있습니다")


def test_parse_request_empty():
    assert parse_request("") == (NARRATION, "")


# ---- 우선순위와 순서 ---------------------------------------------------------


def test_higher_priority_goes_first():
    queue = TtsQueue()
    queue.push(NARRATION, "안내입니다", now=0.0)
    queue.push(RESPONSE, "대답입니다", now=0.1)

    assert queue.pop(now=0.0).text == "대답입니다"
    assert queue.pop(now=0.0).text == "안내입니다"


def test_same_priority_keeps_order():
    queue = TtsQueue()
    queue.push(NARRATION, "첫째", now=0.0)
    queue.push(NARRATION, "둘째", now=0.1)
    queue.push(NARRATION, "셋째", now=0.2)

    assert [queue.pop(now=0.0).text for _ in range(3)] == ["첫째", "둘째", "셋째"]


def test_pop_empty_returns_none():
    assert TtsQueue().pop(now=0.0) is None


# ---- 긴급 발화 선점 ----------------------------------------------------------


def test_emergency_preempts_and_flushes():
    queue = TtsQueue()
    queue.push(NARRATION, "안내입니다", now=0.0)
    queue.push(RESPONSE, "대답입니다", now=0.1)

    result = queue.push(EMERGENCY, "안전을 위해 멈추겠습니다.", now=0.2)
    assert result.accepted and result.preempt

    assert queue.pop(now=0.0).text == "안전을 위해 멈추겠습니다."
    assert queue.pop(now=0.0) is None  # 대기 중이던 일반 발화는 밀려났다


def test_emergency_keeps_other_emergency():
    """안전 관련 발화끼리는 서로 지우지 않는다."""
    queue = TtsQueue()
    queue.push(EMERGENCY, "첫 경고", now=0.0)
    queue.push(EMERGENCY, "둘째 경고", now=0.1)

    assert [queue.pop(now=0.0).text for _ in range(2)] == ["첫 경고", "둘째 경고"]


def test_normal_push_does_not_preempt():
    queue = TtsQueue()
    assert queue.push(RESPONSE, "대답입니다", now=0.0).preempt is False


# ---- 중복 억제 ---------------------------------------------------------------


def test_duplicate_within_window_is_dropped():
    queue = TtsQueue()
    assert queue.push(NARRATION, "같은 말", now=0.0).accepted is True
    assert queue.push(NARRATION, "같은 말", now=1.0).accepted is False
    assert len(queue) == 1


def test_duplicate_after_window_is_accepted():
    queue = TtsQueue(dedup_sec=2.0)
    queue.push(NARRATION, "같은 말", now=0.0)
    assert queue.push(NARRATION, "같은 말", now=2.5).accepted is True
    assert len(queue) == 2


def test_empty_text_is_rejected():
    queue = TtsQueue()
    assert queue.push(NARRATION, "   ", now=0.0).accepted is False
    assert len(queue) == 0


# ---- 적체 방지 ---------------------------------------------------------------


def test_overflow_drops_oldest_narration():
    """정원을 넘으면 가장 오래된 안내부터 버린다 (묵은 안내가 뒤늦게 나오지 않게)."""
    queue = TtsQueue(max_len=3, dedup_sec=0.0)
    for index in range(5):
        queue.push(NARRATION, f"안내 {index}", now=float(index))

    assert [queue.pop(now=0.0).text for _ in range(3)] == ["안내 2", "안내 3", "안내 4"]


def test_overflow_keeps_higher_priority():
    """밀릴 때 답변이 안내보다 먼저 버려지면 안 된다."""
    queue = TtsQueue(max_len=2, dedup_sec=0.0)
    queue.push(RESPONSE, "대답입니다", now=0.0)
    queue.push(NARRATION, "안내 1", now=1.0)
    queue.push(NARRATION, "안내 2", now=2.0)

    assert [queue.pop(now=0.0).text for _ in range(2)] == ["대답입니다", "안내 2"]


def test_overflow_reports_what_was_dropped():
    """버려진 발화를 돌려줘야 노드가 로그로 남길 수 있다.

    조용히 사라지면 "왜 그 안내가 안 나왔는지" 를 사후에 추적할 수 없다.
    """
    queue = TtsQueue(max_len=2, dedup_sec=0.0)
    assert queue.push(NARRATION, "안내 0", now=0.0).dropped == ()
    assert queue.push(NARRATION, "안내 1", now=1.0).dropped == ()

    result = queue.push(NARRATION, "안내 2", now=2.0)
    assert result.accepted
    assert result.dropped == ("안내 0",)      # 가장 오래된 것이 버려졌다


def test_emergency_reports_preempted_utterances():
    """긴급이 밀어낸 대기 발화도 사라진 말이므로 보고한다."""
    queue = TtsQueue(dedup_sec=0.0)
    queue.push(NARRATION, "안내입니다", now=0.0)
    queue.push(RESPONSE, "대답입니다", now=1.0)

    result = queue.push(EMERGENCY, "안전을 위해 멈추겠습니다", now=2.0)
    assert result.preempt
    assert set(result.dropped) == {"안내입니다", "대답입니다"}


def test_no_drop_reported_when_queue_has_room():
    queue = TtsQueue(dedup_sec=0.0)
    for index in range(3):
        assert queue.push(NARRATION, f"안내 {index}", now=float(index)).dropped == ()


def test_unknown_priority_falls_back_to_narration():
    queue = TtsQueue()
    queue.push("이상한값", "문장", now=0.0)
    queue.push(RESPONSE, "대답", now=0.1)
    assert queue.pop(now=0.0).text == "대답"  # 문장이 narration 으로 내려갔다


# ---- 발화 주체 분리 ----------------------------------------------------------


class _FakeIntent:
    """VicaIntent(pydantic)와 ROS 메시지 양쪽을 흉내 내는 최소 객체."""

    def __init__(self, intent, reply, need_confirm=False, safety_flag="normal"):
        self.intent = intent
        self.reply = reply
        self.need_confirm = need_confirm
        self.safety_flag = safety_flag


def test_navigate_confirmed_is_spoken_by_mission_manager():
    """게이트 통과 여부는 Mission Manager 만 안다. LLM 이 먼저 말하면 안 된다."""
    intent = _FakeIntent("navigate", "식당으로 안내하겠습니다.", need_confirm=False)
    assert request_for_intent(intent) is None


def test_navigate_needing_confirm_is_spoken_by_llm():
    """확인 질문은 이동 결과와 무관하므로 LLM 이 말한다."""
    intent = _FakeIntent("navigate", "식당으로 안내해드릴까요?", need_confirm=True)
    assert request_for_intent(intent) == "response:식당으로 안내해드릴까요?"


def test_question_is_spoken_by_llm():
    intent = _FakeIntent("question", "지금 3층입니다.")
    assert request_for_intent(intent) == "response:지금 3층입니다."


def test_emergency_uses_emergency_priority():
    intent = _FakeIntent(
        "unknown", "안전을 위해 멈추겠습니다.", safety_flag="emergency"
    )
    assert request_for_intent(intent) == "emergency:안전을 위해 멈추겠습니다."


def test_empty_reply_is_not_spoken():
    assert request_for_intent(_FakeIntent("question", "")) is None
    assert request_for_intent(_FakeIntent("question", "   ")) is None


def test_build_request_round_trips():
    raw = build_request(RESPONSE, "도착 예정: 3분 뒤")
    assert parse_request(raw) == (RESPONSE, "도착 예정: 3분 뒤")


def test_build_request_unknown_priority_falls_back():
    assert build_request("이상한값", "문장") == "narration:문장"


# ---- 유통기한 (2026-08-26 실기: 밀린 말이 뒤늦게 나오는 문제) -------------------


def test_turn_ment_expires_in_queue():
    """회전 멘트는 3초 지나면 재생하지 않는다 — 회전이 끝난 뒤엔 소음이다."""
    from src.replies import TURN_LEFT
    q = TtsQueue()
    q.push(RESPONSE, TURN_LEFT, now=0.0)
    assert q.pop(now=4.0) is None
    assert TURN_LEFT in q.take_expired()


def test_fresh_turn_ment_plays():
    from src.replies import TURN_RIGHT
    q = TtsQueue()
    q.push(RESPONSE, TURN_RIGHT, now=0.0)
    item = q.pop(now=1.0)
    assert item is not None and item.text == TURN_RIGHT


def test_narration_expires_after_six_seconds():
    q = TtsQueue()
    q.push(NARRATION, "목적지까지 약 3미터 남았습니다.", now=0.0)
    assert q.pop(now=7.0) is None


def test_non_turn_response_never_expires():
    """확인 질문·답변은 낡아도 재생한다 — 상태 결정적 정보다."""
    q = TtsQueue()
    q.push(RESPONSE, "안내를 취소할까요?", now=0.0)
    item = q.pop(now=60.0)
    assert item is not None


def test_emergency_never_expires():
    q = TtsQueue()
    q.push(EMERGENCY, "안전을 위해 멈추겠습니다.", now=0.0)
    assert q.pop(now=60.0) is not None
