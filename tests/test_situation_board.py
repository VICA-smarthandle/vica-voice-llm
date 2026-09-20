"""상황판: 미션 신호 → 사실 블록. 판단 없음, 이력과 무관."""
import json

from src.situation_board import SituationBoard, parse_goal_event_name


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def ev(event, name=""):
    return json.dumps({"event": event, "name": name, "destination_id": "x"}, ensure_ascii=False)


class TestParse:
    def test_reads_event_and_name(self):
        assert parse_goal_event_name(ev("goal_sent", "407호")) == ("goal_sent", "407호")

    def test_garbage_is_none(self):
        assert parse_goal_event_name("not json") == (None, "")
        assert parse_goal_event_name("[1,2]") == (None, "")


class TestBoard:
    def test_empty_board_renders_nothing(self):
        assert SituationBoard(FakeClock()).render() == ""

    def test_guiding_then_arrived(self):
        clock = FakeClock()
        b = SituationBoard(clock)
        b.on_goal_event("goal_sent", "407호")
        clock.t += 120
        text = b.render()
        assert "안내 중: 407호로 이동 중 (2분 전 출발)" in text
        b.on_goal_event("goal_succeeded", "407호")
        clock.t += 200
        text = b.render()
        assert "안내 중: 없음" in text and "마지막 도착: 407호 (3분 전)" in text

    def test_wait_request_survives_until_next_guidance(self):
        class W:
            intent = "wait"
            wait_minutes = 10
        b = SituationBoard(FakeClock())
        b.on_goal_event("goal_accepted", "테스트3")
        b.on_goal_event("goal_succeeded", "테스트3")
        b.on_intent(W())
        assert "대기 요청: 10분" in b.render()
        b.on_goal_event("goal_sent", "화장실")
        assert "대기 요청" not in b.render()

    def test_wait_without_minutes(self):
        class W:
            intent = "wait"
            wait_minutes = -1
        b = SituationBoard(FakeClock())
        b.on_goal_event("goal_succeeded", "화장실")
        b.on_intent(W())
        assert "대기 요청: 시간 미정" in b.render()

    def test_failure_and_cancel_are_recorded(self):
        b = SituationBoard(FakeClock())
        b.on_goal_event("goal_sent", "407호")
        b.on_goal_event("goal_canceled", "407호")
        text = b.render()
        assert "안내 중: 없음" in text and "마지막 안내 결말: 407호 취소" in text

    def test_return_home(self):
        b = SituationBoard(FakeClock())
        b.on_goal_event("goal_succeeded", "화장실")
        b.on_goal_event("return_home_sent", "")
        assert "제자리로 돌아가는 중" in b.render()
        b.on_goal_event("return_home_succeeded", "")
        assert "제자리로 돌아가는 중" not in b.render() and "마지막 도착: 화장실" in b.render()

    def test_unknown_event_ignored(self):
        b = SituationBoard(FakeClock())
        b.on_goal_event(None, "")
        b.on_goal_event("something_else", "x")
        assert b.render() == ""
