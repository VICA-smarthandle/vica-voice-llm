"""tts_queue unit test (테스트 계획: ros_tts_node 큐/우선순위/드롭)."""
from src.tts_queue import TtsPriority, TtsQueue, format_tts_request, parse_tts_request


class TestParse:
    def test_emergency_prefix(self):
        assert parse_tts_request("emergency:비상 정지합니다.") == (
            TtsPriority.EMERGENCY,
            "비상 정지합니다.",
        )

    def test_narration_prefix(self):
        assert parse_tts_request("narration:이동을 시작합니다.") == (
            TtsPriority.NARRATION,
            "이동을 시작합니다.",
        )

    def test_response_prefix(self):
        assert parse_tts_request("response:안내할 수 없습니다.") == (
            TtsPriority.RESPONSE,
            "안내할 수 없습니다.",
        )

    def test_no_prefix_defaults_to_narration(self):
        assert parse_tts_request("407호 방향으로 출발합니다.") == (
            TtsPriority.NARRATION,
            "407호 방향으로 출발합니다.",
        )

    def test_roundtrip(self):
        for p in TtsPriority:
            data = format_tts_request("안내", p)
            assert parse_tts_request(data) == (p, "안내")


class TestQueue:
    def test_priority_order(self):
        q = TtsQueue()
        q.put("응답", TtsPriority.RESPONSE)
        q.put("내레이션", TtsPriority.NARRATION)
        # 긴급은 하위 큐를 비우므로 마지막에 넣어 순서만 검증
        assert q.pop() == (TtsPriority.NARRATION, "내레이션")
        assert q.pop() == (TtsPriority.RESPONSE, "응답")
        assert q.pop() is None

    def test_fifo_within_priority(self):
        q = TtsQueue()
        q.put("첫째", TtsPriority.NARRATION)
        q.put("둘째", TtsPriority.NARRATION)
        assert q.pop()[1] == "첫째"
        assert q.pop()[1] == "둘째"

    def test_cap_drops_oldest(self):
        q = TtsQueue(max_per_priority=3)
        for i in range(5):
            q.put(f"멘트{i}", TtsPriority.RESPONSE)
        assert len(q) == 3
        assert q.pop()[1] == "멘트2"  # 0, 1 은 드롭됨
        assert q.dropped_total == 2

    def test_emergency_clears_lower_queues(self):
        q = TtsQueue()
        q.put("내레이션1", TtsPriority.NARRATION)
        q.put("응답1", TtsPriority.RESPONSE)
        q.put("멈춥니다", TtsPriority.EMERGENCY)
        assert q.pop() == (TtsPriority.EMERGENCY, "멈춥니다")
        assert q.pop() is None  # 하위 멘트는 전부 비워짐
        assert q.dropped_total == 2

    def test_emergency_fifo_kept(self):
        q = TtsQueue()
        q.put("긴급1", TtsPriority.EMERGENCY)
        q.put("긴급2", TtsPriority.EMERGENCY)
        assert q.pop()[1] == "긴급1"
        assert q.pop()[1] == "긴급2"

    def test_empty_text_ignored(self):
        q = TtsQueue()
        q.put("", TtsPriority.NARRATION)
        q.put("   ", TtsPriority.RESPONSE)
        assert len(q) == 0
        assert q.pop() is None
