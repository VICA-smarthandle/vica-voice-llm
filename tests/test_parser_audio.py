"""소리→의도 직행 경로 시험 (Realtime 은 가짜 클라이언트)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _shortcut_intent, parse_intent, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData

DEST = DestinationData(id="starlight_1f_restroom", name="별빛관 1층 화장실",
                       confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?")
PCM = b"\x00\x00" * 1600


class FakeRT:
    def __init__(self, draft, heard, fail=None):
        self.draft, self.heard, self.fail = draft, heard, fail
        self.calls = []

    def ask(self, pcm, history, instructions):
        self.calls.append((pcm, list(history or []), instructions))
        if self.fail:
            raise self.fail
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7,
                              usage={"audio_tokens": 20})


@pytest.fixture
def rt(monkeypatch):
    holder = {}

    def use(draft, heard, fail=None):
        holder["c"] = FakeRT(draft, heard, fail)
        monkeypatch.setattr(parser, "get_realtime_client", lambda: holder["c"])
        return holder["c"]

    return use


class TestShortcutExtraction:
    def test_text_path_unchanged_for_wake_word(self):
        assert parse_intent("비카야", [DEST]).intent == "unknown"

    def test_shortcut_returns_none_for_normal_sentence(self):
        assert _shortcut_intent("화장실로 안내해줘", None, [DEST]) is None

    def test_shortcut_pending_affirm(self):
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        out = _shortcut_intent("그래", hist, [DEST])
        assert out.intent == "navigate" and out.matched_destination_id == DEST.id


class TestAudioPath:
    """소리 모드 = 모델 전결. 코드는 id 매핑·접근 불가·계약상 빈 reply 만 한다."""

    def test_new_destination_uses_model_decision(self, rt):
        c = rt({"intent": "navigate", "destination_candidate": "별빛관 1층 화장실",
                "need_confirm": True, "reply": DEST.confirm_prompt, "confidence": 0.9}, "화장실로 가줘")
        intent, heard, dt, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is True and intent.reply == DEST.confirm_prompt
        assert intent.confidence == pytest.approx(0.9)
        assert heard == "화장실로 가줘" and dt == pytest.approx(0.7)
        assert info["src"] == "llm" and info["usage"] == {"audio_tokens": 20}
        # 지시문: 전결 규칙과 목록의 확인 문구가 들어간다
        assert "need_confirm" in c.calls[0][2] and DEST.confirm_prompt in c.calls[0][2]

    def test_confirmed_navigate_is_silent_for_mission(self, rt):
        rt({"intent": "navigate", "destination_candidate": DEST.name, "need_confirm": False,
            "reply": "출발합니다"}, "네")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is False and intent.reply == ""   # 계약: 출발 안내는 미션이 말한다

    def test_correction_with_negative_word_is_not_deny(self, rt):
        # 09-20 실기 #19: 텍스트 경로는 부정어 규칙으로 '취소'했다. 전결 모드는 모델 결정 그대로.
        rt({"intent": "navigate", "destination_candidate": DEST.name, "need_confirm": True,
            "reply": ""}, "아니 화장실로 가자")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.need_confirm is True
        assert intent.reply == DEST.confirm_prompt      # 모델이 비우면 목록의 확인 문구

    def test_wake_word_is_no_longer_a_shortcut(self, rt):
        rt({"intent": "unknown", "reply": ""}, "비카야")
        intent, _, _, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "unknown" and intent.reply == "" and info["src"] == "llm"

    def test_pending_affirm_is_model_business(self, rt):
        # 확인 대기 판정을 코드가 하지 않는다 — 모델이 affirm 이라 하면 affirm 그대로.
        rt({"intent": "affirm", "reply": "네"}, "넵")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _, _ = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "affirm" and intent.reply == "" and intent.need_confirm is False

    def test_wait_uses_model_minutes(self, rt):
        rt({"intent": "wait", "reply": "알겠습니다", "wait_minutes": 15}, "한 십 분에서 십오 분")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "wait" and intent.wait_minutes == 15
        assert intent.reply == "" and intent.need_confirm is False

    def test_wait_without_minutes_is_minus_one(self, rt):
        rt({"intent": "wait", "reply": ""}, "좀 있다 올게")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.wait_minutes == -1

    @pytest.mark.parametrize("name", ["affirm", "deny", "finish"])
    def test_mission_spoken_intents_have_blank_reply(self, rt, name):
        rt({"intent": name, "reply": "네 알겠습니다", "need_confirm": True}, "응")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == name and intent.reply == "" and intent.need_confirm is False

    def test_unknown_destination_becomes_clarify(self, rt):
        rt({"intent": "navigate", "destination_candidate": "옥상", "need_confirm": True, "reply": ""}, "옥상 가자")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "clarify" and intent.reply == parser.ASK_DESTINATION
        assert intent.need_confirm is False and not intent.matched_destination_id

    def test_unapproachable_destination_keeps_reason(self, rt):
        blocked = DestinationData(id="mech", name="기계실", is_approachable=False,
                                  unavailable_reason="기계실은 안내할 수 없습니다.")
        rt({"intent": "navigate", "destination_candidate": "기계실", "need_confirm": True, "reply": ""}, "기계실")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST, blocked])
        assert intent.intent == "navigate" and intent.matched_destination_id == "mech"
        assert intent.reply == "기계실은 안내할 수 없습니다." and intent.need_confirm is False

    def test_cancel_first_asks_then_confirmed_is_silent(self, rt):
        rt({"intent": "cancel", "need_confirm": True, "reply": ""}, "취소해")
        first, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert first.need_confirm is True and first.reply == parser.CANCEL_CONFIRM
        rt({"intent": "cancel", "need_confirm": False, "reply": "취소했습니다"}, "응")
        second, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert second.need_confirm is False and second.reply == ""

    def test_pause_gets_default_ack(self, rt):
        rt({"intent": "pause", "reply": ""}, "잠깐만")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "pause" and intent.reply == parser.PAUSE_ACK and intent.need_confirm is False

    def test_history_is_handed_to_model_untouched(self, rt):
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt), AIMessage("몇 분쯤 걸리실까요?")]
        c = rt({"intent": "wait", "reply": "", "wait_minutes": 5}, "오 분")
        parse_intent_audio(PCM, [DEST], history=hist)
        assert c.calls[0][1] == hist

    def test_realtime_failure_propagates(self, rt):
        rt({}, "", fail=TimeoutError("8초"))
        with pytest.raises(TimeoutError):
            parse_intent_audio(PCM, [DEST])

    def test_bad_draft_raises_value_error(self, rt):
        rt({"intent": "not-an-intent", "reply": ""}, "x")
        with pytest.raises(ValueError):
            parse_intent_audio(PCM, [DEST])


class TestAudioPrompt:
    def test_prompt_lists_confirm_phrase_and_blocked(self):
        blocked = DestinationData(id="mech", name="기계실", is_approachable=False)
        text = parser.build_audio_prompt([DEST, blocked])
        assert DEST.confirm_prompt in text and "기계실" in text and "접근 불가" in text
        assert "이력에 있는 말을 베껴 적지 마라" in text

    def test_tool_schema_has_need_confirm(self):
        from src.realtime_intent import build_intent_tool
        assert "need_confirm" in build_intent_tool()["parameters"]["properties"]

    def test_prompt_lists_building_and_floor(self):
        d = DestinationData(id="r407", name="407호", aliases=["407호", "윤지영"], building="로봇관", floor=4,
                            confirm_prompt="407호로 안내해드릴까요?")
        text = parser.build_audio_prompt([d])
        assert "위치: 로봇관 4층" in text and "윤지영" in text
        assert "기억이 없다고 하지 마라" in text

    def test_prompt_includes_situation_block(self):
        text = parser.build_audio_prompt([DEST], situation="\n[지금 상황]\n- 마지막 도착: 407호 (3분 전)\n")
        assert "마지막 도착: 407호" in text

    def test_parse_audio_passes_situation(self, rt):
        c = rt({"intent": "question", "reply": "407호예요."}, "아까 어디 갔었지?")
        parse_intent_audio(PCM, [DEST], situation="\n[지금 상황]\n- 마지막 도착: 407호 (방금)\n")
        assert "마지막 도착: 407호" in c.calls[0][2]

    def test_dynamic_blocks_come_last_for_prompt_cache(self):
        # 앞부분(역할·규칙·목적지)이 호출마다 같아야 Realtime 캐시가 먹는다 — 바뀌는 블록은 맨 뒤.
        from src.schema import RobotState
        a = parser.build_audio_prompt([DEST], RobotState(), situation="\n[지금 상황]\n- 안내 중: 없음\n")
        b = parser.build_audio_prompt([DEST], RobotState(is_moving=True), situation="\n[지금 상황]\n- 안내 중: 화장실\n")
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        assert prefix > len(a) * 0.8            # 8할 이상이 공통 접두
        assert a.index("[현재 로봇 상태]") > a.index("[말투]")
        assert a.index("[지금 상황]") > a.index("[현재 로봇 상태]")

    def test_state_block_is_skipped_when_situation_has_ledger(self):
        from src.schema import RobotState
        st = RobotState(current_floor=4, current_building="로봇관", dialog_state="idle")
        with_ledger = parser.build_audio_prompt([DEST], st, situation="\n[지금 상황]\n- 건물/층: 로봇관 4층\n")
        assert "[현재 로봇 상태]" not in with_ledger and "[지금 상황]" in with_ledger
        without = parser.build_audio_prompt([DEST], st, situation="")
        assert "[현재 로봇 상태]" in without

    def test_directory_block_and_elevator_rule(self):
        text = parser.build_audio_prompt([DEST], directory_block="\n[다른 층 장소]\n- 세미나실: 로봇관 3층\n")
        assert "[다른 층 장소]" in text and "엘리베이터" in text
