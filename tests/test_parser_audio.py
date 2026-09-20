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
    def test_navigate_draft_goes_through_finalize(self, rt):
        c = rt({"intent": "navigate", "destination_candidate": "별빛관 1층 화장실", "reply": ""},
               "화장실로 가줘")
        intent, heard, dt, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert heard == "화장실로 가줘" and dt == pytest.approx(0.7)
        assert info["src"] == "model"  # draft 가 _finalize 를 거쳐 확정됐다
        assert info["usage"] == {"audio_tokens": 20}
        assert "heard_text" in c.calls[0][2]  # 추가 지시문이 들어갔다
        assert parser.AUDIO_EXTRA_INSTRUCTIONS in c.calls[0][2]

    def test_wait_minutes_from_heard_text(self, rt):
        # heard("이십 분"=20)가 draft.wait_minutes(5)를 이긴다 — 산수는 코드가 한다.
        rt({"intent": "wait", "reply": "", "wait_minutes": 5}, "이십 분")
        intent, heard, _, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "wait" and intent.wait_minutes == 20
        assert info["src"] == "model"

    def test_affirm_with_pending_becomes_navigate(self, rt):
        # "넵"은 텍스트 지름길 어휘(AFFIRMATIVES/SOFT_AFFIRMATIVES)에 없다 —
        # _shortcut_intent 를 비켜 가서 모델의 affirm 판정이 pending 확정
        # 경로(코드)를 타는지를 검증한다.
        from src.handle_mode import AFFIRMATIVES, SOFT_AFFIRMATIVES
        assert "넵" not in (AFFIRMATIVES | SOFT_AFFIRMATIVES)
        rt({"intent": "affirm", "reply": ""}, "넵")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _, info = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is False
        assert info["src"] == "pending-affirm"

    def test_deny_with_pending_stays_deny(self, rt):
        # "됐거든"은 NEGATIVES 에 없다 — 모델의 deny 판정이 pending 경로를 타는지 검증한다.
        from src.handle_mode import NEGATIVES
        assert "됐거든" not in NEGATIVES
        rt({"intent": "deny", "reply": ""}, "됐거든")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _, info = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "deny"
        assert info["src"] == "pending-deny"

    def test_heard_text_shortcut_wins_over_draft(self, rt):
        # 들린 말이 호출어면 초안(unknown)과 무관하게 호출 응답
        rt({"intent": "unknown", "reply": ""}, "비카야")
        intent, _, _, info = parse_intent_audio(PCM, [DEST])
        assert intent.reply == parser.WAKE_GREETING
        assert info["src"] == "shortcut"

    def test_realtime_failure_propagates(self, rt):
        rt({}, "", fail=TimeoutError("8초"))
        with pytest.raises(TimeoutError):
            parse_intent_audio(PCM, [DEST])

    def test_bad_draft_raises_value_error(self, rt):
        rt({"intent": "not-an-intent", "reply": ""}, "x")
        with pytest.raises(ValueError):
            parse_intent_audio(PCM, [DEST])
