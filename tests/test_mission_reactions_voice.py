"""미션 요청 반응표(2026-10-08) — 음성 쪽: LLM 지시문(규칙 2)·미션 질문 중 대답·문장.

정본 설계: docs/superpowers/specs/2026-10-08-mission-request-reactions-design.md 8절.
"""
from src import langchain_intent_parser as parser
from src import local_rules
from src.langchain_intent_parser import _build_system_prompt
from src.schema import DestinationData

DEST = DestinationData(id="wc", name="화장실", confirm_prompt="화장실로 안내해드릴까요?")


# ---- Task 13: 지시문 --------------------------------------------------------------
class TestPromptRules:
    def test_audio_prompt_allows_destination_answer_to_approach_question(self):
        text = parser.build_audio_prompt([DEST])
        assert "그 답은 반드시" not in text          # 옛 "반드시 affirm/deny" 막음
        assert 'navigate(need_confirm=true, reply="")로 낸다' in text

    def test_audio_prompt_has_wait_refusal_and_negative_question(self):
        text = parser.build_audio_prompt([DEST])
        assert "기다리지 마" in text and "안내가 필요 없으신가요?" in text
        assert '"아니, 필요 없어"' in text

    def test_audio_prompt_bans_promises(self):
        text = parser.build_audio_prompt([DEST])
        assert "약속 금지" in text and "계속 안내할까요?" in text

    def test_audio_prompt_leaves_reasking_to_the_mission(self):
        text = parser.build_audio_prompt([DEST])
        assert "로봇 본체가 같은 질문을 한 번 다시 한다" in text
        assert "그 질문을 다시(\"여기서 기다릴까요?\")" not in text

    def test_text_prompt_has_the_same_rules(self):
        text = _build_system_prompt([DEST])
        for key in ("목적지로 답하면", "기다리지 마", "안내가 필요 없으신가요?", "계속 안내할까요?"):
            assert key in text, key

    def test_local_rule_allows_destination_answer(self):
        assert "목적지를 말하면 navigate" in local_rules.LOCAL_PROMPT_RULES
