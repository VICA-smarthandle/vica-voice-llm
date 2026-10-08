"""미션 요청 반응표(2026-10-08) — 음성 쪽: LLM 지시문(규칙 2)·미션 질문 중 대답·문장.

정본 설계: docs/superpowers/specs/2026-10-08-mission-request-reactions-design.md 8절.
"""
from src import langchain_intent_parser as parser
from src import local_rules
from src.langchain_intent_parser import _build_system_prompt
from langchain_core.messages import AIMessage, HumanMessage

from src.mission_phrases import CONFIRM_SWITCH, WAIT_FINISH_ASK, WAIT_NEED_ASK
from src.mission_question import mission_is_asking, quiet_for_mission
from src.replies import CANCEL_CONFIRM, RETRY_PROMPT, WAKE_GREETING
from src.schema import DestinationData, VicaIntent

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


# ---- Task 14: 미션이 질문 중이면 LLM 은 말하지 않는다 --------------------------------
def _i(kind, reply, need_confirm=False, safety="normal"):
    return VicaIntent(intent=kind, reply=reply, need_confirm=need_confirm, safety_flag=safety)


class TestQuietForMission:
    def test_question_states_are_asking(self):
        for state in ("confirming", "asking_next", "asking_wait_time", "awaiting_user"):
            assert mission_is_asking(state, "", 999.0), state
        assert not mission_is_asking("navigating", "", 1.0)

    def test_mission_questions_in_waiting_count_while_fresh(self):
        for q in (WAIT_FINISH_ASK, WAIT_NEED_ASK, CANCEL_CONFIRM):
            assert mission_is_asking("waiting", q, 10.0), q
            assert not mission_is_asking("waiting", q, 41.0), q

    def test_unknown_reply_is_silenced_while_asking(self):
        out = quiet_for_mission(_i("unknown", RETRY_PROMPT), "asking_next", "", 1.0)
        assert out.intent == "unknown" and out.reply == ""

    def test_clarify_and_question_keep_their_words(self):
        """LLM 이 되묻거나 정보로 답하면 그대로 — 미션도 그때는 끼어들지 않는다."""
        for kind in ("clarify", "question"):
            out = quiet_for_mission(_i(kind, "어느 화장실이요?"), "confirming", "", 1.0)
            assert out.reply == "어느 화장실이요?", kind

    def test_wake_greeting_and_emergency_are_kept(self):
        assert quiet_for_mission(_i("unknown", WAKE_GREETING), "asking_next", "", 1.0).reply
        assert quiet_for_mission(_i("unknown", "멈춥니다", safety="emergency"),
                                 "asking_next", "", 1.0).reply

    def test_not_asking_keeps_the_reply(self):
        out = quiet_for_mission(_i("unknown", RETRY_PROMPT), "idle", "", 1.0)
        assert out.reply == RETRY_PROMPT

    def test_destination_answer_to_approach_is_left_to_the_mission(self):
        out = quiet_for_mission(_i("navigate", "화장실로 안내해드릴까요?", need_confirm=True),
                                "awaiting_user", "", 1.0)
        assert out.reply == ""
        kept = quiet_for_mission(_i("navigate", "화장실로 안내해드릴까요?", need_confirm=True),
                                 "idle", "", 1.0)
        assert kept.reply == "화장실로 안내해드릴까요?"


# ---- Task 15: "네, XX로 안내해드릴까요?"도 확인 질문이다 -----------------------------
ELEV = DestinationData(id="elev", name="엘리베이터", confirm_prompt="엘리베이터로 안내해드릴까요?")


def test_switch_question_points_at_the_new_destination():
    history = [AIMessage(DEST.confirm_prompt), HumanMessage("아니 엘리베이터로 가자"),
               AIMessage(CONFIRM_SWITCH.format(prompt=ELEV.confirm_prompt))]
    assert parser._pending_confirm_destination(history, [DEST, ELEV]) is ELEV
    assert parser._recent_confirm_destination(history, [DEST, ELEV]) is ELEV


def test_yes_to_the_switch_question_confirms_the_new_destination():
    history = [AIMessage(DEST.confirm_prompt), HumanMessage("아니 엘리베이터로 가자"),
               AIMessage(CONFIRM_SWITCH.format(prompt=ELEV.confirm_prompt))]
    result = parser._shortcut_intent("응", history, [DEST, ELEV])
    assert result is not None and result.intent == "navigate"
    assert result.matched_destination_id == "elev"
