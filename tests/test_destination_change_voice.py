"""주행 중 목적지 바꾸기 · 지금 상황 메모 — 음성 쪽 (2026-10-07 사용자 결정).

작업 계획 탭 '주행 중 목적지 변경 흐름': 안내 주행 중 다른 목적지를 말하면 로봇이 서서
"지금 409호로 가는 중이에요. 화장실로 바꿀까요?"라고 묻고, 미션은 물은 목적지를 기억한다.
소리 경로는 LLM 이 이 틀대로 묻고, 글자 경로는 확인 문구를 코드가 만든다(_finalize).
"응"의 목적지는 질문 끝의 '…로 바꿀까요?'다 — 앞의 '지금 가는 곳'을 고르면 엉뚱한 곳.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.ledger_view import render_ledger
from src.schema import DestinationData, RobotState

ROOM = DestinationData(id="r409", name="409호", aliases=[], confirm_prompt="409호로 안내해드릴까요?")
WC = DestinationData(id="wc", name="화장실", aliases=[], confirm_prompt="화장실로 안내해드릴까요?")
MEN = DestinationData(id="men", name="남자 화장실", aliases=[],
                      confirm_prompt="남자 화장실로 안내해드릴까요?")
CAFE = DestinationData(id="cafe", name="식당", aliases=[], confirm_prompt="식당으로 안내해드릴까요?")
DESTS = [ROOM, WC, MEN, CAFE]
GUIDING = RobotState(dialog_state="navigating", active_destination="409호", is_moving=True)


def _draft(name):
    return parser._IntentDraft(intent="navigate", destination_candidate=name, confidence=0.9,
                               reply="", is_confirmation=False)


class TestTextPathQuestion:
    def test_guiding_asks_to_change(self):
        out = parser._finalize(_draft("화장실"), DESTS, robot_state=GUIDING)
        assert out.intent == "navigate" and out.need_confirm
        assert out.matched_destination_id == "wc"
        assert out.reply == "지금 409호로 가는 중이에요. 화장실로 바꿀까요?"

    def test_josa_follows_the_names(self):
        st = RobotState(dialog_state="navigating", active_destination="식당")
        out = parser._finalize(_draft("409호"), DESTS, robot_state=st)
        assert out.reply == "지금 식당으로 가는 중이에요. 409호로 바꿀까요?"

    def test_same_place_is_just_a_fact(self):
        out = parser._finalize(_draft("409호"), DESTS, robot_state=GUIDING)
        assert out.intent == "question" and not out.need_confirm
        assert out.matched_destination_id == ""
        assert out.reply == "지금 409호로 가는 중이에요."

    def test_not_guiding_keeps_the_confirm_prompt(self):
        for st in (None, RobotState(dialog_state="idle"),
                   RobotState(dialog_state="paused", active_destination="409호")):
            out = parser._finalize(_draft("화장실"), DESTS, robot_state=st)
            assert out.reply == "화장실로 안내해드릴까요?"


class TestYesFindsTheAskedPlace:
    QUESTION = "지금 409호로 가는 중이에요. 화장실로 바꿀까요?"

    def test_pending_is_the_new_destination(self):
        history = [HumanMessage("화장실 가고 싶어"), AIMessage(self.QUESTION)]
        assert parser._pending_confirm_destination(history, DESTS) is WC

    def test_longest_name_wins(self):
        history = [AIMessage("지금 409호로 가는 중이에요. 남자 화장실로 바꿀까요?")]
        assert parser._pending_confirm_destination(history, DESTS) is MEN

    def test_llm_worded_change_question_is_found_too(self):
        history = [AIMessage("배가 아프시군요. 지금은 409호로 가는 중인데, 화장실로 바꿀까요?")]
        assert parser._pending_confirm_destination(history, DESTS) is WC

    def test_short_yes_confirms_the_new_destination(self):
        history = [HumanMessage("화장실 가고 싶어"), AIMessage(self.QUESTION)]
        out = parser._shortcut_intent("응", history, DESTS)
        assert out.intent == "navigate" and not out.need_confirm
        assert out.matched_destination_id == "wc"

    def test_local_rules_take_the_latest_question(self):
        """로컬 규칙은 기록을 거슬러 찾는다 — 옛 409호 확인 질문이 아니라 바꾸기 질문."""
        history = [AIMessage("409호로 안내해드릴까요?"), HumanMessage("응"),
                   AIMessage("409호로 안내를 시작합니다."), HumanMessage("화장실 가고 싶어"),
                   AIMessage(self.QUESTION)]
        out = parser._confirming_shortcut("네", history, DESTS)
        assert out.matched_destination_id == "wc"


class TestSituationInPrompts:
    LEDGER = render_ledger(
        RobotState(current_floor=4, current_building="로봇관", dialog_state="waiting",
                   last_destination="화장실", last_arrived_age_sec=20, door_side="오른쪽",
                   wait_minutes=10, wait_left_sec=500, wait_place="입구 오른쪽"),
        awaiting_answer=False, now_text="09:01")

    def test_text_prompt_carries_the_ledger_last(self):
        text = parser._build_system_prompt(DESTS, RobotState(current_floor=4), situation=self.LEDGER)
        assert text.endswith(self.LEDGER)
        assert "[현재 로봇 상태]" not in text            # 옛 2줄 블록 대신 대장
        assert "- 대기 장소: 입구 오른쪽" in text and "- 화장실 방향: 오른쪽" in text

    def test_text_prompt_without_ledger_keeps_the_old_block(self):
        text = parser._build_system_prompt(DESTS, RobotState(current_floor=4))
        assert "[현재 로봇 상태]" in text

    def test_rules_are_in_both_prompts(self):
        audio = parser.build_audio_prompt(DESTS, situation=self.LEDGER)
        text = parser._build_system_prompt(DESTS, situation=self.LEDGER)
        for prompt in (audio, text):
            assert "바꿀까요?" in prompt                   # 주행 중 바꾸기
            assert "'대기 장소' 줄" in prompt              # 어디서 기다린다고?
            assert "방향" in prompt and "question" in prompt
            assert "손 놓기 기다림" in prompt               # 대기 중 목적지는 평소처럼
            assert "다녀왔어" in prompt and "finish" in prompt   # 대기 중 끝말 → 로봇이 묻는다
        assert "navigate(주행 중 바꾸기)" in audio
        assert "…에 도착했습니다" in audio and "… 앞에 도착했습니다" not in audio


class TestReviewFixes:
    """2026-10-07 독립 검토가 찾은 것들."""

    def test_already_going_line_is_a_scripted_reply(self):
        """로컬 규칙의 '못 알아들음' 대체·재청취 기각이 코드가 만든 사실 한 줄을 지우지 않는다."""
        assert parser.is_scripted_reply("지금 409호로 가는 중이에요.", DESTS)
        assert parser.is_scripted_reply("지금 식당으로 가는 중이에요.", DESTS)
        assert not parser.is_scripted_reply("지금 어디로 가는 중이에요.", DESTS)

    def test_location_question_while_guiding_does_not_stop_the_robot(self):
        """안내 중 위치만 물으면 question — 로봇을 세우는 바꾸기는 가고 싶다고 할 때만."""
        audio = parser.build_audio_prompt(DESTS)
        text = parser._build_system_prompt(DESTS)
        for prompt in (audio, text):
            assert "가고 싶다고" in prompt
            assert "로봇을 세우지" in prompt
        assert "안내 중이 아닐 때" in audio          # 위치 질문 → navigate 예외는 안내 밖에서만
