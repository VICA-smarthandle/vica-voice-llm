"""손잡이 터치 × 진동 — LLM 이 알아야 할 것 (2026-09-30, 설계 4.7)."""
from src import langchain_intent_parser as parser
from src.ledger_view import DIALOG_KO, render_ledger
from src.schema import DestinationData, RobotState

DEST = DestinationData(id="toilet", name="화장실", aliases=["화장실"],
                       confirm_prompt="화장실로 안내해드릴까요?")


def test_two_handle_states_are_translated():
    """ROS mission_logic 의 DIALOG_GRIP_WAIT·DIALOG_PAUSED_HANDLE 과 글자가 같아야 한다."""
    assert "grip_wait" in DIALOG_KO and "paused_handle" in DIALOG_KO
    assert "잡으면" in DIALOG_KO["grip_wait"]
    assert "다시 잡으면 출발" in DIALOG_KO["paused_handle"]


def test_handle_pause_reads_differently_from_voice_pause():
    handle = render_ledger(RobotState(dialog_state="paused_handle", is_paused=True),
                           awaiting_answer=False, now_text="09:01")
    voice = render_ledger(RobotState(dialog_state="paused", is_paused=True),
                          awaiting_answer=False, now_text="09:01")
    assert "- 대화 단계: 손잡이를 놓쳐 멈춤(다시 잡으면 출발)" in handle
    assert "- 대화 단계: 일시정지" in voice


def test_facts_cover_the_questions_users_ask():
    for must in ("진동", "손을 놓으면", "다시 잡으면", "잡지 않고", "당기면", "세 번"):
        assert must in parser.HANDLE_FACTS, must
    for number in ("0.5", "80", "2초"):
        assert number not in parser.HANDLE_FACTS, number


def test_facts_are_in_both_prompts():
    assert parser.HANDLE_FACTS in parser.build_audio_prompt([DEST])
    assert parser.HANDLE_FACTS in parser._build_system_prompt([DEST])


def test_facts_sit_in_the_cached_head_of_the_audio_prompt():
    """Realtime 캐시는 앞부분이 같아야 할인된다 — 매번 바뀌는 상황판보다 앞이어야 한다."""
    situation = "\n[지금 상황]\n- 대화 단계: 손잡이 잡기를 기다리는 중(잡으면 안내 시작)\n"
    text = parser.build_audio_prompt([DEST], RobotState(), situation=situation)
    assert text.index(parser.HANDLE_FACTS) < text.rindex("[지금 상황]")
    assert text.endswith(situation)
