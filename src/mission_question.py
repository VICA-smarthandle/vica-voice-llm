"""미션이 질문 중일 때 LLM 대답을 줄이는 판단 (미션 요청 반응표 2026-10-08, 규칙 2·다시 묻기).

미션은 질문마다 못 알아들은 답(unknown)에 같은 질문을 한 번 다시 한다. 그때 LLM 까지 말하면
두 목소리가 된다 — 그래서 LLM 의 unknown 대답을 지운다. 접근 질문·돌아서기 중의 목적지 답은
미션이 수락으로 받고 돌아선 뒤 직접 확인하므로 LLM 의 확인 질문도 지운다. LLM 이 되묻는
clarify("어느 화장실이요?")는 지우지 않는다 — 미션도 그때는 끼어들지 않는다.
ros_node 는 rclpy 없이 시험할 수 없어 판단은 여기 순수 함수로 둔다.
"""
from __future__ import annotations

from .local_rules import ANSWER_WAIT_STATES
from .mission_phrases import WAIT_FINISH_ASK, WAIT_NEED_ASK
from .replies import CANCEL_CONFIRM, WAKE_GREETING

# 대기 중·주행 중에도 미션이 묻는 질문 — dialog_state 만으로는 질문 중인지 모른다.
MISSION_QUESTIONS = frozenset({WAIT_FINISH_ASK, WAIT_NEED_ASK, CANCEL_CONFIRM})
# 그 질문의 답을 기다리는 시간 — 음성 질문 뒤 문맥 창(ros_node.FOLLOWUP_CONTEXT_SEC)과 같다.
ASK_FRESH_SEC = 40.0
# 미션이 수락 뒤 직접 목적지를 묻는 단계 — LLM 의 확인 질문은 겹친다.
APPROACH_ANSWER_STATES = frozenset({"awaiting_user", "turning"})


def mission_is_asking(dialog_state: str, last_robot_text: str, last_robot_age_sec: float) -> bool:
    """미션이 사용자 대답을 기다리는 중인가."""
    if dialog_state in ANSWER_WAIT_STATES:
        return True
    return ((last_robot_text or "").strip() in MISSION_QUESTIONS
            and last_robot_age_sec <= ASK_FRESH_SEC)


def quiet_for_mission(intent, dialog_state: str, last_robot_text: str,
                      last_robot_age_sec: float):
    """미션이 이어서 말할 자리면 LLM reply 를 비운 intent 를, 아니면 그대로 돌려준다."""
    if getattr(intent, "safety_flag", "") == "emergency" or not intent.reply:
        return intent
    if (intent.intent == "unknown" and intent.reply != WAKE_GREETING
            and mission_is_asking(dialog_state, last_robot_text, last_robot_age_sec)):
        return intent.model_copy(update={"reply": ""})
    if (intent.intent == "navigate" and intent.need_confirm
            and dialog_state in APPROACH_ANSWER_STATES):
        return intent.model_copy(update={"reply": ""})
    return intent
