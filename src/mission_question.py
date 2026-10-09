"""미션이 질문 중일 때 LLM 대답을 줄이는 판단 (미션 요청 반응표 2026-10-08, 규칙 2·다시 묻기).

미션은 질문마다 못 알아들은 답(unknown)에 같은 질문을 한 번 다시 한다. 그때 LLM 까지 말하면
두 목소리가 된다 — 그래서 LLM 의 unknown 대답을 지운다. 접근 질문·돌아서기 중의 목적지 답은
미션이 수락으로 받고 돌아선 뒤 직접 확인하므로 LLM 의 확인 질문도 지운다. LLM 이 되묻는
clarify("어느 화장실이요?")는 지우지 않는다 — 미션도 그때는 끼어들지 않는다.
ros_node 는 rclpy 없이 시험할 수 없어 판단은 여기 순수 함수로 둔다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from .local_rules import ANSWER_WAIT_STATES
from .mission_phrases import WAIT_FINISH_ASK, WAIT_NEED_ASK
from .replies import CANCEL_CONFIRM, RESUME_CONFIRM, WAKE_GREETING
from .schema import VicaIntent

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
    """미션이 이어서 말할 자리면 LLM reply 를 비운 intent 를, 아니면 그대로 돌려준다.
    주행 중 "다시 가자"는 확인 질문도 뺀다 — 미션이 가는 곳을 말한다."""
    if getattr(intent, "safety_flag", "") == "emergency" or not intent.reply:
        return intent
    if dialog_state == "awaiting_user" and intent.intent != "question":
        # 접근 질문('안내를 받으시겠어요?') 중에는 질문에만 소리 내어 답한다. 되묻기·"네?"·확인 질문은
        # 지우고 다시 묻기는 미션이 한다(2026-10-09 사용자 결정 — run82 17:19 LLM 이 "어느 곳으로 안내를
        # 원하시나요?"라고 물은 직후 미션은 떠났다). "다시 가자"는 쥐지 않고 미션에 보내 다시 묻게 한다.
        return intent.model_copy(update={"reply": "", "need_confirm": False}
                                 if intent.intent == "resume" else {"reply": ""})
    if intent.intent == "pause":
        # "잠깐"은 미션이 모든 상태에서 답한다(일시정지·"네?"·"지금은 안내 중이 아닙니다").
        # 음성까지 "네, 잠시 설게요."면 두 목소리다(2026-10-09 최종 검토 I-2).
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "resume" and intent.need_confirm and dialog_state == "navigating":
        # 이미 가는 중의 "다시 가자" — 움직일 일이 없어 확인할 것이 없다. 확정으로 보내 미션이
        # "지금 OO로 가는 중이에요"라고 답하게 한다(2026-10-09 사용자 결정, 주행 중만. 그 밖은
        # 일시정지·손 놓침·홈 가다 세운 뒤처럼 이 말에 실제로 움직이는 상태라 묻는다).
        return intent.model_copy(update={"reply": "", "need_confirm": False})
    if (intent.intent == "unknown" and intent.reply != WAKE_GREETING
            and mission_is_asking(dialog_state, last_robot_text, last_robot_age_sec)):
        return intent.model_copy(update={"reply": ""})
    if (intent.intent == "navigate" and intent.need_confirm
            and dialog_state in APPROACH_ANSWER_STATES):
        return intent.model_copy(update={"reply": ""})
    return intent


# ---- 음성이 쥔 "다시 출발할까요?" 다시 묻기 (2026-10-08) -------------------------------
# 미션은 이 질문을 모른다(schema.should_forward_intent 가 확인 전 resume 을 쥔다). 그래서
# 다시 묻기도 음성이 한다 — 못 알아들은 답이나 빈손으로 닫힌 듣기 창에 한 번만.
def _robot_lines(history: list[BaseMessage]) -> list[str]:
    return [m.content for m in history if isinstance(m, AIMessage)]


def resume_reasked(history: list[BaseMessage]) -> bool:
    """이미 한 번 다시 물었나 — 로봇 말 마지막 두 줄이 모두 "다시 출발할까요?"."""
    lines = _robot_lines(history or [])
    return len(lines) >= 2 and lines[-1] == RESUME_CONFIRM and lines[-2] == RESUME_CONFIRM


def reask_held_resume(intent, history: list[BaseMessage]):
    """"다시 출발할까요?"에 못 알아들은 답(unknown)이면 같은 질문을 한 번 다시 묻는 intent 로
    바꾼다. 확인 전 resume 이라 미션에는 가지 않고 말만 나간다. 그 밖에는 그대로."""
    if intent.intent != "unknown" or getattr(intent, "safety_flag", "") == "emergency":
        return intent
    lines = _robot_lines(history or [])
    if not lines or lines[-1] != RESUME_CONFIRM or resume_reasked(history):
        return intent
    return VicaIntent(intent="resume", confidence=1.0, reply=RESUME_CONFIRM, need_confirm=True)


def should_reask_resume_on_silence(history: list[BaseMessage], last_robot_text: str,
                                   last_robot_age_sec: float) -> bool:
    """듣기 창이 빈손으로 닫혔을 때 "다시 출발할까요?"를 한 번 다시 물을까. 질문 뒤 사용자
    말이 없었고(마지막 줄이 그 질문) 아직 다시 묻지 않았을 때만. 그 질문이 방금(ASK_FRESH_SEC 안)
    로봇의 마지막 말이어야 한다 — 글자 모드는 기록에 미션 말이 없어, 2분 뒤 "비카야" → "네?"
    → 조용함에도 옛 질문이 기록 끝에 남아 있다(2026-10-09 검토 I-8)."""
    if not history or not isinstance(history[-1], AIMessage):
        return False
    if (last_robot_text or "").strip() != RESUME_CONFIRM or last_robot_age_sec > ASK_FRESH_SEC:
        return False
    return history[-1].content == RESUME_CONFIRM and not resume_reasked(history)
