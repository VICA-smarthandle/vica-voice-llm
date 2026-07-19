"""사용자 발화를 LangChain(Ollama Cloud)으로 분석해 VicaIntent 로 만든다.

설계 원칙 (docs/design.md):
- LLM 은 intent 분류와 '목적지 표현(destination_candidate)' 까지만 채운다.
- 실제 목적지 확정(matched_destination_id)과 안전 판단은 코드가 한다.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .destination_matcher import match_destination
from .schema import DestinationData, RobotState, VicaIntent, VicaIntentType

load_dotenv()

# LLM 백엔드는 환경변수로 바꿔 낀다 (코드 수정 없이 PC=클라우드 / Jetson=로컬 전환).
#   PC(클라우드):  OLLAMA_HOST=https://ollama.com     VICA_LLM_MODEL=gemma4:cloud  OLLAMA_API_KEY=...
#   Jetson(로컬):  OLLAMA_HOST=http://localhost:11434  VICA_LLM_MODEL=gemma4:e2b    (API 키 불필요)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
DEFAULT_MODEL = os.environ.get("VICA_LLM_MODEL", "gemma4:cloud")

# 직전 확인 질문에 대한 짧은 긍정/부정 답변. 긴급어 필터와 같은 원칙으로
# LLM 을 거치지 않고 코드가 결정한다 — 소형 모델이 is_confirmation 을 놓치면
# 확인 질문이 무한 반복되는 문제가 실기에서 확인됨 (2026-07-19, exaone3.5:2.4b).
_AFFIRMATIVES = frozenset(
    {"네", "예", "응", "어", "그래", "그래요", "맞아", "맞아요",
     "좋아", "좋아요", "네네", "네맞아요", "응응", "가자", "가줘"}
)
_NEGATIVES = frozenset(
    {"아니", "아니요", "아뇨", "아니야", "아니에요", "싫어", "싫어요", "취소"}
)


def _normalize_short_reply(text: str) -> str:
    """STT 가 붙이는 구두점·공백을 제거해 짧은 답변을 비교 가능하게 만든다."""
    return "".join(ch for ch in text if ch.isalnum())


def _pending_confirm_destination(
    history: Optional[list[BaseMessage]], destinations: Sequence[DestinationData]
):
    """직전 AI 발화가 어떤 목적지의 confirm_prompt 였으면 그 목적지를 돌려준다."""
    if not history:
        return None
    last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return None
    for dest in destinations:
        if dest.confirm_prompt and dest.confirm_prompt == last_ai.content:
            return dest
    return None


class _IntentDraft(BaseModel):
    """LLM 이 직접 채우는 부분만 담은 임시 스키마.

    matched_destination_id / need_confirm / safety_flag 는 코드가 채운다.
    """

    intent: VicaIntentType = Field(description="navigate / question / clarify / unknown 중 하나")
    destination_candidate: Optional[str] = Field(
        default=None,
        description="navigate 일 때, 목적지 목록의 name 중 가장 알맞은 하나. 없으면 null.",
    )
    is_confirmation: Optional[bool] = Field(
        default=False,
        description="직전 로봇 제안('OO로 안내할까요?')에 사용자가 긍정(응/네/맞아)한 경우 true",
    )
    confidence: Optional[float] = Field(default=0.0, description="해석 확신도 0~1")
    reply: str = Field(default="", description="사용자에게 들려줄 한국어 답변")


def _format_robot_state(robot_state: Optional[RobotState]) -> str:
    """로봇 현재 상태를 프롬프트용 문자열로 만든다. 없으면 빈 문자열."""
    if robot_state is None:
        return ""
    floor = f"{robot_state.current_floor}층" if robot_state.current_floor is not None else "알 수 없음"
    building = robot_state.current_building or "알 수 없음"
    moving = "예" if robot_state.is_moving else "아니오"
    return (
        "\n[현재 로봇 상태] (question 답변에 활용하라. 모르면 모른다고 답하라)\n"
        f"- 위치: {building} {floor}\n"
        f"- 이동 중: {moving}\n"
    )


def _build_system_prompt(
    destinations: Sequence[DestinationData], robot_state: Optional[RobotState] = None
) -> str:
    lines = []
    for d in destinations:
        aliases = ", ".join(d.aliases)
        approach = "가능" if d.is_approachable else "불가"
        lines.append(f"- {d.name} (별칭: {aliases} / 분류: {d.category2} / 접근: {approach})")
    dest_block = "\n".join(lines)
    state_block = _format_robot_state(robot_state)
    return f"""너는 시각장애인 안내 로봇 'VICA'의 음성 의도 분석기다.
사용자의 한국어 발화를 분석해 아래 규칙으로 분류해라.

[intent 종류]
- navigate: 어딘가로 가고 싶어함. 직접 표현("407호 가자")뿐 아니라 간접 표현("배 아파"->화장실, "배고파"->식당)도 포함.
- question: 이동이 아니라 정보 질문("지금 몇 층이야?").
- clarify: 어디로 갈지 모호해 되물어야 함. reply 에 되묻는 질문을 담아라.
- unknown: 안내와 무관하거나 이해 불가.

[목적지 목록] (navigate 의 destination_candidate 는 반드시 이 name 중 하나여야 한다. 목록에 없으면 clarify)
{dest_block}
{state_block}
[규칙]
- destination_candidate 는 위 목록의 정확한 name 또는 null. 새로 지어내지 마라.
- reply 는 짧고 친절한 한국어.
- 확신이 없으면 confidence 를 낮춰라.

[멀티턴 대화]
- 직전에 로봇이 'OO로 안내해드릴까요?'라고 물었고 사용자가 긍정(응, 네, 맞아, 그래, 좋아)하면:
  intent=navigate, destination_candidate=그 OO 목적지 name, is_confirmation=true 로 답해라.
- 사용자가 부정(아니, 그거 말고)하며 다른 목적지를 말하면 그 목적지로 navigate.
- 부정만 하고 목적지를 안 말하면 clarify."""


def _get_structured_llm(model: str):
    """구조화 출력(_IntentDraft) LLM 을 만든다. (클라우드=키 필요 / 로컬=키 불필요)"""
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    kwargs = {
        "model": model,
        "base_url": OLLAMA_HOST,
        "temperature": 0,
        # gemma4 등 thinking 모델의 내부 추론을 끈다.
        # intent 분류에는 불필요하고, Jetson 에서 응답이 14~20초 -> 3~5초로 줄어든다.
        "reasoning": False,
        # 모델을 메모리에 상주시킨다 (기본 5분 후 언로드 -> 다음 발화가 ~20초 콜드스타트).
        "keep_alive": -1,
    }
    if api_key:  # 클라우드는 인증 헤더 필요, 로컬 Ollama 는 불필요
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    llm = ChatOllama(**kwargs)
    return llm.with_structured_output(_IntentDraft)


def parse_intent(
    user_text: str,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    model: str = DEFAULT_MODEL,
) -> VicaIntent:
    """발화를 분석해 VicaIntent 를 돌려준다. (멀티턴: history, 현재 상태: robot_state)"""
    # 규칙 기반 확인 처리: 직전 제안에 대한 짧은 긍정/부정은 LLM 없이 즉시 결정.
    pending = _pending_confirm_destination(history, destinations)
    if pending is not None:
        word = _normalize_short_reply(user_text)
        if word in _AFFIRMATIVES:
            return VicaIntent(
                intent="navigate",
                destination_candidate=pending.name,
                matched_destination_id=pending.id,
                confidence=1.0,
                reply=f"{pending.name} 안내를 시작합니다.",
                need_confirm=False,
                safety_flag="normal",
            )
        if word in _NEGATIVES:
            return VicaIntent(
                intent="clarify",
                confidence=1.0,
                reply="알겠습니다. 어디로 안내해드릴까요?",
                need_confirm=False,
            )

    structured = _get_structured_llm(model)
    messages: list[BaseMessage] = [SystemMessage(_build_system_prompt(destinations, robot_state))]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(user_text))

    try:
        draft: _IntentDraft = structured.invoke(messages)
    except Exception as exc:
        # LLM/네트워크 실패 -> 크래시 대신 안전한 fallback 응답을 돌려준다.
        # (긴급어는 LLM 이전 단계에서 처리되므로 이 실패의 영향을 받지 않는다.)
        import sys

        print(f"[LLM] 호출 실패: {exc}", file=sys.stderr)
        return VicaIntent(
            intent="unknown",
            reply="죄송합니다. 지금은 요청을 처리할 수 없어요. 잠시 후 다시 말씀해 주세요.",
            confidence=0.0,
            need_confirm=False,
        )
    return _finalize(draft, destinations)


def _finalize(draft: _IntentDraft, destinations: Sequence[DestinationData]) -> VicaIntent:
    """LLM 초안 + 코드 매칭으로 최종 VicaIntent 를 만든다. (결정/안전은 코드 담당)"""
    result = VicaIntent(
        intent=draft.intent,
        destination_candidate=draft.destination_candidate,
        confidence=draft.confidence or 0.0,  # LLM 이 null 을 줘도 안전하게
        reply=draft.reply,
        need_confirm=False,
        safety_flag="normal",
    )

    if draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
        if matched is None:
            # LLM 이 목록에 없는 목적지를 골랐다 -> 되묻기로 안전하게 강등.
            result.intent = "clarify"
            result.reply = result.reply or "어디로 안내해드릴까요?"
        elif not matched.is_approachable:
            # 접근 불가 목적지 -> 코드가 정한 안내 문구, 확인 불필요.
            result.matched_destination_id = matched.id
            result.reply = matched.unavailable_reason or matched.confirm_prompt
            result.need_confirm = False
        elif draft.is_confirmation:
            # 사용자가 직전 제안을 수락 -> 확인 끝, 안내 시작.
            result.matched_destination_id = matched.id
            result.reply = f"{matched.name} 안내를 시작합니다."
            result.need_confirm = False
        else:
            # 정상 목적지 -> 코드가 정한 확인 문구로 통일(LLM 자유 발화 대신).
            result.matched_destination_id = matched.id
            result.reply = matched.confirm_prompt
            result.need_confirm = True

    return result
