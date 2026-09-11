"""사용자 발화를 LangChain(Ollama Cloud)으로 분석해 VicaIntent 로 만든다."""
from __future__ import annotations

import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .destination_matcher import match_destination
from .handle_mode import (
    AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply)
from .replies import (
    ASK_DESTINATION,
    CANCEL_CONFIRM,
    COMMAND_DECLINED,
    LLM_UNAVAILABLE,
    PAUSE_ACK,
    RESUME_CONFIRM,
    RETRY_PROMPT,
    WAKE_GREETING,
)
from .schema import DestinationData, RobotState, VicaIntent, VicaIntentType

load_dotenv()

PROVIDER = os.environ.get("VICA_LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
if PROVIDER == "openai":
    DEFAULT_MODEL = os.environ.get("VICA_OPENAI_MODEL", "gpt-5.4-mini")
else:
    DEFAULT_MODEL = os.environ.get("VICA_LLM_MODEL", "gemma4:cloud")


class _IntentDraft(BaseModel):
    """LLM 이 직접 채우는 부분만 담은 임시 스키마."""

    intent: VicaIntentType = Field(
        description="navigate / question / clarify / unknown / cancel / pause / resume / affirm / deny 중 하나"
    )
    destination_candidate: Optional[str] = Field(
        default=None,
        description="navigate 일 때, 목적지 목록의 name 중 가장 알맞은 하나. 없으면 null.",
    )
    is_confirmation: Optional[bool] = Field(
        default=False,
        description="직전 로봇 제안('OO로 안내할까요?')에 사용자가 긍정(응/네/맞아)한 경우 true",
    )
    confidence: Optional[float] = Field(default=0.0, description="해석 확신도 0~1")
    wait_minutes: Optional[int] = Field(
        default=None,
        description=(
            "intent 가 wait 이고 사용자가 시간을 말했으면 분(minute)으로. "
            "범위('5분에서 10분')면 큰 쪽(10)을 넣어라 — 계산·여유는 시스템이 "
            "한다. 시간을 말하지 않았으면 null. 그 외 intent 는 null."
        ),
    )
    reply: str = Field(
        default="",
        description=(
            "사용자에게 들려줄 한국어 답변. intent 가 navigate 이고 "
            "destination_candidate 를 채웠으면 빈 문자열로 둬라 (확인 문구는 "
            "시스템이 만든다)."
        ),
    )


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
- cancel: 진행 중인 안내를 그만두려 함 ("취소해줘", "안 갈래", "됐어 그만").
- pause: 잠시 서 달라는 요청 ("잠깐 쉬었다 가자", "잠시만 서 줘").
- resume: 멈춘 안내를 다시 시작하려 함 ("다시 가자", "출발해").
- affirm / deny: 로봇이 직전에 던진 안내 제안 질문("안내가 필요하신가요?" 등)에
  대한 수락/거절 ("어… 부탁드려요"->affirm, "괜찮아요, 됐어요"->deny).
  목적지 확인 질문의 답이 아니라, 안내 자체를 받겠냐는 제안에 대한 답일 때만.
- wait: 목적지 도착 후 여기서 기다려 달라는 요청 ("좀 있다 올게", "잠깐 여기 있어").
- finish: 오늘 안내를 다 끝내려 함 ("이제 됐어 고마워", "그만 갈게"). 도착 후
  전체 종료다. cancel(주행 중간에 이 목적지만 그만)과 구분하라.

[목적지 목록] (navigate 의 destination_candidate 는 반드시 이 name 중 하나여야 한다. 목록에 없으면 clarify)
{dest_block}
{state_block}
[규칙]
- destination_candidate 는 위 목록의 정확한 name 또는 null. 새로 지어내지 마라.
- navigate(destination_candidate 포함)·cancel·pause·resume·affirm·deny·wait·finish 로
  분류하면 reply 는 빈 문자열로 둬라. 확인·수락 발화는 시스템이 만든다.
- 그 외(question/clarify/unknown)의 reply 는 짧고 친절한 한국어로 써라.
- 확신이 없으면 confidence 를 낮춰라.

[멀티턴 대화]
- 직전에 로봇이 'OO로 안내해드릴까요?'라고 물었고 사용자가 긍정(응, 네, 맞아, 그래, 좋아)하면:
  intent=navigate, destination_candidate=그 OO 목적지 name, is_confirmation=true 로 답해라.
- 사용자가 부정(아니, 그거 말고)하며 다른 목적지를 말하면 그 목적지로 navigate.
- 부정만 하고 목적지를 안 말하면 clarify."""


_AFFIRMATIVES = AFFIRMATIVES
_NEGATIVES = NEGATIVES
_SOLO_AFFIRMATIVES = AFFIRMATIVES | SOFT_AFFIRMATIVES
_normalize_short_reply = normalize_short_reply

SHORTCUT_REPLIES = frozenset({WAKE_GREETING})


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


_COMMAND_CONFIRMS = {
    CANCEL_CONFIRM: "cancel",
    RESUME_CONFIRM: "resume",
}

_CANCEL_WORDS = {"취소", "취소해줘", "취소해주세요", "취소할래", "안내취소"}
_PAUSE_WORDS = {"잠깐만", "잠깐만요", "잠시만", "잠시만요"}
_WAKE_WORDS = {"비카야", "피카야", "비까야",
               "미카야", "리카야", "비켜야", "비кая"}
_SINO_UNITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
               "육": 6, "칠": 7, "팔": 8, "구": 9}
_NATIVE_NUM = {"한": 1, "두": 2, "세": 3, "네": 4}


def _sino_number(token: str):
    """한자어 수사 -> 값. "십오"=15, "이십"=20 같은 합성도 푼다. 실패면 None."""
    if not token:
        return None
    if "십" in token:
        head, _, tail = token.partition("십")
        if head and head not in _SINO_UNITS:
            return None
        if tail and tail not in _SINO_UNITS:
            return None
        return (_SINO_UNITS[head] if head else 1) * 10 + _SINO_UNITS.get(tail, 0)
    return _SINO_UNITS.get(token)


def parse_wait_minutes(text: str):
    """한국어 시간 표현에서 분(minute)을 뽑는다. 없으면 None."""
    import re
    t = (text or "").replace(" ", "")
    if not t:
        return None
    if "반시간" in t:
        return 30
    values = []
    for num, unit in re.findall(r"(\d+|[일이삼사오육칠팔구십]+|[한두세네])(분|시간)", t):
        if num.isdigit():
            value = int(num)
        else:
            value = _NATIVE_NUM.get(num) or _sino_number(num)
        if value is None:
            continue
        values.append(value * (60 if unit == "시간" else 1))
    if not values:
        return None
    if len(values) >= 2 or "에서" in t or "~" in t:
        return int(max(values) * 1.5 + 0.5)
    return values[0]


def is_instant_utterance(user_text: str) -> bool:
    """LLM 없이 0초에 판정되는 짧은 말인가."""
    word = _normalize_short_reply(user_text)
    return bool(word) and (
        word in _SOLO_AFFIRMATIVES or word in _NEGATIVES
        or word in _CANCEL_WORDS or word in _PAUSE_WORDS
        or word in _WAKE_WORDS)


def _pending_command(history: Optional[list[BaseMessage]]) -> Optional[str]:
    """직전 AI 발화가 제어 확인 질문이었으면 해당 intent 를 돌려준다."""
    if not history:
        return None
    last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return None
    return _COMMAND_CONFIRMS.get(last_ai.content)


def _get_structured_llm(model: str):
    """구조화 출력(_IntentDraft) LLM 을 만든다. 백엔드는 PROVIDER 가 정한다."""
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            timeout=15,
            max_retries=1,
        )
        return llm.with_structured_output(_IntentDraft, method="json_schema", strict=True)

    api_key = os.environ.get("OLLAMA_API_KEY", "")
    kwargs = {
        "model": model,
        "base_url": OLLAMA_HOST,
        "temperature": 0,
        "reasoning": False,
        "keep_alive": -1,
    }
    if api_key:
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
    pending_command = _pending_command(history)
    if pending_command is not None:
        word = _normalize_short_reply(user_text)
        if word in _AFFIRMATIVES:
            return VicaIntent(
                intent=pending_command,
                confidence=1.0,
                reply="",
                need_confirm=False,
            )
        if word in _NEGATIVES:
            return VicaIntent(
                intent="unknown",
                confidence=1.0,
                reply=COMMAND_DECLINED,
                need_confirm=False,
            )

    pending = _pending_confirm_destination(history, destinations)
    if pending is not None:
        word = _normalize_short_reply(user_text)
        tokens = user_text.split()
        first = _normalize_short_reply(tokens[0]) if tokens else ""
        denied = (word in _NEGATIVES
                  or any(_normalize_short_reply(t) in _NEGATIVES for t in tokens))
        if not denied and (word in _SOLO_AFFIRMATIVES or first in _AFFIRMATIVES):
            return VicaIntent(
                intent="navigate",
                destination_candidate=pending.name,
                matched_destination_id=pending.id,
                confidence=1.0,
                reply=f"{pending.name} 안내를 시작합니다.",
                need_confirm=False,
                safety_flag="normal",
            )
        if denied:
            return VicaIntent(
                intent="deny",
                confidence=1.0,
                reply="",
                need_confirm=False,
            )

    word = _normalize_short_reply(user_text)
    if word in _WAKE_WORDS:
        return VicaIntent(intent="unknown", reply=WAKE_GREETING,
                          need_confirm=False, confidence=1.0)
    if word in _CANCEL_WORDS:
        return VicaIntent(intent="cancel", confidence=1.0, reply=CANCEL_CONFIRM, need_confirm=True)
    if word in _PAUSE_WORDS:
        return VicaIntent(intent="pause", confidence=1.0, reply=PAUSE_ACK, need_confirm=False)

    if word in _SOLO_AFFIRMATIVES:
        return VicaIntent(intent="affirm", confidence=1.0, reply="", need_confirm=False)
    if word in _NEGATIVES:
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)

    structured = _get_structured_llm(model)
    messages: list[BaseMessage] = [SystemMessage(_build_system_prompt(destinations, robot_state))]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(user_text))

    try:
        draft: _IntentDraft = structured.invoke(messages)
    except Exception as exc:
        import sys

        print(f"[LLM] 호출 실패: {exc}", file=sys.stderr)
        return VicaIntent(
            intent="unknown",
            reply=LLM_UNAVAILABLE,
            confidence=0.0,
            need_confirm=False,
        )
    return _finalize(draft, destinations, pending=pending,
                     pending_command=pending_command, user_text=user_text)


def _finalize(
    draft: _IntentDraft,
    destinations: Sequence[DestinationData],
    pending: Optional[DestinationData] = None,
    pending_command: Optional[str] = None,
    user_text: str = "",
) -> VicaIntent:
    """LLM 초안 + 코드 매칭으로 최종 VicaIntent 를 만든다. (결정/안전은 코드 담당)"""
    result = VicaIntent(
        intent=draft.intent,
        destination_candidate=draft.destination_candidate,
        confidence=draft.confidence or 0.0,
        reply=draft.reply,
        need_confirm=False,
        safety_flag="normal",
    )

    if draft.intent in ("affirm", "deny", "finish"):
        result.reply = ""
        result.matched_destination_id = ""
        result.need_confirm = False
        return result

    if draft.intent == "wait":
        result.reply = ""
        result.need_confirm = False
        minutes = parse_wait_minutes(user_text)
        if minutes is None and draft.wait_minutes and draft.wait_minutes > 0:
            minutes = draft.wait_minutes
        result.wait_minutes = minutes if minutes else -1
        return result

    if draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
        if matched is None:
            result.intent = "clarify"
            result.reply = result.reply or ASK_DESTINATION
        elif not matched.is_approachable:
            result.matched_destination_id = matched.id
            result.reply = matched.unavailable_reason or matched.confirm_prompt
            result.need_confirm = False
        elif draft.is_confirmation and pending is not None and matched.id == pending.id:
            result.matched_destination_id = matched.id
            result.reply = f"{matched.name} 안내를 시작합니다."
            result.need_confirm = False
        else:
            result.matched_destination_id = matched.id
            result.reply = matched.confirm_prompt
            result.need_confirm = True

    if draft.intent in ("cancel", "pause", "resume"):
        if draft.intent == "pause":
            result.reply = PAUSE_ACK
            result.need_confirm = False
        elif pending_command == draft.intent:
            result.reply = ""
            result.need_confirm = False
        else:
            result.reply = {
                "cancel": CANCEL_CONFIRM,
                "resume": RESUME_CONFIRM,
            }[draft.intent]
            result.need_confirm = True

    if not result.reply:
        result.reply = ASK_DESTINATION if result.intent == "clarify" else RETRY_PROMPT

    return result
