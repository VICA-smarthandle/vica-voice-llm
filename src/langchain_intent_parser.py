"""사용자 발화를 LangChain(Ollama Cloud)으로 분석해 VicaIntent 로 만든다."""
from __future__ import annotations

import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from .destination_loader import _josa_euro
from .destination_matcher import match_destination, tied_destinations
from .mission_phrases import CONFIRM_SWITCH
from .handle_mode import (
    AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply)
from . import local_rules
from .ledger_view import FLOOR_LABEL
from .llm_backend import LlmBackendManager, ProbeResult, http_probe, ollama_warm
from .realtime_intent import get_realtime_client
from .replies import (
    ALREADY_GOING,
    ASK_DESTINATION,
    CANCEL_CONFIRM,
    CHANGE_DESTINATION_QUESTION,
    CHANGE_DESTINATION_TAIL,
    COMMAND_DECLINED,
    LLM_UNAVAILABLE,
    PAUSE_ACK,
    RESUME_CONFIRM,
    RETRY_PROMPT,
    WAKE_GREETING,
    choose_between,
)
from .schema import DestinationData, RobotState, VicaIntent, VicaIntentType

load_dotenv()

PROVIDER = os.environ.get("VICA_LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
if PROVIDER == "openai":
    DEFAULT_MODEL = os.environ.get("VICA_OPENAI_MODEL", "gpt-5.4-mini")
else:
    DEFAULT_MODEL = os.environ.get("VICA_LLM_MODEL", "gemma4:cloud")

FALLBACK_MODEL = os.environ.get("VICA_LLM_FALLBACK_MODEL", "").strip()
FALLBACK_HOST = os.environ.get("VICA_LLM_FALLBACK_HOST", "http://localhost:11434")
CLOUD_TIMEOUT_SEC = float(os.environ.get("VICA_LLM_CLOUD_TIMEOUT", "6"))
CLOUD_PROBE_SEC = float(os.environ.get("VICA_LLM_CLOUD_PROBE_SEC", "30"))
LOCAL_TIMEOUT_SEC = float(os.environ.get("VICA_LLM_LOCAL_TIMEOUT", "45"))


class _IntentDraft(BaseModel):
    """LLM 이 직접 채우는 부분만 담은 임시 스키마."""

    intent: VicaIntentType = Field(
        description="navigate / question / clarify / unknown / cancel / pause / resume / affirm / deny / wait / finish / help 중 하나"
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
    need_confirm: Optional[bool] = Field(
        default=None,
        description=(
            "소리 모드(모델 전결)에서만 쓴다: 되물어야 하면 true(새 목적지 제안·취소·"
            "다시 출발), 확정·단답이면 false. 텍스트 경로의 _finalize 는 이 값을 무시한다."
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


HANDLE_FACTS = """[손잡이] 사용자가 물으면 이 사실로 답한다.
- 로봇 뒤쪽에 손잡이가 있다. 안내를 시작할 때 손잡이가 진동해 자리를 알려 준다.
- 손잡이를 잡으면 손잡이 모드로 안내한다. 이때 손을 놓으면 로봇이 그 자리에 서고
  손잡이가 진동한다. 다시 잡으면 이어서 간다.
- 손잡이를 잡지 않고 따라와도 안내한다. 이때는 손을 놓아도 서지 않는다.
- 손잡이 줄을 당기면 로봇이 느려진다. 끝까지 당기면 선다. (말로 속도를 바꾸는 기능은 없다)
- 도착하면 손잡이가 짧게 세 번 떨린다."""


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
    destinations: Sequence[DestinationData], robot_state: Optional[RobotState] = None,
    local: bool = False, situation: str = "",
) -> str:
    """글자 경로 지시문. situation 은 LLM 노드의 [지금 상황] 블록(대장, ledger_view)."""
    lines = []
    for d in destinations:
        aliases = ", ".join(d.aliases)
        approach = "가능" if d.is_approachable else "불가"
        lines.append(f"- {d.name} (별칭: {aliases} / 분류: {d.category2} / 접근: {approach})")
    dest_block = "\n".join(lines)
    ledger = situation if FLOOR_LABEL in situation else ""
    state_block = "" if ledger else _format_robot_state(robot_state)
    return f"""너는 시각장애인 안내 로봇 'VICA'의 음성 의도 분석기다.
사용자의 한국어 발화를 분석해 아래 규칙으로 분류해라.

[intent 종류]
- navigate: 어딘가로 가고 싶어함. 직접 표현("407호 가자")뿐 아니라 간접 표현("배 아파"->화장실, "배고파"->식당)도 포함.
- question: 이동이 아니라 정보 질문("지금 몇 층이야?").
- clarify: 어디로 갈지 모호해 되물어야 함. reply 에 되묻는 질문을 담아라.
- unknown: 안내와 무관하거나 이해 불가.
- cancel: 진행 중인 안내를 그만두려 함 ("취소해줘", "안 갈래", "됐어 그만").
- pause: 잠시 서 달라는 요청 ("잠깐 쉬었다 가자", "잠시만 서 줘").
  "대기해줘"·"여기서 기다려줘"는 pause 가 아니라 wait 다.
- resume: 멈춘 안내를 다시 시작하려 함 ("다시 가자", "출발해").
- affirm / deny: 로봇이 직전에 던진 안내 제안 질문("안내가 필요하신가요?" 등)에
  대한 수락/거절 ("어… 부탁드려요"->affirm, "괜찮아요, 됐어요"->deny).
  목적지 확인 질문의 답이 아니라, 안내 자체를 받겠냐는 제안에 대한 답일 때만.
  "안내를 받으시겠어요?"에 목적지로 답하면("응, 화장실 가고 싶어") affirm 이 아니라 navigate 다.
  "기다리지 마"·"대기 안 해도 돼"는 대기를 묻는 질문 뒤면 deny, "안내를 마칠까요?" 뒤면 affirm.
  "안내가 필요 없으신가요?"는 부정 질문이라 뜻으로 고른다 — 필요 없다는 뜻("네", "아니, 필요 없어",
  "괜찮아")은 affirm, 필요하다는 뜻("아니요", "필요해")은 deny.
- wait: 목적지 도착 후 여기서 기다려 달라는 요청 ("좀 있다 올게", "잠깐 여기 있어", "대기해줘"·"여기서 기다려줘").
  홈으로 돌아가다 "비카야"로 세운 뒤에도 wait 다(로봇 본체가 직전에 간 곳의 대기 장소로 간다).
- finish: 오늘 안내를 다 끝내려 함 ("이제 됐어 고마워", "그만 갈게"). 도착 후
  전체 종료다. cancel(주행 중간에 이 목적지만 그만)과 구분하라.
- help: 직원이나 관리자의 도움을 원함 ("도움이 필요해", "관리자 불러줘", "사람 좀 불러 주세요",
  "직원 불러줘"). reply 는 비운다 — 확인 질문과 안내는 로봇 본체가 한다.

{HANDLE_FACTS}

[목적지 목록] (navigate 의 destination_candidate 는 반드시 이 name 중 하나여야 한다. 목록에 없으면 clarify)
{dest_block}
{state_block}
[규칙]
- destination_candidate 는 위 목록의 정확한 name 또는 null. 새로 지어내지 마라.
- 들은 말이 목록의 여러 곳에 똑같이 걸리면(같은 별칭 — 예: "화장실"이 남자 화장실·여자 화장실 둘 다)
  navigate 로 하나를 짐작해 고르지 말고 clarify 로 그 곳들의 name 을 모두 들어 묻는다("남자 화장실과
  여자 화장실 중 어디로 갈까요?"). 하나만 들어 묻지도 않는다(2026-10-11 실기).
- navigate(destination_candidate 포함)·cancel·pause·resume·affirm·deny·wait·finish 로
  분류하면 reply 는 빈 문자열로 둬라. 확인·수락 발화는 시스템이 만든다.
- 그 외(question/clarify/unknown)의 reply 는 짧고 친절한 한국어로, 가급적 한 문장으로 써라. 알려 줄 말과
  물을 말은 마침표로 나누지 말고 "~인데,"로 잇는다 — 로봇은 문장마다 목소리를 새로 만들어서 문장 사이에
  빈틈이 생긴다(2026-10-10 사용자). 정보와 되묻기까지만이다 —
  "알겠습니다"·"할게요" 같은 약속, "안내해드릴까요?"·"계속 안내할까요?"처럼 로봇이 할 일을 권하는
  말은 쓰지 않는다(로봇 본체가 모르는 약속이 된다). 목록에 있는 곳을 권하려면 navigate 로 내라.
  되묻는 말만 "?"로 끝내고, 대답·안내·모른다는 말은 "."로 끝낸다 — 평서문에 "?"를 붙이면 질문처럼
  들리고 로봇이 쓸데없이 귀를 연다(2026-10-11).
- 로봇의 마지막 말이 "…안내를 받으시겠어요?"(대화 단계 '인사 답 대기')이면 question 의 reply 는
  한 문장으로 답만 하고 "?"로 끝내지 않는다 — 로봇 본체가 이어서 다시 묻는다(2026-10-09 실기).
- 확신이 없으면 confidence 를 낮춰라.

[지금 상황 쓰기] 맨 뒤에 [지금 상황]이 있으면 미션이 확인한 사실이다.
- 대화 단계가 '안내 중(주행)'일 때 안내 중인 곳과 다른 목적지로 가고 싶다고 하면 navigate 다(로봇이
  서서 바꿀지 묻는다 — 그 질문은 시스템이 만든다). 위치만 물으면("화장실 어디예요?") 로봇을 세우지
  않도록 question 으로 위치만 답한다.
- "OO 어디야?"에서 OO 가 직전에 간 곳(방금 도착한 곳)이고 'OO 방향' 줄이 있으면 navigate 가 아니라
  question 이다. reply 는 그 줄대로("화장실은 오른쪽에 있어요."). 이 방향은 도착했을 때 로봇(뒤에서
  손잡이를 잡은 사용자) 기준 입구 쪽이다.
- "어디서 기다린다고?"·"어디서 기다려?"는 question 이다. '대기 장소' 줄로 답하고, 줄이 없으면
  지금 있는 곳에서 기다린다고 답한다.
- 대화 단계가 '대기 중'·'손 놓기 기다림'일 때 목적지를 말하면 평소처럼 navigate 다(대기는 출발할 때 끝난다).
  "다 됐어"·"다녀왔어"처럼 볼일이 끝났다고만 하면 finish 다(reply 는 빈 문자열 — 로봇이 "어디로
  모실까요?"라고 묻는다).

[멀티턴 대화]
- 직전에 로봇이 'OO로 안내해드릴까요?' 또는 '…OO로 바꿀까요?'라고 물었고 사용자가 긍정(응, 네,
  맞아, 그래, 좋아)하면: intent=navigate, destination_candidate=그 OO 목적지 name, is_confirmation=true
  로 답해라. '지금 XX로 가는 중이에요. OO로 바꿀까요?'의 답이면 OO 다(XX 가 아니다).
- 사용자가 부정(아니, 그거 말고)하며 다른 목적지를 말하면 그 목적지로 navigate.
- 부정만 하고 목적지를 안 말하면 clarify.""" + (local_rules.LOCAL_PROMPT_RULES if local else "") + ledger


_AFFIRMATIVES = AFFIRMATIVES
_NEGATIVES = NEGATIVES
_SOLO_AFFIRMATIVES = AFFIRMATIVES | SOFT_AFFIRMATIVES
_normalize_short_reply = normalize_short_reply

SHORTCUT_REPLIES = frozenset({WAKE_GREETING})


def is_scripted_reply(reply: str, destinations: Sequence[DestinationData]) -> bool:
    """코드가 정한 문구(호출 응답·목적지 확인 질문)인가 — 재청취 기각 예외 판별."""
    if not reply:
        return False
    return reply in SHORTCUT_REPLIES or any(
        (d.confirm_prompt and d.confirm_prompt == reply)
        or reply == ALREADY_GOING.format(cur=d.name, cur_josa=_josa_euro(d.name))
        for d in destinations)


def _question_destination(text: str, destinations: Sequence[DestinationData]):
    """이 로봇 말이 어떤 목적지로 가자는 확인 질문이면 그 목적지, 아니면 None."""
    text = (text or "").strip()
    if not text:
        return None
    for dest in destinations:
        if dest.confirm_prompt and text in (
                dest.confirm_prompt, CONFIRM_SWITCH.format(prompt=dest.confirm_prompt)):
            return dest
    matches = [
        dest for dest in destinations
        if text.endswith(CHANGE_DESTINATION_TAIL.format(
            new=dest.name, new_josa=_josa_euro(dest.name)))
    ]
    return max(matches, key=lambda d: len(d.name)) if matches else None


def _guiding_destination(robot_state: Optional[RobotState]) -> str:
    """미션이 안내 주행 중이면 가는 곳 이름, 아니면 "" (주행 중 바꾸기 질문용)."""
    if robot_state is None or robot_state.dialog_state != "navigating":
        return ""
    return robot_state.active_destination or ""


def _pending_confirm_destination(
    history: Optional[list[BaseMessage]], destinations: Sequence[DestinationData]
):
    """직전 AI 발화가 어떤 목적지의 확인 질문이었으면 그 목적지를 돌려준다."""
    if not history:
        return None
    last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return None
    return _question_destination(last_ai.content, destinations)


def _recent_confirm_destination(
    history: Optional[list[BaseMessage]], destinations: Sequence[DestinationData]
):
    """기록에서 **가장 최근의 확인 질문**이 가리키는 목적지 (로컬 규칙 전용, 2026-10-05)."""
    if not history:
        return None
    for m in reversed(history):
        if not isinstance(m, AIMessage):
            continue
        dest = _question_destination(m.content, destinations)
        if dest is not None:
            return dest
    return None


def _pending_for(history, destinations, local: bool, dialog_state: str):
    """'확인 대기 중인 목적지'. 로컬 규칙 + 미션 상태를 알면 미션 상태가 정한다."""
    if not local or not dialog_state:
        return _pending_confirm_destination(history, destinations)
    if dialog_state == "confirming":
        return _recent_confirm_destination(history, destinations)
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


def parse_wait_minutes(text: str, range_factor: float = 1.5):
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
        return int(max(values) * range_factor + 0.5)
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


def _get_structured_llm(model: str, *, timeout: float = 15, max_retries: int = 1):
    """구조화 출력(_IntentDraft) LLM 을 만든다. 백엔드는 PROVIDER 가 정한다."""
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
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
    if local_rules.enabled():
        kwargs["num_ctx"] = local_rules.LOCAL_NUM_CTX
    client_kwargs: dict = {"timeout": timeout}
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    kwargs["client_kwargs"] = client_kwargs
    llm = ChatOllama(**kwargs)
    return llm.with_structured_output(_IntentDraft)


def _build_local_structured():
    """로컬 Ollama 구조화 LLM. 09-19 실측: gemma4-e2b-text 8/8, 발화당 3.0초."""
    extra = {"num_ctx": local_rules.LOCAL_NUM_CTX} if local_rules.enabled() else {}
    llm = ChatOllama(
        model=FALLBACK_MODEL,
        base_url=FALLBACK_HOST,
        temperature=0,
        reasoning=False,
        keep_alive=-1,
        client_kwargs={"timeout": LOCAL_TIMEOUT_SEC},
        **extra,
    )
    return llm.with_structured_output(_IntentDraft)


def _cloud_probe() -> ProbeResult:
    """토큰을 쓰지 않는 접속 확인. openai 는 /models, ollama 클라우드는 /api/version."""
    if PROVIDER == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        return http_probe(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    key = os.environ.get("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    return http_probe(f"{OLLAMA_HOST.rstrip('/')}/api/version", headers=headers)


def _log(level: str, msg: str) -> None:
    import sys

    if level == "info":
        print(msg, file=sys.stderr)
    else:
        print(f"[{level.upper()}] {msg}", file=sys.stderr)


_MANAGER: Optional[LlmBackendManager] = None


def get_backend_manager() -> LlmBackendManager:
    """전환 담당 모듈 싱글턴. 처음 부를 때 만든다(시험이 임포트만으로 클라이언트를 만들지 않게)."""
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    if FALLBACK_MODEL:
        try:
            local = _build_local_structured().invoke
        except Exception as exc:
            _log("error", f"[LLM] 로컬 백엔드 준비 실패({exc}) — 폴백 없이 클라우드만 씀")
            local = None
    else:
        local = None
    fallback_on = local is not None
    try:
        cloud_llm = _get_structured_llm(
            DEFAULT_MODEL,
            timeout=CLOUD_TIMEOUT_SEC if fallback_on else 15,
            max_retries=0 if fallback_on else 1,
        )
        cloud = cloud_llm.invoke
    except Exception as build_error:
        def cloud(messages, _err=build_error):
            raise _err
    _MANAGER = LlmBackendManager(
        cloud, local, _cloud_probe,
        warm_local=(lambda: ollama_warm(FALLBACK_HOST, FALLBACK_MODEL, logger=_log)) if fallback_on else None,
        logger=_log,
        local_name=FALLBACK_MODEL if fallback_on else "-",
        probe_interval_sec=CLOUD_PROBE_SEC,
    )
    return _MANAGER


def reset_backend_manager() -> None:
    """시험용. 다음 get_backend_manager() 가 새로 만든다."""
    global _MANAGER
    _MANAGER = None


def _shortcut_intent(user_text: str, history: Optional[list[BaseMessage]],
                     destinations: Sequence[DestinationData],
                     dialog_state: str = "", local: bool = False) -> Optional[VicaIntent]:
    """LLM 없이 코드가 확정하는 경우. 없으면 None."""
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
            if pending_command == "cancel":
                return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
            return VicaIntent(
                intent="unknown",
                confidence=1.0,
                reply=COMMAND_DECLINED,
                need_confirm=False,
            )

    if local and dialog_state == "confirming":
        confirming = _confirming_shortcut(user_text, history, destinations)
        if confirming is not None:
            return confirming

    pending = _pending_for(history, destinations, local, dialog_state)
    if pending is not None:
        word = _normalize_short_reply(user_text)
        tokens = user_text.split()
        first = _normalize_short_reply(tokens[0]) if tokens else ""
        denied = (word in _NEGATIVES
                  or any(_normalize_short_reply(t) in _NEGATIVES for t in tokens))
        if not denied and word in SOFT_AFFIRMATIVES and word not in _AFFIRMATIVES:
            return VicaIntent(
                intent="clarify",
                confidence=1.0,
                reply=pending.confirm_prompt,
                need_confirm=False,
                safety_flag="normal",
            )
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

    if local and not local_rules.short_answer_allowed(dialog_state):
        return None
    if word in _SOLO_AFFIRMATIVES:
        return VicaIntent(intent="affirm", confidence=1.0, reply="", need_confirm=False)
    if word in _NEGATIVES:
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
    return None


def _confirming_shortcut(user_text: str, history: Optional[list[BaseMessage]],
                         destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """로컬 규칙 + 미션 confirming 일 때의 짧은 대답. 짧은 대답이 아니면 None(LLM 몫)."""
    word = _normalize_short_reply(user_text)
    tokens = user_text.split()
    first = _normalize_short_reply(tokens[0]) if tokens else ""
    denied = (word in _NEGATIVES
              or any(_normalize_short_reply(t) in _NEGATIVES for t in tokens))
    dest = _recent_confirm_destination(history, destinations)
    if denied:
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
    if word in SOFT_AFFIRMATIVES and word not in _AFFIRMATIVES:
        return VicaIntent(intent="clarify", confidence=1.0,
                          reply=dest.confirm_prompt if dest else RETRY_PROMPT,
                          need_confirm=False, safety_flag="normal")
    if word in _SOLO_AFFIRMATIVES or first in _AFFIRMATIVES:
        if dest is None:
            return VicaIntent(intent="affirm", confidence=1.0, reply="", need_confirm=False)
        return VicaIntent(intent="navigate", destination_candidate=dest.name,
                          matched_destination_id=dest.id, confidence=1.0,
                          reply=f"{dest.name} 안내를 시작합니다.",
                          need_confirm=False, safety_flag="normal")
    return None


def is_scripted_intent(intent: VicaIntent, destinations: Sequence[DestinationData]) -> bool:
    """재청취 기각 예외 판별 — is_scripted_reply 에 로컬 규칙의 되묻기 하나를 더한다."""
    if is_scripted_reply(intent.reply, destinations):
        return True
    return (local_rules.enabled() and intent.intent == "clarify"
            and intent.reply == RETRY_PROMPT and intent.confidence == 1.0)


def parse_intent(
    user_text: str,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    model: Optional[str] = None,
    situation: str = "",
) -> VicaIntent:
    """발화를 분석해 VicaIntent 를 돌려준다. (멀티턴: history, 현재 상태: robot_state)"""
    local = local_rules.enabled()
    dialog_state = (robot_state.dialog_state if (local and robot_state) else "") or ""
    shortcut = _shortcut_intent(user_text, history, destinations,
                                dialog_state=dialog_state, local=local)
    if shortcut is not None:
        return shortcut
    pending_command = _pending_command(history)
    pending = _pending_for(history, destinations, local, dialog_state)

    messages: list[BaseMessage] = [SystemMessage(
        _build_system_prompt(destinations, robot_state, local=local, situation=situation))]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(user_text))

    try:
        if model is not None:
            draft: _IntentDraft = _get_structured_llm(model).invoke(messages)
        else:
            draft = get_backend_manager().invoke(messages)
    except Exception as exc:
        if model is not None:
            _log("error", f"[LLM] 호출 실패(모델 {model}): {exc}")
        elif not get_backend_manager().has_local:
            _log("error", f"[LLM] 호출 실패(클라우드만, 폴백 꺼짐): {exc}")
        else:
            _log("error", f"[LLM] 호출 실패(로컬까지): {exc}")
        return VicaIntent(
            intent="unknown",
            reply=LLM_UNAVAILABLE,
            confidence=0.0,
            need_confirm=False,
        )
    return _finalize(draft, destinations, pending=pending,
                     pending_command=pending_command, user_text=user_text, local=local,
                     robot_state=robot_state)


def build_audio_prompt(
    destinations: Sequence[DestinationData], robot_state: Optional[RobotState] = None,
    situation: str = "", directory_block: str = "",
) -> str:
    """소리 모드 지시문. 모델이 대화 판단자다 — 확인·정정·침묵까지 모델 몫."""
    lines = []
    for d in destinations:
        aliases = ", ".join(d.aliases)
        where = " ".join(x for x in (d.building, f"{d.floor}층" if d.floor else "") if x)
        place = f" / 위치: {where}" if where else ""
        if d.is_approachable:
            lines.append(f'- {d.name} (별칭: {aliases}{place}) — 확인 질문: "{d.confirm_prompt}"')
        else:
            lines.append(f"- {d.name} (별칭: {aliases}{place}) — 접근 불가")
    dest_block = "\n".join(lines)
    state_block = "" if FLOOR_LABEL in situation else _format_robot_state(robot_state)
    return f"""너는 시각장애인 안내 로봇 '비카(VICA)'다. 차분하고 짧게 말하는 안내원처럼 말하고, 사용자의
목소리를 직접 듣고, 다음에 무엇을 할지와 무슨 말을 할지를 set_intent 함수 한 번으로 정한다.
코드는 네 결정을 고치지 않고 그대로 로봇 본체(미션 관리자)에 전달한다. 확인 질문·정정·
되묻기·침묵·말투까지 네가 책임진다.

[대화 이력 = 네 기억] 앞에 오는 assistant 줄은 네가(로봇이) 실제로 소리 내어 말한 문장이고,
user 줄은 그 전에 들린 사용자의 말이다. 로봇의 마지막 말이 질문이면 지금 들리는 말은 대개
그 답이다. "아까 어디 갔었지?", "방금 도착한 곳이 어디야?" 같은 질문은 이력의 "…로 안내를
시작합니다"·"…에 도착했습니다" 줄을 보고 답한다 — 기억이 없다고 하지 마라.

[heard_text] 이번 소리에서 실제로 들린 말만 한국어로 그대로 적는다(결정을 내렸다면 반드시
채운다 — "그래"를 듣고 출발시키면서 heard_text 를 비우지 마라). 사람 말이 아니면
(기침·소음·로봇 자신의 목소리·다른 사람들끼리의 잡담) 빈 문자열로 두고 intent=unknown,
reply="" 로 답한다. 이력에 있는 말을 베껴 적지 마라 — 이번 소리에 없는 말은 없는 것이다.
애매하면 confidence 를 낮추고, 답이 꼭 필요한 자리면 clarify 로 되묻는다.

[intent 별 규칙 — need_confirm 과 reply 까지 네가 정한다]
- navigate(새 목적지): 가고 싶은 곳을 말했다. 직접("407호 가자")도 간접("배 아파"→화장실)도 된다.
  destination_candidate=목록의 name 그대로, need_confirm=true, reply=네 말로 만든 확인 질문
  (목적지 name 을 그대로 넣고 반드시 "?"로 끝낸다. 예: "화장실로 안내해드릴까요?").
  들은 말이 목록의 여러 곳에 똑같이 걸리면(같은 별칭 — 예: "화장실 가고 싶어"가 남자 화장실·여자 화장실
  둘 다) 하나를 짐작해 고르지 말고 clarify 로 그 곳들의 name 을 모두 들어 묻는다("남자 화장실과 여자 화장실
  중 어디로 갈까요?"). 하나만 들어 "남자 화장실인가요?"처럼 묻지도 않는다(2026-10-11 실기: 같은 말에
  "여자 화장실로 안내해드릴까요?"·"남자 화장실인가요?").
- navigate(주행 중 바꾸기): [지금 상황]의 대화 단계가 '안내 중(주행)'인데 '안내 중' 줄의 곳(OO)이
  아닌 다른 목적지(XX)로 가고 싶다고 했다("화장실 가고 싶어", "식당으로 가자", "배 아파").
  destination_candidate=XX, need_confirm=true, reply="지금 OO로 가는 중이에요. XX로 바꿀까요?"
  (조사는 받침에 맞춰 '로/으로'). 로봇은 서서 이 답을 기다린다.
  이 질문은 대화 단계가 '안내 중(주행)'일 때만 쓴다. 대기 중·안내 없음·도착 질문 중에는 로봇이 어디로도
  가지 않으니 "가는 중"이라고 말하지 않는다 — '직전에 간 곳'과 '대기 장소'는 지금 가는 곳이 아니다
  (2026-10-10 실기). 지금 가는 곳(OO)을 또 말하면
  navigate 가 아니라 question 으로 "지금 OO로 가는 중이에요."만 답한다. 안내 중에 위치만 물으면
  ("화장실 어디예요?") 로봇을 세우지 않는다 — 아래 question 규칙의 위치 질문 예외와 달리 question
  으로 위치만 답한다(가고 싶다고 하면 그때 바꾸기다).
- navigate(확정): 로봇의 마지막 말이 "OO로 안내해드릴까요?" 또는 "…OO로 바꿀까요?"이고 사용자가
  긍정(네·응·그래·맞아·좋아·어, "그래 빨리 가자"처럼 재촉이 붙어도 긍정)했다. destination_candidate=OO,
  need_confirm=false, reply="" (출발 안내는 미션이 말한다). "지금 XX로 가는 중이에요. OO로 바꿀까요?"의
  답이면 OO 다(XX 가 아니다). 확인 질문을 또 하지 마라 — 긍정에 다시 물으면 사용자는 두 번 답해야 한다.
- 대기 중 목적지: 대화 단계가 '대기 중'·'손 놓기 기다림'이어도 목적지를 말하면 평소처럼
  navigate(새 목적지)다 — 대기는 출발할 때 끝난다. reply 는 목록의 확인 질문 하나다(예: "화장실로
  안내해드릴까요?"). "다 됐어"·"다녀왔어"처럼 볼일이 끝났다고만
  하면 finish, reply="" 다(로봇 본체가 "네, 어디로 모실까요?"라고 묻는다).
- 정정: 확인 질문에 "아니 XX로 가자"처럼 다른 목적지를 말하면 XX 로 navigate, need_confirm=true,
  reply=XX 로 다시 묻는 확인 질문("XX로 안내해드릴까요?").
- deny: 확인 질문이나 제안에 부정만 하고 대안이 없다("아니", "아니요", "됐어"). reply="".
  부정 뒤에 **목록에 없는 곳**을 말하면("아니, 학과장님 사무실") 방금 거절당한 목적지를 그대로 다시
  제안하지 마라 — 로봇 본체는 같은 곳의 재제안을 승낙으로 보고 출발한다(21:21 실기). clarify 로
  "학과장님 사무실은 목록에 없는데, 학과사무실이 맞을까요?"처럼 되묻는다.
  "기다리지 마"·"대기 안 해도 돼"는 대기를 거절하는 말이다: 로봇의 마지막 말이 대기를 묻는 질문
  ("…여기서 기다릴까요?"·"여기서 대기할까요?"·"몇 분쯤 걸리실까요?")이면 deny, "여기까지 안내를
  마칠까요?"처럼 끝낼지 묻는 질문이면 affirm 이다(2026-10-08 실기 "대기하지 마").
  로봇의 마지막 말이 "안내가 필요 없으신가요?"이면 글자가 아니라 뜻으로 고른다 — 필요 없다는 뜻
  ("네", "응", "필요 없어", "아니, 필요 없어", "괜찮아", "됐어")은 affirm, 필요하다는 뜻("아니요",
  "필요해", "더 기다려 줘")은 deny. reply="".
- affirm: 로봇의 제안 질문("안내를 받으시겠어요?", "여기서 기다릴까요?", "여기서 대기할까요?")에 긍정. reply="".
  로봇의 마지막 말이 "…안내를 받으시겠어요?"(사람에게 다가가 건 첫 질문)이면 긍정은 affirm, 부정은
  deny 다 — 그 전에 네가 "OO로 안내해드릴까요?"라고 물었던 것과 무관하다. "네. 네." 도 affirm. 그
  질문에 목적지로 답하면("응, 화장실 가고 싶어") navigate(need_confirm=true, reply="")로 낸다 — 로봇
  본체가 수락으로 받고, 돌아서 손잡이를 내준 뒤 그 목적지를 직접 묻는다(2026-10-08).
- wait: 도착 뒤 기다려 달라는 말이나 시간("오 분", "한 10분에서 15분", "반시간"). wait_minutes 에 분을 넣는다
  (범위면 큰 쪽, 시간이 없으면 null). "몇 분쯤 걸리실까요?"의 답은 시간만 말해도 wait 다. reply="".
  "대기해줘"·"여기서 기다려줘"처럼 머물러 달라는 말도 wait 다. 홈으로 돌아가다 "비카야"로 세운 뒤에도 wait 다
  — 로봇 본체가 직전에 간 곳의 대기 장소로 간다(2026-10-10 실기: pause 로 내서 "지금은 안내 중이
  아닙니다"로 거절됐다).
- finish: 도착 뒤 오늘 안내를 다 끝낸다("이제 됐어 고마워", "그만 갈게"). reply="".
- help: 직원이나 관리자의 도움을 원한다("도움이 필요해", "관리자 불러줘", "사람 좀 불러 주세요",
  "직원 불러줘"). reply="" — 확인 질문("직원의 도움이 필요하신가요?")과 안내는 로봇 본체가 한다.
- cancel: 진행 중인 안내를 그만두려 한다. 처음 말했으면 need_confirm=true, reply="{CANCEL_CONFIRM}".
  로봇이 방금 그렇게 물었고 사용자가 긍정하면 need_confirm=false, reply="".
- pause: 잠시 서 달라("잠깐만", "잠시 서 줘"). need_confirm=false, reply="{PAUSE_ACK}".
  "대기해줘"·"여기서 기다려줘"는 pause 가 아니라 wait 다.
- resume: 다시 출발. 처음이면 need_confirm=true, reply="{RESUME_CONFIRM}". 방금 그렇게 물었고 긍정이면 need_confirm=false, reply="".
- question: 이동이 아닌 정보 질문이나 로봇에게 건 가벼운 말(인사·고맙다·농담·"왜 안 가?").
  reply 에 **한 문장으로 짧게(30자 안팎)** 답한다. 아는 것 = 목적지 목록(별칭·위치)·[지금
  상황]·대화 이력. 안내 중이 아닐 때 "OO 교수님 방이 어디야?"·"세미나실이 어디지?"처럼 **목록에 있는 곳의
  위치를 물으면 intent 는 question 이 아니라 navigate** 다(need_confirm=true, destination_candidate=
  그 곳, reply="407호는 로봇관 4층에 있는데, 안내해드릴까요?"). question 으로 두면 뒤따르는 "응"이
  갈 곳 없는 긍정이 되어 로봇이 움직이지 않는다(21:11 실기). 단, 물은 곳이 [지금 상황]의 직전에 간
  곳(방금 도착한 곳)이고 그곳의 '방향' 줄("- 화장실 방향: 오른쪽")이 있으면 navigate 가 아니라 question
  으로 그 줄대로 답한다("화장실은 오른쪽에 있어요."). 이 방향은 도착했을 때 로봇(뒤에서 손잡이를 잡은
  사용자) 기준 입구 쪽이다. "어디서 기다린다고?"·"어디서 기다려?"는 '대기 장소' 줄로 답하고(줄이
  없으면 지금 있는 곳에서 기다린다고 답한다). "몇 층이야?"는 [지금 상황]의 건물/층
  (목적지 목록에 같은 방 번호가 다른 건물로 있어도 절대 섞지 말고 이 줄의 건물만 답한다), "지금 어디
  있어?"는 지금 있는 곳, "지금 어디 가?"는 안내 중,
  "아까 어디 갔었지?"는 직전에 간 곳, "어디 가려고 했더라?"는 하려다 만 곳, "몇 시야?"는 시각,
  "배터리 얼마나 남았어?"는 배터리 줄로 답한다. 줄이 "모름"이면 한 문장으로 모른다고 한다. 정말 모르는 것만 한 문장으로 모른다고 한다 —
  다른 사람에게 물어보라는 식의 조언은 하지 않는다.
- 약속 금지(question·clarify·unknown 공통, 2026-10-08): reply 는 정보와 되묻기까지만이다. "알겠습니다"·
  "할게요"·"해 드릴게요" 같은 약속, "계속 안내할까요?"·"방향 맞춰드릴까요?"처럼 로봇이 할 일을 권하는
  말은 쓰지 않는다 — 로봇 본체가 모르는 약속이 된다. 목록에 있는 곳으로 안내를 권하려면 question 이
  아니라 navigate(need_confirm=true)로 낸다(위 위치 질문 규칙처럼).
- 접근 질문 중의 question: 로봇의 마지막 말이 "…안내를 받으시겠어요?"(대화 단계 '인사 답 대기')이면
  묻는 말에 한 문장으로 답만 하고 "?"로 끝내지 않는다. 되묻거나 말을 청하지 않는다("말씀해 주세요"·
  "안내를 시작할까요?" 같은 말 금지) — 로봇 본체가 이어서 "안내를 받으시겠어요?"를 다시 묻는다(2026-10-09
  실기: 답 끝에 질문을 붙여 질문이 두 번 겹쳤다). 예: "누구세요?" → "저는 시각장애인 안내로봇 비카입니다.",
  "뭐라고요?" → "안내를 받으시겠냐고 여쭤봤어요."
- clarify: 어디로 갈지 모호하거나 목록에 없는 곳. reply 에 되묻는 질문(반드시 "?"로 끝).
  질문 하나만 짧게 하고 예시를 늘어놓지 않는다. 들은 말이 목록의 여러 곳에 걸리면(같은 별칭)
  그 곳들의 name 만 들어 어느 쪽인지 묻는다.
- unknown + 침묵(reply=""): 로봇에게 한 말이 **아닌 것이 분명할 때**(다른 사람들끼리의 대화,
  기침·소음, 로봇 자신의 목소리), 그리고 **로봇이 방금 질문하고 답을 기다리는데 못 알아들었을
  때**다. 질문 중이면 로봇 본체가 같은 질문을 한 번 다시 한다 — 네가 또 물으면 두 목소리가 된다
  (2026-10-08). 로봇이 묻고 있지 않은데 사용자가 로봇에게 말한 것 같고 못 알아들었으면 침묵하지
  말고 intent=clarify 로 "잘 못 들었는데, 다시 말씀해 주시겠어요?"(2026-09-20 사용자: 아무 말도
  안 하면 무슨 상황인지 알 수 없다). 호출어 "비카야"만 들리면 로봇을 부른 것이다 — intent=unknown, reply 는
  "{WAKE_GREETING}" 나 "네, 말씀하세요." 같은 짧은 응답.

[목적지 목록] destination_candidate 는 반드시 아래 name 중 하나. 목록에 없는 곳은 clarify.
(단, 아래 '다른 층 장소' 에 있는 곳은 clarify 가 아니라 [다른 층] 규칙)
reply 에서 장소를 말할 때도 아래 name·별칭과 '다른 층 장소'만 쓴다. 목록에 없는 장소·부서·사람
이름은 지어내지 않는다 — 예로도 들지 않는다. 이 지시문의 규칙 속 장소 이름(407호·식당·세미나실
등)은 설명용 예일 뿐이다(2026-10-09).
접근 불가인 곳을 사용자가 말하면 그 name 으로 navigate(need_confirm=false, reply="")를 낸다 — 로봇
본체가 갈 수 없다고 알린다. 비슷한 다른 곳으로 바꾸지 않고, 되묻기 후보로도 들지 않는다(2026-10-10
실기: 목록에 없던 비공개 408호를 407호로 바꿔 출발했다).
{dest_block}
{directory_block}
[다른 층] 위 "다른 층 장소"에 있는 곳을 물으면 몇 층인지와 층 이동은 못 한다는 것을 한 문장으로 알리고,
목록에 "엘리베이터"가 있으면 같은 문장에서 그리로 안내를 제안한다(navigate, need_confirm=true, 예: "OO는
N층이라 제가 못 가는데, 엘리베이터로 안내해드릴까요?"). 없으면 "OO는 N층이라 엘리베이터로 가시면 돼요." 한 문장.

[확신이 낮을 때] 안내를 끝내거나 접는 결정(deny·finish·cancel, 확정 navigate)은 되돌리기 어렵다.
들린 말이 짧고 불분명해 confidence 가 0.7 미만이면 그 결정을 내리지 말고 intent=unknown,
reply="" 로 둔다 — 로봇 본체가 마지막 질문을 다시 한다(2026-10-08). "그럴래?"·"그럴까?"·"어어"처럼 부드러운 긍정을 부정으로 오해하지 마라.

[숫자] "테스트1·테스트2·테스트3"처럼 숫자로 갈리는 목적지는 숫자가 핵심이다. "쓰리·스리·삼"=3,
"투·이"=2, "원·일"=1. 숫자가 확실치 않으면 confidence 를 낮추고 clarify 로 몇 번인지 되묻는다.

{HANDLE_FACTS}

[말투] 차분하고 짧게, 존댓말로 필요한 말만 한다. **reply 는 30자 안팎, 가급적 한 문장이고 질문은 하나만
한다** — 소리로만 듣는 사람에게 긴 말은 짐이다(2026-10-09 사용자: 대답이 길고 자유분방하다). 알려 줄 말과
물을 말이 함께 있으면 마침표로 나누지 말고 "~인데,"·"~이라"로 이어 한 문장으로 만든다 — 로봇은 문장마다
목소리를 새로 만들어서 문장과 문장 사이에 빈틈이 생긴다(2026-10-10 사용자). 농담·추측·맞장구("~하겠죠", "~하시군요")와 예시 늘어놓기, 변명·긴 설명은
넣지 않는다. "죄송해요", "제가 헷갈리게 했나요", "이름표나 안내판을 확인해 보세요" 같은 사과·자책·
조언은 쓰지 않는다. 모르면 "그건 제가 몰라요." 한 문장이면 된다. 같은 자리에서는 같은 말로 묻는다.
답을 들어야 하는 말(되묻기·확인 질문)만 "?"로 끝낸다(그래야 로봇이 듣는다 — 접근 질문 중 question 답은 예외,
로봇 본체가 다시 묻는다). 묻는 말에 대한 대답·안내·모른다는 말은 평서문이니 "."로 끝낸다 — 평서문에 "?"를
붙이면 말끝이 올라가 질문처럼 들리고 로봇이 쓸데없이 귀를 연다(2026-10-11: "성별은 추정할 수 없어요?",
"목록에 408호는 없어요?" — 질문 대답 30개 중 18개가 "?"로 끝났다).
reply="" 로 정한 경우(확정 navigate·affirm·deny·wait·finish·확정 cancel)는 로봇 본체가 이어서
말하므로 네가 말을 보태지 않는다.
{state_block}{situation}"""


def parse_intent_audio(
    pcm16_16k: bytes,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    situation: str = "",
    directory_block: str = "",
) -> tuple[VicaIntent, str, float, dict]:
    """발화 소리 -> Realtime(set_intent) -> 모델 결정을 그대로 VicaIntent 로 (모델 전결)."""
    instructions = build_audio_prompt(destinations, robot_state, situation, directory_block)
    result = get_realtime_client().ask(pcm16_16k, history, instructions)
    heard = result.heard_text.strip()
    try:
        draft = _IntentDraft(**result.draft)
    except ValidationError as exc:
        raise ValueError(f"set_intent 인자가 _IntentDraft 와 맞지 않는다: {exc}") from exc
    intent = deliver_audio_draft(draft, destinations, robot_state.dialog_state if robot_state else "",
                                 robot_state.active_destination if robot_state else "")
    return intent, heard, result.latency_sec, {"src": "llm", "usage": result.usage}


def deliver_audio_draft(draft: _IntentDraft, destinations: Sequence[DestinationData],
                        dialog_state: str = "", active: str = "") -> VicaIntent:
    """모델 결정 → VicaIntent. 판단은 하지 않는다: id 매핑·접근 불가·계약상 빈 reply 만."""
    result = VicaIntent(
        intent=draft.intent,
        destination_candidate=draft.destination_candidate,
        confidence=draft.confidence or 0.0,
        reply=(draft.reply or "").strip(),
        need_confirm=bool(draft.need_confirm),
        safety_flag="normal",
    )
    tied = tied_destinations(draft.destination_candidate, list(destinations)) if draft.intent == "navigate" else []
    asked = _choice_in_reply(result.reply, destinations, active) if draft.intent == "navigate" else []
    if len(tied) > 1:
        _ask_which(result, tied, dialog_state)
    elif len(asked) > 1:
        _ask_which(result, asked, dialog_state)
    elif draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
        if matched is None:
            result.intent = "clarify"
            result.need_confirm = False
            result.reply = result.reply or ASK_DESTINATION
        elif not matched.is_approachable:
            result.matched_destination_id = matched.id
            result.reply = matched.unavailable_reason or matched.confirm_prompt
            result.need_confirm = False
        else:
            result.matched_destination_id = matched.id
            if not result.need_confirm:
                result.reply = ""
            elif not result.reply:
                result.reply = matched.confirm_prompt
    elif draft.intent == "wait":
        result.need_confirm = False
        result.reply = ""
        minutes = draft.wait_minutes
        result.wait_minutes = minutes if minutes and minutes > 0 else -1
    elif draft.intent in ("affirm", "deny", "finish", "help"):
        result.need_confirm = False
        result.reply = ""
        result.matched_destination_id = ""
    elif draft.intent == "pause":
        result.need_confirm = False
        result.reply = result.reply or PAUSE_ACK
    elif draft.intent in ("cancel", "resume"):
        if result.need_confirm:
            result.reply = result.reply or (CANCEL_CONFIRM if draft.intent == "cancel" else RESUME_CONFIRM)
        else:
            result.reply = ""
    elif draft.intent == "clarify":
        result.need_confirm = False
        result.reply = result.reply or ASK_DESTINATION
    else:
        result.need_confirm = False
    return result

_APPROACH_STATES = frozenset({"awaiting_user", "turning"})


def _choice_in_reply(reply: str, destinations: Sequence[DestinationData], active: str = "") -> list[DestinationData]:
    """되묻는 말(끝 "?")에 든 목적지들 — 다른 이름 속에 든 짧은 이름은 빼고, 지금 가는 곳(active)도 뺀다."""
    reply = (reply or "").strip()
    if not reply.endswith("?"):
        return []
    named = [d for d in destinations if d.name and d.name in reply]
    outer = [d for d in named if not any(o.name != d.name and d.name in o.name for o in named)]
    return [d for d in outer if d.name != active]


def _named_destination(user_text: str, destinations: Sequence[DestinationData]) -> Optional[DestinationData]:
    """사용자 말에 이름이 딱 하나 든 목적지 — 띄어쓰기는 보지 않고, 다른 이름 속에 든 짧은 이름은 뺀다."""
    def squash(s: str) -> str:
        return "".join((s or "").split())

    said = squash(user_text)
    if not said:
        return None
    named = [d for d in destinations if d.name and squash(d.name) in said]
    outer = [d for d in named if not any(o.name != d.name and squash(d.name) in squash(o.name) for o in named)]
    return outer[0] if len(outer) == 1 else None


def _ask_which(result: VicaIntent, tied: Sequence[DestinationData], dialog_state: str = "") -> VicaIntent:
    """두 곳 이상에 똑같이 걸린 navigate 를 되묻기로 바꾼다 — 한쪽을 고르지 않는다(2026-10-11 사용자 결정)."""
    if dialog_state in _APPROACH_STATES:
        result.destination_candidate = None
        result.matched_destination_id = ""
        result.reply = ""
        return result
    names = [d.name for d in tied]
    keep = result.reply.strip().endswith("?") and all(n in result.reply for n in names)
    result.intent = "clarify"
    result.destination_candidate = None
    result.matched_destination_id = ""
    result.need_confirm = False
    result.reply = result.reply if keep else choose_between(names)
    return result


def _finalize(
    draft: _IntentDraft,
    destinations: Sequence[DestinationData],
    pending: Optional[DestinationData] = None,
    pending_command: Optional[str] = None,
    user_text: str = "",
    local: bool = False,
    robot_state: Optional[RobotState] = None,
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

    if draft.intent in ("affirm", "deny", "finish", "help"):
        result.reply = ""
        result.matched_destination_id = ""
        result.need_confirm = False
        return result

    if draft.intent == "wait":
        result.reply = ""
        result.need_confirm = False
        if local:
            minutes = parse_wait_minutes(user_text, range_factor=1.0)
        else:
            minutes = parse_wait_minutes(user_text)
            if minutes is None and draft.wait_minutes and draft.wait_minutes > 0:
                minutes = draft.wait_minutes
        result.wait_minutes = minutes if minutes else -1
        return result

    named = _named_destination(user_text, destinations) if local and draft.intent == "navigate" else None
    tied = (tied_destinations(draft.destination_candidate, list(destinations))
            if draft.intent == "navigate" and named is None else [])
    if len(tied) > 1:
        _ask_which(result, tied, robot_state.dialog_state if robot_state else "")
    elif draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
        if named is not None and (matched is None or matched.id != named.id):
            matched = named
            result.destination_candidate = named.name
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
            current = _guiding_destination(robot_state)
            if current and current == matched.name:
                result.intent = "question"
                result.destination_candidate = None
                result.matched_destination_id = ""
                result.need_confirm = False
                result.reply = ALREADY_GOING.format(cur=current, cur_josa=_josa_euro(current))
            elif current:
                result.reply = CHANGE_DESTINATION_QUESTION.format(
                    cur=current, cur_josa=_josa_euro(current),
                    new=matched.name, new_josa=_josa_euro(matched.name))

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
