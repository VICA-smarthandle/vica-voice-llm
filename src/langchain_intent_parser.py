"""사용자 발화를 LangChain(Ollama Cloud)으로 분석해 VicaIntent 로 만든다.

설계 원칙 (docs/design.md):
- LLM 은 intent 분류와 '목적지 표현(destination_candidate)' 까지만 채운다.
- 실제 목적지 확정(matched_destination_id)과 안전 판단은 코드가 한다.
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, field_validator

from .destination_loader import _josa_euro
from .destination_matcher import match_destination, tied_destinations
from .mission_phrases import CONFIRM_SWITCH
from .handle_mode import (
    AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply)
from . import answer_field, intent_recut, local_rules
from .ledger_view import FLOOR_LABEL
from .llm_backend import LlmBackendManager, ProbeResult, http_probe, ollama_warm
from .realtime_intent import audio_tools_active, get_realtime_client
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
from .question_record import OpenQuestion
from .schema import BaseIntentType, DestinationData, RobotState, VicaIntent, VicaIntentType
from .reservation_flow import ReserveCall
from .reservation_store import default_facilities
from .tool_calls import RESERVE, SET_INTENT, reserve_call, reserve_tool, set_intent_tool, text_tools_on

load_dotenv()

# LLM 백엔드는 환경변수로 바꿔 낀다 (코드 수정 없이 전환). 기본은 Ollama Cloud 다.
#   기본(ollama 클라우드): OLLAMA_HOST=https://ollama.com     VICA_LLM_MODEL=gemma4:cloud  OLLAMA_API_KEY=...
#   (선택) ollama 로컬:    OLLAMA_HOST=http://localhost:11434  VICA_LLM_MODEL=gemma4:e2b    (키 불필요)
#   (선택) openai:         VICA_LLM_PROVIDER=openai  VICA_OPENAI_MODEL=gpt-5.4-mini  OPENAI_API_KEY=...
#
# 모델 변수는 백엔드별로 분리한다 — VICA_LLM_MODEL 은 ollama 전용이다. 하나를
# 겸용하면 .env 의 ollama 태그(gemma4:cloud)가 openai 로 넘어가 404 가 난다
# (2026-08-15 스모크에서 실제 발생).
#
# openai 경로 도입 근거 (2026-08-15 젯슨 실측, logs/llm_bench.jsonl): gpt-5.4-mini
# 지연 중앙값 0.92초·최대 1.86초로 gemma4:cloud(0.95초, 꼬리 7.95초)보다 꼬리가
# 짧고, 판정 18/18, strict json_schema 구조화 출력 완전 준수, 발화당 ~1.9k/40 토큰.
PROVIDER = os.environ.get("VICA_LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
if PROVIDER == "openai":
    DEFAULT_MODEL = os.environ.get("VICA_OPENAI_MODEL", "gpt-5.4-mini")
else:
    DEFAULT_MODEL = os.environ.get("VICA_LLM_MODEL", "gemma4:cloud")

# ----- 로컬 폴백(2026-09-19 설계) --------------------------------------------
# 비어 있으면 폴백이 꺼진다 = 예전 동작(클라우드 실패 = LLM_UNAVAILABLE).
FALLBACK_MODEL = os.environ.get("VICA_LLM_FALLBACK_MODEL", "").strip()
FALLBACK_HOST = os.environ.get("VICA_LLM_FALLBACK_HOST", "http://localhost:11434")
# 폴백이 켜졌을 때의 클라우드 대기 시간. 8월 실측 최대 1.86초의 3배. 재시도 없음.
CLOUD_TIMEOUT_SEC = float(os.environ.get("VICA_LLM_CLOUD_TIMEOUT", "6"))
CLOUD_PROBE_SEC = float(os.environ.get("VICA_LLM_CLOUD_PROBE_SEC", "30"))
# 로컬 Ollama 호출 상한. 스택 없이 잰 적재 7초+추론 3초의 여유 배수.
LOCAL_TIMEOUT_SEC = float(os.environ.get("VICA_LLM_LOCAL_TIMEOUT", "45"))


class _IntentDraft(BaseModel):
    """LLM 이 직접 채우는 부분만 담은 임시 스키마.

    matched_destination_id / need_confirm / safety_flag 는 코드가 채운다.
    """

    intent: BaseIntentType = Field(
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
        # navigate 의 확인 문구는 코드가 confirm_prompt 로 갈아끼우므로(_finalize),
        # 모델이 문장을 써 봐야 폐기된다. 빈 문자열이면 그만큼 생성 토큰이 줄어
        # 지연이 준다 (Jetson 로컬 22tok/s 실측 기준 ~1초 이상).
        description=(
            "사용자에게 들려줄 한국어 답변. intent 가 navigate 이고 "
            "destination_candidate 를 채웠으면 빈 문자열로 둬라 (확인 문구는 "
            "시스템이 만든다)."
        ),
    )


class _AnswerDraft(_IntentDraft):
    """VICA_ANSWER=on 일 때의 출력 — _IntentDraft 에 대답 칸 하나를 더한다(설계 2026-11-18 answer-interpretation 4.1).

    형은 str 이다: 외부 API(도구 호출·Realtime)는 형식을 강제하지 않아 허용 밖의 값이 올 수 있다 — 검증에서 죽지 않고
    answer_field.fill 이 unclear 로 바꾼다. 로컬은 질문 종류마다 값을 좁힌 스키마를 쓴다(_ollama_structured).
    """

    answer: str = Field(default="", description=answer_field.ANSWER_DESCRIPTION,
                        json_schema_extra={"enum": ["", *answer_field.ANSWERS]})

    @field_validator("answer", mode="before")
    @classmethod
    def _answer_text(cls, value):
        # 도구 호출·Realtime 이 null·숫자를 넣어도 발화 하나가 통째로 실패하지 않게 빈 값으로 읽는다(검토 M2).
        return value if isinstance(value, str) else ""


class _RecutDraft(_IntentDraft):
    """VICA_RECUT=on 일 때의 출력 — 새 의도 값 둘과 clarify 상황 칸 셋을 더한다(원칙 4·9·10·11).

    새 칸은 _AnswerDraft.answer 처럼 형이 다른 값(null·숫자·문자열↔목록)에도 죽지 않고 빈 값으로 읽는다.
    clarify_kind 는 허용 밖의 값도 그대로 받는다 — 뒤처리가 no_destination 으로 읽는다.
    """

    intent: VicaIntentType = Field(
        description=("navigate / question / clarify / unknown / cancel / pause / resume / affirm / deny / wait / finish / help"
                     " / not_for_robot / dismiss 중 하나"))
    clarify_kind: str = Field(
        default="",
        description="intent 가 clarify 일 때만: several(목적지 후보가 둘 이상) / not_available(목록에 없는 곳) / no_destination(목적지를 못 알아들음). 그 외 빈 문자열.",
        json_schema_extra={"enum": ["", *intent_recut.CLARIFY_KINDS]})
    clarify_options: list[str] = Field(default_factory=list, description="clarify_kind 가 several 일 때 목록의 name 을 모두")
    clarify_wanted: str = Field(default="", description="clarify_kind 가 not_available 일 때 사용자가 찾는 것을 말한 낱말 그대로")

    @field_validator("clarify_kind", "clarify_wanted", mode="before")
    @classmethod
    def _recut_text(cls, value):
        return value if isinstance(value, str) else ""

    @field_validator("clarify_options", mode="before")
    @classmethod
    def _recut_options(cls, value):
        return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


class _RecutAnswerDraft(_RecutDraft, _AnswerDraft):
    """두 스위치가 모두 켜졌을 때 — 새 칸과 대답 칸."""


def _base_draft_cls() -> type[_IntentDraft]:
    """로컬(Ollama) 구조화 출력의 바탕 스키마 — 재단 스위치와 상관없이 늘 _IntentDraft(원칙 16, 2단계 측정 P3)."""
    return _IntentDraft


def _draft_cls() -> type[_IntentDraft]:
    """LLM 출력 스키마 — (대답 칸, 재단) 스위치 조합마다 늘 같은 클래스. 둘 다 꺼졌으면 지금의 _IntentDraft."""
    if intent_recut.on():
        return _RecutAnswerDraft if answer_field.on() else _RecutDraft
    return _AnswerDraft if answer_field.on() else _IntentDraft


# 이번 해석 호출의 열린 질문 종류(로컬 스키마 고르기). 호출마다 parse_intent 가 넣고 지운다 — 그림자 비교 스레드·
# 워밍업이 동시에 불러도 섞이지 않게 ContextVar 다.
_ANSWER_KIND: ContextVar[str] = ContextVar("vica_answer_kind", default="")

# 이번 해석 호출을 Ollama 가 답했는가(원칙 16). 로컬 호출체가 불릴 때 켜고 parse_intent 가 호출 앞에 끄고 끝에 되돌린다.
# 모듈 전역이 아니라 ContextVar 인 까닭은 _ANSWER_KIND 와 같다 — 그림자 비교 스레드·워밍업이 동시에 불러도 안 섞인다.
_LOCAL_ANSWERED: ContextVar[bool] = ContextVar("vica_local_answered", default=False)


class _LocalCall:
    """_ollama_structured 가 만든 호출체를 감싼다 — 불리면 이 호출은 Ollama 가 답한 것으로 적는다.

    RunnableLambda 로 감싸지 않는다: langchain 이 컨텍스트를 복사해 돌려 ContextVar 쓰기가 호출부로 안 돌아온다.
    """

    def __init__(self, inner):
        self._inner = inner

    def invoke(self, messages, *args, **kwargs):
        _LOCAL_ANSWERED.set(True)
        return self._inner.invoke(messages, *args, **kwargs)


# 손잡이 사용법 사실 (2026-09-30). 사용자가 "손잡이 어디 있어요?", "손 놓으면
# 어떻게 돼요?", "좀 천천히 가 주세요"라고 물을 때 답할 근거다 — 지시문에 손잡이
# 이야기가 한 줄도 없어 LLM 이 지어내거나 모른다고 했다. 말투는 지시하지 않는다
# (09-20 자유 대사 선호). 수치(0.5초·80 %)는 넣지 않는다 — 바뀔 값이고 사용자에게
# 필요 없다. 두 지시문(글자·소리)이 같은 상수를 쓴다. 정본: 루트 docs/superpowers/
# specs/2026-09-28-touch-haptic-integration-final.md 4.7 (나).
# 2026-10-21 미션의 주행 중 놓침 정지를 껐다(터치 칩이 8초쯤 쥐면 저절로 놓음이 됨) —
# 터치는 처음 잡기만 본다. "놓으면 선다"를 다시 넣지 않는다.
HANDLE_FACTS = """[손잡이] 사용자가 물으면 이 사실로 답한다.
- 로봇 뒤쪽에 손잡이가 있다. 안내를 시작할 때 손잡이가 진동해 자리를 알려 준다.
- 손잡이를 잡으면 안내를 시작한다. 잡지 않고 따라와도 안내한다.
- 걷는 동안 손을 놓아도 로봇은 서지 않는다. 서고 싶으면 손잡이 줄을 끝까지 당기거나
  '잠깐만'이라고 말하면 된다.
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
    local: bool = False, situation: str = "", question: Optional[OpenQuestion] = None,
    recut: Optional[bool] = None, ask_question: bool = True,
) -> str:
    """글자 경로 지시문.

    ask_question: False 면 대답 칸이 켜져 있어도 [지금 상황] 끝에 '로봇이 물은 질문' 줄을 붙이지 않는다(로컬 Ollama 가
    답하는 호출, VICA_ANSWER_LOCAL 이 꺼졌을 때 — 3.5). 기본 True 는 지금 동작 그대로다.

    recut: 재단(원칙 16) 지시문을 쓸지. None 이면 스위치(intent_recut.on())를 따르고, 호출부가 False 로 못 박으면
    스위치가 켜져 있어도 꺼진 글자로 만든다(로컬 Ollama 가 답하는 호출). situation 은 LLM 노드의 [지금 상황] 블록(대장, ledger_view).

    question 은 열린 질문(질문 기록) — 대답 칸이 켜졌으면 맨 끝 [지금 상황]에 한 줄로 붙는다(answer_field.with_question).

    2026-10-07 부터 글자 경로에도 메모를 붙인다(사용자 결정) — 와이파이가 끊겨 글자 경로
    (받아쓰기 → gpt/로컬)로 넘어가도 "어디서 기다린다고?"·"화장실 어디야?"에 답하게.
    대장 블록이 오면 옛 2줄 상태 블록 대신 쓰고, 소리 경로처럼 맨 뒤에 붙인다(매번 바뀌는
    부분이 뒤여야 앞부분 캐시가 산다).
    """
    lines = []
    for d in destinations:
        aliases = ", ".join(d.aliases)
        approach = "가능" if d.is_approachable else "불가"
        lines.append(f"- {d.name} (별칭: {aliases} / 분류: {d.category2} / 접근: {approach})")
    dest_block = "\n".join(lines)
    ledger = situation if FLOOR_LABEL in situation else ""
    state_block = "" if ledger else _format_robot_state(robot_state)
    # 원칙 1·4·12·31: 켰을 때만 새 판단 순서·의도 정의·clarify 칸 이름을 싣는다. 꺼졌으면 아래 변수가 옛 글 그대로다.
    recut = intent_recut.on() if recut is None else recut
    order_block = intent_recut.PROMPT_ORDER + "\n\n" if recut else ""
    clarify_unknown = intent_recut.PROMPT_INTENTS if recut else """- clarify: 어디로 갈지 모호해 되물어야 함. reply 에 되묻는 질문을 담아라.
- unknown: 안내와 무관하거나 이해 불가."""
    not_avail = intent_recut.NOT_AVAILABLE_TAG if recut else ""
    several_rule = intent_recut.SEVERAL_RULE if recut else """clarify 로 그 곳들의 name 을 모두 들어 묻는다("남자 화장실과
  여자 화장실 중 어디로 갈까요?"). 하나만 들어 묻지도 않는다(2026-10-11 실기)."""
    empty_kinds = "·clarify·unknown·not_for_robot·dismiss" if recut else ""
    other_kinds = "question 의 " if recut else "그 외(question/clarify/unknown)의 "
    no_dest = intent_recut.NO_DESTINATION_TAG if recut else ""
    return f"""너는 시각장애인 안내 로봇 'VICA'의 음성 의도 분석기다.
사용자의 한국어 발화를 분석해 아래 규칙으로 분류해라.

{order_block}[intent 종류]
- navigate: 어딘가로 가고 싶어함. 직접 표현("407호 가자")뿐 아니라 간접 표현("배 아파"->화장실, "배고파"->식당)도 포함.
- question: 이동이 아니라 정보 질문("지금 몇 층이야?").
{clarify_unknown}
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

[목적지 목록] (navigate 의 destination_candidate 는 반드시 이 name 중 하나여야 한다. 목록에 없으면 clarify{not_avail})
{dest_block}
{state_block}
[규칙]
- destination_candidate 는 위 목록의 정확한 name 또는 null. 새로 지어내지 마라.
- 들은 말이 목록의 여러 곳에 똑같이 걸리면(같은 별칭 — 예: "화장실"이 남자 화장실·여자 화장실 둘 다)
  navigate 로 하나를 짐작해 고르지 말고 {several_rule}
- navigate(destination_candidate 포함)·cancel·pause·resume·affirm·deny·wait·finish{empty_kinds} 로
  분류하면 reply 는 빈 문자열로 둬라. 확인·수락 발화는 시스템이 만든다.
- {other_kinds}reply 는 짧고 친절한 한국어로, 가급적 한 문장으로 써라. 알려 줄 말과
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
- 대화 단계가 '대기 중'·'대기 장소로 떠나기 직전'일 때 목적지를 말하면 평소처럼 navigate 다(대기는 출발할 때 끝난다).
  "다 됐어"·"다녀왔어"처럼 볼일이 끝났다고만 하면 finish 다(reply 는 빈 문자열 — 로봇이 "어디로
  모실까요?"라고 묻는다).

[멀티턴 대화]
- 직전에 로봇이 'OO로 안내해드릴까요?' 또는 '…OO로 바꿀까요?'라고 물었고 사용자가 긍정(응, 네,
  맞아, 그래, 좋아)하면: intent=navigate, destination_candidate=그 OO 목적지 name, is_confirmation=true
  로 답해라. '지금 XX로 가는 중이에요. OO로 바꿀까요?'의 답이면 OO 다(XX 가 아니다).
- 사용자가 부정(아니, 그거 말고)하며 다른 목적지를 말하면 그 목적지로 navigate.
- 부정만 하고 목적지를 안 말하면 clarify{no_dest}.""" + (local_rules.prompt_rules(recut) if local else "") + (
        answer_field.with_question(intent_recut.with_clue(ledger, question) if recut else ledger,
                                   question if ask_question else None))


# 직전 확인 질문에 대한 짧은 긍정/부정. 긴급어 필터와 같은 원칙으로 LLM 을 거치지
# 않고 코드가 결정한다 — 소형 모델이 is_confirmation 을 놓치면 확인 질문이 무한
# 반복되는 문제가 실기에서 확인됐다 (2026-07-19, exaone3.5:2.4b).
#
# 목록의 정본은 `handle_mode.py` 다. 스마트핸들 모드 질문도 같은 "네/아니요"를
# 받으므로, 두 곳에 따로 두면 한쪽만 고쳐져 어긋난다. 아래 이름은 이 모듈의
# 기존 호출부와 테스트가 쓰던 것이라 별칭으로 남긴다.
_AFFIRMATIVES = AFFIRMATIVES
_NEGATIVES = NEGATIVES
# 받아쓰기가 "아니"를 잘못 적은 꼴 — 대답을 기다리는 단계(ANSWER_WAIT_STATES)에서만 거절로 읽는다. 공용 NEGATIVES 에는
# 넣지 않는다(귀의 짧은 답 구제·점검기도 그 목록을 쓴다). 10-24 12:44:36 Qwen3-ASR 이 "아니"를 말레이어 'hani' → '하니.'.
_ASR_SLIP_NEGATIVES = frozenset({"하니"})
# 혼잣소리('음'·'어어')는 발화 전체가 그것뿐일 때만 긍정으로 본다.
# 첫 단어 지름길에는 넣지 않는다 — "음… 아니야"가 승낙이 되면 안 된다.
_SOLO_AFFIRMATIVES = AFFIRMATIVES | SOFT_AFFIRMATIVES
_normalize_short_reply = normalize_short_reply

# 코드(지름길)가 직접 채운 응답 문구. LLM 이 지어낸 잡담 대꾸와 구별하는
# 유일한 표지다 — ros_node 의 재청취 기각이 이 답은 삼키지 않는다.
# 목록은 지름길이 늘 때만 늘린다 (LLM 생성 문구는 절대 넣지 않는다).
SHORTCUT_REPLIES = frozenset({WAKE_GREETING})


def is_scripted_reply(reply: str, destinations: Sequence[DestinationData]) -> bool:
    """코드가 정한 문구(호출 응답·목적지 확인 질문)인가 — 재청취 기각 예외 판별.

    확인 질문 재질문('음' 단독, 2026-09-28)은 clarify 로 나가는데, 재청취 창의
    무의미 발화 기각이 intent 이름만 보고 삼키면 되묻기가 침묵이 된다.
    LLM 이 지어낸 대꾸는 여기서 거짓이라 종전대로 버려진다.
    """
    if not reply:
        return False
    # "지금 409호로 가는 중이에요."(_finalize 가 만든 사실 한 줄, 2026-10-07)도 코드 문구다 —
    # 로컬 규칙의 '못 알아들음' 대체·재청취 기각이 이것을 지우면 안 된다.
    return reply in SHORTCUT_REPLIES or _is_choice_question(reply, destinations) or any(
        (d.confirm_prompt and d.confirm_prompt == reply)
        or reply == ALREADY_GOING.format(cur=d.name, cur_josa=_josa_euro(d.name))
        for d in destinations)


def _is_choice_question(reply: str, destinations: Sequence[DestinationData]) -> bool:
    """choose_between 이 만든 고르기 질문인가 — 목록에 있는 두 곳 이상을 고정 문장으로 묻는다(2026-10-24).

    재청취 기각이 이것을 지우면 확인 질문에 "아니, 나 배 아파"라고 답해도 침묵이 된다.
    """
    suffix = " 중 어디로 갈까요?"
    if not reply.endswith(suffix):
        return False
    names = [d.name for d in destinations if d.name]
    head = reply[:-len(suffix)]
    if ", " in head:
        parts = head.split(", ")
        return all(p in names for p in parts) and reply == choose_between(parts)
    return any(reply == choose_between([a, b]) for a in names for b in names if a != b)


def is_relisten_noise(intent: VicaIntent, destinations: Sequence[DestinationData], *,
                      in_window: bool, last_robot_text: str) -> bool:
    """재청취 기각(ros_node 2-1) 판정 — 대답 창 안의, 할 말을 들고 오지 않은 판정만 참.

    로봇이 마지막으로 한 말이 호출 응답("네?")이면 대답 창이 아니라 호출 창이다(2026-10-24). 대답 창 안의
    "비카야"는 창 안 호출 구제로 글자 '비카야'만 오고 _followup_until 이 남는다 — 창 밖 호출(/vica/wake 가
    비움)과 같게 다음 요청을 기각하지 않는다. 10-24 음성만 시험: 고르기 질문 3번이 이 자리에서 지워졌다.
    """
    return (in_window
            and (last_robot_text or "").strip() != WAKE_GREETING
            and intent.intent in ("unknown", "clarify")
            and not intent.need_confirm
            and not is_scripted_intent(intent, destinations)
            and intent.safety_flag != "emergency")


def _question_destination(text: str, destinations: Sequence[DestinationData]):
    """이 로봇 말이 어떤 목적지로 가자는 확인 질문이면 그 목적지, 아니면 None.

    확인 문구(confirm_prompt)와 글자가 같거나, 주행 중 바꾸기 질문("지금 409호로 가는
    중이에요. 화장실로 바꿀까요?", 2026-10-07)이면 끝의 '…로 바꿀까요?' 목적지다. 바꾸기
    질문엔 목적지가 둘 들어 있어 앞의 것(지금 가는 곳)을 고르면 "네"가 엉뚱한 곳이 된다.
    """
    text = (text or "").strip()
    if not text:
        return None
    for dest in destinations:
        # 미션이 확인 질문 중 다른 목적지로 다시 묻는 "네, XX로 안내해드릴까요?"(2026-10-08 결정
        # 4)도 그 목적지의 확인 질문이다 — 못 알아보면 "응"이 앞서 물은 곳으로 붙는다.
        if dest.confirm_prompt and text in (
                dest.confirm_prompt, CONFIRM_SWITCH.format(prompt=dest.confirm_prompt)):
            return dest
    # 이름이 다른 이름의 끝과 겹치면("화장실"·"남자 화장실") 긴 쪽이 맞다.
    matches = [
        dest for dest in destinations
        if text.endswith(CHANGE_DESTINATION_TAIL.format(
            new=dest.name, new_josa=_josa_euro(dest.name)))
    ]
    return max(matches, key=lambda d: len(d.name)) if matches else None


def _guiding_destination(robot_state: Optional[RobotState]) -> str:
    """미션이 안내 주행 중이면 가는 곳 이름, 아니면 "" (주행 중 바꾸기 질문용).

    대장(dialog_state)이 'navigating' 일 때만 — 일시정지·확인 질문 중은 바꾸기가 아니다.
    """
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
    """기록에서 **가장 최근의 확인 질문**이 가리키는 목적지 (로컬 규칙 전용, 2026-10-05).

    로컬 규칙은 로봇이 소리 낸 말 전부(미션 말 포함)를 기록에 넣는다. 그러면 확인
    질문 뒤에 미션의 다른 말("지금은 다른 응대 중…")이 끼어 '마지막 로봇 말'이
    확인 질문이 아닐 수 있다. 확인 중인지는 미션의 dialog_state 가 정하고, 여기서는
    어느 목적지인지만 찾는다 — 그래서 마지막 말이 아니라 거슬러 올라가 찾는다.
    """
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
    """'확인 대기 중인 목적지'. 로컬 규칙 + 미션 상태를 알면 미션 상태가 정한다.

    - 로컬 규칙이 꺼졌거나 dialog_state 를 모르면: 예전 그대로(마지막 로봇 말).
    - 로컬 규칙 + confirming: 기록의 가장 최근 확인 질문 목적지.
    - 로컬 규칙 + 그 밖의 단계: 없음 — 확인 중이 아닌데 "그래"가 지나간 확인
      질문에 붙어 출발이 확정되던 문제(10-02 시연 ①, 수리안 가)를 막는다.
    """
    if not local or not dialog_state:
        return _pending_confirm_destination(history, destinations)
    if dialog_state == "confirming":
        return _recent_confirm_destination(history, destinations)
    return None


# 제어 확인 문구 -> intent. 로봇의 직전 발화가 이 중 하나면 "네" 한마디로 확정된다.
# pause 는 여기 없다 — 서는 방향은 되묻지 않고 즉시 요청한다 (replies.PAUSE_ACK).
_COMMAND_CONFIRMS = {
    CANCEL_CONFIRM: "cancel",
    RESUME_CONFIRM: "resume",
}

# LLM 없이 잡는 명백한 취소·일시정지 발화 (normalize_short_reply 적용 후 비교).
# 긴급어 필터와 같은 원칙 — 결정적 명령의 감지는 룰이 빠르고 확실하다.
# 간접 표현("아 됐어, 안 가도 돼")은 LLM 이 분류한다.
_CANCEL_WORDS = {"취소", "취소해줘", "취소해주세요", "취소할래", "안내취소"}
_PAUSE_WORDS = {"잠깐만", "잠깐만요", "잠시만", "잠시만요", "잠깐", "잠시"}
# "몇 분쯤 걸리실까요?"의 대답일 수 있는 꼴 — 시간 질문 단계에서는 일시정지로 읽지 않고 LLM 에 둔다(3-나-1 최종 검토).
_BARE_PAUSE_WORDS = {"잠깐", "잠시"}
# 일시정지 중의 재개 말(3-나-1, 설계 2026-11-18 4.2). 11-18 실기 재생에서 로컬 LLM 은 "다시 가자"류를 0/8 —
# 목록의 다른 곳(407호 등)으로 가자는 말로 읽었다. 낱말은 그 실기와 미션 안내("다시 출발하려면 말씀해
# 주세요")에서 뽑았고 통째 일치만 본다. 받아쓰기가 깨진 꼴("다시 가져")은 넣지 않는다 — LLM 몫으로 둔다.
_RESUME_WORDS = {
    "다시가자", "다시가", "다시가줘", "다시가요", "다시가줘요", "다시가주세요",
    "다시출발", "다시출발해", "다시출발해줘", "다시출발하자", "다시출발해요", "다시출발해줘요", "다시출발해주세요",
    "출발", "출발해", "출발해줘", "출발하자", "출발해요", "출발해줘요", "출발해주세요",
    "계속가자", "계속가", "계속가줘", "계속가요", "계속가줘요", "계속가주세요",
}
# 청취 창 안에서 부른 호출어 — 창이 열린 동안 호출 감지기는 잠들어 있어
# "비카야"가 발화로 전사돼 들어온다. LLM 을 거치면 1초+ 우회이므로 즉시
# "네?"로 받아 밖에서 부른 것과 똑같이 느껴지게 한다 (2026-08-29).
# 오전사 단골('피카야' 실측 2026-08-28)도 함께 받는다.
#
# 2026-09-02 실기에서 나온 변형을 더 넣는다(짧은답 30회 + 거절 10회에서
# 관측). '비켜야'는 실재하는 한국어("비켜야 해")지만, 청취 창 안에서 그
# 한마디만 오는 경우는 사실상 호출이라 받는다 — 대가는 "네?" 한 번이다.
# '비кая'는 whisper 다국어 모델이 키릴 문자를 섞어 적은 것이다.
#
# 어휘 추가는 대증요법이다. 변형은 무한하고 근본 원인은 따로 있다 —
# **창이 열린 동안 호출 감지기(모델 A)가 잠들어 있어** 창 안의 "비카야"가
# 오직 STT 전사로만 판별된다(_step 의 listen 분기: 긴급 모델 B 만 검사).
_WAKE_WORDS = {"비카야", "피카야", "비까야",
               "미카야", "리카야", "비켜야", "비кая"}
# 한국어 수 파싱용 (5분, 이십 분, 십오 분, 반시간, 한 시간).
_SINO_UNITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
               "육": 6, "칠": 7, "팔": 8, "구": 9}
_NATIVE_NUM = {"한": 1, "두": 2, "세": 3, "네": 4}   # "한 시간" 등 고유어 수


def _sino_number(token: str):
    """한자어 수사 -> 값. "십오"=15, "이십"=20 같은 합성도 푼다. 실패면 None.

    끝 글자만 보던 옛 방식은 "십오 분"을 5분으로 오파싱했다(잠재 결함,
    2026-08-30 발견). 십의 자리와 일의 자리를 분리해 계산한다.
    """
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
    """한국어 시간 표현에서 분(minute)을 뽑는다. 없으면 None.

    산수는 전부 여기서 한다 — LLM 에게 계산을 맡겼더니 범위("십 분에서
    십오 분")에 평균(13분)을 냈다(2026-08-30 실기). 규칙:
    - 단일 값("20분", "십오 분", "한 시간")은 그대로.
    - 범위(숫자 2개 이상 또는 "에서"·"~")는 넉넉한 쪽(최댓값) x1.5 반올림
      (사용자 결정: "5분에서 10분" -> 15분 — 여유를 주는 게 센스다).
    - 상한(30분)은 여기서 걸지 않는다 — 판정 권한은 Mission 에 있다.
    - range_factor: 범위에 곱하는 수. 로컬 규칙은 1.0 을 넘겨 OpenAI 지시문과
      같은 "범위면 큰 쪽"으로 맞춘다(2026-10-05). 기본값 1.5 는 그대로다.
    """
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
    """LLM 없이 0초에 판정되는 짧은 말인가.

    이런 말에는 접수 신호("확인할게요")가 군더더기다 — 진짜 답이 바로
    뒤따르기 때문 (2026-08-28 사용자 결정). 지름길 단어 목록을 그대로
    재사용한다 — 목록이 갈라지면 신호 생략과 실제 지름길이 어긋난다.
    """
    word = _normalize_short_reply(user_text)
    return bool(word) and (
        word in _SOLO_AFFIRMATIVES or word in _NEGATIVES
        or word in _CANCEL_WORDS or word in _PAUSE_WORDS
        or word in _WAKE_WORDS)

# 제어가 확정됐을 때의 reply. 실행이 아니라 요청이다 — MissionCommand 서비스
# 호출은 ROS 노드, 수락/거절 판정은 Mission Manager 몫이며 이 문구는 로그용이다.


def _pending_command(history: Optional[list[BaseMessage]]) -> Optional[str]:
    """직전 AI 발화가 제어 확인 질문이었으면 해당 intent 를 돌려준다."""
    if not history:
        return None
    last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return None
    return _COMMAND_CONFIRMS.get(last_ai.content)


def _get_structured_llm(model: str, *, timeout: float = 15, max_retries: int = 1):
    """구조화 출력(_IntentDraft) LLM 을 만든다. 백엔드는 PROVIDER 가 정한다.

    timeout 은 openai·ollama 두 경로 모두에 적용한다. max_retries 는 openai 에만
    쓴다(ollama 경로는 재시도 개념이 없다). 폴백이 켜지면 관리자가
    (CLOUD_TIMEOUT_SEC, 0) 을 넘긴다 — 로컬이 받아 주므로 오래 기다릴 이유가 없다.
    로봇 대화에서 무한 대기는 곧 침묵이다. 기본값(15, 1)은 실측 꼬리(1.86초)의
    여유 배수에서 끊고, 실패는 parse_intent 의 LLM_UNAVAILABLE 폴백이 받는다.
    """
    if PROVIDER == "openai":
        # 지연 import — ollama 만 쓰는 환경에 openai 패키지를 요구하지 않는다.
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
        )
        # strict=True: 스키마를 서버가 문법 수준에서 강제한다 (실측 준수 18/18).
        return llm.with_structured_output(_draft_cls(), method="json_schema", strict=True)

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
    if local_rules.enabled():
        # 로컬 규칙: 문맥 길이를 Ollama 기본값에 맡기지 않고 명시한다(2026-10-05).
        kwargs["num_ctx"] = local_rules.LOCAL_NUM_CTX
    client_kwargs: dict = {"timeout": timeout}
    if api_key:  # 클라우드는 인증 헤더 필요, 로컬 Ollama 는 불필요
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    kwargs["client_kwargs"] = client_kwargs
    llm = ChatOllama(**kwargs)
    return _ollama_structured(llm)


def _ollama_structured(llm):
    """로컬(Ollama) 구조화 출력. 대답 칸이 꺼졌거나 로컬 대답 칸(VICA_ANSWER_LOCAL)이 꺼졌으면 지금 그대로 _IntentDraft 하나다.

    둘 다 켜졌으면 호출 때의 열린 질문 종류(_ANSWER_KIND)로 answer 값을 좁힌 스키마를 고른다 — 구조화 출력이라 그 값만
    나온다(설계 2026-11-18 4.1). 열린 질문이 없으면 칸 없는 _IntentDraft 그대로라 토큰이 늘지 않는다. 종류마다 한 번 만든다.
    """
    if not answer_field.local_on():   # 3.5 — 대답 칸이 켜져도 VICA_ANSWER_LOCAL 이 꺼졌으면 칸 없는 스키마(규칙 대답만 채운다)
        return _LocalCall(llm.with_structured_output(_base_draft_cls()))
    by_kind: dict = {}

    def invoke(messages):
        kind = _ANSWER_KIND.get()
        if kind not in by_kind:
            by_kind[kind] = llm.with_structured_output(answer_field.narrowed(_base_draft_cls(), kind))
        return by_kind[kind].invoke(messages)

    return _LocalCall(RunnableLambda(invoke))


def _set_intent_tool() -> dict:
    """set_intent 도구. 인자는 지금 구조화 출력(_IntentDraft)의 칸과 설명 그대로다."""
    return set_intent_tool(_draft_cls().model_json_schema())


def _draft_from_tool_message(msg) -> "_IntentDraft | ReserveCall":
    """도구 호출 응답 → 해석 결과. 첫 호출만 본다(설계 2026-11-18 5.1).

    호출이 없거나, 목록에 없는 도구 이름이거나, set_intent 인자가 _IntentDraft 와 안 맞으면 ValueError —
    지금의 LLM 호출 실패와 같은 길(로컬 대피 또는 LLM_UNAVAILABLE)로 간다.
    """
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        raise ValueError("도구 호출이 없다")
    name, args = calls[0].get("name"), calls[0].get("args") or {}
    if name == RESERVE:
        return reserve_call(args)
    if name == SET_INTENT:
        try:
            return _draft_cls().model_validate(args)
        except ValidationError as exc:
            raise ValueError(f"set_intent 인자가 맞지 않다: {exc}") from exc
    raise ValueError(f"목록에 없는 도구: {name}")


def _get_tool_llm(model: str, *, timeout: float = 15, max_retries: int = 1):
    """도구 출력 LLM(openai 전용). 발화마다 set_intent 나 reserve_facility 하나를 부르게 한다.

    2026-11-18 측정(~/vica_data/tool_call_test_20261118)에 쓴 방식 그대로다: bind_tools + tool_choice="required",
    32개 중 32개. 로컬 모델에는 도구를 주지 않는다(같은 측정 16/32, 사용자 결정 D11).
    """
    from langchain_openai import ChatOpenAI

    names = [f.name for f in default_facilities()]
    if not names:
        # 시설 파일이 깨졌거나 비었다 — 빈 enum 은 API 가 요청을 통째로 거절한다.
        raise ValueError("예약할 시설 목록이 비어 있다(facilities.yaml)")
    llm = ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=max_retries)
    tools = [_set_intent_tool(), reserve_tool(names)]
    # 발화마다 도구 하나 — 여러 개를 한꺼번에 부르지 못하게 한다.
    bound = llm.bind_tools(tools, tool_choice="required", parallel_tool_calls=False)

    def invoke(messages):
        answer = bound.invoke(messages)          # 연결 실패·시간 초과는 그대로 올린다(전환 담당 모듈이 로컬로 받는다)
        try:
            return _draft_from_tool_message(answer)
        except ValueError as exc:
            # 형식 오류(도구 없음·모르는 이름·안 맞는 인자)는 연결 실패가 아니다. 구조화 출력(strict)과 달리
            # 도구 호출은 서버가 형식을 강제하지 않는다 — 한 번의 형식 오류로 로컬 대피(최소 30초)까지 가지 않게
            # 같은 발화를 한 번만 다시 부른다. 또 틀리면 실패로 올린다.
            _log("warning", f"[LLM] 도구 응답 형식 오류({exc}) — 한 번 다시 부른다")
            return _draft_from_tool_message(bound.invoke(messages))

    return RunnableLambda(invoke)


def text_tools_active() -> bool:
    """글자 경로의 클라우드 호출이 도구 출력을 쓰는가 — openai 이고 VICA_TOOLS_TEXT=on 일 때만."""
    return PROVIDER == "openai" and text_tools_on()


def _get_cloud_llm(model: str, *, timeout: float = 15, max_retries: int = 1):
    """클라우드 쪽 LLM 을 고른다 — 도구 출력(text_tools_active) 또는 지금의 구조화 출력.

    도구 출력을 못 만들면(시설 파일이 깨짐 등) 클라우드를 잃지 않게 구조화 출력으로 돈다 — 예약만 안 된다.
    """
    if text_tools_active():
        try:
            return _get_tool_llm(model, timeout=timeout, max_retries=max_retries)
        except Exception as exc:
            _log("error", f"[LLM] 도구 출력 준비 실패({exc}) — 구조화 출력으로 돈다(예약 안 됨)")
    return _get_structured_llm(model, timeout=timeout, max_retries=max_retries)


def _build_local_structured():
    """로컬 Ollama 구조화 LLM. 09-19 실측: gemma4-e2b-text 8/8, 발화당 3.0초.

    client_kwargs 의 timeout 이 없으면 ollama 클라이언트 기본값(None)이 적용돼
    Ollama 가 멈춰도 무한정 기다린다 — LOCAL_TIMEOUT_SEC 으로 상한을 둔다.
    """
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
    return _ollama_structured(llm)


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
    # 로컬을 먼저 만든다 — 로컬 빌드가 실패해도(모델 없음 등) 클라우드는 반드시
    # 살려야 한다("클라우드는 되는데 로컬만 실패"에서 아예 못 쓰게 되면 폴백
    # 설계 취지(가용성)와 어긋난다).
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
        cloud_llm = _get_cloud_llm(
            DEFAULT_MODEL,
            timeout=CLOUD_TIMEOUT_SEC if fallback_on else 15,
            max_retries=0 if fallback_on else 1,
        )
        cloud = cloud_llm.invoke
    except Exception as build_error:  # 키 없음 등 — 호출 때 실패로 드러나 로컬로 대피한다
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
    """LLM 없이 코드가 확정하는 경우. 없으면 None.

    parse_intent 의 지름길(제어 확인 대기 -> 목적지 확인 대기 긍/부정 -> 호출어 ->
    취소 -> 잠깐 -> 단독 긍/부정)을 그대로 옮긴 것 — parse_intent_audio 도
    들린 말(heard_text)에 이 함수를 그대로 써서 텍스트 경로와 같은 지름길을 쓴다.
    """
    # 직전 확인 질문에 대한 짧은 긍정/부정은 LLM 없이 코드가 결정한다 (아래 참고).
    pending_command = _pending_command(history)
    if pending_command is not None:
        word = _normalize_short_reply(user_text)
        if word in _AFFIRMATIVES:
            # reply 는 침묵 — 결과 발화("다시 출발합니다"·"안내를 취소했습니다")는
            # 미션 몫이다. 예전 "주행 제어를 요청합니다"는 2단 발화로 어색했다.
            return VicaIntent(
                intent=pending_command,
                confidence=1.0,
                reply="",
                need_confirm=False,
            )
        if word in _NEGATIVES:
            if pending_command == "cancel" or (pending_command == "resume" and answer_field.on()):
                # "안내를 취소할까요?"에 아니요 — 미션이 받고 "안내를 계속하겠습니다."라고
                # 답한다(2026-10-08 반응표). 여기서 따로 말하면 두 목소리다.
                # 대답 칸이 켜졌으면 "다시 출발할까요?"도 미션이 물은 질문이다(Task 3.4) — 미션이 받고
                # "알겠습니다. 다시 출발하려면 말씀해 주세요."라고 답한다(원칙 9).
                return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
            return VicaIntent(
                intent="unknown",
                confidence=1.0,
                reply=COMMAND_DECLINED,
                need_confirm=False,
            )

    # 로컬 규칙 + 미션이 확인 중(confirming): 확인 질문의 짧은 대답을 여기서
    # 끝낸다. 기록의 마지막 말이 확인 질문이 아니어도(미션 말이 끼어도) 확인
    # 중이라는 사실은 dialog_state 가 안다 — 이때 "음"이 아래 일반 경로로 새서
    # affirm 이 되면 미션은 곧바로 출발한다(CONFIRMING + affirm = 확정, 08-31).
    # 09-28 "음" 되묻기의 의도("음"으로는 절대 출발 안 함)를 지키는 자리다.
    if local and dialog_state == "confirming":
        confirming = _confirming_shortcut(user_text, history, destinations)
        if confirming is not None:
            return confirming

    pending = _pending_for(history, destinations, local, dialog_state)
    if pending is not None:
        word = _normalize_short_reply(user_text)
        # 첫 단어 기준 판정 (2026-09-01): "응 화장실로 가자"처럼 긍정어 뒤에
        # 말이 붙으면 전체 정규화("응화장실로가자")로는 지름길에 안 걸려
        # LLM 으로 갔고, LLM 이 재제안(need_confirm=True)으로 되돌려 같은
        # 확인 질문이 반복됐다(8/31 야간 실기 — 3수째에야 출발). 첫 단어가
        # 긍정/부정이면 그 자리에서 결정한다. 첫 단어 전체 일치라 "어디로
        # 가?"("어" 접두)류 오폭은 없다.
        tokens = user_text.split()
        first = _normalize_short_reply(tokens[0]) if tokens else ""
        # 정정 방어 (2026-09-02 실기): 사람은 말하다 마음을 바꾼다.
        # "네. 아니. 아니. 아니." 를 첫 단어만 보고 승낙으로 읽어 **거절한
        # 사용자를 태우고 출발했다.** 부정어가 한 토막이라도 섞여 있으면
        # 승낙 지름길을 쓰지 않는다 — 애매하면 출발하지 않는 쪽이 안전하다
        # (fail-closed). 판정은 토막 전체 일치라 "화장실 아니고 안내소"의
        # '아니고' 같은 말은 걸리지 않는다.
        denied = (word in _NEGATIVES
                  or any(_normalize_short_reply(t) in _NEGATIVES for t in tokens))
        # 혼잣소리 단독('음'·'어어')은 출발 승낙으로 쓰지 않고 같은 확인
        # 질문을 한 번 더 한다 (2026-09-28 사용자 결정 A안). whisper 가 적은
        # "응"일 수도, 망설임일 수도 있어 글자로는 못 가른다 — 마이크 실기에서
        # 혼잣소리 "음."이 안내소 출발을 확정했다. 모드 질문·짧은 답 구제 등
        # 움직이지 않는 자리의 '음' 승낙(2026-09-02)은 그대로 둔다.
        #
        # navigate 로 보내면 안 된다 — Mission 은 확인 중인 같은 목적지의
        # navigate(need_confirm=True)를 '다시 말한 승낙'으로 보고 출발한다
        # (mission_logic.on_intent, 2026-09-01). clarify 는 Mission 이 무시해
        # 확인 대기가 그대로 살고, 질문만 한 번 더 나간다.
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
                # navigate + need_confirm=False 는 Mission Manager 가 말한다.
                # 이 reply 는 로그·기록용이며 TTS 로 나가지 않는다 (tts_queue).
                reply=f"{pending.name} 안내를 시작합니다.",
                need_confirm=False,
                safety_flag="normal",
            )
        if denied:
            # deny 로 보낸다 (2026-09-02). 종전의 clarify 는 두 곳에서 막혔다:
            # ① 미션의 on_intent 는 navigate 가 아니면 아무것도 하지 않아
            #    CONFIRMING 이 타임아웃까지 남았고 ② ros_node 의 재청취 기각이
            #    clarify 를 잡담으로 보고 삼켜 거절이 통째로 증발했다(9/1~9/2
            #    실기 3/3 + 3회). 미션에는 이미 정식 통로가 있다 —
            #    on_confirm_answer(False) 가 CONFIRMING 을 접고 말한다.
            # reply 는 비운다 — 계약(VicaIntent.msg affirm/deny 절): 발화는
            # 상태를 아는 미션 몫이라 여기서 채우면 두 번 말한다.
            return VicaIntent(
                intent="deny",
                confidence=1.0,
                reply="",
                need_confirm=False,
            )

    # 명백한 취소·일시정지 발화는 LLM 없이 직행한다 (0초, 오판 없음).
    # 목적지·제어 확인 대기 중이면 위 블록들이 먼저 처리하므로 여기 오지 않는다.
    word = _normalize_short_reply(user_text)
    if word in _WAKE_WORDS:
        # 창 안에서 부른 호출어 — "네?"는 ?로 끝나 재청취 창이 다시 열리고,
        # 녹음 캐시(wake_greeting.wav)가 있어 0초에 재생된다.
        # intent 는 unknown 이지만 **할 말을 들고 있다** — ros_node 의 재청취
        # 기각이 이 답까지 삼켜 '피카야'가 무응답이 됐다(9/2 실기). 그쪽은
        # SHORTCUT_REPLIES 로 가려낸다.
        return VicaIntent(intent="unknown", reply=WAKE_GREETING,
                          need_confirm=False, confidence=1.0)
    if word in _CANCEL_WORDS:
        return VicaIntent(intent="cancel", confidence=1.0, reply=CANCEL_CONFIRM, need_confirm=True)
    if word in _PAUSE_WORDS and not (dialog_state == "asking_wait_time" and word in _BARE_PAUSE_WORDS):
        return VicaIntent(intent="pause", confidence=1.0, reply=PAUSE_ACK, need_confirm=False)
    if dialog_state == "paused" and word in _RESUME_WORDS:
        # 손 놓침 정지(paused_handle)는 여기 오지 않는다 — 그 재개는 손잡이를 다시 잡는 것이다. 출발은 묻고 한다:
        # "다시 출발할까요?"가 이미 열려 있을 때만 승낙으로 읽는다(_finalize 의 pending_command 와 같은 동작).
        if pending_command == "resume":
            return VicaIntent(intent="resume", confidence=1.0, reply="", need_confirm=False)
        return VicaIntent(intent="resume", confidence=1.0, reply=RESUME_CONFIRM, need_confirm=True)

    # 도착 후 대기·종료(wait/finish)는 지름길을 두지 않는다 — 발화 의도가
    # 맥락에 좌우돼("그만 좀 물어봐" ≠ 종료) 단어 매칭이 위험하다. LLM 이
    # 판단하고, 시간 숫자만 코드가 원문에서 뽑는다 (2026-08-30 사용자 결정).

    # 확인 대기가 없는 짧은 긍/부정은 affirm/deny 로 발행한다 (0초, LLM 없이).
    # 어느 질문의 답인지는 판정하지 않는다 — 상태를 가진 Mission 이 소비하거나
    # 무시한다 (계약: VicaIntent.msg affirm/deny 절, "아무 때나 보내도 안전").
    # reply 는 빈 문자열 — 수락/거절 발화는 Mission 몫이라 채우면 두 번 말한다.
    # '취소'는 NEGATIVES 에도 있으나 위 _CANCEL_WORDS 직행이 먼저 잡는다.
    # 로컬 규칙: 미션이 대답을 기다리는 단계일 때만 즉시 처리하고, 아니면 LLM 이
    # 앞뒤 대화를 보고 판단한다(2026-10-05 인수인계 "짧은 대답").
    if local and not local_rules.short_answer_allowed(dialog_state):
        return None
    if word in _SOLO_AFFIRMATIVES:
        return VicaIntent(intent="affirm", confidence=1.0, reply="", need_confirm=False)
    if word in _NEGATIVES:
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
    return None


def _confirming_shortcut(user_text: str, history: Optional[list[BaseMessage]],
                         destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """로컬 규칙 + 미션 confirming 일 때의 짧은 대답. 짧은 대답이 아니면 None(LLM 몫).

    - 부정어가 한 토막이라도 있으면 deny (09-02 정정 방어와 같다).
    - '음'·'어어' 단독은 출발 대신 같은 확인 질문을 한 번 더 (09-28 A안).
      목적지를 기록에서 못 찾으면 고정 문구 RETRY_PROMPT 로 되묻는다 — 어느 쪽이든
      Mission 이 무시하는 clarify 라 출발하지 않는다(안전한 쪽으로 실패).
    - 진짜 긍정(단독 또는 첫 단어): 목적지를 알면 그 목적지로 확정 navigate,
      모르면 affirm — Mission 은 CONFIRMING 에서 affirm 을 자기가 아는 목적지의
      확정으로 받는다.
    """
    word = _normalize_short_reply(user_text)
    tokens = user_text.split()
    first = _normalize_short_reply(tokens[0]) if tokens else ""
    # 확인 중(confirming)은 대답을 기다리는 단계라 받아쓰기 오기 '하니'도 거절이다(2026-10-24) — 출발하지 않는 쪽.
    no_words = _NEGATIVES | _ASR_SLIP_NEGATIVES
    denied = (word in no_words
              or any(_normalize_short_reply(t) in no_words for t in tokens))
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
    """재청취 기각 예외 판별 — is_scripted_reply 에 로컬 규칙의 되묻기 하나를 더한다.

    로컬 규칙의 '음' 되묻기는 목적지를 못 찾으면 RETRY_PROMPT 로 나간다. 같은 문구를
    _finalize 가 빈 대답 메움에도 쓰므로, 지름길이 만든 것(clarify + 확신 1.0)만 통과시킨다.
    """
    if is_scripted_reply(intent.reply, destinations):
        return True
    return (local_rules.enabled() and intent.intent == "clarify"
            and intent.reply == RETRY_PROMPT and intent.confidence == 1.0)


# 대기를 권하는 질문("…여기서 기다릴까요?"·"여기서 대기할까요?")에 '필요 없다'는 대답 — 정규화(띄어쓰기·끝 구두점 뺌)한 글자.
# '괜찮아'는 넣지 않는다 — 권유에 "괜찮아"는 승낙으로도 거절로도 쓴다.
_NO_NEED_TO_WAIT = frozenset({"필요없어", "필요없어요", "필요없습니다", "안기다려도돼", "안기다려도돼요", "안기다려도되요",
                              "안해도돼", "안해도돼요", "됐어", "됐어요"})
_WAIT_OFFER_ENDS = ("기다릴까요?", "대기할까요?")


def _mission_question_answer(user_text: str, history: Optional[list[BaseMessage]],
                             robot_state: Optional[RobotState], local: bool) -> Optional[VicaIntent]:
    """미션이 방금 물은 질문의 답을 단어로 바로 읽는다(2026-10-24 사용자 "시연 대사 고정은 별로 — 말이 조금만 달라도
    이상해진다"). LLM 앞에서 — 10-24 재생에서 midm2-mini 가 "한 십 분 정도"·"삼십 분"을 목적지로, "필요 없어"를 취소로 봤다.

    - 대기 시간 질문(asking_wait_time): 말에서 분을 읽을 수 있으면 wait. 못 읽으면 LLM.
    - 대기 권유(asking_next, 로봇의 마지막 말이 '기다릴까요?'·'대기할까요?')에 '필요 없어'류 → deny.
      "여기까지 안내를 마칠까요?"에는 뜻이 갈려 LLM 에 맡긴다.
    - 대답을 기다리는 단계(ANSWER_WAIT_STATES)의 받아쓰기 오기 '하니'(= "아니") → deny. 로컬 규칙이 꺼진 ⑫ 예비
      경로는 지름길에 대화 상태를 넘기지 않아 여기서 본다.
    """
    state = robot_state.dialog_state if robot_state else ""
    if state in local_rules.ANSWER_WAIT_STATES and _normalize_short_reply(user_text) in _ASR_SLIP_NEGATIVES:
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
    if state == "asking_wait_time":
        # 범위 처리는 _finalize 의 wait 와 같다(로컬 규칙이면 큰 쪽 그대로, 아니면 ×1.5)
        minutes = parse_wait_minutes(user_text, range_factor=1.0) if local else parse_wait_minutes(user_text)
        if minutes:
            return VicaIntent(intent="wait", wait_minutes=minutes, confidence=1.0, reply="", need_confirm=False)
    if state == "asking_next" and _normalize_short_reply(user_text) in _NO_NEED_TO_WAIT:
        last = next((m.content for m in reversed(history or []) if isinstance(m, AIMessage)), "")
        if str(last).strip().endswith(_WAIT_OFFER_ENDS):
            return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False)
    return None


def parse_intent(
    user_text: str,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    model: Optional[str] = None,
    situation: str = "",
    question: Optional[OpenQuestion] = None,
) -> "VicaIntent | ReserveCall":
    """발화를 분석해 VicaIntent 를 돌려준다. (멀티턴: history, 현재 상태: robot_state)

    situation 은 LLM 노드의 [지금 상황] 블록(2026-10-07 글자 경로에도 붙임).
    도구 출력이 켜져 있고(VICA_TOOLS_TEXT=on, openai) LLM 이 reserve_facility 를 부르면 ReserveCall 을 돌려준다.
    question 은 이 말이 들어올 때 열려 있던 질문(질문 기록). 대답 칸이 켜졌으면(VICA_ANSWER=on) 돌려주는 의도에
    question_id·answer·answer_value 가 채워진다 — 이미 있는 짧은 대답 규칙이 서면 그것, 그 밖은 LLM 이 낸 값이다(설계
    2026-11-18 4.3). 의도 칸은 지금 길로 정하고 규칙은 그 읽기를 뒤집지 않는다(분 대답은 wait, 네/아니요로 보이는 not_for_robot 은
    unknown) — 나가는 곳마다 같은 answer_field.fill 을 지난다(Task 3.5b, 원칙 6·7). 예약 호출(ReserveCall)은 그대로 돌려준다.
    """
    local = local_rules.enabled()
    dialog_state = (robot_state.dialog_state if (local and robot_state) else "") or ""
    ruled = answer_field.short_answer(user_text, question, local)
    shortcut = _shortcut_intent(user_text, history, destinations,
                                dialog_state=dialog_state, local=local)
    if shortcut is not None:
        return answer_field.fill(shortcut, question, ruled)
    answer = _mission_question_answer(user_text, history, robot_state, local)
    if answer is not None:
        return answer_field.fill(answer, question, ruled)
    pending_command = _pending_command(history)
    pending = _pending_for(history, destinations, local, dialog_state)

    def build_messages(recut: Optional[bool], ask: bool = True) -> list[BaseMessage]:
        msgs: list[BaseMessage] = [SystemMessage(_build_system_prompt(
            destinations, robot_state, local=local, situation=situation, question=question, recut=recut,
            ask_question=ask))]
        if history:
            msgs.extend(history)
        msgs.append(HumanMessage(user_text))
        return msgs

    # 원칙 16: 재단이 켜져도 Ollama 가 답하는 호출은 꺼진 지시문·스키마·뒤처리를 쓴다. 공급자가 ollama 면 처음부터 꺼진
    # 쪽 messages 를 넘기고, openai 면 재단 켠 messages 를 cloud 에·꺼진 것을 local_messages(쓸 때만 만듦)로 넘긴다.
    # 3.5: 대답 칸이 켜져도 VICA_ANSWER_LOCAL 이 꺼졌으면 Ollama 호출은 질문 줄도 대답 칸도 없는 꺼진 글자를 쓴다 — 같은 구조.
    recut_on = intent_recut.on()
    ask_local = answer_field.local_on() or not answer_field.on()   # 로컬 호출에 질문 줄을 줄지(꺼진 상태에선 어차피 줄이 없다)
    plain_local = recut_on or not ask_local
    messages = build_messages(False, ask_local) if plain_local and PROVIDER != "openai" else build_messages(None)
    local_messages = (lambda: build_messages(False, ask_local)) if plain_local and PROVIDER == "openai" else None

    kind_token = _ANSWER_KIND.set(question.kind if question is not None and answer_field.on() else "")
    answered_token = _LOCAL_ANSWERED.set(False)
    try:
        if model is not None:
            # 벤치·수동 지정 경로(scripts/bench_models.py): 그 모델 하나로 직접 부른다.
            draft: _IntentDraft | ReserveCall = _get_cloud_llm(model).invoke(messages)
        else:
            manager = get_backend_manager()
            draft = (manager.invoke(messages) if local_messages is None
                     else manager.invoke(messages, local_messages=local_messages))
    except Exception as exc:
        # 클라우드→로컬까지 실패했거나 폴백이 꺼진 상태의 실패.
        # 크래시 대신 안전한 fallback 응답을 돌려준다.
        # (긴급어는 LLM 이전 단계에서 처리되므로 이 실패의 영향을 받지 않는다.)
        if model is not None:
            _log("error", f"[LLM] 호출 실패(모델 {model}): {exc}")
        elif not get_backend_manager().has_local:
            _log("error", f"[LLM] 호출 실패(클라우드만, 폴백 꺼짐): {exc}")
        else:
            _log("error", f"[LLM] 호출 실패(로컬까지): {exc}")
        return answer_field.fill(VicaIntent(
            intent="unknown",
            reply=LLM_UNAVAILABLE,
            confidence=0.0,
            need_confirm=False,
        ), question, ruled)
    finally:
        _ANSWER_KIND.reset(kind_token)
    # 이번 호출을 Ollama 가 답했는지(원칙 16) — 뒤처리가 끝나면 되돌린다.
    local_answered = _LOCAL_ANSWERED.get()
    _LOCAL_ANSWERED.reset(answered_token)
    if isinstance(draft, ReserveCall):
        # 예약 요청·대답 — 값 검사와 묻는 문장은 reservation_flow 가 한다(_finalize 는 의도 전용).
        return draft
    # Ollama 가 답했으면 재단 뒤처리(early·settle)를 건너뛴다 — 스위치를 끈 것과 같은 길이다.
    finalize = _finalize_core if local_answered else _finalize
    result = finalize(draft, destinations, pending=pending,
                      pending_command=pending_command, user_text=user_text, local=local,
                      robot_state=robot_state)
    return answer_field.fill(result, question, ruled, getattr(draft, "answer", "") or "")


# ---------------------------------------------------------------------------
# 소리 모드 = 모델 전결 (feat/realtime-intent, 2026-09-20 사용자 결정 "아주 실험적으로")
#
# 소리 입력에서는 판단을 전부 모델(Realtime)에 맡기고 코드는 배달만 한다. 텍스트
# 경로의 지름길(_shortcut_intent)·확인 대기 판정·_finalize 는 여기서 쓰지 않는다 —
# whisper 시절의 울타리(글자를 못 믿어 코드가 모델을 덮어쓰던 규칙)가 정확히 들은
# 말을 버렸다("아니 테스트3로 가자" → 부정어 규칙으로 취소, 09-20 실기 #19).
#
# 코드가 남긴 일 3가지 (판단이 아니라 사실·계약):
#   ① 목적지 이름 → id 매핑 (미션은 id 로만 움직인다)
#   ② 접근 불가 목적지는 기존과 같이 사유 문구로 답한다 (물리적 사실)
#   ③ 메시지 계약상 빈 reply — affirm/deny/wait/finish·확정 navigate·확정 cancel 은
#      상태를 아는 미션이 말한다 (VicaIntent.msg). 모델이 채워도 비운다.
# 나머지(확인 여부·정정·시간·침묵·되묻기)는 프롬프트의 규칙이고 모델이 정한다.
# 로봇이 실제로 소리 낸 말(미션의 질문 포함)은 LLM 노드가 /vica/tts_done 으로
# 이력에 넣어 준다 — 모델이 "몇 분쯤 걸리실까요?"의 답을 답으로 볼 수 있게.
# ---------------------------------------------------------------------------


# 소리 지시문의 예약 부분(2026-11-19, 사용자 승인) — VICA_TOOLS_AUDIO=on 이고 세션에 reserve_facility 가 있을 때만
# 들어간다. 꺼져 있으면 지시문은 글자 하나까지 예전 그대로다. 원인 실험(~/vica_data/tool_call_test_20261118/README.md):
# 이 규칙이 없으면 "아니 세 시로 해 줘"·"모레로 바꿔 줘"를 목적지 정정으로 읽었다(0/4 → 4/4).
# 예시 문장은 측정 묶음(tools/reserve_cases.py)과 겹치지 않게 골랐다.
AUDIO_FIRST_LINE = "다음에 무엇을 할지와 무슨 말을 할지를 set_intent 함수 한 번으로 정한다."
AUDIO_FIRST_LINE_WITH_RESERVE = ("다음에 무엇을 할지와 무슨 말을 할지를 함수 하나(set_intent 또는 reserve_facility)를 "
                                 "한 번 호출해 정한다.")
AUDIO_RESERVE_RULES = """
[시설 예약] 시설 예약은 reserve_facility 함수로 한다. [지금 상황]에 '[진행 중인 예약]' 줄이 있으면 지금은 예약
대화 중이고, 로봇의 마지막 말은 그 예약의 질문이다.
- 그 질문에 값으로 답하면(이름·날짜·시각·시설) reserve_facility 에 이번에 말한 값만 넣어 호출한다.
- 값을 바꿔 달라는 말("아니 다섯 시로 해 줘", "다음 주 화요일로 바꿔 줘")도 reserve_facility 에 바뀐 값만 넣어 호출한다.
- "…예약할까요?"에 긍정만 하면 set_intent 의 affirm, 부정만 하면 deny 다(reply=""). 이때는 reserve_facility 를 부르지 않는다.
- 예약 대화에 나오는 시설 이름은 가려는 목적지가 아니다 — destination_candidate 에 넣지 않는다.
"""


def build_audio_prompt(
    destinations: Sequence[DestinationData], robot_state: Optional[RobotState] = None,
    situation: str = "", directory_block: str = "",
) -> str:
    """소리 모드 지시문. 모델이 대화 판단자다 — 확인·정정·침묵까지 모델 몫.

    situation 은 LLM 노드의 상황판(situation_board.render) — 이동 중·마지막 도착·대기
    요청처럼 코드가 미션 신호로 확인한 사실. 이력이 비어도 남는 기억이다.

    순서가 비용이다(2026-09-21): Realtime 은 직전 호출과 **앞부분이 똑같은** 토큰을
    반값(캐시)으로 친다. 그래서 매 호출 바뀌는 로봇 상태·상황판은 맨 뒤에 붙이고,
    역할·규칙·목적지 목록(파일이 안 바뀌면 동일)은 앞에 둔다. 절약량은 `[RT]` 로그의
    cached 토큰으로 본다.
    """
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
    # 대장 블록(ledger_view)이 왔으면 층이 거기 있으므로 옛 상태 블록은 뺀다. 옛 상황판은 같은
    # "[지금 상황]" 머리글을 쓰지만 층 줄이 없으니 이 판정에 안 걸린다.
    state_block = "" if FLOOR_LABEL in situation else _format_robot_state(robot_state)
    reserve_on = audio_tools_active()
    first_line = AUDIO_FIRST_LINE_WITH_RESERVE if reserve_on else AUDIO_FIRST_LINE
    reserve_rules = AUDIO_RESERVE_RULES if reserve_on else ""
    # 원칙 1·4·12·31: 켰을 때만 새 판단 순서·의도 정의·clarify 칸 이름을 싣는다. 꺼졌으면 아래 변수가 옛 글 그대로다.
    recut = intent_recut.on()
    heard_intent = "not_for_robot" if recut else "unknown"
    ambiguous = ("애매하면 confidence 를 낮추고 unknown 으로 둔다." if recut else
                 "애매하면 confidence 를 낮추고, 답이 꼭 필요한 자리면 clarify 로 되묻는다.")
    order_block = "\n\n" + intent_recut.PROMPT_ORDER if recut else ""
    several_rule = intent_recut.SEVERAL_RULE if recut else """clarify 로 그 곳들의 name 을 모두 들어 묻는다("남자 화장실과 여자 화장실
  중 어디로 갈까요?"). 하나만 들어 "남자 화장실인가요?"처럼 묻지도 않는다(2026-10-11 실기: 같은 말에
  "여자 화장실로 안내해드릴까요?"·"남자 화장실인가요?")."""
    not_avail_deny = ("clarify(clarify_kind=not_available)로 내고 clarify_wanted 에 그 말을 적는다. 비슷한 다른 곳을 권하지 않는다." if recut else
                      """clarify 로
  "학과장님 사무실은 목록에 없는데, 학과사무실이 맞을까요?"처럼 되묻는다.""")
    promise_scope = "question 의 reply" if recut else "question·clarify·unknown 공통"
    clarify_unknown = (intent_recut.PROMPT_INTENTS + "\n" + intent_recut.WAKE_LINE) if recut else f"""- clarify: 어디로 갈지 모호하거나 목록에 없는 곳. reply 에 되묻는 질문(반드시 "?"로 끝).
  질문 하나만 짧게 하고 예시를 늘어놓지 않는다. 들은 말이 목록의 여러 곳에 걸리면(같은 별칭)
  그 곳들의 name 만 들어 어느 쪽인지 묻는다.
- unknown + 침묵(reply=""): 로봇에게 한 말이 **아닌 것이 분명할 때**(다른 사람들끼리의 대화,
  기침·소음, 로봇 자신의 목소리), 그리고 **로봇이 방금 질문하고 답을 기다리는데 못 알아들었을
  때**다. 질문 중이면 로봇 본체가 같은 질문을 한 번 다시 한다 — 네가 또 물으면 두 목소리가 된다
  (2026-10-08). 로봇이 묻고 있지 않은데 사용자가 로봇에게 말한 것 같고 못 알아들었으면 침묵하지
  말고 intent=clarify 로 "잘 못 들었는데, 다시 말씀해 주시겠어요?"(2026-09-20 사용자: 아무 말도
  안 하면 무슨 상황인지 알 수 없다). 호출어 "비카야"만 들리면 로봇을 부른 것이다 — intent=unknown, reply 는
  "{WAKE_GREETING}" 나 "네, 말씀하세요." 같은 짧은 응답."""
    not_avail = intent_recut.NOT_AVAILABLE_TAG if recut else ""
    digit_rule = ("clarify(clarify_kind=several)로 내고 clarify_options 에 그 번호들의 name 을 적는다." if recut else
                  "clarify 로 몇 번인지 되묻는다.")
    empty_kinds = "·clarify·unknown·not_for_robot·dismiss" if recut else ""
    return f"""너는 시각장애인 안내 로봇 '비카(VICA)'다. 차분하고 짧게 말하는 안내원처럼 말하고, 사용자의
목소리를 직접 듣고, {first_line}
코드는 네 결정을 고치지 않고 그대로 로봇 본체(미션 관리자)에 전달한다. 확인 질문·정정·
되묻기·침묵·말투까지 네가 책임진다.

[대화 이력 = 네 기억] 앞에 오는 assistant 줄은 네가(로봇이) 실제로 소리 내어 말한 문장이고,
user 줄은 그 전에 들린 사용자의 말이다. 로봇의 마지막 말이 질문이면 지금 들리는 말은 대개
그 답이다. "아까 어디 갔었지?", "방금 도착한 곳이 어디야?" 같은 질문은 이력의 "…로 안내를
시작합니다"·"…에 도착했습니다" 줄을 보고 답한다 — 기억이 없다고 하지 마라.

[heard_text] 이번 소리에서 실제로 들린 말만 한국어로 그대로 적는다(결정을 내렸다면 반드시
채운다 — "그래"를 듣고 출발시키면서 heard_text 를 비우지 마라). 사람 말이 아니면
(기침·소음·로봇 자신의 목소리·다른 사람들끼리의 잡담) 빈 문자열로 두고 intent={heard_intent},
reply="" 로 답한다. 이력에 있는 말을 베껴 적지 마라 — 이번 소리에 없는 말은 없는 것이다.
{ambiguous}{order_block}

[intent 별 규칙 — need_confirm 과 reply 까지 네가 정한다]
- navigate(새 목적지): 가고 싶은 곳을 말했다. 직접("407호 가자")도 간접("배 아파"→화장실)도 된다.
  destination_candidate=목록의 name 그대로, need_confirm=true, reply=네 말로 만든 확인 질문
  (목적지 name 을 그대로 넣고 반드시 "?"로 끝낸다. 예: "화장실로 안내해드릴까요?").
  들은 말이 목록의 여러 곳에 똑같이 걸리면(같은 별칭 — 예: "화장실 가고 싶어"가 남자 화장실·여자 화장실
  둘 다) 하나를 짐작해 고르지 말고 {several_rule}
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
- 대기 중 목적지: 대화 단계가 '대기 중'·'대기 장소로 떠나기 직전'이어도 목적지를 말하면 평소처럼
  navigate(새 목적지)다 — 대기는 출발할 때 끝난다. reply 는 목록의 확인 질문 하나다(예: "화장실로
  안내해드릴까요?"). "다 됐어"·"다녀왔어"처럼 볼일이 끝났다고만
  하면 finish, reply="" 다(로봇 본체가 "네, 어디로 모실까요?"라고 묻는다).
- 정정: 확인 질문에 "아니 XX로 가자"처럼 다른 목적지를 말하면 XX 로 navigate, need_confirm=true,
  reply=XX 로 다시 묻는 확인 질문("XX로 안내해드릴까요?").
- deny: 확인 질문이나 제안에 부정만 하고 대안이 없다("아니", "아니요", "됐어"). reply="".
  부정 뒤에 **목록에 없는 곳**을 말하면("아니, 학과장님 사무실") 방금 거절당한 목적지를 그대로 다시
  제안하지 마라 — 로봇 본체는 같은 곳의 재제안을 승낙으로 보고 출발한다(21:21 실기). {not_avail_deny}
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
- 약속 금지({promise_scope}, 2026-10-08): reply 는 정보와 되묻기까지만이다. "알겠습니다"·
  "할게요"·"해 드릴게요" 같은 약속, "계속 안내할까요?"·"방향 맞춰드릴까요?"처럼 로봇이 할 일을 권하는
  말은 쓰지 않는다 — 로봇 본체가 모르는 약속이 된다. 목록에 있는 곳으로 안내를 권하려면 question 이
  아니라 navigate(need_confirm=true)로 낸다(위 위치 질문 규칙처럼).
- 접근 질문 중의 question: 로봇의 마지막 말이 "…안내를 받으시겠어요?"(대화 단계 '인사 답 대기')이면
  묻는 말에 한 문장으로 답만 하고 "?"로 끝내지 않는다. 되묻거나 말을 청하지 않는다("말씀해 주세요"·
  "안내를 시작할까요?" 같은 말 금지) — 로봇 본체가 이어서 "안내를 받으시겠어요?"를 다시 묻는다(2026-10-09
  실기: 답 끝에 질문을 붙여 질문이 두 번 겹쳤다). 예: "누구세요?" → "저는 시각장애인 안내로봇 비카입니다.",
  "뭐라고요?" → "안내를 받으시겠냐고 여쭤봤어요."
{clarify_unknown}

[목적지 목록] destination_candidate 는 반드시 아래 name 중 하나. 목록에 없는 곳은 clarify{not_avail}.
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
"투·이"=2, "원·일"=1. 숫자가 확실치 않으면 confidence 를 낮추고 {digit_rule}

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
reply="" 로 정한 경우(확정 navigate·affirm·deny·wait·finish·확정 cancel{empty_kinds})는 로봇 본체가 이어서
말하므로 네가 말을 보태지 않는다.
{reserve_rules}{state_block}{situation}"""


def parse_intent_audio(
    pcm16_16k: bytes,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    situation: str = "",
    directory_block: str = "",
    question: Optional[OpenQuestion] = None,
    tool_trail: Optional[list[dict]] = None,
) -> "tuple[VicaIntent | ReserveCall, str, float, dict]":
    """발화 소리 -> Realtime(set_intent) -> 모델 결정을 그대로 VicaIntent 로 (모델 전결).

    (의도, 들린 말, 지연초, info) 를 돌려준다. info = {"src": "llm", "usage": ...}.
    실패(예외·timeout)는 그대로 올린다 — LLM 노드가 그 발화를 텍스트 경로로 넘긴다.
    도구 출력이 켜져 있고(VICA_TOOLS_AUDIO=on) 모델이 reserve_facility 를 부르면 의도 대신 ReserveCall 을 돌려준다.
    question 은 열린 질문(질문 기록) — 대답 칸이 켜졌으면 [지금 상황] 끝에 한 줄로 가고, 모델이 낸 answer 가 의도에 실린다.
    """
    instructions = build_audio_prompt(destinations, robot_state,
                                      answer_field.with_question(intent_recut.with_clue(situation, question), question), directory_block)
    # tool_trail: 진행 중인 예약에서 부른 reserve_facility 기록(ReservationFlow.tool_trail) — 대화 기록에 싣는다.
    # 없으면 예전과 같은 호출이다(인자를 넘기지 않는다).
    result = get_realtime_client().ask(pcm16_16k, history, instructions,
                                       **({"tool_trail": tool_trail} if tool_trail else {}))
    return _deliver_realtime_result(result, destinations, robot_state, question)


def parse_intent_realtime_text(
    text: str,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
    situation: str = "",
    directory_block: str = "",
    question: Optional[OpenQuestion] = None,
    tool_trail: Optional[list[dict]] = None,
) -> "tuple[VicaIntent | ReserveCall, str, float, dict]":
    """재생 측정용(tools/reserve_replay) — 소리 대신 글자를 Realtime 에 넣는다. 지시문과 그 뒤 처리는
    parse_intent_audio 와 같다. 운영 경로는 부르지 않는다."""
    instructions = build_audio_prompt(destinations, robot_state,
                                      answer_field.with_question(intent_recut.with_clue(situation, question), question), directory_block)
    result = get_realtime_client().ask_text(text, history, instructions,
                                            **({"tool_trail": tool_trail} if tool_trail else {}))
    return _deliver_realtime_result(result, destinations, robot_state, question)


def _deliver_realtime_result(result, destinations: Sequence[DestinationData], robot_state: Optional[RobotState],
                             question: Optional[OpenQuestion] = None,
                             ) -> "tuple[VicaIntent | ReserveCall, str, float, dict]":
    """Realtime 응답(도구 인자) → (의도 또는 ReserveCall, 들린 말, 지연초, info).

    대답 칸이 켜졌으면 Realtime 이 들은 글자(heard)에 짧은 대답 규칙을 대고 글자 경로와 같은 answer_field.fill 을 지난다
    (Task 3.5b, 원칙 6·7) — 규칙은 모델의 읽기를 뒤집지 않는다(같은 읽기일 때 대답 칸 채우기·분·not_for_robot 되묻기만).
    규칙이 의도나 대답 칸을 실제로 바꿨으면 info["src"] 에 "+대답규칙" 을 붙이고 바뀌기 전 값을 로그 한 줄로 남긴다."""
    heard = result.heard_text.strip()
    if result.tool == RESERVE:
        # 예약 요청·대답 — 값 검사와 묻는 문장은 reservation_flow 가 한다(의도 배달 규칙은 의도 전용).
        return reserve_call(result.draft), heard, result.latency_sec, {"src": "llm", "usage": result.usage}
    try:
        draft = _draft_cls()(**result.draft)
    except ValidationError as exc:
        raise ValueError(f"set_intent 인자가 _IntentDraft 와 맞지 않는다: {exc}") from exc
    dialog_state = robot_state.dialog_state if robot_state else ""
    shared = _audio_shared_guess(draft, heard, destinations, dialog_state)
    intent = deliver_audio_draft(draft, destinations, dialog_state,
                                 robot_state.active_destination if robot_state else "", shared=shared,
                                 heard=heard)
    # 규칙이 모델 결정을 바꿨으면 [RT] 로그 줄의 src 로 보이게 한다.
    src = "llm+고르기규칙" if shared else "llm"
    if _audio_short_answer(intent, heard, dialog_state):
        src += "+짧은답규칙"
    # 원칙 6·7: 규칙에 대는 글자는 Realtime 이 들은 글자다 — 로컬 받아쓰기(/user_text)는 기다리지 않는다(다른 토픽으로
    # 다른 때에 와서 기다리면 지연·순서에 따라 결과가 갈린다). 노드의 '[대답] 받아쓴 글자 규칙=…' 줄은 견주는 기록으로 남는다.
    llm_answer = getattr(draft, "answer", "") or ""
    ruled = answer_field.short_answer(heard, question)
    filled = answer_field.fill(intent, question, ruled, llm_answer)
    if ruled is not None:
        by_llm = answer_field.fill(intent, question, None, llm_answer)
        if (filled.intent, filled.answer, filled.answer_value) != (by_llm.intent, by_llm.answer, by_llm.answer_value):
            src += "+대답규칙"
            # '규칙 전' 은 _audio_short_answer 가 지나간 뒤의 값이다(검토 M4) — Realtime 이 처음 낸 의도는 src 의 +짧은답규칙 으로 본다.
            _log("info", f"[대답규칙] 규칙 전(짧은답규칙 뒤) intent={by_llm.intent} "
                         f"answer={by_llm.answer or '-'}/{by_llm.answer_value or '-'}"
                         f" → intent={filled.intent} answer={filled.answer}/{filled.answer_value or '-'} heard='{heard}'")
    return filled, heard, result.latency_sec, {"src": src, "usage": result.usage}


def _audio_short_answer(intent: VicaIntent, heard: str, dialog_state: str) -> bool:
    """대답을 기다리는 단계에서 짧은 긍정·부정만 말했는데 모델이 대답으로 안 봤으면 affirm·deny 로 바꾼다.

    10-23 리허설(run105) 10:03:39·47 — 접근 질문 "안내를 받으시겠어요?"에 "그래"를 question 으로 봐 같은 질문이
    세 번 나왔다. 글자 모드의 짧은 대답 지름길(_shortcut_intent, local_rules.ANSWER_WAIT_STATES)과 같은 판단이다.
    모델이 대답 의도(affirm·deny·navigate·wait …)를 냈으면 건드리지 않는다. 바꿨으면 True.
    """
    if dialog_state not in local_rules.ANSWER_WAIT_STATES or intent.intent not in ("question", "unknown", "clarify"):
        return False
    word = _normalize_short_reply(heard)
    if word in _SOLO_AFFIRMATIVES:
        kind = "affirm"
    elif word in _NEGATIVES:
        kind = "deny"
    else:
        return False
    intent.intent = kind
    intent.reply = ""                    # 계약: affirm·deny 의 말은 미션 몫
    intent.need_confirm = False
    intent.destination_candidate = None
    intent.matched_destination_id = ""
    return True


def deliver_audio_draft(draft: _IntentDraft, destinations: Sequence[DestinationData],
                        dialog_state: str = "", active: str = "",
                        shared: Sequence[DestinationData] = (), heard: str = "") -> VicaIntent:
    """재단 스위치(VICA_RECUT)가 켜졌으면 새 값을 맨 앞에서 돌려보내고 clarify·unknown 문장을 코드 틀에서 만든다(원칙 4·7·8·11)."""
    early = intent_recut.early(draft)
    if early is not None:
        return early
    return intent_recut.settle(_deliver_audio_draft(draft, destinations, dialog_state, active, shared),
                               draft, destinations, heard)


def _deliver_audio_draft(draft: _IntentDraft, destinations: Sequence[DestinationData],
                         dialog_state: str = "", active: str = "",
                         shared: Sequence[DestinationData] = ()) -> VicaIntent:
    """모델 결정 → VicaIntent. 판단은 하지 않는다: id 매핑·접근 불가·계약상 빈 reply 만.

    shared 는 _audio_shared_guess 가 찾은 '여러 곳에 걸린 말에 모델이 한쪽을 짐작한' 후보들이다 — 둘 이상이면
    고르기 질문으로 바꾼다(2026-10-21 사용자 요청).
    """
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
    if len(shared) > 1:
        _ask_which(result, shared, dialog_state)
    elif len(tied) > 1:
        _ask_which(result, tied, dialog_state)
    elif len(asked) > 1:
        # 입으로는 "남자 화장실과 여자 화장실 중 어디로…?"라고 고르게 하면서 목적지는 한쪽 정확한 이름으로 냈다
        # (2026-10-11 run90 18:44:43 — 미션이 그곳을 확인하는 중이 돼 다음 "남자 화장실로 가자"에 출발 안내와 LLM
        # 확인 질문이 겹쳤다). 고르기 질문이면 목적지를 보내지 않는다.
        _ask_which(result, asked, dialog_state)
    elif draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
        if matched is None:
            # 목록에 없는 이름 — id 가 없으니 미션에 보낼 수 없다. 되묻기로.
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
                result.reply = ""            # 출발 안내는 미션이 말한다 (계약)
            elif "".join(matched.name.split()) not in "".join(result.reply.split()):
                # 모델이 비웠거나 고른 곳의 이름을 말하지 않았으면 목록 문구 — 10-23 리허설 10:11:54 사회복지를 고르고
                # "장애인 서류로 안내해드릴까요?"라고 물었다. 이름이 든 모델 말(주행 중 바꾸기 질문 포함)은 그대로 둔다.
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
            result.reply = ""                # 결과 발화는 미션 몫 (계약)
    elif draft.intent == "clarify":
        result.need_confirm = False
        result.reply = result.reply or ASK_DESTINATION
        named = _choice_in_reply(result.reply, destinations, active)
        if len(named) > 1:
            result.reply = choose_between([d.name for d in named])   # 고르기 질문은 정해진 문장(10-23 리허설)
    else:                                    # question / unknown — 모델의 reply 그대로 (빈 말이면 침묵)
        result.need_confirm = False
    return result

# 접근 질문("안내를 받으시겠어요?")에 답하는 단계 — mission_question.APPROACH_ANSWER_STATES 와 같다.
_APPROACH_STATES = frozenset({"awaiting_user", "turning"})


def _choice_in_reply(reply: str, destinations: Sequence[DestinationData], active: str = "") -> list[DestinationData]:
    """되묻는 말(끝 "?")에 든 목적지들 — 다른 이름 속에 든 짧은 이름은 빼고, 지금 가는 곳(active)도 뺀다.

    둘 이상이면 고르기 질문이다. 주행 중 "지금 OO로 가는 중이에요. XX로 바꿀까요?"의 OO 는 지금 가는 곳이라
    세지 않는다(XX 하나만 남아 그대로 바꾸기 질문).
    """
    reply = (reply or "").strip()
    if not reply.endswith("?"):
        return []
    named = [d for d in destinations if d.name and d.name in reply]
    outer = [d for d in named if not any(o.name != d.name and d.name in o.name for o in named)]
    return [d for d in outer if d.name != active]


def _named_destination(user_text: str, destinations: Sequence[DestinationData]) -> Optional[DestinationData]:
    """사용자 말에 이름이 딱 하나 든 목적지 — 띄어쓰기는 보지 않고, 다른 이름 속에 든 짧은 이름은 뺀다.

    별칭은 보지 않는다 — '화장실'·'사무실'처럼 여러 곳에 걸린다. 둘 이상이거나 없으면 None.
    """
    def squash(s: str) -> str:
        return "".join((s or "").split())

    said = squash(user_text)
    if not said:
        return None
    named = [d for d in destinations if d.name and squash(d.name) in said]
    outer = [d for d in named if not any(o.name != d.name and squash(d.name) in squash(o.name) for o in named)]
    return outer[0] if len(outer) == 1 else None


def _alias_only_group(user_text: str, destinations: Sequence[DestinationData]) -> list[DestinationData]:
    """사용자 말이 여러 갈 수 있는 곳에 같이 걸린 별칭만 담았으면 그 곳들(목록 순서), 아니면 빈 목록.

    "화장실로 가자"·"나 배 아파"는 남자·여자 화장실 둘 다의 별칭이다. 목적지 이름을 말했거나("남자 화장실"),
    한 곳에만 있는 말을 했으면("남자 쪽으로"의 '남자', "서류"의 사회복지) 사용자가 이미 고른 것이라 빈 목록이다.
    띄어쓰기는 보지 않는다. 갈 수 없는 곳은 후보로 들지 않는다(tied_destinations 와 같은 기준).
    """
    def squash(s: str) -> str:
        return "".join((s or "").split())

    said = squash(user_text)
    if not said or any(d.name and squash(d.name) in said for d in destinations):
        return []
    open_ = [d for d in destinations if d.is_approachable]
    alias_sets = {d.id: {squash(a) for a in d.aliases if squash(a)} for d in open_}
    heard = {a for aliases in alias_sets.values() for a in aliases if a in said}
    # 걸린 별칭 중 다른 걸린 별칭 안에 든 짧은 것은 뺀다 — 사용자가 말한 것은 긴 쪽이다(2026-10-29 행사 지도:
    # "서울로봇고 가자"에 세 곳 공통 '로봇고'가 함께 걸려 경남로봇고까지 물었다).
    heard = {a for a in heard if not any(a != b and a in b for b in heard)}
    if not heard:
        return []
    if any(sum(a in aliases for aliases in alias_sets.values()) == 1 for a in heard):
        return []
    group = [d for d in open_ if alias_sets[d.id] & heard]
    return group if len(group) > 1 else []


def _audio_shared_guess(draft: _IntentDraft, heard: str, destinations: Sequence[DestinationData],
                        dialog_state: str = "") -> list[DestinationData]:
    """소리 모드에서 여러 곳에 걸린 말에 모델이 한쪽을 짐작했으면 그 후보들(2026-10-21 실기·사용자 요청).

    10-21 10:24 "화장실로 가자"→남자 화장실 바로 제안, 10:37 주행 중 "나 화장실 가고 싶어"→"여자 화장실로
    바꿀까요". 글자 모드의 로컬 규칙(_shared_alias_tie)은 소리 모드를 거치지 않는다. 판단 근거는 모델이 들은 말
    (heard_text)이다 — 그 말로 모델이 결정했기 때문이다.
    - navigate: 고른 곳이 후보 중 하나면 고르기로. 확인 질문 중의 승낙(need_confirm false)은 물어 둔 그곳이라 둔다.
    - clarify·question: 대답이 후보 중 한 곳만 들면 짐작이라 고르기로. 접근 질문 중에는 두지 않는다 — 미션이
      그 되묻기에 "안내를 받으시겠어요?"를 다시 묻는다.
    """
    group = _alias_only_group(heard, destinations)
    if not group:
        return []
    if draft.intent == "navigate":
        if dialog_state == "confirming" and not draft.need_confirm:
            return []
        chosen = match_destination(draft.destination_candidate, list(destinations))
        return group if chosen is not None and chosen.id in {d.id for d in group} else []
    if draft.intent in ("clarify", "question") and dialog_state not in _APPROACH_STATES:
        named = [d for d in group if d.name in (draft.reply or "")]
        return group if len(named) == 1 else []
    return []


def _ask_which(result: VicaIntent, tied: Sequence[DestinationData], dialog_state: str = "") -> VicaIntent:
    """두 곳 이상에 똑같이 걸린 navigate 를 되묻기로 바꾼다 — 한쪽을 고르지 않는다(2026-10-11 사용자 결정).

    10-11 실기: "화장실 가고 싶어"(남자·여자 화장실 둘 다 별칭 '화장실')에 한쪽을 짐작했다. 지시문으로 막아도
    모델이 이름 대신 '화장실'로 navigate 를 내면 match_destination 이 목록 앞쪽을 골라 미션이 그곳을 확인하는
    중이 됐다. 모델의 되묻는 말이 후보를 다 들면 그대로, 아니면 코드가 묻는다.
    되묻는 말은 늘 choose_between 의 정해진 문장이다(2026-10-23).
    접근 질문 중이면 되묻지 않고 목적지 없는 수락(navigate)으로 둔다 — 미션은 되묻기를 수락으로 안 받고
    "안내를 받으시겠어요?"를 다시 묻는데, 목적지 없는 navigate 는 수락으로 받고 돌아선 뒤 직접 목적지를 묻는다.
    """
    if dialog_state in _APPROACH_STATES:
        result.destination_candidate = None
        result.matched_destination_id = ""
        result.reply = ""
        return result
    result.intent = "clarify"
    result.destination_candidate = None
    result.matched_destination_id = ""
    result.need_confirm = False
    # 늘 정해진 문장으로 묻는다 — 10-23 리허설 09:54:32 모델 말을 그대로 두었더니 "남자 화장실로인지 여자 화장실로인지요,
    # 남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"가 나갔다(옛 동작: 후보를 다 들면 모델 말 그대로).
    result.reply = choose_between([d.name for d in tied])
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
    """재단 스위치(VICA_RECUT)가 켜졌으면 새 값을 맨 앞에서 돌려보내고 clarify·unknown 문장을 코드 틀에서 만든다(원칙 4·7·8·11)."""
    early = intent_recut.early(draft)
    if early is not None:
        return early
    return intent_recut.settle(
        _finalize_core(draft, destinations, pending, pending_command, user_text, local, robot_state),
        draft, destinations, user_text)


def _finalize_core(
    draft: _IntentDraft,
    destinations: Sequence[DestinationData],
    pending: Optional[DestinationData] = None,
    pending_command: Optional[str] = None,
    user_text: str = "",
    local: bool = False,
    robot_state: Optional[RobotState] = None,
) -> VicaIntent:
    """LLM 초안 + 코드 매칭으로 최종 VicaIntent 를 만든다. (결정/안전은 코드 담당)

    pending 은 코드가 대화 이력에서 확인한 '대기 중인 확인 질문'의 목적지다.
    draft.is_confirmation 은 이것과 대조해서만 믿는다 — 확인 질문이 없었는데
    LLM 이 true 를 줘도(모델 오판) 확인 절차를 건너뛸 수 없다.

    robot_state 는 주행 중 바꾸기 질문(2026-10-07)에만 쓴다 — 안내 주행 중이면 확인
    문구 대신 "지금 OO로 가는 중이에요. XX로 바꿀까요?"를 만든다.
    """
    result = VicaIntent(
        intent=draft.intent,
        destination_candidate=draft.destination_candidate,
        confidence=draft.confidence or 0.0,  # LLM 이 null 을 줘도 안전하게
        reply=draft.reply,
        need_confirm=False,
        safety_flag="normal",
    )

    if draft.intent in ("affirm", "deny", "finish", "help"):
        # LLM 이 reply 를 채워도 비운다 — 응답 발화는 Mission 몫 (계약). help 도 미션이 묻고 답한다.
        result.reply = ""
        result.matched_destination_id = ""
        result.need_confirm = False
        return result

    if draft.intent == "wait":
        # 시간 병합 (판정 권한 원칙): 단일 값도 범위(상단x1.5)도 코드가
        # 계산한다 — LLM 에게 산수를 맡겼더니 평균을 냈다(2026-08-30 실기,
        # "십 분에서 십오 분" -> 13분). LLM 제안값은 숫자가 아예 없는
        # 발화("좀 있다가")의 폴백일 뿐. 둘 다 없으면 -1 로 두고 Mission 이
        # 후속 질문. 상한(30분) 강제는 Mission. reply 는 Mission 몫.
        result.reply = ""
        result.need_confirm = False
        if local:
            # 로컬 규칙(2026-10-05): OpenAI 지시문 기준에 맞춘다 — 범위면 큰 쪽(×1.5
            # 없음), 하나면 그 값, 숫자가 없으면 비움(Mission 이 "몇 분쯤?" 다시 물음).
            # 숫자 계산은 계속 코드가 한다 — 작은 모델은 범위에 평균을 냈다.
            minutes = parse_wait_minutes(user_text, range_factor=1.0)
        else:
            minutes = parse_wait_minutes(user_text)
            if minutes is None and draft.wait_minutes and draft.wait_minutes > 0:
                minutes = draft.wait_minutes
        result.wait_minutes = minutes if minutes else -1
        return result

    # 여러 곳에 같이 걸린 별칭만 말했는데 모델이 한쪽을 짐작했으면 고르기 질문(2026-10-24 사용자 "'나 배아파' → 화장실을
    # 가자는 것"). 소리 모드(_audio_shared_guess)와 같은 규칙을 글자 경로(⑬ 오프라인·⑫ 예비)에도 — 10-24 재생에서 글자
    # 경로만 '나 배 아파.'에 남자 화장실을 짐작했다. 판단 근거는 사용자 말(user_text)이다.
    dialog = robot_state.dialog_state if robot_state else ""
    shared = _audio_shared_guess(draft, user_text, destinations, dialog)
    if not shared and draft.intent == "unknown" and dialog not in _APPROACH_STATES:
        # 모델이 아예 못 알아들어도(unknown) 말이 여러 곳의 별칭만 담았으면 고르기 질문(2026-10-24 미션 대역 시험
        # 12:49:16 '나 배 아파.'에 midm2-mini 가 "잘 듣지 못했습니다"). 글자 경로만 — 소리 모드의 unknown 은
        # '사람 말 아님'(옆사람·메아리) 판단이기도 해 _audio_shared_guess 가 그대로 둔다.
        shared = _alias_only_group(user_text, destinations)
    if len(shared) > 1:
        return _ask_which(result, shared, dialog)

    # 로컬 규칙(2026-10-16): 사용자가 목적지 이름을 하나만 말했으면 그곳을 따른다 — 10-16 2단계 실기 재생에서
    # 로봇 안 모델이 "남자 화장실."에 407호를 골랐다(모델 원답). 이름이 둘 이상이거나 없으면 모델 판단 그대로.
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
            # LLM 이 목록에 없는 목적지를 골랐다 -> 되묻기로 안전하게 강등.
            result.intent = "clarify"
            result.reply = result.reply or ASK_DESTINATION
        elif not matched.is_approachable:
            # 접근 불가 목적지 -> 코드가 정한 안내 문구, 확인 불필요.
            result.matched_destination_id = matched.id
            result.reply = matched.unavailable_reason or matched.confirm_prompt
            result.need_confirm = False
        elif draft.is_confirmation and pending is not None and matched.id == pending.id:
            # 사용자가 직전 제안을 수락 -> 확인 끝, 안내 시작.
            # (코드가 아는 pending 과 목적지까지 일치할 때만. 불일치·부재면 아래
            #  else 로 떨어져 정상 확인 질문을 다시 한다.)
            result.matched_destination_id = matched.id
            result.reply = f"{matched.name} 안내를 시작합니다."
            result.need_confirm = False
        else:
            # 정상 목적지 -> 코드가 정한 확인 문구로 통일(LLM 자유 발화 대신).
            result.matched_destination_id = matched.id
            result.reply = matched.confirm_prompt
            result.need_confirm = True
            current = _guiding_destination(robot_state)
            if current and current == matched.name:
                # 지금 가는 곳을 또 말했다 — 바꿀 것이 없다. 미션에 보낼 요청이 아니다.
                result.intent = "question"
                result.destination_candidate = None
                result.matched_destination_id = ""
                result.need_confirm = False
                result.reply = ALREADY_GOING.format(cur=current, cur_josa=_josa_euro(current))
            elif current:
                # 안내 주행 중 다른 곳 — 로봇이 서서 바꿀지 묻는다(2026-10-07 사용자 결정).
                # 미션은 이 제안에 말하지 않고 물은 목적지를 기억한다.
                result.reply = CHANGE_DESTINATION_QUESTION.format(
                    cur=current, cur_josa=_josa_euro(current),
                    new=matched.name, new_josa=_josa_euro(matched.name))

    if draft.intent in ("cancel", "pause", "resume"):
        if draft.intent == "pause":
            # 서는 방향은 되묻지 않는다 — 오분류해도 잠시 서는 것뿐(안전한 실패).
            # 감속 정지 실행과 수락/거절 판정은 Mission Manager 몫이다. 이 reply 는
            # ros_node 2-3(quiet_for_mission)이 지운다 — 대답도 미션이 한다(2026-10-09).
            result.reply = PAUSE_ACK
            result.need_confirm = False
        elif pending_command == draft.intent:
            # 방금 이 제어를 물었고 사용자가 말로 다시 수락했다 ("응, 취소해 줘").
            # reply 는 침묵 — 결과 발화는 미션 몫이다 (2단 발화 방지).
            result.reply = ""
            result.need_confirm = False
        else:
            # 버리는 쪽(cancel)과 움직이는 쪽(resume)은 반드시 되묻는다 —
            # LLM 제안만으로는 실행되지 않는다 (is_confirmation 게이팅과 같은 원칙).
            result.reply = {
                "cancel": CANCEL_CONFIRM,
                "resume": RESUME_CONFIRM,
            }[draft.intent]
            result.need_confirm = True

    if not result.reply:
        # reply 생략은 navigate·제어 전용 지시인데, 모델이 다른 intent 에서도 비울
        # 수 있다. 침묵은 소리로만 상태를 아는 사용자에게 최악이므로 고정 문구로 메운다.
        result.reply = ASK_DESTINATION if result.intent == "clarify" else RETRY_PROMPT

    return result
