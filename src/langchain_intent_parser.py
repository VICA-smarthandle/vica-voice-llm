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


class _IntentDraft(BaseModel):
    """LLM 이 직접 채우는 부분만 담은 임시 스키마.

    matched_destination_id / need_confirm / safety_flag 는 코드가 채운다.
    """

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
        # navigate 의 확인 문구는 코드가 confirm_prompt 로 갈아끼우므로(_finalize),
        # 모델이 문장을 써 봐야 폐기된다. 빈 문자열이면 그만큼 생성 토큰이 줄어
        # 지연이 준다 (Jetson 로컬 22tok/s 실측 기준 ~1초 이상).
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


# 직전 확인 질문에 대한 짧은 긍정/부정. 긴급어 필터와 같은 원칙으로 LLM 을 거치지
# 않고 코드가 결정한다 — 소형 모델이 is_confirmation 을 놓치면 확인 질문이 무한
# 반복되는 문제가 실기에서 확인됐다 (2026-07-19, exaone3.5:2.4b).
#
# 목록의 정본은 `handle_mode.py` 다. 스마트핸들 모드 질문도 같은 "네/아니요"를
# 받으므로, 두 곳에 따로 두면 한쪽만 고쳐져 어긋난다. 아래 이름은 이 모듈의
# 기존 호출부와 테스트가 쓰던 것이라 별칭으로 남긴다.
_AFFIRMATIVES = AFFIRMATIVES
_NEGATIVES = NEGATIVES
# 혼잣소리('음'·'어어')는 발화 전체가 그것뿐일 때만 긍정으로 본다.
# 첫 단어 지름길에는 넣지 않는다 — "음… 아니야"가 승낙이 되면 안 된다.
_SOLO_AFFIRMATIVES = AFFIRMATIVES | SOFT_AFFIRMATIVES
_normalize_short_reply = normalize_short_reply

# 코드(지름길)가 직접 채운 응답 문구. LLM 이 지어낸 잡담 대꾸와 구별하는
# 유일한 표지다 — ros_node 의 재청취 기각이 이 답은 삼키지 않는다.
# 목록은 지름길이 늘 때만 늘린다 (LLM 생성 문구는 절대 넣지 않는다).
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
_PAUSE_WORDS = {"잠깐만", "잠깐만요", "잠시만", "잠시만요"}
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


def parse_wait_minutes(text: str):
    """한국어 시간 표현에서 분(minute)을 뽑는다. 없으면 None.

    산수는 전부 여기서 한다 — LLM 에게 계산을 맡겼더니 범위("십 분에서
    십오 분")에 평균(13분)을 냈다(2026-08-30 실기). 규칙:
    - 단일 값("20분", "십오 분", "한 시간")은 그대로.
    - 범위(숫자 2개 이상 또는 "에서"·"~")는 넉넉한 쪽(최댓값) x1.5 반올림
      (사용자 결정: "5분에서 10분" -> 15분 — 여유를 주는 게 센스다).
    - 상한(30분)은 여기서 걸지 않는다 — 판정 권한은 Mission 에 있다.
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
        return int(max(values) * 1.5 + 0.5)
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


def _get_structured_llm(model: str):
    """구조화 출력(_IntentDraft) LLM 을 만든다. 백엔드는 PROVIDER 가 정한다."""
    if PROVIDER == "openai":
        # 지연 import — ollama 만 쓰는 환경에 openai 패키지를 요구하지 않는다.
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            # 로봇 대화에서 무한 대기는 곧 침묵이다. 실측 꼬리(1.86초)의 여유
            # 배수에서 끊고, 실패는 parse_intent 의 LLM_UNAVAILABLE 폴백이 받는다.
            timeout=15,
            max_retries=1,
        )
        # strict=True: 스키마를 서버가 문법 수준에서 강제한다 (실측 준수 18/18).
        return llm.with_structured_output(_IntentDraft, method="json_schema", strict=True)

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
            return VicaIntent(
                intent="unknown",
                confidence=1.0,
                reply=COMMAND_DECLINED,
                need_confirm=False,
            )

    pending = _pending_confirm_destination(history, destinations)
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
        if word in _SOLO_AFFIRMATIVES or first in _AFFIRMATIVES:
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
        if word in _NEGATIVES or first in _NEGATIVES:
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
    if word in _PAUSE_WORDS:
        return VicaIntent(intent="pause", confidence=1.0, reply=PAUSE_ACK, need_confirm=False)

    # 도착 후 대기·종료(wait/finish)는 지름길을 두지 않는다 — 발화 의도가
    # 맥락에 좌우돼("그만 좀 물어봐" ≠ 종료) 단어 매칭이 위험하다. LLM 이
    # 판단하고, 시간 숫자만 코드가 원문에서 뽑는다 (2026-08-30 사용자 결정).

    # 확인 대기가 없는 짧은 긍/부정은 affirm/deny 로 발행한다 (0초, LLM 없이).
    # 어느 질문의 답인지는 판정하지 않는다 — 상태를 가진 Mission 이 소비하거나
    # 무시한다 (계약: VicaIntent.msg affirm/deny 절, "아무 때나 보내도 안전").
    # reply 는 빈 문자열 — 수락/거절 발화는 Mission 몫이라 채우면 두 번 말한다.
    # '취소'는 NEGATIVES 에도 있으나 위 _CANCEL_WORDS 직행이 먼저 잡는다.
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
        # LLM/네트워크 실패 -> 크래시 대신 안전한 fallback 응답을 돌려준다.
        # (긴급어는 LLM 이전 단계에서 처리되므로 이 실패의 영향을 받지 않는다.)
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
    """LLM 초안 + 코드 매칭으로 최종 VicaIntent 를 만든다. (결정/안전은 코드 담당)

    pending 은 코드가 대화 이력에서 확인한 '대기 중인 확인 질문'의 목적지다.
    draft.is_confirmation 은 이것과 대조해서만 믿는다 — 확인 질문이 없었는데
    LLM 이 true 를 줘도(모델 오판) 확인 절차를 건너뛸 수 없다.
    """
    result = VicaIntent(
        intent=draft.intent,
        destination_candidate=draft.destination_candidate,
        confidence=draft.confidence or 0.0,  # LLM 이 null 을 줘도 안전하게
        reply=draft.reply,
        need_confirm=False,
        safety_flag="normal",
    )

    if draft.intent in ("affirm", "deny", "finish"):
        # LLM 이 reply 를 채워도 비운다 — 응답 발화는 Mission 몫 (계약).
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
        minutes = parse_wait_minutes(user_text)
        if minutes is None and draft.wait_minutes and draft.wait_minutes > 0:
            minutes = draft.wait_minutes
        result.wait_minutes = minutes if minutes else -1
        return result

    if draft.intent == "navigate":
        matched = match_destination(draft.destination_candidate, list(destinations))
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

    if draft.intent in ("cancel", "pause", "resume"):
        if draft.intent == "pause":
            # 서는 방향은 되묻지 않는다 — 오분류해도 잠시 서는 것뿐(안전한 실패).
            # 감속 정지 실행과 수락/거절 판정은 Mission Manager 몫이다.
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
