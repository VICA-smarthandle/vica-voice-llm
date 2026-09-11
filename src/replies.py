"""로봇이 말하는 고정 문구 모음 (LLM 이 만들지 않는 말)."""
from __future__ import annotations

EMERGENCY_REPLY = "안전을 위해 멈추겠습니다."

ACK_LISTENING_POOL = (
    "확인할게요.",
    "잠시만요.",
    "찾아볼게요.",
    "잠시 기다려주세요.",
    "알아볼게요.",
)

RETRY_PROMPT = "잘 듣지 못했습니다. 다시 말씀해 주세요."

LLM_UNAVAILABLE = "죄송합니다. 지금은 요청을 처리할 수 없어요. 잠시 후 다시 말씀해 주세요."

ASK_DESTINATION = "어디로 안내해드릴까요?"

CONFIRM_DECLINED = "알겠습니다. 어디로 안내해드릴까요?"

CANCEL_CONFIRM = "안내를 취소할까요?"
RESUME_CONFIRM = "다시 출발할까요?"

APPROACH_QUESTION = (
    "안녕하세요? 저는 시각장애인 안내로봇 비카입니다! "
    "저와 함께 목적지까지 동행해보시는건 어떠세요? 안내를 받으시겠어요?"
)
APPROACH_ONBOARDING = (
    "안녕하세요? 반갑습니다! 저에게 말을 거실 때는 '비카야'라고 불러주세요. "
    "자, 이제 어디로 가고 싶으신가요?"
)

APPROACH_TURN_NOTICE = "네, 잠시만 기다려주세요. 로봇이 회전하니 주의하세요."
APPROACH_TURN_DONE = "회전이 완료되었습니다."
APPROACH_FAREWELL = "알겠습니다. 이만 물러납니다."

PAUSE_ACK = "네, 잠시 설게요."

USAGE_GUIDE = "안내 중에 잠깐 쉬려면 '잠깐만', 다시 가려면 '다시 가자', 그만두려면 '취소'라고 말씀해 주세요."

COMMAND_DECLINED = "알겠습니다. 계속 진행할게요."

MODE_ASK = "손잡이를 잡고 스마트핸들 모드를 시작하시겠어요?"

MODE_ENTER_REQUEST = "손잡이를 3초간 잡아 주세요."

MODE_READY = "스마트핸들 모드입니다."

MODE_DECLINED = "알겠습니다."

HANDLE_UNAVAILABLE = "손잡이 연결에 문제가 있어 일반 안내로 진행합니다."

HANDLE_LOST = "손잡이를 잡아 주세요."

WAKE_GREETING = "네?"


def expects_answer(reply: str) -> bool:
    """이 발화가 사용자 답을 기대하는가 — 재청취 창을 열지 판정."""
    reply = reply.strip()
    return reply.endswith("?") or reply.endswith("말씀해 주세요.")


def all_phrases() -> dict[str, str]:
    """이 모듈이 가진 고정 문구 전체 (검사·감수용). 문구 묶음(tuple)도 편다."""
    phrases: dict[str, str] = {}
    for name, value in globals().items():
        if not name.isupper():
            continue
        if isinstance(value, str):
            phrases[name] = value
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    phrases[f"{name}[{i}]"] = item
    return phrases
