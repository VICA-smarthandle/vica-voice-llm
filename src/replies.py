"""로봇이 말하는 고정 문구 모음 (LLM 이 만들지 않는 말)."""
from __future__ import annotations

from . import vica_facts
from .destination_loader import _josa_euro
from .mission_phrases import josa_eun_neun

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

APPROACH_QUESTION = "안녕하세요? 시각장애인 안내로봇 비카입니다. 안내를 받으시겠어요?"
APPROACH_ONBOARDING = (
    "저에게 말을 거실 때는 '비카야'라고 불러주세요. 어디로 가고 싶으신가요?"
)

APPROACH_TURN_NOTICE = "네, 잠시만 기다려주세요. 로봇이 회전하니 주의하세요."
APPROACH_TURN_DONE = "회전이 완료되었습니다."
APPROACH_FAREWELL = "알겠습니다. 이만 물러납니다."

PAUSE_ACK = "네, 잠시 설게요."

USAGE_GUIDE = "안내 중에 잠깐 쉬려면 '잠깐만', 다시 가려면 '다시 가자', 그만두려면 '취소'라고 말씀해 주세요."

COMMAND_DECLINED = "알겠습니다. 계속 진행할게요."

CHANGE_DESTINATION_QUESTION = "지금 {cur}{cur_josa} 가는 중이에요. {new}{new_josa} 바꿀까요?"
CHANGE_DESTINATION_TAIL = "{new}{new_josa} 바꿀까요?"
ALREADY_GOING = "지금 {cur}{cur_josa} 가는 중이에요."

MODE_ASK = "손잡이를 잡고 스마트핸들 모드를 시작하시겠어요?"

MODE_ENTER_REQUEST = "손잡이를 3초간 잡아 주세요."

MODE_READY = "스마트핸들 모드입니다."

MODE_DECLINED = "알겠습니다."

HANDLE_UNAVAILABLE = "손잡이 연결에 문제가 있어 일반 안내로 진행합니다."

HANDLE_LOST = "손잡이를 잡아 주세요."

WAKE_GREETING = "네?"


INVITE_ENDINGS = ("말씀하세요", "말씀해주세요", "말씀을해주세요", "말씀해주시겠어요", "알려주세요",
                  "다시해주세요", "말해주세요")


def expects_answer(reply: str) -> bool:
    """이 발화가 사용자 답을 기대하는가 — 재청취 창을 열지 판정."""
    reply = reply.strip()
    if _about_sentence(reply):
        return False
    if reply.endswith("?"):
        return True
    tail = "".join(ch for ch in reply if not ch.isspace()).rstrip(".!~…")
    return tail.endswith(INVITE_ENDINGS)


def _about_sentence(reply: str) -> bool:
    """비카에 대한 정해진 대답(vica_facts.ABOUT_SENTENCES)과 글자까지 같은가 — 정해진 대답이 켜졌을 때만(fixed_answers.on())."""
    if reply not in vica_facts.ABOUT_SENTENCES.values():
        return False
    from . import fixed_answers
    return fixed_answers.on()


def choose_between(names) -> str:
    """두 곳 이상에 똑같이 걸리는 말에 되묻는 말 — "남자 화장실과 여자 화장실 중 어디로 갈까요?"(2026-10-11)."""
    names = [n for n in names if n]
    if len(names) == 2:
        a, b = names
        return f"{a}{'과' if josa_eun_neun(a) == '은' else '와'} {b} 중 어디로 갈까요?"
    return f"{', '.join(names)} 중 어디로 갈까요?"


NOT_AVAILABLE = "제가 아는 목적지 중에서 {what}{josa} 없는데, 어디로 안내해드릴까요?"
UNNAMED_PLACE = "말씀하신 곳"
_WANTED_MAX_CHARS = 20


def not_available(wanted: str) -> str:
    """없는 곳을 찾을 때 되묻는 말. 찾는 곳 이름이 비었거나 길거나 줄바꿈·물음표·긴급어가 있거나 한글이 없으면 '말씀하신 곳'으로 말한다."""
    from .emergency_filter import detect_emergency

    what = (wanted or "").strip()
    if (not what or len(what) > _WANTED_MAX_CHARS or "\n" in what or "?" in what or "？" in what
            or detect_emergency(what)
            or not any("가" <= ch <= "힣" for ch in what)):
        what = UNNAMED_PLACE
    return NOT_AVAILABLE.format(what=what, josa=josa_eun_neun(what))


PLACES_REPLY = "{names}{josa} 안내할 수 있습니다."
_PLACES_MAX = 3


def places_sentence(names) -> str:
    """갈 수 있는 곳을 알리는 말. 이름이 없으면 빈 글자(말하지 않는다 — 부르는 쪽이 지금 길로 둔다)."""
    names = [n for n in names if n]
    if not names:
        return ""
    head = ", ".join(names[:_PLACES_MAX]) + (" 등" if len(names) > _PLACES_MAX else "")
    return PLACES_REPLY.format(names=head, josa=_josa_euro(head))


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
