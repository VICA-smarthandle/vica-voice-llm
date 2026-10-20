"""미션이 말하는 문장의 음성 쪽 사본 — 미리 합성·녹음 굽기용 (2026-10-07 대기 장소)."""
from __future__ import annotations

from typing import Iterable

ARRIVED_FALLBACK = "{name}에 도착했습니다."
DOOR_SIDE = "{name}{eun} {side}에 있습니다."
DOOR_SIDE_WORDS = ("앞", "뒤", "오른쪽", "왼쪽")
ASK_RESTROOM = "다녀오시는 동안 여기서 기다릴까요?"
ASK_ENTRANCE = "여기까지 안내를 마칠까요?"
ASK_GENERIC = "여기서 대기할까요?"
ASK_WAIT_TIME = "몇 분쯤 걸리실까요?"
ARRIVAL_QUESTIONS = (ASK_RESTROOM, ASK_ENTRANCE, ASK_GENERIC, ASK_WAIT_TIME)

WAIT_SPOT_CONFIRM = (
    "{minutes}분 동안 {place}에서 기다리겠습니다. 돌아오시면 '비카야'라고 불러 주세요.")
WAIT_SPOT_DEFAULT = (
    "최대 30분 동안 {place}에서 기다리겠습니다. 돌아오시면 '비카야'라고 불러 주세요.")
WAIT_PLACE_PHRASES = {"right": "입구 오른쪽", "left": "입구 왼쪽", "across": "입구 맞은편"}
BAKED_WAIT_MINUTES = (5, 10, 15, 20, 30)
WAIT_BEACON = "비카가 대기 중입니다."
WAIT_SPOT_BLOCKED = "대기 자리가 막혀 입구 앞에서 기다리겠습니다."
WAIT_EXPIRED = "대기 시간이 종료되어 제자리로 돌아갑니다."
WAIT_FINISH_ASK = "네, 어디로 모실까요?"
ESTOP_WAKE = "지금은 비상 멈춤 상태입니다."
OBSTACLE_AVOID = "앞에 장애물이 있어 비켜갈게요."
OBSTACLE_SLOW = "앞에 장애물이 있어 감속할게요."
OBSTACLE_STOP = "앞에 장애물이 있어 정지합니다."
APPROACH_REASK = "안내를 받으시겠어요?"
APPROACH_UNANSWERED = "필요하시면 '비카야'라고 불러 주세요."

WAIT_NEED_ASK = "안내가 필요 없으신가요?"
CONFIRM_SWITCH = "네, {prompt}"
WAIT_PLACE_AT_DESTINATION = "입구 앞"
HELP_CONFIRM = "직원의 도움이 필요하신가요?"
HELP_CANCEL_CONFIRM = "도움 요청을 취소할까요?"
HELP_SENTENCES = (
    HELP_CONFIRM,
    "지금 도움 요청을 보냈습니다",
    "관리자가 오고있습니다. 잠시만 기다려주세요",
    HELP_CANCEL_CONFIRM,
    "도움 요청을 취소했습니다.",
    "지금은 도움 요청을 보낼 수 없습니다",
)

_DIGIT_HAS_BATCHIM = {
    "0": True, "1": True, "2": False, "3": True, "4": False,
    "5": False, "6": True, "7": True, "8": True, "9": False,
}


def josa_eun_neun(word: str) -> str:
    """'은 / 는' — 미션 mission_logic.josa_eun_neun 사본(같은 답을 내야 미리 합성이 맞는다)."""
    word = (word or "").rstrip()
    if not word:
        return "는"
    last = word[-1]
    if "가" <= last <= "힣":
        return "은" if (ord(last) - 0xAC00) % 28 else "는"
    if last in _DIGIT_HAS_BATCHIM:
        return "은" if _DIGIT_HAS_BATCHIM[last] else "는"
    return "는"


def door_side_sentences(name: str) -> list[str]:
    """한 목적지의 M1 네 문장(앞·뒤·오른쪽·왼쪽)."""
    eun = josa_eun_neun(name)
    return [DOOR_SIDE.format(name=name, eun=eun, side=side) for side in DOOR_SIDE_WORDS]


def wait_spot_sentences() -> dict[str, str]:
    """구워 둘 M2·M2′ 18문장. 파일 이름(확장자 없음) → 문장."""
    out: dict[str, str] = {}
    for side, place in WAIT_PLACE_PHRASES.items():
        for minutes in BAKED_WAIT_MINUTES:
            out[f"mission_msg_wait_spot_{side}_{minutes}"] = WAIT_SPOT_CONFIRM.format(
                minutes=minutes, place=place)
        out[f"mission_msg_wait_spot_{side}_default"] = WAIT_SPOT_DEFAULT.format(place=place)
    return out


def wait_front_sentences() -> dict[str, str]:
    """구워 둘 '입구 앞' M2·M2′ 6문장(2026-10-08 결정 1). 파일 이름 → 문장."""
    out = {
        f"mission_msg_wait_front_{minutes}": WAIT_SPOT_CONFIRM.format(
            minutes=minutes, place=WAIT_PLACE_AT_DESTINATION)
        for minutes in BAKED_WAIT_MINUTES
    }
    out["mission_msg_wait_front_default"] = WAIT_SPOT_DEFAULT.format(
        place=WAIT_PLACE_AT_DESTINATION)
    return out


def baked_mission_ments() -> dict[str, str]:
    """이번 작업에서 녹음으로 굽는 미션 문장 전부(작업 계획 탭 '소리 준비')."""
    out = {
        "mission_msg_wait_beacon": WAIT_BEACON,
        "mission_msg_wait_spot_blocked": WAIT_SPOT_BLOCKED,
        "mission_msg_wait_expired": WAIT_EXPIRED,
        "mission_msg_estop_wake": ESTOP_WAKE,
        "mission_msg_wait_finish_ask": WAIT_FINISH_ASK,
        "mission_msg_wait_need_ask": WAIT_NEED_ASK,
        "mission_msg_obstacle_avoid": OBSTACLE_AVOID,
        "mission_msg_obstacle_slow": OBSTACLE_SLOW,
        "mission_msg_obstacle_stop": OBSTACLE_STOP,
        "mission_msg_approach_reask": APPROACH_REASK,
        "mission_msg_approach_unanswered": APPROACH_UNANSWERED,
    }
    out.update(wait_spot_sentences())
    out.update(wait_front_sentences())
    return out


def _unique(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [p for p in phrases if p and not (p in seen or seen.add(p))]


def standalone_prewarm(destinations: Iterable) -> list[str]:
    '''혼자 말해지는 문장 — 목적지 확인 질문과, 확인 질문 중 다른 목적지로 다시 묻는 "네, …"'''
    dests = list(destinations)
    phrases = [d.confirm_prompt for d in dests]
    phrases += [CONFIRM_SWITCH.format(prompt=d.confirm_prompt) for d in dests if d.confirm_prompt]
    phrases += list(HELP_SENTENCES)
    return _unique(phrases)


def merged_prewarm(destinations: Iterable) -> list[str]:
    """도착 발화에 합쳐 나가는 문장 — 도착 멘트 → 도착 질문 → M1(입구 방향이 있는 목적지만)."""
    dests = list(destinations)
    phrases: list[str] = [d.arrival_message for d in dests]
    phrases += list(ARRIVAL_QUESTIONS)
    for d in dests:
        if getattr(d, "door_yaw", None) is not None and d.is_approachable:
            phrases += door_side_sentences(d.name)
    return _unique(phrases)
