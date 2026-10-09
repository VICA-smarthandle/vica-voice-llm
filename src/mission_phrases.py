"""미션이 말하는 문장의 음성 쪽 사본 — 미리 합성·녹음 굽기용 (2026-10-07 대기 장소).

정본은 vica_ros2_ws 의 vica_mission_manager/mission_logic.py MSG_* 상수다. TTS 는 요청
문장이 글자 하나까지 같아야 구운 녹음(MentCache)·미리 합성(SynthCache)을 쓰고, 다르면
조용히 그때 합성으로 넘어가 0.9초쯤 늦어진다. 그래서 미션이 만들 문장을 여기서 똑같이
만든다. 저장소가 달라 import 하지 않고 사본을 두며(destination_loader._josa_euro 와 같은
방식), tests/test_mission_phrases.py 가 미션 파일을 직접 읽어 글자를 대조한다.
"""
from __future__ import annotations

from typing import Iterable

# ---- 도착 (M1 은 도착 멘트·질문과 한 발화로 이어 말한다, 10-07 사용자 확정) -------------
# 도착 멘트가 빈 목적지의 미션 기본 문장. 앱이 이제 도착 멘트를 비워 저장한다.
ARRIVED_FALLBACK = "{name}에 도착했습니다."
# M1 — 입구 방향이 있는 목적지만. {eun} 은 josa_eun_neun 이 채운다.
DOOR_SIDE = "{name}{eun} {side}에 있습니다."
DOOR_SIDE_WORDS = ("앞", "뒤", "오른쪽", "왼쪽")
# 도착 후 유형별 질문과 시간 질문. 도착 멘트와 한 발화로 나가 문장별로 미리 합성한다.
ASK_RESTROOM = "다녀오시는 동안 여기서 기다릴까요?"
ASK_ENTRANCE = "여기까지 안내를 마칠까요?"
ASK_GENERIC = "여기서 대기할까요?"
ASK_WAIT_TIME = "몇 분쯤 걸리실까요?"
ARRIVAL_QUESTIONS = (ASK_RESTROOM, ASK_ENTRANCE, ASK_GENERIC, ASK_WAIT_TIME)

# ---- 대기 장소 멘트 (작업 계획 탭 '멘트') ------------------------------------------
# M2 / M2′ — {place} 는 WAIT_PLACE_PHRASES.
WAIT_SPOT_CONFIRM = (
    "{minutes}분 동안 {place}에서 기다리겠습니다. 돌아오시면 '비카야'라고 불러 주세요.")
WAIT_SPOT_DEFAULT = (
    "최대 30분 동안 {place}에서 기다리겠습니다. 돌아오시면 '비카야'라고 불러 주세요.")
WAIT_PLACE_PHRASES = {"right": "입구 오른쪽", "left": "입구 왼쪽", "across": "입구 맞은편"}
# 녹음으로 굽는 대기 시간(분). 그 밖의 시간은 그때 합성한다(대기 멘트 수준의 지연).
BAKED_WAIT_MINUTES = (5, 10, 15, 20, 30)
WAIT_BEACON = "비카가 대기 중입니다."               # M3 (대기 10초·홈 1분마다)
WAIT_SPOT_BLOCKED = "대기 자리가 막혀 입구 앞에서 기다리겠습니다."   # M6
WAIT_EXPIRED = "대기 시간이 종료되어 제자리로 돌아갑니다."         # M7
# 대기 중 돌아와 "다 됐어" — 다음 목적지를 묻는다(2026-10-07 사용자 결정).
WAIT_FINISH_ASK = "네, 어디로 모실까요?"
# 비상 정지 중 "비카야" — "네?" 없이 이 한 마디만(호출 반응표).
ESTOP_WAKE = "지금은 비상 멈춤 상태입니다."
# 주행 중 장애물 안내(2026-10-09 미션에 넣음, 사용자 선택 A1·S1). 미션 MSG_OBSTACLE_* 와 같은 글자.
OBSTACLE_AVOID = "앞에 장애물이 있어 피해 갈게요."
OBSTACLE_SLOW = "앞에 장애물이 있어 천천히 갈게요."
# 접근 질문을 한 번 다시 묻는 말(2026-10-09 사용자 결정). 미션 MSG_APPROACH_REASK 와 같은 글자.
APPROACH_REASK = "안내를 받으시겠어요?"

# ---- 미션 요청 반응표 (2026-10-08) --------------------------------------------------
# 대기 중 "취소" — 안내를 끝낼지 묻는다(사용자 결정 3, 사용자 문구).
WAIT_NEED_ASK = "안내가 필요 없으신가요?"
# 확인 질문 중 다른 목적지를 확정하면 "네, " + 그 목적지의 확인 질문(사용자 결정 4).
CONFIRM_SWITCH = "네, {prompt}"
# 대기 장소가 없는 목적지로 돌아가 기다릴 때(홈 가는 중 "기다려", 결정 1)의 장소 말 — 미션
# WAIT_PLACE_AT_DESTINATION. 대기 장소가 막혔을 때(M6 뒤) 상황판의 말과 같다.
WAIT_PLACE_AT_DESTINATION = "입구 앞"

# 숫자로 끝나는 이름을 읽을 때 마지막 숫자의 받침(영·일·삼·육·칠·팔 있음, 이·사·오·구 없음).
_DIGIT_HAS_BATCHIM = {
    "0": True, "1": True, "2": False, "3": True, "4": False,
    "5": False, "6": True, "7": True, "8": True, "9": False,
}


def josa_eun_neun(word: str) -> str:
    """'은 / 는' — 미션 mission_logic.josa_eun_neun 사본(같은 답을 내야 미리 합성이 맞는다).

    한글로 끝나면 받침 있음 → '은', 없음 → '는'. 숫자로 끝나면 읽는 소리의 받침을
    본다. 그 밖의 글자(영문 등)는 읽는 소리를 알 수 없어 '는'.
    """
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
    """이번 작업에서 녹음으로 굽는 미션 문장 전부(작업 계획 탭 '소리 준비').

    M3·M6·M7·비상 한 마디·대기 중 끝말 질문·M2/M2′ 18개. "네?"는 이미 구워져 있다
    (reply_wake_greeting).
    scripts/bake_one_cv.py --batch 가 이 표를 굽고, 시험이 manifest 에 다 있는지 본다.
    """
    out = {
        "mission_msg_wait_beacon": WAIT_BEACON,
        "mission_msg_wait_spot_blocked": WAIT_SPOT_BLOCKED,
        "mission_msg_wait_expired": WAIT_EXPIRED,
        "mission_msg_estop_wake": ESTOP_WAKE,
        "mission_msg_wait_finish_ask": WAIT_FINISH_ASK,
        # 미션 요청 반응표(2026-10-08) — 대기 중 "취소"의 질문(결정 3).
        "mission_msg_wait_need_ask": WAIT_NEED_ASK,
        # 주행 중 장애물 안내(2026-10-09) — ambient 라 바로 나가야 한다. 실시간 합성은 늦다.
        "mission_msg_obstacle_avoid": OBSTACLE_AVOID,
        "mission_msg_obstacle_slow": OBSTACLE_SLOW,
        # 접근 질문 다시 묻기(2026-10-09) — 첫 질문과 같은 목소리로 굽는다.
        "mission_msg_approach_reask": APPROACH_REASK,
    }
    out.update(wait_spot_sentences())
    out.update(wait_front_sentences())
    return out


def _unique(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [p for p in phrases if p and not (p in seen or seen.add(p))]


def standalone_prewarm(destinations: Iterable) -> list[str]:
    """혼자 말해지는 문장 — 목적지 확인 질문과, 확인 질문 중 다른 목적지로 다시 묻는 "네, …"
    (2026-10-08 결정 4). 통문장이 구워져 있으면 미리 합성할 필요가 없다."""
    dests = list(destinations)
    phrases = [d.confirm_prompt for d in dests]
    phrases += [CONFIRM_SWITCH.format(prompt=d.confirm_prompt) for d in dests if d.confirm_prompt]
    return _unique(phrases)


def merged_prewarm(destinations: Iterable) -> list[str]:
    """도착 발화에 합쳐 나가는 문장 — 도착 멘트 → 도착 질문 → M1(입구 방향이 있는 목적지만).

    미션은 "화장실에 도착했습니다. 화장실은 오른쪽에 있습니다. 다녀오시는 동안 여기서
    기다릴까요?"를 한 발화로 보내고, TTS 는 통문장 녹음이 없으면 문장마다 나눠 합성
    보관함만 찾는다. 그래서 이 문장들은 따로 구워져 있어도 문장 단위로 미리 합성한다.
    """
    dests = list(destinations)
    phrases: list[str] = [d.arrival_message for d in dests]
    phrases += list(ARRIVAL_QUESTIONS)
    for d in dests:
        if getattr(d, "door_yaw", None) is not None and d.is_approachable:
            phrases += door_side_sentences(d.name)
    return _unique(phrases)
