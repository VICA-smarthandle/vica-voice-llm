"""예약 묶음 38개(설계 2026-11-18 7절) — 도구 출력의 켜는 기준을 재는 발화.

첫 요청 15(R1~R12 · E1~E3) · 그 밖 13(N1~N13) · 이어진 대답 10(F1~F10).
R·N 은 11-18 측정(~/vica_data/tool_call_test_20261118/measure.py)의 발화 그대로다(R8 의 이름만 정현우).
기준일은 NOW(2026-11-18 수요일) — "내일" = 11-19, "모레" = 11-20, "다음 주 월요일" = 11-23.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src import reservation_phrases as p
from src.reservation_flow import ReserveCall

NOW = datetime(2026, 11, 18, 8, 0)      # 수요일 아침 — 그날의 모든 시각이 아직 지나지 않았다
CONFIRM = "confirm"                     # expect["say"] — 확인 질문(문장 ②)이 나와야 한다


@dataclass(frozen=True)
class Case:
    id: str
    setup: tuple        # 그 발화 전에 진행 상태에 넣을 (ReserveCall, 글자) 들. 빈 tuple = 첫 요청
    text: str
    expect: dict        # reserve · facility · date · start · end · name · say · saved · active · intent · destination


def _first(case_id: str, text: str, facility: str, date: str = "", start: str = "", end: str = "",
           name: str = "") -> Case:
    return Case(case_id, (), text, {"reserve": True, "facility": facility, "date": date, "start": start,
                                    "end": end, "name": name})


def _other(case_id: str, text: str, intents: tuple, destination: str = "") -> Case:
    return Case(case_id, (), text, {"reserve": False, "intent": intents, "destination": destination})


# 이어진 대답의 앞 상황 — 흐름에 차례로 넣으면 그 질문이 열린다.
ASKED_NAME = ((ReserveCall(facility="세미나실", date="내일", start_time="14:00"), "내일 오후 두 시에 세미나실 예약해 줘."),)   # ①
ASKED_WHEN = ((ReserveCall(facility="세미나실"), "세미나실 예약해 줘."),)                                               # ⑤
ASKED_WHERE = ((ReserveCall(date="내일", start_time="14:00"), "내일 오후 두 시에 예약해 줘."),)                           # ⑦
ASKED_CONFIRM = ASKED_NAME + ((ReserveCall(name="김민수"), "김민수"),)                                                 # ②

CASES: list[Case] = [
    _first("R1", "내일 오후 두 시에 세미나실 예약해 줘.", "세미나실", "2026-11-19", "14:00", "16:00"),
    _first("R2", "세미나실 좀 예약하고 싶은데, 모레 오전 열 시.", "세미나실", "2026-11-20", "10:00", "12:00"),
    _first("R3", "상담실 오늘 오후 네 시에 쓸 수 있어?", "상담실", "2026-11-18", "16:00", "18:00"),
    _first("R4", "내일 세 시에 상담실 잡아 줘.", "상담실", "2026-11-19", "15:00", "17:00"),
    _first("R5", "다음 주 월요일 오전 아홉 시 세미나실 예약.", "세미나실", "2026-11-23", "09:00", "11:00"),
    _first("R6", "세미나실 예약해 줘.", "세미나실"),
    _first("R7", "오늘 저녁 여섯 시 반에 세미나실 쓸 수 있을까?", "세미나실", "2026-11-18", "18:30", "20:30"),
    _first("R8", "정현우 이름으로 내일 오전 열한 시 상담실 예약해 줘.", "상담실", "2026-11-19", "11:00", "13:00", "정현우"),
    _first("R9", "모레 오후 한 시 반 세미나실 예약 부탁해요.", "세미나실", "2026-11-20", "13:30", "15:30"),
    _first("R10", "세미나실 내일 열 시부터 예약할게.", "세미나실", "2026-11-19", "10:00", "12:00"),
    _first("R11", "상담실 예약하고 싶어요. 내일 오후 다섯 시요.", "상담실", "2026-11-19", "17:00", "19:00"),
    _first("R12", "11월 20일 오후 두 시 세미나실 예약해 줘.", "세미나실", "2026-11-20", "14:00", "16:00"),
    _first("E1", "내일 두 시부터 세 시까지 세미나실 예약해 줘.", "세미나실", "2026-11-19", "14:00", "15:00"),
    _first("E2", "모레 오전 열 시부터 열두 시까지 상담실 예약해 줘.", "상담실", "2026-11-20", "10:00", "12:00"),
    _first("E3", "내일 오후 두 시에 한 시간만 세미나실 쓸게요.", "세미나실", "2026-11-19", "14:00", "15:00"),
    _other("N1", "화장실로 가자.", ("navigate",), "화장실"),
    _other("N2", "사회복지로 안내해 줘.", ("navigate",), "사회복지"),
    _other("N3", "세미나실로 가자.", ("navigate",), "세미나실"),
    _other("N4", "식당 가고 싶어.", ("navigate",), "식당"),
    _other("N5", "잠깐만.", ("pause",)),
    _other("N6", "다시 가자.", ("resume",)),
    _other("N7", "안내 취소해 줘.", ("cancel",)),
    _other("N8", "네.", ("affirm",)),
    _other("N9", "아니.", ("deny",)),
    _other("N10", "너 이름이 뭐야?", ("question",)),
    _other("N11", "도와줄 사람 불러 줘.", ("help",)),
    _other("N12", "세미나실이 어디 있어?", ("question", "navigate")),
    _other("N13", "예약은 됐고 화장실 가자.", ("navigate",), "화장실"),
    Case("F1", ASKED_NAME, "김민수", {"reserve": True, "name": "김민수", "say": CONFIRM}),
    Case("F2", ASKED_NAME, "이서연입니다.", {"reserve": True, "name": "이서연", "say": CONFIRM}),
    Case("F3", ASKED_NAME, "제 이름은 박준호예요.", {"reserve": True, "name": "박준호", "say": CONFIRM}),
    Case("F4", ASKED_NAME, "아니, 됐어.", {"say": p.NOT_RESERVED, "active": False, "saved": 0}),
    Case("F5", ASKED_NAME, "화장실 먼저 가자.", {"reserve": False, "intent": ("navigate",), "destination": "화장실",
                                           "active": False, "saved": 0}),
    Case("F6", ASKED_WHEN, "내일 오후 세 시", {"reserve": True, "facility": "세미나실", "date": "2026-11-19",
                                        "start": "15:00", "end": "17:00", "say": p.ASK_NAME}),
    Case("F7", ASKED_WHERE, "세미나실이요.", {"reserve": True, "facility": "세미나실", "date": "2026-11-19",
                                       "start": "14:00", "say": p.ASK_NAME}),
    Case("F8", ASKED_CONFIRM, "그래.", {"say": p.DONE, "saved": 1, "active": False}),
    Case("F9", ASKED_CONFIRM, "아니요.", {"say": p.NOT_RESERVED, "saved": 0, "active": False}),
    Case("F10", ASKED_CONFIRM, "아니 네 시로 해 줘.", {"reserve": True, "name": "김민수", "start": "16:00", "end": "18:00",
                                               "say": CONFIRM, "saved": 0}),
]
