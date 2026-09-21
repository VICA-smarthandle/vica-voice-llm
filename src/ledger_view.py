"""대장 방송(/vica/robot_state) → 지시문 맨 뒤 `[지금 상황]` 블록 (스펙 3.2).

미션이 적은 사실을 옮겨 적을 뿐 판단하지 않는다. 옛 미션(대장 칸이 비어 옴)이면 빈
문자열을 돌려주고, 노드는 goal-event 추정 상황판(situation_board)으로 폴백한다.
"""
from __future__ import annotations

from .schema import RobotState

DIALOG_KO = {
    "idle": "대기(안내 없음)",
    "awaiting_user": "인사 답 대기",
    "confirming": "목적지 확인 대기",
    "seeking": "호출 방향으로 회전 중",
    "turning": "회전 중",
    "approaching": "사람에게 다가가는 중",
    "navigating": "안내 중(주행)",
    "paused": "일시정지",
    "arrived": "도착",
    "asking_next": "도착 질문 중",
    "asking_wait_time": "대기 시간 질문 중",
    "waiting": "대기 중",
    "returning": "제자리로 복귀 중",
    "estopped": "비상 정지",
    "failed": "이동 실패",
}

HEADER = ("\n[지금 상황] (미션이 확인한 사실 — 대화 이력이 비어 있어도 이것은 맞다. "
          "\"지금 어디 가?\"·\"아까 어디 갔었지?\"·\"몇 층이야?\"·\"몇 시야?\"는 이것으로 답한다. "
          "대화 이력과 다르면 이것이 맞다)\n")

FLOOR_LABEL = "건물/층"   # 대장만 항상 내는 줄. parser 가 이 상수로 "대장이 왔는가"를 판정한다.


def _ago(sec: int) -> str:
    if sec < 0:
        return "시각 모름"
    if sec < 60:
        return "방금"
    if sec < 3600:
        return f"{sec // 60}분 전"
    return f"{sec // 3600}시간 전"


def _left(sec: int) -> str:
    return f"{sec // 60}분 {sec % 60}초" if sec >= 60 else f"{sec}초"


def render_ledger(state: RobotState, *, awaiting_answer: bool, now_text: str) -> str:
    if not state.dialog_state:
        return ""
    lines = []
    if state.current_building or state.current_floor is not None:
        floor = f" {state.current_floor}층" if state.current_floor is not None else ""
        lines.append(f"- {FLOOR_LABEL}: {state.current_building or '건물 모름'}{floor}")
    else:
        lines.append(f"- {FLOOR_LABEL}: 모름")
    lines.append(f"- 지금 있는 곳: {state.place_here or '위치 미확인'}")
    if state.active_destination:
        lines.append(f"- 안내 중: {state.active_destination}로 이동 중")
    else:
        lines.append("- 안내 중: 없음")
    if state.last_destination:
        lines.append(f"- 직전에 간 곳: {state.last_destination} ({_ago(state.last_arrived_age_sec)} 도착)")
    else:
        lines.append("- 직전에 간 곳: 아직 없음")
    if state.aborted_destination:
        lines.append(f"- 하려다 만 곳: {state.aborted_destination}")
    lines.append(f"- 대화 단계: {DIALOG_KO.get(state.dialog_state, state.dialog_state)}")
    if state.wait_minutes >= 0:
        left = f", {_left(state.wait_left_sec)} 남음" if state.wait_left_sec >= 0 else ""
        lines.append(f"- 대기: {state.wait_minutes}분 요청{left}")
    else:
        lines.append("- 대기: 없음")
    lines.append(f"- 시각: {now_text}")
    lines.append(f"- 배터리: {state.battery_pct}%" if state.battery_pct >= 0 else "- 배터리: 모름")
    if awaiting_answer:
        lines.append("- 로봇이 방금 질문하고 답을 기다리는 중: 예 (지금 들리는 말은 그 답일 가능성이 높다)")
    return HEADER + "\n".join(lines) + "\n"
