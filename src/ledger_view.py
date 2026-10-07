"""대장 방송(/vica/robot_state) → 지시문 맨 뒤 `[지금 상황]` 블록 (스펙 3.2).

미션이 적은 사실을 옮겨 적을 뿐 판단하지 않는다. 옛 미션(대장 칸이 비어 옴)이면 빈
문자열을 돌려주고, 노드는 goal-event 추정 상황판(situation_board)으로 폴백한다.
"""
from __future__ import annotations

from .schema import RobotState

DIALOG_KO = {
    # "대기(안내 없음)"였다. 대기 장소 작업(2026-10-07)으로 진짜 '대기'(waiting)가 셋
    # 생겨 헷갈리므로 "안내 없음"으로 바꿨다(사용자 결정).
    "idle": "안내 없음",
    "awaiting_user": "인사 답 대기",
    "confirming": "목적지 확인 대기",
    "seeking": "호출 방향으로 회전 중",
    "turning": "회전 중",
    "approaching": "사람에게 다가가는 중",
    "navigating": "안내 중(주행)",
    "paused": "일시정지",
    # 손잡이 두 단계(2026-09-30, ROS mission_logic DIALOG_GRIP_WAIT·DIALOG_PAUSED_HANDLE).
    # 손 놓쳐 선 것과 "잠깐"으로 선 것이 둘 다 "일시정지"로 보이면 "왜 안 가요?"에
    # "다시 가자라고 말하세요"로 엉뚱하게 답한다. 정본: 루트 docs/superpowers/specs/
    # 2026-09-28-touch-haptic-integration-final.md 4.7절.
    "grip_wait": "손잡이 잡기를 기다리는 중(잡으면 안내 시작)",
    "paused_handle": "손잡이를 놓쳐 멈춤(다시 잡으면 출발)",
    "arrived": "도착",
    "asking_next": "도착 질문 중",
    "asking_wait_time": "대기 시간 질문 중",
    "waiting": "대기 중",
    # 대기 장소 (2026-10-07, ROS mission_logic State 의 새 값 셋). 대기의 일부라 남은
    # 시간이 흐른다. 빠지면 영어 이름이 그대로 LLM 에 간다 — 시험이 미션 상태 전부를 본다.
    "waiting_release": "손 놓기 기다림(손을 놓으면 혼자 대기 장소로 감)",
    "moving_to_wait_spot": "대기 장소로 혼자 이동 중",
    "moving_back_to_dest": "대기 장소가 막혀 입구 앞으로 돌아가는 중",
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
    if state.door_side:
        # 도착할 때 미션이 정한 입구 쪽(로봇 = 뒤에서 손잡이를 잡은 사용자 기준). 도착
        # 멘트 M1 과 같은 값이라 "화장실 어디야?"에 같은 답이 나온다(2026-10-07).
        lines.append(f"- {state.last_destination or '목적지'} 방향: {state.door_side}")
    if state.aborted_destination:
        lines.append(f"- 하려다 만 곳: {state.aborted_destination}")
    lines.append(f"- 대화 단계: {DIALOG_KO.get(state.dialog_state, state.dialog_state)}")
    if state.wait_minutes >= 0:
        left = f", {_left(state.wait_left_sec)} 남음" if state.wait_left_sec >= 0 else ""
        lines.append(f"- 대기: {state.wait_minutes}분 요청{left}")
    else:
        lines.append("- 대기: 없음")
    if state.wait_place:
        # "어디서 기다린다고?"의 답. 대기 장소 없는 제자리 대기는 칸이 비어 줄이 없다.
        lines.append(f"- 대기 장소: {state.wait_place}")
    lines.append(f"- 시각: {now_text}")
    lines.append(f"- 배터리: {state.battery_pct}%" if state.battery_pct >= 0 else "- 배터리: 모름")
    if awaiting_answer:
        lines.append("- 로봇이 방금 질문하고 답을 기다리는 중: 예 (지금 들리는 말은 그 답일 가능성이 높다)")
    return HEADER + "\n".join(lines) + "\n"
