"""대장 방송(RobotState) → 지시문 [지금 상황] 블록. 옛 미션(칸 없음)이면 빈 문자열."""
from types import SimpleNamespace

from src.ros_convert import msg_to_robot_state
from src.schema import RobotState


def _msg(**kw):
    base = dict(current_floor=4, current_building="로봇관", is_moving=False, is_paused=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestConvert:
    def test_new_fields_are_converted(self):
        st = msg_to_robot_state(_msg(dialog_state="waiting", place_here="407호 앞", place_here_dist_m=1.5,
                                     active_destination="", last_destination="407호", last_arrived_age_sec=660,
                                     aborted_destination="화장실", wait_minutes=10, wait_left_sec=340, battery_pct=-1))
        assert st.dialog_state == "waiting" and st.place_here == "407호 앞" and st.last_destination == "407호"
        assert st.wait_minutes == 10 and st.wait_left_sec == 340 and st.battery_pct == -1
        assert st.current_floor == 4

    def test_old_message_without_fields_still_converts(self):
        st = msg_to_robot_state(_msg())
        assert st.dialog_state == "" and st.place_here == "" and st.last_arrived_age_sec == -1
        assert st == RobotState(current_floor=4, current_building="로봇관")


from src.ledger_view import render_ledger  # noqa: E402


class TestRender:
    def test_full_board(self):
        st = RobotState(current_floor=4, current_building="로봇관", dialog_state="waiting", place_here="407호 앞",
                        place_here_dist_m=1.5, last_destination="407호", last_arrived_age_sec=660,
                        aborted_destination="화장실", wait_minutes=10, wait_left_sec=340, battery_pct=-1)
        text = render_ledger(st, awaiting_answer=False, now_text="15:40")
        assert text.startswith("\n[지금 상황]")
        for line in ("- 건물/층: 로봇관 4층", "- 지금 있는 곳: 407호 앞", "- 안내 중: 없음",
                     "- 직전에 간 곳: 407호 (11분 전 도착)", "- 하려다 만 곳: 화장실",
                     "- 대화 단계: 대기 중", "- 대기: 10분 요청, 5분 40초 남음",
                     "- 시각: 15:40", "- 배터리: 모름"):
            assert line in text, line

    def test_navigating_and_unknowns(self):
        st = RobotState(dialog_state="navigating", active_destination="화장실", is_moving=True)
        text = render_ledger(st, awaiting_answer=True, now_text="09:01")
        assert "- 건물/층: 모름" in text and "- 지금 있는 곳: 위치 미확인" in text
        assert "- 안내 중: 화장실로 이동 중" in text and "- 직전에 간 곳: 아직 없음" in text
        assert "- 대화 단계: 안내 중(주행)" in text and "- 대기: 없음" in text
        assert "답을 기다리는 중: 예" in text

    def test_old_mission_gives_empty(self):
        assert render_ledger(RobotState(current_floor=4), awaiting_answer=False, now_text="09:01") == ""

    def test_unknown_state_word_passes_through(self):
        text = render_ledger(RobotState(dialog_state="seeking_x"), awaiting_answer=False, now_text="09:01")
        assert "- 대화 단계: seeking_x" in text

    def test_ago_negative_is_unknown(self):
        st = RobotState(dialog_state="idle", last_destination="407호", last_arrived_age_sec=-1)
        text = render_ledger(st, awaiting_answer=False, now_text="09:01")
        assert "시각 모름" in text
        assert "방금" not in text


class TestWaitSpotLines:
    """대기 장소 (2026-10-07): 칸 2개를 받아 메모에 줄 2개, 새 상태 3개, idle 은 "안내 없음"."""

    def test_new_fields_are_converted(self):
        st = msg_to_robot_state(_msg(dialog_state="waiting", door_side="오른쪽",
                                     wait_place="입구 오른쪽"))
        assert st.door_side == "오른쪽" and st.wait_place == "입구 오른쪽"
        old = msg_to_robot_state(_msg(dialog_state="waiting"))
        assert old.door_side == "" and old.wait_place == ""

    def test_idle_reads_no_guidance(self):
        text = render_ledger(RobotState(dialog_state="idle"), awaiting_answer=False, now_text="09:01")
        assert "- 대화 단계: 안내 없음" in text

    def test_new_states_are_korean(self):
        for state, word in (("waiting_release", "손 놓기 기다림"),
                            ("moving_to_wait_spot", "대기 장소로 혼자 이동 중"),
                            ("moving_back_to_dest", "입구 앞으로 돌아가는 중")):
            text = render_ledger(RobotState(dialog_state=state), awaiting_answer=False, now_text="09:01")
            assert word in text, state
            assert state not in text, state

    def test_door_and_wait_place_lines(self):
        st = RobotState(dialog_state="waiting", last_destination="화장실", last_arrived_age_sec=30,
                        door_side="오른쪽", wait_minutes=10, wait_left_sec=500,
                        wait_place="입구 오른쪽")
        text = render_ledger(st, awaiting_answer=False, now_text="09:01")
        assert "- 화장실 방향: 오른쪽" in text
        assert "- 대기 장소: 입구 오른쪽" in text
        # 방향 줄은 직전에 간 곳 바로 다음이다(같은 곳의 사실끼리 붙인다).
        assert text.index("- 직전에 간 곳") < text.index("- 화장실 방향")

    def test_no_lines_when_empty(self):
        st = RobotState(dialog_state="waiting", last_destination="화장실", wait_minutes=10)
        text = render_ledger(st, awaiting_answer=False, now_text="09:01")
        assert "방향:" not in text and "- 대기 장소" not in text
