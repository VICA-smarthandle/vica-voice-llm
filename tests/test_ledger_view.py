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
