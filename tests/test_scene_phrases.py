"""장면 분석기가 쓰는 미션 문장 사본이 미션 파일에 글자 그대로 있는가 (R2 — src/mission_phrases.py 는 안 건드림)."""
from pathlib import Path

import pytest

from tools.scene import phrases

MISSION = (Path(__file__).resolve().parents[2] / "vica_ros2_ws" / "src" / "vica_mission_manager"
           / "vica_mission_manager" / "mission_logic.py")


@pytest.mark.parametrize("sentence", phrases.MISSION_COPIES)
def test_copy_appears_verbatim_in_mission(sentence):
    if not MISSION.exists():
        pytest.skip("옆 저장소 vica_ros2_ws 가 없다")
    assert f'"{sentence}"' in MISSION.read_text(encoding="utf-8")
