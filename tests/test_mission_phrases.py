"""미션 문장 ↔ 음성 쪽 사본 계약 (2026-10-07 대기 장소, 작업 계획 탭 '4. 음성 — 시험').

TTS 는 요청 문장이 글자 하나까지 같아야 구운 녹음·미리 합성을 쓴다. 다르면 조용히
그때 합성으로 넘어가 0.9초쯤 늦는다. 그래서 세 가지를 미션 파일과 직접 대조한다.
  1. 미션 문장(MSG_*·"네?"·비상 한 마디)과 음성 쪽 사본·녹음 목록이 글자까지 같은가
  2. 두 저장소의 은/는(그리고 로/으로) 함수가 같은 답을 내는가
  3. 미션 상태(dialog_state) 전부가 LLM 단어장(DIALOG_KO)에 있는가
미션 로직은 ROS 없이 불러올 수 있다(순수 파이썬). 옆 저장소가 없는 곳에서는 건너뛴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from src import mission_phrases as mp
from src import replies
from src.destination_loader import _josa_euro, load_destinations
from src.ledger_view import DIALOG_KO
from src.ment_cache import ASSETS_DIR
from src.schema import DestinationData
from src.tts_queue import AMBIENT, PRIORITIES
from src.tts_text import split_sentences

MISSION_PKG = (Path(__file__).resolve().parents[2]
               / "vica_ros2_ws" / "src" / "vica_mission_manager")
DESTINATION_ROOT = Path.home() / "vica_data" / "destinations"


@pytest.fixture(scope="module")
def ml():
    if not (MISSION_PKG / "vica_mission_manager" / "mission_logic.py").exists():
        pytest.skip("옆 저장소 vica_ros2_ws 가 없다 — 계약 대조는 두 저장소가 함께 있는 곳에서")
    sys.path.insert(0, str(MISSION_PKG))
    try:
        from vica_mission_manager import mission_logic
    finally:
        sys.path.remove(str(MISSION_PKG))
    return mission_logic


def _names_on_disk() -> list[str]:
    names: list[str] = []
    for path in sorted(DESTINATION_ROOT.glob("*/destinations.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names += [str(d.get("name", "")) for d in raw.get("destinations", []) or []]
    return names


JOSA_WORDS = [
    "화장실", "식당", "안내센터", "407호", "방2", "방3", "테스트1", "회의실B1", "B1",
    "ATM", "남자 화장실", "회의실(소)", "엘리베이터", "출입구", "1", "", " ",
]


class TestSentences:
    def test_arrival_and_question_constants(self, ml):
        assert mp.ARRIVED_FALLBACK == ml.MSG_ARRIVED_FALLBACK
        assert mp.DOOR_SIDE == ml.MSG_DOOR_SIDE
        assert mp.ASK_RESTROOM == ml.MSG_ASK_RESTROOM
        assert mp.ASK_ENTRANCE == ml.MSG_ASK_ENTRANCE
        assert mp.ASK_GENERIC == ml.MSG_ASK_GENERIC
        assert mp.ASK_WAIT_TIME == ml.MSG_ASK_WAIT_TIME

    def test_wait_spot_constants(self, ml):
        assert mp.WAIT_SPOT_CONFIRM == ml.MSG_WAIT_SPOT_CONFIRM
        assert mp.WAIT_SPOT_DEFAULT == ml.MSG_WAIT_SPOT_DEFAULT
        assert mp.WAIT_PLACE_PHRASES == ml.WAIT_PLACE_PHRASES
        assert mp.WAIT_BEACON == ml.MSG_WAIT_BEACON
        assert mp.WAIT_SPOT_BLOCKED == ml.MSG_WAIT_SPOT_BLOCKED
        assert mp.WAIT_EXPIRED == ml.MSG_WAIT_EXPIRED
        assert mp.ESTOP_WAKE == ml.MSG_ESTOP_WAKE
        assert mp.OBSTACLE_AVOID == ml.MSG_OBSTACLE_AVOID   # 주행 중 장애물 안내(2026-10-09)
        assert mp.OBSTACLE_SLOW == ml.MSG_OBSTACLE_SLOW
        assert mp.APPROACH_REASK == ml.MSG_APPROACH_REASK   # 접근 질문 다시 묻기(2026-10-09)
        assert mp.WAIT_FINISH_ASK == ml.MSG_WAIT_FINISH_ASK
        assert replies.WAKE_GREETING == ml.MSG_WAKE_GREETING
        # 빈 확인 문구를 미션이 직접 물을 때의 문장 = 음성 기본 확인 문구(미리 합성됨).
        dest = DestinationData(id="x", name="식당")
        from src.destination_loader import _fill_defaults
        assert _fill_defaults(dest).confirm_prompt == ml.say_destination(
            ml.MSG_CONFIRM_PROMPT_FALLBACK, "식당")

    def test_mission_reaction_sentences(self, ml):
        """미션 요청 반응표(2026-10-08) 새 문장·장소 말이 미션과 같은 글자다."""
        assert mp.WAIT_NEED_ASK == ml.MSG_WAIT_NEED_ASK
        assert mp.CONFIRM_SWITCH == ml.MSG_CONFIRM_SWITCH
        assert mp.WAIT_PLACE_AT_DESTINATION == ml.WAIT_PLACE_AT_DESTINATION
        # 안내 주행 중 "다시 가자"의 답 — 음성 replies.ALREADY_GOING 과 같은 글자.
        for name in ("409호", "식당", "화장실"):
            assert ml.say_destination(ml.MSG_ALREADY_GOING, name) == replies.ALREADY_GOING.format(
                cur=name, cur_josa=_josa_euro(name))

    def test_front_sentences_match_mission_formatting(self, ml):
        mission = {ml.MSG_WAIT_SPOT_CONFIRM.format(minutes=m, place=ml.WAIT_PLACE_AT_DESTINATION)
                   for m in mp.BAKED_WAIT_MINUTES}
        mission.add(ml.MSG_WAIT_SPOT_DEFAULT.format(place=ml.WAIT_PLACE_AT_DESTINATION))
        assert set(mp.wait_front_sentences().values()) == mission
        assert len(mission) == 6

    def test_switch_question_is_prewarmed(self):
        dests = [DestinationData(id="x", name="식당", confirm_prompt="식당으로 안내해드릴까요?")]
        assert "네, 식당으로 안내해드릴까요?" in mp.standalone_prewarm(dests)

    def test_door_side_words_cover_every_mission_answer(self, ml):
        seen = {ml.door_side_word(door, robot)
                for door in range(0, 360, 5) for robot in range(0, 360, 7)}
        assert seen == set(mp.DOOR_SIDE_WORDS)

    def test_m2_sentences_match_mission_formatting(self, ml):
        mission = set()
        for place in ml.WAIT_PLACE_PHRASES.values():
            for minutes in mp.BAKED_WAIT_MINUTES:
                mission.add(ml.MSG_WAIT_SPOT_CONFIRM.format(minutes=minutes, place=place))
            mission.add(ml.MSG_WAIT_SPOT_DEFAULT.format(place=place))
        assert set(mp.wait_spot_sentences().values()) == mission
        assert len(mission) == 18

    def test_beacon_goes_out_as_ambient(self, ml):
        assert AMBIENT in PRIORITIES
        logic = ml.MissionLogic()
        logic._wait_place = "spot"
        logic._beacon_next_at = 0.0
        acts = logic._beacon_tick(1.0)
        assert [(a.text, a.priority) for a in acts] == [(mp.WAIT_BEACON, AMBIENT)]


class TestJosaParity:
    def test_eun_neun_matches_mission(self, ml):
        for word in JOSA_WORDS + _names_on_disk():
            assert mp.josa_eun_neun(word) == ml.josa_eun_neun(word), word

    def test_euro_matches_mission(self, ml):
        for word in JOSA_WORDS + _names_on_disk():
            assert _josa_euro(word) == ml.josa_euro(word), word


class TestArrivalUtteranceIsPrewarmed:
    """미션이 실제로 내는 도착 발화를 문장으로 나누면, 전부 미리 합성 목록에 있다."""

    @pytest.mark.parametrize("category", ["restroom", "entrance", "reception"])
    def test_every_chunk_is_prewarmed(self, ml, category, tmp_path):
        mission_dest = ml.Destination(
            id="d1", name="화장실", pose=ml.Pose2D(3.0, 2.0, 0.0, "map"), calibrated=True,
            category=category, door_yaw_deg=90.0)
        logic = ml.MissionLogic(arrival_dialog=True)
        logic.on_intent(ml.IntentData(intent="navigate", matched_destination_id="d1",
                                      need_confirm=False, safety_flag="normal"),
                        mission_dest, None, True, 0.0)
        logic.robot_yaw_deg = 0.0
        says = [a for a in logic.on_tick(1.0, ml.NavStatus.SUCCEEDED) if isinstance(a, ml.Say)]
        assert len(says) == 1 and "에 있습니다." in says[0].text   # M1 이 합쳐 나갔다
        path = tmp_path / "destinations.yaml"
        path.write_text(yaml.safe_dump({"destinations": [
            {"id": "d1", "name": "화장실", "category2": category, "door_yaw": 90.0}]},
            allow_unicode=True), encoding="utf-8")
        warm = set(mp.merged_prewarm(load_destinations(path)))
        for chunk in split_sentences(says[0].text):
            assert chunk in warm, chunk

    def test_destination_without_door_yaw_gets_no_m1(self):
        dests = [DestinationData(id="a", name="식당", arrival_message="식당에 도착했습니다."),
                 DestinationData(id="b", name="화장실", door_yaw=180.0,
                                 arrival_message="화장실에 도착했습니다.")]
        warm = mp.merged_prewarm(dests)
        assert "화장실은 오른쪽에 있습니다." in warm
        assert not any(p.startswith("식당은") for p in warm)


class TestDialogWords:
    def test_every_mission_state_has_a_korean_word(self, ml):
        states = {s.value for s in ml.State} | {ml.DIALOG_GRIP_WAIT, ml.DIALOG_PAUSED_HANDLE}
        missing = states - set(DIALOG_KO)
        assert not missing, f"DIALOG_KO 에 없는 미션 상태: {sorted(missing)}"
        assert len(states) == 20


class TestBaked:
    """이번에 굽는 미션 문장이 전부 녹음 목록(manifest)에 같은 글자로 있다."""

    def test_baked_list_is_in_the_manifest(self):
        manifest = json.loads((ASSETS_DIR / "baked" / "manifest.json").read_text(encoding="utf-8"))
        for stem, text in mp.baked_mission_ments().items():
            assert manifest.get(f"{stem}.wav") == text, stem
            assert (ASSETS_DIR / "baked" / f"{stem}.wav").exists(), stem

    def test_wake_greeting_is_baked(self):
        manifest = json.loads((ASSETS_DIR / "baked" / "manifest.json").read_text(encoding="utf-8"))
        assert replies.WAKE_GREETING in manifest.values()
