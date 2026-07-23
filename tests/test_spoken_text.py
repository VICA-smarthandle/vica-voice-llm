"""로봇이 말하는 문장에 긴급어가 섞이지 않는지 검사하는 회귀 테스트.

배경: emergency_monitor 는 마이크를 계속 열어 두므로 스피커로 나간 로봇 자기
목소리도 듣는다. 발화 문장에 긴급어가 들어 있으면 로봇이 자기 말을 듣고
EmergencyEvent 를 만드는 자가 트리거가 생긴다.

목적지 이름은 confirm_prompt / arrival_message 로 자동 확장돼 그대로 발화되므로
(destination_loader._fill_defaults) 함께 검사한다.

실행 (프로젝트 루트에서):
    .venv/bin/python -m unittest tests.test_spoken_text -v
"""
import unittest

from src.destination_loader import load_destinations
from src.emergency_filter import detect_emergency
from src.replies import all_phrases

# 목적지 항목 중 실제로 사용자에게 들려주는 필드.
SPOKEN_FIELDS = ("name", "confirm_prompt", "arrival_message", "unavailable_reason")


class SpokenTextTest(unittest.TestCase):
    def test_fixed_phrases_have_no_keyword(self):
        """고정 문구 전체를 훑는다. 새 문구가 추가돼도 자동으로 포함된다."""
        offenders = []
        for name, text in sorted(all_phrases().items()):
            keyword = detect_emergency(text)
            if keyword:
                offenders.append(f"{name}={text!r} <- '{keyword}'")

        self.assertEqual(
            offenders,
            [],
            "고정 문구에 긴급어가 있어 자가 트리거가 발생한다:\n  "
            + "\n  ".join(offenders),
        )

    def test_phrase_collection_is_not_empty(self):
        """introspection 이 조용히 0건이 되어 위 검사가 무력화되는 것을 막는다."""
        self.assertGreaterEqual(len(all_phrases()), 3)

    def test_destination_spoken_fields_have_no_keyword(self):
        offenders = []
        for dest in load_destinations():
            for field in SPOKEN_FIELDS:
                text = getattr(dest, field, None) or ""
                keyword = detect_emergency(text)
                if keyword:
                    offenders.append(f"{dest.id}.{field}={text!r} <- '{keyword}'")

        self.assertEqual(
            offenders,
            [],
            "목적지 발화 문구에 긴급어가 있어 자가 트리거가 발생한다:\n  "
            + "\n  ".join(offenders),
        )

    def test_place_name_containing_keyword_is_safe(self):
        """낱말 속에 긴급어 글자가 들어간 장소명이 오탐을 내지 않는지 고정한다.

        "행정지원실"(행정+지원실) 안의 "정지"가 대표 사례다. 경계 판정이 다시
        느슨해지면 도착 안내를 하다가 스스로 비상정지가 걸린다.
        """
        name = "행정대학건물 1층 행정지원실"
        self.assertIn(name, {dest.name for dest in load_destinations()})
        self.assertIsNone(detect_emergency(f"{name} 앞에 도착했습니다."))


if __name__ == "__main__":
    unittest.main()
