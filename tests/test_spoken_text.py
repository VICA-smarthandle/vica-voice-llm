"""로봇이 말하는 문장에 긴급어가 섞이지 않는지 검사하는 회귀 테스트."""
import unittest

from src.destination_loader import load_destinations
from src.emergency_filter import detect_emergency
from src.replies import all_phrases

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
        """낱말 속에 긴급어 글자가 들어간 장소명이 오탐을 내지 않는지 고정한다."""
        name = "행정대학건물 1층 행정지원실"
        self.assertIn(name, {dest.name for dest in load_destinations()})
        self.assertIsNone(detect_emergency(f"{name} 앞에 도착했습니다."))


if __name__ == "__main__":
    unittest.main()
