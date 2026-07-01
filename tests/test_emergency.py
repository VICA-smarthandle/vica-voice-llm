"""emergency_filter 단위 테스트.

실행 (프로젝트 루트에서):
    .venv/bin/python -m unittest tests.test_emergency -v
"""
import unittest

from src.emergency_filter import detect_emergency


class EmergencyTest(unittest.TestCase):
    def test_detects_basic(self):
        self.assertEqual(detect_emergency("멈춰"), "멈춰")
        self.assertEqual(detect_emergency("정지"), "정지")

    def test_detects_in_sentence(self):
        self.assertEqual(detect_emergency("지금 당장 멈춰줘"), "멈춰")
        self.assertEqual(detect_emergency("좀 천천히 가자"), "천천히")

    def test_no_false_positive(self):
        self.assertIsNone(detect_emergency("407호 데려다줘"))
        self.assertIsNone(detect_emergency("배 아파"))

    def test_empty(self):
        self.assertIsNone(detect_emergency(""))


if __name__ == "__main__":
    unittest.main()
