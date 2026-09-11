"""emergency_filter 단위 테스트."""
import unittest

from src.emergency_filter import (
    EMERGENCY_KEYWORDS,
    SOFT_KEYWORDS,
    detect_emergency,
)


class EmergencyTest(unittest.TestCase):
    def test_detects_basic(self):
        self.assertEqual(detect_emergency("멈춰"), "멈춰")
        self.assertEqual(detect_emergency("정지"), "정지")

    def test_detects_in_sentence(self):
        self.assertEqual(detect_emergency("지금 당장 멈춰줘"), "멈춰")
        self.assertEqual(detect_emergency("정지해 주세요"), "정지")

    def test_detects_across_spacing(self):
        """STT 가 띄어쓰기를 다르게 내도 같은 판정이어야 한다."""
        self.assertEqual(detect_emergency("안돼요"), "안돼")
        self.assertEqual(detect_emergency("안 돼요"), "안돼")

    def test_detects_after_an_interjection(self):
        """위급하면 사람은 긴급어만 딱 외치지 않고 감탄사를 붙인다."""
        self.assertEqual(detect_emergency("아 안 돼"), "안돼")
        self.assertEqual(detect_emergency("야 안 돼"), "안돼")
        self.assertEqual(detect_emergency("아 안 돼 안 돼 안 돼"), "안돼")
        self.assertEqual(detect_emergency("아, 안 돼!"), "안돼")

    def test_detects_interjection_with_other_keywords(self):
        """"멈춰" 계열은 원래도 잡혔다. 회귀로 빠지지 않는지 확인한다."""
        self.assertEqual(detect_emergency("야 멈춰"), "멈춰")
        self.assertEqual(detect_emergency("어어 멈춰"), "멈춰")
        self.assertEqual(detect_emergency("잠깐 멈춰봐"), "멈춰")
        self.assertEqual(detect_emergency("아 위험해"), "위험해")

    def test_no_false_positive(self):
        self.assertIsNone(detect_emergency("407호 데려다줘"))
        self.assertIsNone(detect_emergency("배 아파"))

    def test_keyword_inside_word_is_not_emergency(self):
        """낱말 속에 우연히 들어간 글자로 비상정지가 걸리면 안 된다."""
        self.assertIsNone(detect_emergency("행정대학건물 1층 행정지원실"))
        self.assertIsNone(detect_emergency("행정지원실로 가줘"))
        self.assertIsNone(detect_emergency("감정지수가 뭐야"))
        self.assertIsNone(detect_emergency("결정지어 주세요"))

    def test_no_false_positive_on_measured_transcriptions(self):
        """실기에서 "행정지원실"을 말했을 때 whisper 가 실제로 낸 받아쓰기들."""
        for heard in [
            "행정지원",
            "행정지",
            "행동 지원실",
            "청중지원지",
            "현중지원실",
            "탕정지원시?",
            "생존 지원 주의",
            "생동지원증",
            "행정지원실에 가자",
            "행정지원실로 가주세요",
            "MBC 뉴스 이덕영입니다.",
            "시청해주셔서 감사합니다.",
            "다음 영상에서 만나요",
        ]:
            self.assertIsNone(detect_emergency(heard), heard)

    def test_soft_words_are_not_emergency(self):
        """감속 요청은 정지가 아니다."""
        self.assertIsNone(detect_emergency("좀 천천히 가자"))
        self.assertIsNone(detect_emergency("잠깐만요"))
        self.assertIsNone(detect_emergency("느리게 가주세요"))

    def test_soft_and_hard_lists_are_disjoint(self):
        self.assertEqual(set(EMERGENCY_KEYWORDS) & set(SOFT_KEYWORDS), set())

    def test_hard_list_matches_mission_manager_contract(self):
        """정본은 vica_ros2_ws 의 mission_logic.HARD_EMERGENCY_KEYWORDS 다."""
        self.assertEqual(
            set(EMERGENCY_KEYWORDS),
            {"멈춰", "정지", "스탑", "스톱", "안돼", "위험해"},
        )

    def test_empty(self):
        self.assertIsNone(detect_emergency(""))


if __name__ == "__main__":
    unittest.main()
