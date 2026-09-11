"""wakeword_gate 단위 테스트."""
import unittest

from src.wakeword_gate import FrameGate, match_emergency_transcript


class MatchTest(unittest.TestCase):
    def test_exact_keywords(self):
        self.assertEqual(match_emergency_transcript("멈춰"), "멈춰")
        self.assertEqual(match_emergency_transcript("정지"), "정지")
        self.assertEqual(match_emergency_transcript("스톱"), "스톱")
        self.assertEqual(match_emergency_transcript("스탑"), "스탑")

    def test_punctuation_and_space(self):
        self.assertEqual(match_emergency_transcript("멈춰!"), "멈춰")
        self.assertEqual(match_emergency_transcript(" 정지! "), "정지")
        self.assertEqual(match_emergency_transcript("스톱..."), "스톱")

    def test_repetition(self):
        self.assertEqual(match_emergency_transcript("멈춰 멈춰"), "멈춰")
        self.assertEqual(match_emergency_transcript("정지! 정지!"), "정지")
        self.assertEqual(match_emergency_transcript("멈춰멈춰멈춰"), "멈춰")

    def test_interjection_prefix(self):
        self.assertEqual(match_emergency_transcript("어어 멈춰!"), "멈춰")
        self.assertEqual(match_emergency_transcript("아 정지"), "정지")
        self.assertEqual(match_emergency_transcript("야 스톱"), "스톱")

    def test_variants_map_to_canonical(self):
        self.assertEqual(match_emergency_transcript("종지"), "정지")
        self.assertEqual(match_emergency_transcript("종지 종지"), "정지")
        self.assertEqual(match_emergency_transcript("중지"), "정지")
        self.assertEqual(match_emergency_transcript("맘차"), "멈춰")
        self.assertEqual(match_emergency_transcript("마음차"), "멈춰")

    def test_rejects_traps(self):
        for text in ("멈춤", "정지야", "정지선", "정지하", "정지하오",
                     "멈췄어요", "멈춰야 되나", "행정지원실", "멍청아",
                     "멍충멍충", "스톡 확인해줘"):
            self.assertIsNone(match_emergency_transcript(text), text)

    def test_rejects_keyword_in_sentence(self):
        for text in ("지금 당장 멈춰줘", "정지 화면이 왜 이래", "스톱워치 눌러"):
            self.assertIsNone(match_emergency_transcript(text), text)

    def test_empty_and_silence(self):
        self.assertIsNone(match_emergency_transcript(""))
        self.assertIsNone(match_emergency_transcript("   "))
        self.assertIsNone(match_emergency_transcript("어어"))

    def test_returned_keyword_is_hard_keyword(self):
        hard = {"멈춰", "정지", "스탑", "스톱", "안돼", "위험해"}
        for text in ("멈춰", "정지!", "스톱 스톱", "스탑", "종지", "맘차"):
            self.assertIn(match_emergency_transcript(text), hard)


class FrameGateTest(unittest.TestCase):
    def test_requires_persist_frames(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        self.assertFalse(g.feed(0.9, now=0.00))
        self.assertTrue(g.feed(0.9, now=0.08))

    def test_spike_does_not_fire(self):
        g = FrameGate(threshold=0.5, persist=2)
        self.assertFalse(g.feed(0.9, now=0.00))
        self.assertFalse(g.feed(0.1, now=0.08))
        self.assertFalse(g.feed(0.9, now=0.16))
        self.assertTrue(g.feed(0.9, now=0.24))

    def test_cooldown_blocks_refire(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        g.feed(0.9, now=0.00)
        self.assertTrue(g.feed(0.9, now=0.08))
        self.assertFalse(g.feed(0.9, now=0.16))
        self.assertFalse(g.feed(0.9, now=1.00))
        self.assertTrue(g.feed(0.9, now=2.20))

    def test_cooldown_then_new_shout_needs_persist_again(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        g.feed(0.9, now=0.00)
        self.assertTrue(g.feed(0.9, now=0.08))
        self.assertFalse(g.feed(0.1, now=0.16))
        self.assertFalse(g.feed(0.9, now=2.50))
        self.assertTrue(g.feed(0.9, now=2.58))

    def test_reset_clears_streak(self):
        g = FrameGate(threshold=0.5, persist=2)
        g.feed(0.9, now=0.00)
        g.reset()
        self.assertFalse(g.feed(0.9, now=0.08))
        self.assertTrue(g.feed(0.9, now=0.16))

    def test_below_threshold_never_fires(self):
        g = FrameGate(threshold=0.5, persist=2)
        for i in range(10):
            self.assertFalse(g.feed(0.49, now=i * 0.08))


if __name__ == "__main__":
    unittest.main()
