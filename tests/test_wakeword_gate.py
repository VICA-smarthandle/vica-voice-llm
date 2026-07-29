"""wakeword_gate 단위 테스트.

실행 (프로젝트 루트에서):
    .venv/bin/python -m unittest tests.test_wakeword_gate -v

사례는 실측(vica-wakeword/docs/stt-gate-findings.md)에서 실제로 나온 전사들이다.
"""
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
        # 위급 시 자연 발화 (녹음 상황 'repeat')
        self.assertEqual(match_emergency_transcript("멈춰 멈춰"), "멈춰")
        self.assertEqual(match_emergency_transcript("정지! 정지!"), "정지")
        self.assertEqual(match_emergency_transcript("멈춰멈춰멈춰"), "멈춰")

    def test_interjection_prefix(self):
        # 녹음 상황 'interjection' — 양성 처리 근거는 명세 8-3절
        self.assertEqual(match_emergency_transcript("어어 멈춰!"), "멈춰")
        self.assertEqual(match_emergency_transcript("아 정지"), "정지")
        self.assertEqual(match_emergency_transcript("야 스톱"), "스톱")

    def test_variants_map_to_canonical(self):
        # whisper 오전사 흡수 (실측 근거) — 정본 키워드로 돌려준다
        self.assertEqual(match_emergency_transcript("종지"), "정지")
        self.assertEqual(match_emergency_transcript("종지 종지"), "정지")
        self.assertEqual(match_emergency_transcript("중지"), "정지")
        self.assertEqual(match_emergency_transcript("맘차"), "멈춰")
        self.assertEqual(match_emergency_transcript("마음차"), "멈춰")

    def test_rejects_traps(self):
        # 라이브·오프라인에서 관문을 뚫었던 함정 전사들 — STT 층이 막아야 한다
        for text in ("멈춤", "정지야", "정지선", "정지하", "정지하오",
                     "멈췄어요", "멈춰야 되나", "행정지원실", "멍청아",
                     "멍충멍충", "스톡 확인해줘"):
            self.assertIsNone(match_emergency_transcript(text), text)

    def test_rejects_keyword_in_sentence(self):
        # 문장 속 키워드는 짧은 외침이 아니다 — 정확 매칭이 거른다
        for text in ("지금 당장 멈춰줘", "정지 화면이 왜 이래", "스톱워치 눌러"):
            self.assertIsNone(match_emergency_transcript(text), text)

    def test_empty_and_silence(self):
        self.assertIsNone(match_emergency_transcript(""))
        self.assertIsNone(match_emergency_transcript("   "))
        self.assertIsNone(match_emergency_transcript("어어"))  # 감탄사만

    def test_returned_keyword_is_hard_keyword(self):
        # 반환값은 로봇 쪽 정본(HARD_EMERGENCY_KEYWORDS)에 있어야 브리지가 동작한다
        hard = {"멈춰", "정지", "스탑", "스톱", "안돼", "위험해"}
        for text in ("멈춰", "정지!", "스톱 스톱", "스탑", "종지", "맘차"):
            self.assertIn(match_emergency_transcript(text), hard)


class FrameGateTest(unittest.TestCase):
    def test_requires_persist_frames(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        self.assertFalse(g.feed(0.9, now=0.00))   # 1프레임째 — 아직
        self.assertTrue(g.feed(0.9, now=0.08))    # 2연속 — 발동

    def test_spike_does_not_fire(self):
        g = FrameGate(threshold=0.5, persist=2)
        self.assertFalse(g.feed(0.9, now=0.00))   # 스파이크 한 번
        self.assertFalse(g.feed(0.1, now=0.08))   # 끊김 — 리셋
        self.assertFalse(g.feed(0.9, now=0.16))   # 다시 1프레임째
        self.assertTrue(g.feed(0.9, now=0.24))

    def test_cooldown_blocks_refire(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        g.feed(0.9, now=0.00)
        self.assertTrue(g.feed(0.9, now=0.08))
        # 같은 외침이 이어져도 쿨다운 안에서는 재발동하지 않는다
        self.assertFalse(g.feed(0.9, now=0.16))
        self.assertFalse(g.feed(0.9, now=1.00))
        # 외침이 '계속' 이어지는 중이면 쿨다운이 끝나는 즉시 재발동한다
        # (지속 조건은 이미 충족 상태 — "멈춰!!!" 를 계속 외치는 상황)
        self.assertTrue(g.feed(0.9, now=2.20))

    def test_cooldown_then_new_shout_needs_persist_again(self):
        g = FrameGate(threshold=0.5, persist=2, cooldown_sec=2.0)
        g.feed(0.9, now=0.00)
        self.assertTrue(g.feed(0.9, now=0.08))
        self.assertFalse(g.feed(0.1, now=0.16))    # 외침 끝 — 조용해짐
        # 쿨다운이 지난 뒤 새 외침은 지속 조건을 처음부터 채워야 한다
        self.assertFalse(g.feed(0.9, now=2.50))
        self.assertTrue(g.feed(0.9, now=2.58))

    def test_reset_clears_streak(self):
        g = FrameGate(threshold=0.5, persist=2)
        g.feed(0.9, now=0.00)
        g.reset()                                  # 긴급 우선 등으로 취소
        self.assertFalse(g.feed(0.9, now=0.08))    # 처음부터 다시 세야 함
        self.assertTrue(g.feed(0.9, now=0.16))

    def test_below_threshold_never_fires(self):
        g = FrameGate(threshold=0.5, persist=2)
        for i in range(10):
            self.assertFalse(g.feed(0.49, now=i * 0.08))


if __name__ == "__main__":
    unittest.main()
