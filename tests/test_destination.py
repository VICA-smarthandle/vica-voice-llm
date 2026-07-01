"""destination_loader / destination_matcher / schema 의 작은 단위 테스트.

실행 (프로젝트 루트에서):
    .venv/bin/python -m unittest tests.test_destination -v
"""
import unittest

from src.destination_loader import _fill_defaults, _josa_euro, load_destinations
from src.destination_matcher import match_destination
from src.schema import DestinationData


class JosaTest(unittest.TestCase):
    def test_josa_euro(self):
        self.assertEqual(_josa_euro("화장실"), "로")    # ㄹ 받침
        self.assertEqual(_josa_euro("안내센터"), "로")  # 받침 없음
        self.assertEqual(_josa_euro("식당"), "으로")    # ㅇ 받침


class FillDefaultsTest(unittest.TestCase):
    def test_auto_fill(self):
        d = DestinationData(id="x", name="안내센터")  # confirm/arrival 비어 있음
        _fill_defaults(d)
        self.assertEqual(d.confirm_prompt, "안내센터로 안내해드릴까요?")
        self.assertEqual(d.arrival_message, "안내센터 앞에 도착했습니다.")

    def test_keep_existing(self):
        d = DestinationData(id="x", name="식당", confirm_prompt="이미 있음")
        _fill_defaults(d)
        self.assertEqual(d.confirm_prompt, "이미 있음")  # 기존 값은 유지

    def test_not_approachable_has_no_arrival(self):
        d = DestinationData(id="x", name="식당", is_approachable=False)
        _fill_defaults(d)
        self.assertEqual(d.arrival_message, "")  # 갈 수 없는 곳은 도착 메시지 없음


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.dests = load_destinations()

    def test_loaded_count(self):
        self.assertGreaterEqual(len(self.dests), 5)

    def test_no_safety_level_field(self):
        # safety_level 은 제거됐으므로 스키마에 존재하지 않아야 한다.
        self.assertFalse(hasattr(self.dests[0], "safety_level"))


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.dests = load_destinations()

    def test_exact_alias(self):
        d = match_destination("407호", self.dests)
        self.assertIsNotNone(d)
        self.assertEqual(d.id, "engineering_4f_room_407_prof_yoon_jiyoung_office")

    def test_restroom(self):
        d = match_destination("화장실", self.dests)
        self.assertIsNotNone(d)
        self.assertEqual(d.category2, "restroom")

    def test_match_inside_sentence(self):
        d = match_destination("안내센터로 가줘", self.dests)
        self.assertIsNotNone(d)
        self.assertEqual(d.category2, "information_center")

    def test_no_match_returns_none(self):
        # "배 아파" 같은 간접 표현은 matcher 가 직접 풀지 않는다 (LLM 역할).
        self.assertIsNone(match_destination("배 아파", self.dests))


if __name__ == "__main__":
    unittest.main()
