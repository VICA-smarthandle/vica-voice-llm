"""여러 곳 별칭 고르기 — 말 속에서 걸린 별칭 중 다른 걸린 별칭 안에 들어간 짧은 것은 뺀다(2026-10-29 사용자).

행사 지도(map_1029_074730)에서 "서울로봇고 가자"가 서울로봇고 두 팀 + 경남로봇고 세 곳 고르기 질문이 됐다 —
짧은 별칭 '로봇고'(세 곳 공통)가 긴 별칭 '서울로봇고'(두 곳 공통) 안에 들어 있어 함께 걸렸다. 사용자가 말한 것은
긴 쪽이므로 긴 별칭만 본다. "로봇고로 가자"(세 곳)·"나 배 아파"(남녀 화장실) 같은 기존 고르기는 그대로다.
"""
import pytest

from src.langchain_intent_parser import _alias_only_group, _IntentDraft
from src import langchain_intent_parser as parser
from src.replies import choose_between
from src.schema import DestinationData, RobotState


def dest(i, name, aliases):
    return DestinationData(id=i, name=name, aliases=aliases, confirm_prompt=f"{name}로 안내해드릴까요?")


OHARAM = dest("oharam", "서울로봇고 오하람과 아이들",
              ["서울로봇고등학교 오하람과 아이들", "서울로봇고", "오하람", "오하람과 아이들", "아이들", "로봇고", "서울"])
ALLROUND = dest("allround", "서울로봇고 올라운더",
                ["서울로봇고등학교 올라운더", "올라운더", "라운더", "로봇고", "서울", "서울로봇고", "서울로봇고등학교"])
GYEONGNAM = dest("gyeongnam", "경남로봇고 모빌로스",
                 ["경남로봇고등학교", "경남로봇고", "로봇고", "경남", "모빌로스", "모삘로스", "모바일", "mobilos"])
YC = dest("yc", "영천고 독수리 오형제", ["영천고등학교", "영천고", "영천", "독수리 오형제", "독수리", "오형제", "이글"])
SUNHWA = dest("sunhwa", "선화여고 헤리테지",
              ["선화여자고등학교", "선화여고", "선화", "영천 헤리테지 에이아이 탐구단", "헤리테지", "영천", "탐구단"])
YCSN = dest("ycsn", "영천성남여고 미피", ["영천성남여자고등학교", "영천성남여고", "영천", "성남", "성남여고", "미피"])
EVENT = [OHARAM, ALLROUND, GYEONGNAM, YC, SUNHWA, YCSN]

MEN = dest("men", "남자 화장실", ["남자 화장실", "남자", "화장실", "대변", "소변", "배아파", "아파"])
WOMEN = dest("women", "여자 화장실", ["여자 화장실", "여자", "화장실", "대변", "소변", "배아파", "아파"])
WC = [MEN, WOMEN]


def ids(group):
    return [d.id for d in group]


class TestLongerAliasWins:
    @pytest.mark.parametrize("said", ["서울로봇고 가자", "서울로봇고로 가 줘.", "서울 로봇고 보고 싶어"])
    def test_seoul_robot_high_asks_only_the_two_seoul_teams(self, said):
        assert ids(_alias_only_group(said, EVENT)) == ["oharam", "allround"]

    def test_robot_high_alone_still_asks_all_three(self):
        assert ids(_alias_only_group("로봇고로 가자", EVENT)) == ["oharam", "allround", "gyeongnam"]

    def test_yeongcheon_alone_still_asks_all_three(self):
        assert ids(_alias_only_group("영천 가고 싶어", EVENT)) == ["yc", "sunhwa", "ycsn"]

    @pytest.mark.parametrize("said", ["서울로봇고 올라운더로 가자",          # 이름을 말함
                                      "경남로봇고로 가자",                    # 한 곳에만 있는 긴 별칭
                                      "영천 헤리테지 에이아이 탐구단 보러 가자",  # 한 곳에만 있는 긴 별칭이 '영천'을 품음
                                      "모빌로스 보러 가자"])
    def test_one_place_is_left_to_the_model(self, said):
        assert _alias_only_group(said, EVENT) == []

    @pytest.mark.parametrize("said", ["나 배 아파.", "나 배아파", "나 배가 아파", "화장실 가고 싶어"])
    def test_restroom_choice_unchanged(self, said):
        assert ids(_alias_only_group(said, WC)) == ["men", "women"]


def test_text_path_question_names_only_the_two_seoul_teams():
    draft = _IntentDraft(intent="navigate", destination_candidate="서울로봇고 올라운더", need_confirm=True,
                         reply="서울로봇고 올라운더로 안내해드릴까요?", confidence=0.8)
    out = parser._finalize(draft, EVENT, user_text="서울로봇고 가자", local=True,
                           robot_state=RobotState(dialog_state="idle"))
    assert out.intent == "clarify"
    assert out.reply == choose_between(["서울로봇고 오하람과 아이들", "서울로봇고 올라운더"])
