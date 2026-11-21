"""새 답지 채점 — 로봇이 한 일(행동·대상)을 답지와 견준다 (설계 2026-11-20 G2 · 8절, 계획 Task 1-3)."""
from __future__ import annotations

from types import SimpleNamespace

from src.question_record import OpenQuestion
from src.schema import DestinationData, VicaIntent
from tools.key_grade import Reaction, grade_key, load_key, reaction_of

TOILET = DestinationData(id="t", name="화장실")
WELFARE = DestinationData(id="s", name="사회복지")
PRIVATE = DestinationData(id="p", name="비공개 408호", is_approachable=False)
MEN = DestinationData(id="m", name="남자 화장실", aliases=["화장실"])
WOMEN = DestinationData(id="w", name="여자 화장실", aliases=["화장실"])
DESTS = [TOILET, WELFARE, PRIVATE, MEN, WOMEN]


def nav(dest: DestinationData, need_confirm: bool) -> VicaIntent:
    return VicaIntent(intent="navigate", matched_destination_id=dest.id, need_confirm=need_confirm)


def clarify(text: str) -> VicaIntent:
    return VicaIntent(intent="clarify", need_confirm=False, ask_text=text, ask_kind="open")


def asked(kind: str, text: str, options: tuple = ()) -> OpenQuestion:
    return OpenQuestion("q", kind, text, options, 0.0)


def test_confirmed_departure_is_go_and_breaks_principle_3():
    r = reaction_of(nav(TOILET, need_confirm=False), None, DESTS)
    assert r == Reaction("go", TOILET.id)
    assert grade_key({"action": "confirm", "target": TOILET.id}, r) == ("wrong", 3)


def test_proposal_is_a_confirm_question():
    r = reaction_of(nav(TOILET, need_confirm=True), None, DESTS)
    assert r == Reaction("confirm", TOILET.id)
    assert grade_key({"action": "confirm", "target": TOILET.id}, r) == ("ok", 0)
    assert grade_key({"action": "confirm", "target": WELFARE.id}, r) == ("wrong", 5)      # 다른 곳을 물음


def test_naming_the_place_the_robot_asked_about_is_yes():
    q = asked("confirm", "화장실로 안내해드릴까요?", (TOILET.id,))
    assert reaction_of(nav(TOILET, need_confirm=False), q, DESTS) == Reaction("yes")
    llm_made = asked("open", "사회복지는 1층에 있는데, 안내해드릴까요?")              # LLM 이 지은 확인 질문
    assert reaction_of(nav(WELFARE, need_confirm=True), llm_made, DESTS) == Reaction("yes")


def test_pick_from_choice_question():
    q = asked("choice", "남자 화장실과 여자 화장실 중 어디로 갈까요?", (MEN.id, WOMEN.id))
    r = reaction_of(nav(MEN, need_confirm=True), q, DESTS)
    assert r == Reaction("pick", MEN.id)
    assert grade_key({"action": "pick", "target": MEN.id}, r) == ("ok", 0)
    assert grade_key({"action": "pick", "target": WOMEN.id}, r) == ("wrong", 6)


def test_pick_and_confirm_of_the_same_place_are_the_same_reaction():
    # 고르기 질문을 LLM 이 지었으면 질문 기록의 종류가 open 이라 반응이 confirm 으로 읽힌다 — 로봇이 하는 일은 같다
    assert grade_key({"action": "pick", "target": MEN.id}, Reaction("confirm", MEN.id)) == ("ok", 0)
    assert grade_key({"action": "confirm", "target": MEN.id}, Reaction("pick", MEN.id)) == ("ok", 0)


def test_place_the_robot_cannot_go_is_notavail():
    assert reaction_of(nav(PRIVATE, need_confirm=False), None, DESTS) == Reaction("notavail", PRIVATE.id)


def test_clarify_sentences_split_into_which_notavail_reask():
    assert reaction_of(clarify("남자 화장실과 여자 화장실 중 어디로 갈까요?"), None, DESTS).action == "which"
    assert reaction_of(clarify("제가 아는 목적지 중에서 스무디는 없는데, 어디로 안내해드릴까요?"), None, DESTS).action == "notavail"
    assert reaction_of(clarify("어디로 안내해드릴까요?"), None, DESTS).action == "reask"
    spoken_by_voice = VicaIntent(intent="clarify", need_confirm=False, reply="남자 화장실과 여자 화장실 중 어디로 갈까요?")
    assert reaction_of(spoken_by_voice, None, DESTS).action == "which"                   # 넘기지 않은 문장(reply)도 같다


def test_silent_unknown_while_mission_question_open_is_reask():
    q = asked("confirm", "화장실로 안내해드릴까요?", (TOILET.id,))
    assert reaction_of(VicaIntent(intent="unknown"), q, DESTS).action == "reask"         # 미션이 같은 질문을 다시 한다
    assert reaction_of(VicaIntent(intent="unknown"), None, DESTS).action == "silent"
    assert reaction_of(VicaIntent(intent="unknown"), asked("open", "네?"), DESTS).action == "silent"


def test_unknown_with_a_sentence():
    assert reaction_of(VicaIntent(intent="unknown", ask_text="네?", ask_kind="open"), None, DESTS).action == "wake"
    assert reaction_of(VicaIntent(intent="unknown", reply="네?"), None, DESTS).action == "wake"
    retry = VicaIntent(intent="unknown", ask_text="잘 듣지 못했습니다. 다시 말씀해 주세요.", ask_kind="open")
    assert reaction_of(retry, None, DESTS).action == "reask"


def test_short_answers_and_new_values():
    assert reaction_of(VicaIntent(intent="affirm"), None, DESTS) == Reaction("yes")
    assert reaction_of(VicaIntent(intent="deny"), None, DESTS) == Reaction("no")
    # 새 값(2단계에 스키마에 들어간다) — 지금은 VicaIntent 가 받지 않아 같은 꼴의 객체로 본다
    for value, action in (("not_for_robot", "silent"), ("dismiss", "bye"), ("repeat", "repeat")):
        assert reaction_of(SimpleNamespace(intent=value, reply="", ask_text=""), None, DESTS) == Reaction(action)
    assert reaction_of(VicaIntent(intent="question", reply="지금 4층이에요."), None, DESTS) == Reaction("answer", say="지금 4층이에요.")
    assert reaction_of(VicaIntent(intent="help"), None, DESTS).action == "etc"


def test_wait_cancel_finish_follow_the_question():
    minutes = asked("minutes", "몇 분쯤 걸리실까요?")
    assert reaction_of(VicaIntent(intent="wait", wait_minutes=10), minutes, DESTS) == Reaction("wait", minutes="10")
    assert reaction_of(VicaIntent(intent="wait"), asked("yesno", "여기서 기다릴까요?"), DESTS) == Reaction("yes")
    assert reaction_of(VicaIntent(intent="wait"), asked("yesno", "여기까지 안내를 마칠까요?"), DESTS) == Reaction("no")
    assert reaction_of(VicaIntent(intent="finish"), asked("yesno", "여기까지 안내를 마칠까요?"), DESTS) == Reaction("yes")
    assert reaction_of(VicaIntent(intent="cancel"), asked("yesno_inverted", "안내가 필요 없으신가요?"), DESTS) == Reaction("yes")
    assert reaction_of(VicaIntent(intent="finish"), None, DESTS) == Reaction("finish")


def test_wait_minutes_must_match_when_the_key_has_them():
    assert grade_key({"action": "wait", "minutes": "10"}, Reaction("wait", minutes="10")) == ("ok", 0)
    assert grade_key({"action": "wait", "minutes": "10"}, Reaction("wait", minutes="30")) == ("wrong", 6)
    assert grade_key({"action": "wait", "minutes": ""}, Reaction("wait", minutes="30")) == ("ok", 0)


def test_opposite_answer_and_unasked_yes():
    assert grade_key({"action": "yes"}, Reaction("no")) == ("wrong", 6)
    assert grade_key({"action": "wake"}, Reaction("yes"), kind="yesno") == ("wrong", 13)
    assert grade_key({"action": "wake"}, Reaction("yes"), kind="open") == ("mismatch", 13)   # "네?"에 승낙 — 아무 일도 안 일어난다


def test_dropping_a_real_request_is_wrong_and_talking_to_bystanders_breaks_7():
    assert grade_key({"action": "confirm", "target": "a"}, Reaction("silent")) == ("wrong", 5)
    assert grade_key({"action": "silent"}, Reaction("confirm", "a")) == ("wrong", 7)
    assert grade_key({"action": "silent"}, Reaction("reask")) == ("reask", 7)
    assert grade_key({"action": "reask"}, Reaction("silent")) == ("mismatch", 8)         # 되물어야 하는데 가만히 있음


def test_drop_and_pending_rows_are_not_graded():
    assert grade_key({"action": "yes", "drop": "mapping"}, Reaction("no"))[0] == "drop"
    assert grade_key({"action": "etc", "source": "pending"}, Reaction("no"))[0] == "pending"


def test_load_key_reads_rows_by_ts(tmp_path):
    path = tmp_path / "answer_key.csv"
    path.write_text("ts,date,time,set,action,target,minutes,say,source,drop,memo,no\n"
                    "100.5,11-04,10:00:00,seen,confirm,t,,화장실로 안내해드릴까요?,rule,,,7\n", encoding="utf-8")
    key = load_key(path)
    assert key["100.5"]["action"] == "confirm" and key["100.5"]["set"] == "seen"
