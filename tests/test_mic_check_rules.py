"""마이크 점검기 합격 기준(설계 6절 초기값)·저장 검사(7절)."""
from __future__ import annotations

import pytest

from src.mic_check_rules import (FAIL, PASS, REDO, WAIT, WARN, AnswerTry, CallTry, Measurements, MicValues,
                                 RobotRun, StopTry, check_values, echo_name, judge, noise_floor)

V = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
KEYS = ("noise", "call", "answer", "stop", "robot")


def good() -> Measurements:
    return Measurements(
        noise=[0.0030, 0.0032, 0.0035],
        calls=[CallTry(True, True, 0.93, "화장실로 가자")] * 3,
        runs=[CallTry(True, True, 0.91, "비카야 화장실로 가자")] * 2,
        answers=[AnswerTry("yes", True, True, 0.045, "그래")] * 2 + [AnswerTry("no", True, True, 0.040, "아니")] * 2,
        stops=[StopTry(True, True, 0.86, "멈춰")] * 5,
        robot=RobotRun(0, 0.031),
        chips={k: V.chip() for k in KEYS},
        robot_thresholds=(0.70, 0.50),
    )


def row(m, key, v=V, prev=None):
    return next(r for r in judge(m, v, prev).rows if r.key == key)


def test_all_good_is_pass():
    verdict = judge(good(), V)
    assert verdict.overall == PASS and verdict.title == "시연 가능"


def test_empty_is_wait():
    verdict = judge(Measurements(), V)
    assert verdict.overall == WAIT and verdict.title == "점검 중"


@pytest.mark.parametrize("floor,expected", [(0.0040, PASS), (0.00401, WARN), (0.0061, WARN), (0.0062, FAIL)])
def test_noise_ratio_boundaries(floor, expected):
    m = good()
    m.noise = [floor * 0.9, floor * 0.95, floor]
    assert row(m, "noise").grade == expected


def test_noise_spread_warns():
    m = good()
    m.noise = [0.0010, 0.0020, 0.0030]      # 세 측정이 3배 차이
    r = row(m, "noise")
    assert r.grade == WARN and "세 번 차이" in r.detail


GO = "화장실로 가자"


def test_call_margin_and_miss():
    m = good()
    m.calls = [CallTry(True, True, 0.79, GO)] * 3
    assert row(m, "call").grade == WARN      # 0.79 < 0.70 + 0.1
    m.calls = [CallTry(True, True, 0.80, GO)] * 3
    assert row(m, "call").grade == PASS
    m.calls = [CallTry(True, True, 0.9, GO)] * 2 + [CallTry(False, False, 0.6)]
    assert row(m, "call").grade == FAIL
    m.calls = [CallTry(True, True, 0.9, GO)] * 2 + [CallTry(True, False, 0.9)]
    r = row(m, "call")
    assert r.grade == FAIL and "말을 못 받음" in r.detail


def test_call_needs_the_place_in_what_was_heard():
    """호출 뒤 받은 말에 '화장실'이 없으면 목적지가 안 간 것이다 — 이어 말하면 머리가 잘린다(10-13 call_1 '로 가자')."""
    m = good()
    m.calls = [CallTry(True, True, 0.9, GO)] * 2 + [CallTry(True, True, 0.95, "로 가자")]
    r = row(m, "call")
    assert r.grade == FAIL and "목적지" in r.detail and "로 가자" in r.detail
    m.calls = [CallTry(True, True, 0.9, "화장 실로 가자.")] * 3     # 띄어 써도 같은 말
    assert row(m, "call").grade == PASS


def test_run_on_call_row_is_reference_only():
    """이어 부르기는 지금 귀가 못 받는 것을 안다 — 참고 줄이라 '주의'까지만, 할 일은 "네?" 듣고 말하기."""
    m = good()
    m.runs = [CallTry(False, False, 0.61), CallTry(True, True, 0.95, "로 가자")]
    r = row(m, "run")
    assert r.grade == WARN and "0/2" in r.detail and "네?" in r.todo
    assert judge(m, V).overall == WARN
    m.runs = m.runs[:1]
    assert row(m, "run").grade == WAIT
    m = good()
    assert row(m, "run").grade == PASS


def test_run_on_call_counts_cross_trigger():
    m = good()
    m.runs = [CallTry(True, True, 0.9, "비카야 화장실로 가자", emergency=True)] * 2
    assert row(m, "cross").grade == FAIL


def test_answer_row_names_the_words_people_use():
    """실사용 대답은 '그래' 400·'아니' 151 vs '네' 107·'아니요' 22(10-13 실로그) — 그 말로 점검한다."""
    assert "그래" in row(good(), "answer").name and "아니" in row(good(), "answer").name


def test_answer_margin_and_wrong_word():
    m = good()
    m.answers = [AnswerTry("yes", True, True, 0.029)] * 4
    assert row(m, "answer").grade == WARN    # 0.029 < 0.02 × 1.5
    m.answers = [AnswerTry("yes", True, True, 0.030)] * 4
    assert row(m, "answer").grade == PASS
    m.answers = [AnswerTry("yes", True, True, 0.04)] * 3 + [AnswerTry("no", True, False, 0.04, "네")]
    r = row(m, "answer")
    assert r.grade == FAIL and "'네'" in r.detail
    m.answers = [AnswerTry("yes", True, True, 0.04)] * 3 + [AnswerTry("no", False, False, None)]
    assert row(m, "answer").grade == FAIL


@pytest.mark.parametrize("ok,expected", [(5, PASS), (4, WARN), (3, FAIL)])
def test_stop_counts(ok, expected):
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰")] * ok + [StopTry(False, False, 0.44)] * (5 - ok)
    assert row(m, "stop").grade == expected


def test_stop_miss_reasons():
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰")] * 3 + [StopTry(False, False, 0.44), StopTry(False, True, 0.8, "멈추라")]
    r = row(m, "stop")
    assert "점수 0.44" in r.detail and "확인 기각 '멈추라'" in r.detail
    assert "잡음" in r.todo and "말투" in r.todo


def test_robot_rows():
    m = good()
    m.robot = RobotRun(1, 0.03)
    assert row(m, "robot").grade == FAIL
    m.robot = RobotRun(0, 0.046)
    assert row(m, "robot", prev=0.030).grade == WARN     # 0.046 > 0.030 × 1.5
    assert row(m, "robot", prev=0.031).grade == PASS     # 0.046 < 0.0465
    assert row(m, "robot", prev=None).grade == PASS


def test_robot_play_failed_is_fail_and_overall_fail():
    m = good()
    m.robot = RobotRun(0, 0.003, played_ok=False)
    r = row(m, "robot")
    assert r.grade == FAIL and "재생 실패" in r.detail and "스피커 연결" in r.todo
    assert judge(m, MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1), None).overall == FAIL


def test_robot_redo_when_thresholds_changed():
    m = good()
    v = MicValues(0.008, 0.02, 0.70, 0.43, 0.007, 1.5, 1)   # 멈춰 문턱만 바꿈
    r = row(m, "robot", v=v)
    assert r.grade == REDO and "6단계" in r.todo


def test_cross_trigger_fails():
    m = good()
    m.answers = list(m.answers)
    m.answers[3] = AnswerTry("no", True, True, 0.04, "아니요", emergency=True)
    assert row(m, "cross").grade == FAIL
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰", woke=True)] + [StopTry(True, True, 0.9, "멈춰")] * 4
    assert row(m, "cross").grade == FAIL


def test_agc_change_needs_rerecord_echo_change_only_robot():
    m = good()
    agc = MicValues(0.008, 0.02, 0.70, 0.50, 0.010, 1.5, 1)
    v = judge(m, agc)
    assert {r.key: r.grade for r in v.rows} == {
        "noise": REDO, "call": REDO, "run": REDO, "answer": REDO, "stop": REDO, "robot": REDO, "cross": REDO}
    assert v.overall == REDO and v.title == "다시 녹음 필요"
    echo = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.0, 1)
    v = judge(m, echo)
    grades = {r.key: r.grade for r in v.rows}
    assert grades["robot"] == REDO
    assert all(g == PASS for k, g in grades.items() if k != "robot")


def test_overall_takes_worst():
    m = good()
    m.robot = RobotRun(2, 0.03)
    m.stops = [StopTry(True, True, 0.9)] * 4 + [StopTry(False, False, 0.4)]
    assert judge(m, V).overall == FAIL


def test_check_values():
    assert check_values(V) == []
    bad = MicValues(0.014, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
    assert any("1.5배" in p for p in check_values(bad))
    out = MicValues(0.008, 0.02, 0.95, 0.30, 0.007, 1.5, 1)
    problems = check_values(out)
    assert any("호출 문턱" in p for p in problems) and any("멈춰" in p for p in problems)


def test_noise_floor_and_echo_name():
    assert noise_floor([]) is None
    assert noise_floor([0.001 * i for i in range(1, 11)]) == pytest.approx(0.010)
    assert echo_name(1.5, 1) == "strong" and echo_name(1.0, 0) == "factory" and echo_name(2.0, 1) is None
