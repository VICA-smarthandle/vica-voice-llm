"""마이크 점검기 합격 기준(설계 6절 초기값)·저장 검사(7절)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.mic_check_rules import (FAIL, PASS, REDO, WAIT, WARN, AnswerTry, CallTry, Measurements, MicValues,
                                 NearTry, RobotRun, StopTry, check_values, echo_name, judge, noise_floor)

V = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
KEYS = ("noise", "call", "answer", "stop", "far", "near", "robot")
QUIET_NEAR = NearTry(False, False, 0.004)


def good() -> Measurements:
    return Measurements(
        noise=[0.0030, 0.0032, 0.0035],
        calls=[CallTry(True, True, 0.93, "화장실로 가자")] * 3,
        runs=[CallTry(True, True, 0.91, "비카야 화장실로 가자")] * 2,
        answers=[AnswerTry("yes", True, True, 0.045, "그래")] * 2 + [AnswerTry("no", True, True, 0.040, "아니")] * 2,
        stops=[StopTry(True, True, 0.86, "멈춰")] * 5,
        far=[AnswerTry("yes", True, True, 0.015, "그래"), AnswerTry("no", True, True, 0.014, "아니")],
        near=[QUIET_NEAR] * 3,
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
        "noise": REDO, "call": REDO, "run": REDO, "answer": REDO, "stop": REDO, "far": REDO, "near": REDO,
        "robot": REDO, "cross": REDO}
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
    # 2026-10-22: 짧은 답 최소 ≥ 수음 최소 × 1.0 — 같아도 되고, 옛 1.5배 규칙에서 막히던 0.014·0.02 도 된다
    assert check_values(replace(V, listen_min_rms=0.02)) == []
    assert check_values(replace(V, listen_min_rms=0.014)) == []
    problems = check_values(replace(V, listen_min_rms=0.025))
    assert any("짧은 답 최소는 수음 최소 이상" in p and "0.80배" in p for p in problems)
    out = MicValues(0.008, 0.02, 0.95, 0.30, 0.007, 1.5, 1)
    problems = check_values(out)
    assert any("호출 문턱" in p for p in problems) and any("멈춰" in p for p in problems)


def test_noise_floor_and_echo_name():
    assert noise_floor([]) is None
    assert noise_floor([0.001 * i for i in range(1, 11)]) == pytest.approx(0.010)
    assert echo_name(1.5, 1) == "strong" and echo_name(1.0, 0) == "factory" and echo_name(2.0, 1) is None


# ---------------------------------------------------------------- 10-21: 최대 증폭·접근 대답 최소·대역 최소, 먼 대답·옆사람
def test_new_values_default_to_off():
    """세 값의 기본은 None — 7개로 만든 값은 지금까지와 같다(최대 증폭 안 씀·접근 = 수음 최소·대역 관문 끔)."""
    assert (V.agc_max_gain, V.approach_min_rms, V.band_min) == (None, None, None)


def test_max_gain_change_needs_rerecord():
    v = judge(good(), replace(V, agc_max_gain=2.0))
    grades = {r.key: r.grade for r in v.rows}
    assert all(grades[k] == REDO for k in ("noise", "call", "run", "answer", "stop", "far", "near", "robot", "cross"))


def test_far_and_near_wait_before_recording():
    m = Measurements()
    assert row(m, "far").grade == WAIT and row(m, "near").grade == WAIT


def test_far_row_judged_against_approach_min():
    m = good()
    v = replace(V, approach_min_rms=0.006)
    m.far = [AnswerTry("yes", True, True, 0.0065, "그래"), AnswerTry("no", True, True, 0.010, "아니")]
    r = row(m, "far", v)
    assert r.grade == WARN and "0.0065" in r.detail and "접근 대답 최소 0.0060" in r.detail   # 0.0065 < 0.006 × 1.5
    m.far = [AnswerTry("yes", True, True, 0.0095, "그래"), AnswerTry("no", True, True, 0.010, "아니")]
    assert row(m, "far", v).grade == PASS
    assert "수음 최소 0.0080" in row(m, "far", V).detail          # 줄이 없으면 수음 최소로 판단한다


def test_far_row_fails_when_dropped_or_misheard():
    m = good()
    m.far = [AnswerTry("yes", False, False, 0.005), AnswerTry("no", True, True, 0.02, "아니")]
    r = row(m, "far")
    assert r.grade == FAIL and "접근 대답 최소" in r.todo
    m.far = [AnswerTry("yes", True, True, 0.02, "그래"), AnswerTry("no", True, False, 0.02, "그래")]
    r = row(m, "far")
    assert r.grade == FAIL and "'그래'" in r.detail


def test_near_leak_into_question_window_fails():
    m = good()
    m.near = [NearTry(True, True, 0.030, "어제 그거 봤어")] + [QUIET_NEAR] * 2
    r = row(m, "near")
    assert r.grade == FAIL and "질문 창 통과 1/3" in r.detail and "어제 그거 봤어" in r.detail
    assert "수음 최소" in r.todo and "최대 증폭" in r.todo
    assert judge(m, V).overall == FAIL


def test_near_leak_only_in_approach_window_warns():
    m = good()
    m.near = [NearTry(False, True, 0.007, "응 그래")] + [QUIET_NEAR] * 2
    r = row(m, "near")
    assert r.grade == WARN and "접근 창 통과 1/3" in r.detail and "접근 대답 최소" in r.todo


def test_near_detail_shows_gap_to_user_and_split_value():
    """손잡이 사용자 최소(짧은 답)와 옆사람 최대 사이 dB, 둘을 가르는 수음 최소(기하 평균) — 10-15 주변 대화 시험기와 같은 셈."""
    m = good()
    m.answers = [AnswerTry("yes", True, True, 0.040)] * 4
    m.near = [NearTry(False, False, 0.004), NearTry(False, False, 0.002), NearTry(False, False, None)]
    r = row(m, "near")
    assert r.grade == PASS
    assert "옆사람 최대 0.0040" in r.detail and "+20.0 dB" in r.detail and "가를 값 0.0126" in r.detail
    assert "먼 대답 최소 0.0140" in r.detail and "+10.9 dB" in r.detail


def test_near_cross_trigger_counts():
    m = good()
    m.near = [NearTry(False, False, 0.004, emergency=True)] + [QUIET_NEAR] * 2
    assert row(m, "cross").grade == FAIL
    m = good()
    m.far = [AnswerTry("yes", True, True, 0.015, "그래", emergency=True), m.far[1]]
    assert row(m, "cross").grade == FAIL


def test_check_values_new_ranges():
    assert check_values(replace(V, approach_min_rms=0.006, band_min=0.004, agc_max_gain=2.0)) == []
    assert any("접근 대답 최소" in p for p in check_values(replace(V, approach_min_rms=0.001)))
    assert any("말소리 대역 최소" in p for p in check_values(replace(V, band_min=0.02)))


def test_factory_gain_is_the_same_chip_as_no_line():
    """줄 없음이면 점검기가 칩에 공장값 ×31.6 을 쓴다 — 둘은 같은 칩 상태다(검토 10-21)."""
    factory = replace(V, agc_max_gain=31.6)
    assert factory.chip() == V.chip() and factory.gain() == V.gain()
    assert all(r.grade != REDO for r in judge(good(), factory).rows)


def test_near_gap_skips_recordings_made_at_another_gain():
    m = good()
    m.answers = [AnswerTry("yes", True, True, 0.040)] * 4
    m.chips["answer"] = replace(V, agc_max_gain=2.0).chip()                   # 4단계는 ×2 로 녹음했다
    r = row(m, "near")
    assert "옆사람 최대" in r.detail and "사용자 최소" not in r.detail and "먼 대답 최소" in r.detail


def test_ranges_cover_hallway_levels():
    """10-22 복도(×4): 옆사람 최대 0.021·먼 대답 최소 0.049 → 가를 수음 최소 ≈ 0.032 를 고칠 수 있어야 한다."""
    ok = replace(V, listen_min_rms=0.040, short_answer_min_rms=0.050, approach_min_rms=0.030)
    assert check_values(ok) == []
    assert check_values(replace(V, short_answer_min_rms=0.010)) == []
    for over in (dict(listen_min_rms=0.0405, short_answer_min_rms=0.050), dict(short_answer_min_rms=0.051),
                 dict(short_answer_min_rms=0.009), dict(approach_min_rms=0.0305)):
        assert check_values(replace(V, **over)), over
