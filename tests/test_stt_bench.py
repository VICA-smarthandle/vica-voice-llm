"""STT 비교기(tools/stt_bench.py)의 문제지·채점·요약 — 마이크·모델 없이 순수 로직만."""
import numpy as np

from tools.stt_bench import (FREE, FREE_ITEM, LONG, ROBOT_QUIET, ROBOT_STOP, SCRIPT, SHORT, STOP, build_order,
                             judge, normalize, peak_db, summarize)


def _item(item_id):
    return next(i for i in SCRIPT if i.id == item_id)


def test_script_has_28_items_in_five_groups():
    counts = {}
    for i in SCRIPT:
        counts[i.cat] = counts.get(i.cat, 0) + 1
    assert counts == {SHORT: 10, LONG: 8, STOP: 4, ROBOT_QUIET: 3, ROBOT_STOP: 3}
    assert len({i.id for i in SCRIPT}) == len(SCRIPT)


def test_order_is_shuffled_but_repeatable():
    a, b = build_order(), build_order()
    assert [i.id for i in a] == [i.id for i in b]
    assert sorted(i.id for i in a) == sorted(i.id for i in SCRIPT)
    assert [i.id for i in a] != [i.id for i in SCRIPT]


def test_normalize_drops_spaces_and_punctuation():
    assert normalize(" 407호로 가자. ") == "407호로가자"
    assert normalize("멈춰!") == "멈춰"
    assert normalize(None) == ""


def test_short_answer_needs_the_same_word():
    yes = _item("short_그래")
    assert judge(yes, "그래.") is True
    assert judge(yes, "그래 가자") is False
    assert judge(yes, "") is False
    assert judge(_item("short_네"), "예") is True


def test_long_answer_accepts_digits_or_spelled_numbers():
    go = _item("long_407호로가자")
    assert judge(go, "407호로 가자.") is True
    assert judge(go, "사백칠호로 가자") is True
    assert judge(go, "408호로 가자") is False


def test_stop_word_counts_even_with_extra_sound():
    assert judge(_item("stop_loud"), "어 멈춰!") is True
    assert judge(_item("stop_normal"), "멈추어") is False
    assert judge(_item("stop_jeongji"), "정지") is True


def test_robot_quiet_is_right_only_when_nothing_is_written():
    quiet = _item("robot_quiet_1")
    assert judge(quiet, "") is True
    assert judge(quiet, " . ") is True
    assert judge(quiet, "잠시 멈추겠습니다") is False


def test_robot_stop_needs_the_stop_word():
    assert judge(_item("robot_stop_1"), "다시 출발하려면 멈춰") is True
    assert judge(_item("robot_stop_1"), "다시 출발하려면") is False


def test_free_is_not_scored():
    assert judge(FREE_ITEM, "아무 말") is None
    assert FREE_ITEM.cat == FREE


def _rec(item, cat, t_ok, q_ok, t_sec=1.3, q_sec=0.3):
    return {"item": item, "cat": cat, "turbo": {"ok": t_ok, "sec": t_sec}, "qwen": {"ok": q_ok, "sec": q_sec}}


def test_summary_counts_last_try_per_item_and_skips_free():
    recs = [_rec("short_그래", SHORT, True, False),
            _rec("short_그래", SHORT, True, True),          # 다시 한 것 — 이것만 센다
            _rec("long_다시출발하자", LONG, False, True),
            _rec("free", FREE, None, None, 1.0, 0.2)]
    s = summarize(recs)
    assert s["by_cat"][SHORT] == {"name": "짧은 대답", "n": 1, "turbo": 1, "qwen": 1}
    assert s["by_cat"][LONG]["n"] == 1 and s["by_cat"][LONG]["turbo"] == 0
    assert s["by_cat"][STOP]["n"] == 0
    assert s["sec"]["turbo"] == 1.3                          # 시간은 자유까지 모든 시도의 중앙값
    assert s["sec"]["qwen"] == 0.3


def test_summary_of_nothing():
    s = summarize([])
    assert s["sec"] == {"turbo": None, "qwen": None}
    assert all(c["n"] == 0 for c in s["by_cat"].values())


def test_peak_db_is_loudest_20ms_slice():
    quiet = np.zeros(1600, dtype=np.int16)
    loud = quiet.copy()
    loud[320:640] = 3277                                   # 20 ms 한 조각만 최대의 1/10 → −20 dBFS
    assert peak_db(loud) == -20.0
    assert peak_db(quiet) < -150
    assert peak_db(np.zeros(100, dtype=np.int16)) is None  # 20 ms 도 안 되면 재지 않는다
