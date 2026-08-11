"""웨이크워드 측정의 채점 검증. 마이크/whisper 불필요.

측정 도구가 틀린 숫자를 내면 그 숫자를 믿고 잘못 결정하게 되므로, 채점 규칙만
따로 떼어 검사한다.
"""
from __future__ import annotations

from tools.wakeword_score import (
    DEFAULT_VARIANTS,
    Trial,
    format_report,
    heard_counts,
    match_wake_word,
    overall_verdict,
    summarize,
)


def _trial(phrase: str, should_wake: bool, heard: str) -> Trial:
    return Trial(
        phrase=phrase,
        should_wake=should_wake,
        heard=heard,
        matched=match_wake_word(heard),
    )


# -- 판정 -----------------------------------------------------------------


def test_exact_wake_word_matches():
    assert match_wake_word("비카야") == "비카야"


def test_wake_word_in_a_sentence_matches():
    assert match_wake_word("비카야 화장실로 가줘") == "비카야"


def test_trailing_particle_still_matches():
    """받아쓰기가 조사·어미를 붙여도 낱말 첫머리면 인정한다."""
    assert match_wake_word("비카야가 어디 있지") == "비카야"


def test_empty_text_does_not_match():
    assert match_wake_word("") is None
    assert match_wake_word("   ") is None


def test_bikyeo_must_not_match():
    """복도에서 흔한 "비켜"가 깨움으로 잡히면 안 된다 — 가장 중요한 함정."""
    assert match_wake_word("비켜") is None
    assert match_wake_word("비켜주세요") is None
    assert match_wake_word("좀 비켜봐") is None


def test_similar_words_do_not_match():
    for text in ("비상", "이거야", "비가 와요", "카야크"):
        assert match_wake_word(text) is None, text


def test_not_matched_inside_a_longer_word():
    """긴 낱말 속에 우연히 들어간 경우는 인정하지 않는다(운영 긴급어와 같은 규칙)."""
    assert match_wake_word("나비카야드") is None


def test_variants_can_be_widened():
    """받아쓰기 흔들림을 인정하려면 후보를 넓힐 수 있어야 한다."""
    assert match_wake_word("비까야", ("비카야",)) is None
    assert match_wake_word("비까야", ("비카야", "비까야")) == "비까야"


def test_default_variants_are_strict():
    assert DEFAULT_VARIANTS == ("비카야",)


# -- 집계 -----------------------------------------------------------------


def test_summarize_groups_by_phrase():
    trials = [
        _trial("비카야", True, "비카야"),
        _trial("비카야", True, "비가야"),
        _trial("비켜", False, "비켜"),
    ]
    results = summarize(trials)
    assert [r.phrase for r in results] == ["비카야", "비켜"]
    assert results[0].total == 2 and results[0].woke == 1
    assert results[1].total == 1 and results[1].woke == 0


def test_hit_rate_and_false_rate():
    hit = summarize([_trial("비카야", True, "비카야")] * 4 + [_trial("비카야", True, "")])[0]
    assert hit.rate == 0.8
    assert hit.label == "인식률"

    false = summarize([_trial("비켜", False, "비켜")] * 5)[0]
    assert false.rate == 0.0
    assert false.label == "오인율"


def test_empty_result_has_zero_rate_not_crash():
    from tools.wakeword_score import PhraseResult

    assert PhraseResult(phrase="비카야", should_wake=True).rate == 0.0


# -- 판정 등급 -------------------------------------------------------------


def test_high_hit_rate_is_good():
    result = summarize([_trial("비카야", True, "비카야")] * 10)[0]
    assert result.verdict == "좋음"


def test_low_hit_rate_is_unusable():
    trials = [_trial("비카야", True, "비카야")] * 5 + [_trial("비카야", True, "")] * 5
    assert summarize(trials)[0].verdict == "부적합"


def test_any_false_wake_is_flagged():
    """함정 문구가 한 번이라도 깨우면 '좋음'이 아니어야 한다."""
    trials = [_trial("비켜", False, "비켜")] * 19 + [_trial("비켜", False, "비카야")]
    result = summarize(trials)[0]
    assert result.woke == 1
    assert result.verdict != "좋음"


def test_frequent_false_wake_is_unusable():
    trials = [_trial("비켜", False, "비카야")] * 5 + [_trial("비켜", False, "비켜")] * 5
    assert summarize(trials)[0].verdict == "부적합"


def test_overall_verdict_takes_the_worst():
    good = summarize([_trial("비카야", True, "비카야")] * 10)
    bad = summarize([_trial("비켜", False, "비카야")] * 10)
    assert overall_verdict(good).startswith("양호")
    assert overall_verdict(good + bad).startswith("부적합")


def test_overall_verdict_with_no_data():
    assert overall_verdict([]) == "측정 없음"


# -- 받아쓰기 내용 ---------------------------------------------------------


def test_heard_counts_shows_what_whisper_wrote():
    """후보 목록을 추측이 아니라 실측으로 정하기 위한 정보다."""
    trials = [
        _trial("비카야", True, "비카야"),
        _trial("비카야", True, "비카야"),
        _trial("비카야", True, "비까야"),
    ]
    assert heard_counts(trials, "비카야") == [("비카야", 2), ("비까야", 1)]


def test_heard_counts_labels_empty_recognition():
    assert heard_counts([_trial("비카야", True, "")], "비카야") == [("(빈 인식)", 1)]


# -- 보고서 ---------------------------------------------------------------


def test_report_contains_both_rates_and_verdict():
    trials = [_trial("비카야", True, "비카야")] * 9 + [_trial("비켜", False, "비켜")] * 9
    report = format_report(trials)
    assert "인식률" in report and "오인율" in report
    assert "비카야" in report and "비켜" in report
    assert "종합:" in report


def test_report_without_data_does_not_crash():
    assert "측정 결과가 없습니다" in format_report([])
