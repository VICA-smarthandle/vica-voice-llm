"""유령 STT 방어 검증 — 무음 환각이 사용자 발화로 둔갑하면 안 된다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stt_guard import accept_segments, is_hallucination
from src.wakeword_monitor import WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)


@dataclass
class Seg:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def test_drops_segment_only_when_both_signals_say_silence():
    """no_speech 높음 + logprob 낮음이 '동시에' 참일 때만 버린다 —"""
    both_bad = Seg("시청해주셔서 감사합니다", no_speech_prob=0.9, avg_logprob=-1.5)
    quiet_real = Seg("화장실", no_speech_prob=0.3, avg_logprob=-1.5)
    noisy_real = Seg("가고 싶어요", no_speech_prob=0.9, avg_logprob=-0.4)

    assert accept_segments([both_bad]) == ""
    assert accept_segments([quiet_real]) == "화장실"
    assert accept_segments([both_bad, quiet_real, noisy_real]) == "화장실가고 싶어요"


def test_accepts_plain_segments_and_stub_without_attrs():
    class Bare:
        text = "응"

    assert accept_segments([Seg("네 맞아요")]) == "네 맞아요"
    assert accept_segments([Bare()]) == "응"
    assert accept_segments([]) == ""


def test_known_hallucinations_full_match():
    assert is_hallucination("시청해 주셔서 감사합니다.")
    assert is_hallucination("시청해주셔서 감사합니다")
    assert is_hallucination("구독과 좋아요 부탁드립니다!")
    assert is_hallucination("MBC 뉴스 김성현입니다.")


def test_real_speech_is_not_flagged():
    assert not is_hallucination("화장실 데려다줘")
    assert not is_hallucination("안내해 주셔서 감사합니다")
    assert not is_hallucination("")


def test_hallucinated_listen_window_closes_quietly():
    """소음이 RMS 문지기를 뚫고 whisper 가 환각을 지어내도, user_text 로"""
    texts = []
    m = WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=texts.append,
        predict=lambda f: {"a": 0.0, "b": 0.0},
        transcribe=lambda a: "시청해 주셔서 감사합니다.",
    )
    m._open_listen(followup=False, now=0.0)

    results = []
    for i in range(5):
        results.append(m.process_frame(LOUD, now=i * 0.08, vad=True))
    for i in range(30):
        results.append(m.process_frame(QUIET, now=0.4 + i * 0.08))

    assert "wake_silent" in results
    assert texts == []


def test_listen_prefers_guarded_transcribe_but_emergency_uses_raw():
    """전사 경로 분리: 대화 청취는 필터판(_transcribe_listen), 긴급 검증은"""
    texts, events = [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        predict=lambda f: {"a": 0.0, "b": 0.0},
        transcribe=lambda a: "멈춰",
    )
    m._transcribe_listen = lambda a: "화장실 가자"

    m._open_listen(followup=False, now=0.0)
    for i in range(5):
        m.process_frame(LOUD, now=i * 0.08, vad=True)
    for i in range(30):
        m.process_frame(QUIET, now=0.4 + i * 0.08)
    assert texts == ["화장실 가자"]

    m._enter_postroll()
    for i in range(4):
        m.process_frame(LOUD, now=2.0 + i * 0.08)
    assert len(events) == 1
    assert events[0].keyword == "멈춰"


def test_capture_stats_numbers():
    from src.wakeword_monitor import capture_stats

    silence = np.zeros(1600, dtype=np.int16)
    s = capture_stats(silence)
    assert s == {"rms": 0.0, "peak": 0.0, "clip_ratio": 0.0}

    clipped = np.full(100, 32767, dtype=np.int16)
    s = capture_stats(clipped)
    assert s["peak"] > 0.99 and s["clip_ratio"] == 1.0

    normal = np.full(100, 3000, dtype=np.int16)
    s = capture_stats(normal)
    assert 0.05 < s["rms"] < 0.15 and s["clip_ratio"] == 0.0


class TestStripRobotEcho:
    """에코 방어(2026-09-01) — 야간 실기의 실제 전사 4건이 기준이다."""

    ROBOT = [
        "화장실로 안내해 드릴까요?",
        "잘 듣지 못했습니다. 다시 말씀해 주세요.",
        "무슨 뜻인지 잘 모르겠어요.",
        "네?",
    ]

    def test_full_echo_is_dropped(self):
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("무슨 뜻인지 잘 모르겠어요.", self.ROBOT) == ""

    def test_partial_transcript_of_robot_ment_is_dropped(self):
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("다시 말씀해주세요.", self.ROBOT) == ""

    def test_answer_after_echo_survives(self):
        from src.stt_guard import strip_robot_echo
        out = strip_robot_echo("화장실로 안내해 드릴까요? 어.", self.ROBOT)
        assert "어" in out and "안내해" not in out

    def test_glued_answer_survives_without_punctuation(self):
        from src.stt_guard import strip_robot_echo
        out = strip_robot_echo("화장실로 안내해 드릴까요 어", self.ROBOT)
        assert out == "어"

    def test_short_affirmation_is_never_treated_as_echo(self):
        """로봇이 방금 "네?" 라고 했어도 사용자의 "네."는 살아야 한다."""
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("네.", self.ROBOT) == "네."
        assert strip_robot_echo("어.", self.ROBOT) == "어."

    def test_normal_utterance_passes_unchanged(self):
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("화장실 가고 싶어.", self.ROBOT) == "화장실 가고 싶어."

    def test_no_recent_robot_speech_passes(self):
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("응 대기해 줘.", []) == "응 대기해 줘."

    def test_answer_repeating_words_of_the_question_survives(self):
        """질문 속 장소 이름을 따라 한 대답은 메아리가 아니다 — 10-16 실기에서 "남자 화장실"이 두 번 버려졌다."""
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("남자 화장실.", ["남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"]) == "남자 화장실."
        assert strip_robot_echo("사회 복지창구.", ["사회 복지창구로 안내해드릴까요?"]) == "사회 복지창구."
        assert strip_robot_echo("407호.", ["407호로 안내해드릴까요?"]) == "407호."

    def test_tail_of_any_robot_sentence_is_still_echo(self):
        """여러 문장짜리 로봇 말은 문장마다 끝부분을 본다 — 가운데 문장이 통째로 새어 들어도 버린다."""
        from src.stt_guard import strip_robot_echo
        robot = ["남자 화장실에 도착했습니다. 남자 화장실은 왼쪽에 있습니다. 다녀오시는 동안 여기서 기다릴까요?"]
        assert strip_robot_echo("남자 화장실은 왼쪽에 있습니다.", robot) == ""
        assert strip_robot_echo("여기서 기다릴까요?", robot) == ""

    def test_spelled_numbers_still_match_robot_text(self):
        """로봇은 '407로' 를 말했는데 Qwen 은 '사백칠로' 로 적는다(2026-10-15) — 숫자 정리 뒤 대조한다."""
        from src.stt_guard import strip_robot_echo
        assert strip_robot_echo("사백칠로 안내해 드릴까요?", ["407로 안내해 드릴까요?"]) == ""
        assert strip_robot_echo("사백칠로 안내해 드릴까요? 응.", ["407로 안내해 드릴까요?"]) == "응."


def test_short_answer_ghosts_2026_09_02():
    """짧은답 30회 실기: 사용자가 "네"·"대기해줘"라고 말한 자리에서 나온"""
    assert is_hallucination("다 됐어.") is True
    assert is_hallucination("고맙습니다") is True
    assert is_hallucination("감사합니다.") is True
    assert is_hallucination("다 됐어. 고마워.") is True
    assert is_hallucination("다 됐어. 화장실로 가자.") is False
    assert is_hallucination("이제 됐어") is False


def test_gomawo_alone_is_hallucination_but_mixed_survives():
    """"고마워" 단독은 유령(2026-09-01 실기 — 말한 적 없는데 반복 등장),"""
    from src.stt_guard import is_hallucination
    assert is_hallucination("고마워.") is True
    assert is_hallucination("고마워") is True
    assert is_hallucination("고마워. 대기해.") is False


def test_number_loop_is_hallucination_2026_09_28():
    """2026-09-28 마이크 실기: 사용자 말 대신 숫자 몇 개가 끝없이 반복되는"""
    loop_a = ", ".join(["2", "3"] * 40 + ["4", "3"] * 20)
    loop_b = "1, 2, 1, 2, 1, 2, " + ", ".join(["1", "3", "2"] * 30)
    assert is_hallucination(loop_a) is True
    assert is_hallucination(loop_b) is True
    assert is_hallucination("입구, " + ", ".join(["화장실"] * 20)) is True
    assert is_hallucination(", ".join(["입구", "4층", "실험실 입구"] * 6)) is True


def test_real_repetition_survives():
    """사람도 되풀이한다 — 짧은 반복·섞인 말은 살아야 한다. '네. 아니. 아니."""
    assert is_hallucination("네. 아니. 아니. 아니.") is False
    assert is_hallucination("네 네 네 네") is False
    assert is_hallucination("아니 아니 아니 아니 아니 아니") is False
    assert is_hallucination("2층 3층 말고 4층으로 가 주세요") is False
    assert is_hallucination("방2로 가 줘") is False
