"""유령 STT 방어 검증 — 무음 환각이 사용자 발화로 둔갑하면 안 된다.

stt_guard 순수 함수와, 청취 창에서 환각 전사가 wake_silent 로 처리되는
monitor 통합을 본다.
"""
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


# ---------------------------------------------------------------- 신뢰도 필터
def test_drops_segment_only_when_both_signals_say_silence():
    """no_speech 높음 + logprob 낮음이 '동시에' 참일 때만 버린다 —
    작게 말한 진짜 발화(낮은 확신 하나만)는 살아야 한다."""
    both_bad = Seg("시청해주셔서 감사합니다", no_speech_prob=0.9, avg_logprob=-1.5)
    quiet_real = Seg("화장실", no_speech_prob=0.3, avg_logprob=-1.5)   # 확신만 낮음
    noisy_real = Seg("가고 싶어요", no_speech_prob=0.9, avg_logprob=-0.4)  # 무음확률만 높음

    assert accept_segments([both_bad]) == ""
    assert accept_segments([quiet_real]) == "화장실"
    assert accept_segments([both_bad, quiet_real, noisy_real]) == "화장실가고 싶어요"


def test_accepts_plain_segments_and_stub_without_attrs():
    class Bare:
        text = "응"

    assert accept_segments([Seg("네 맞아요")]) == "네 맞아요"
    assert accept_segments([Bare()]) == "응"
    assert accept_segments([]) == ""


# ---------------------------------------------------------------- 수배 전단
def test_known_hallucinations_full_match():
    assert is_hallucination("시청해 주셔서 감사합니다.")
    assert is_hallucination("시청해주셔서 감사합니다")
    assert is_hallucination("구독과 좋아요 부탁드립니다!")
    assert is_hallucination("MBC 뉴스 김성현입니다.")


def test_real_speech_is_not_flagged():
    assert not is_hallucination("화장실 데려다줘")
    # "감사합니다" 단독은 2026-09-02 사용자 결정으로 환각 목록에 들어갔다.
    # 섞인 발화는 여전히 살아야 한다 — 그 경계가 이 시험의 몫이다.
    assert not is_hallucination("안내해 주셔서 감사합니다")  # 부분 유사도 아님
    assert not is_hallucination("")


# ---------------------------------------------------------------- monitor 통합
def test_hallucinated_listen_window_closes_quietly():
    """소음이 RMS 문지기를 뚫고 whisper 가 환각을 지어내도, user_text 로
    LLM 에 가면 안 된다 — 조용히 닫힌다(wake_silent)."""
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
        # 칩이 발화로 오판한 소음 (사람 말 유사음 등)
        results.append(m.process_frame(LOUD, now=i * 0.08, vad=True))
    for i in range(30):   # 최소 개방 2.5초(9/1)를 넘겨 마감
        results.append(m.process_frame(QUIET, now=0.4 + i * 0.08))

    assert "wake_silent" in results   # 최소 개방(9/1) 탓에 마감이 루프 중간에 온다
    assert texts == []


def test_listen_prefers_guarded_transcribe_but_emergency_uses_raw():
    """전사 경로 분리: 대화 청취는 필터판(_transcribe_listen), 긴급 검증은
    무필터판(_transcribe)을 써야 한다 — 작은 외침 보존이 우선이다
    (외침 10회 실측에서 빈 전사 기각 1건)."""
    texts, events = [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        predict=lambda f: {"a": 0.0, "b": 0.0},
        transcribe=lambda a: "멈춰",          # 무필터판 (긴급 검증용)
    )
    m._transcribe_listen = lambda a: "화장실 가자"  # 필터판 (대화용)

    m._open_listen(followup=False, now=0.0)
    for i in range(5):
        m.process_frame(LOUD, now=i * 0.08, vad=True)
    for i in range(30):   # 최소 개방 2.5초(9/1)를 넘겨 마감
        m.process_frame(QUIET, now=0.4 + i * 0.08)
    assert texts == ["화장실 가자"]           # 대화 경로가 필터판을 썼다

    m._enter_postroll()
    for i in range(4):
        m.process_frame(LOUD, now=2.0 + i * 0.08)
    assert len(events) == 1                   # 긴급 경로는 무필터판("멈춰")
    assert events[0].keyword == "멈춰"


# ---------------------------------------------------------------- 수음 계측
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
        "네?",   # 짧은 발화는 대조에서 빠져야 한다
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


def test_short_answer_ghosts_2026_09_02():
    """짧은답 30회 실기: 사용자가 "네"·"대기해줘"라고 말한 자리에서 나온
    문구들(사용자 증언 — 한 적 없는 말). 둘 다 finish 로 해석돼 안내가
    통째로 끝났다. 진짜 말이 한 조각이라도 섞이면 살려 둔다."""
    assert is_hallucination("다 됐어.") is True
    assert is_hallucination("고맙습니다") is True
    assert is_hallucination("감사합니다.") is True
    # 조각이 전부 환각이면 붙어 있어도 기각한다 (30회차 실측).
    assert is_hallucination("다 됐어. 고마워.") is True
    # 진짜 말이 섞이면 통과 — 기각은 사용자의 말을 통째로 버리는 일이다.
    assert is_hallucination("다 됐어. 화장실로 가자.") is False
    assert is_hallucination("이제 됐어") is False   # 진짜 종료 표현은 산다


def test_gomawo_alone_is_hallucination_but_mixed_survives():
    """"고마워" 단독은 유령(2026-09-01 실기 — 말한 적 없는데 반복 등장),
    진짜 내용이 섞인 발화는 살아야 한다."""
    from src.stt_guard import is_hallucination
    assert is_hallucination("고마워.") is True
    assert is_hallucination("고마워") is True
    assert is_hallucination("고마워. 대기해.") is False


def test_number_loop_is_hallucination_2026_09_28():
    """2026-09-28 마이크 실기: 사용자 말 대신 숫자 몇 개가 끝없이 반복되는
    전사가 3분 새 두 번 나왔다(whisper 반복 루프). 목록 문구가 아니라 통과했고,
    LLM 이 '2'를 방2로 읽어 안내를 제안했다. 실측 전사 그대로 쓴다."""
    loop_a = ", ".join(["2", "3"] * 40 + ["4", "3"] * 20)
    loop_b = "1, 2, 1, 2, 1, 2, " + ", ".join(["1", "3", "2"] * 30)
    assert is_hallucination(loop_a) is True
    assert is_hallucination(loop_b) is True
    # 같은 루프가 장소 이름으로도 나온다 — STT 장소 귀띔을 무음에서 되뇐 것.
    # 과거 실기 전사 1,053개 중 58개가 반복 루프였고 대부분이 이 꼴이다.
    assert is_hallucination("입구, " + ", ".join(["화장실"] * 20)) is True
    assert is_hallucination(", ".join(["입구", "4층", "실험실 입구"] * 6)) is True


def test_real_repetition_survives():
    """사람도 되풀이한다 — 짧은 반복·섞인 말은 살아야 한다. '네. 아니. 아니.
    아니.'는 09-02 실기에서 정정 방어의 근거가 된 진짜 발화다."""
    assert is_hallucination("네. 아니. 아니. 아니.") is False
    assert is_hallucination("네 네 네 네") is False
    assert is_hallucination("아니 아니 아니 아니 아니 아니") is False
    assert is_hallucination("2층 3층 말고 4층으로 가 주세요") is False
    assert is_hallucination("방2로 가 줘") is False
