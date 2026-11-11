"""세션 주인 목소리 판정 핵심(설계 2026-11-11-speaker-follow-design.md 4.1·4.2·9절 B1·B3, D13·D14).

가짜 임베딩: 소리 평균이 양수면 A[1,0], 음수면 B[0,1], 0.1 s(1600 샘플) 미만이면 None.
A·B 의 코사인은 1·0 이라 판정 문턱과 무관하게 통과·실패가 갈린다. 문턱 경계는 judge 표 시험이 맡는다.
"""
from __future__ import annotations

import numpy as np

from src.speaker_follow import SessionTracker, SpeakerFollow, SpkConfig, Verdict, judge, spk_config_from_env

ON = SpkConfig(mode="on", doa_center=180.0, doa_width=60.0)


def voice(sign: float, sec: float = 1.0) -> np.ndarray:
    return np.full(int(16000 * sec), int(sign * 0.2 * 32768), dtype=np.int16)


class FakeEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        if len(audio) < 1600:
            return None
        return np.array([1.0, 0.0]) if float(np.mean(audio)) > 0 else np.array([0.0, 1.0])


def follow(cfg: SpkConfig = ON):
    emb = FakeEmbed()
    return SpeakerFollow(cfg, emb, log=lambda m: None), emb


def started(f: SpeakerFollow, now: float = 10.0) -> SpeakerFollow:
    """A 가 '비카야 … 화장실 가자' 를 말해 세션이 시작된 상태."""
    f.note_wake_audio(voice(+1, 0.8), now - 1.0)
    f.check(voice(+1, 1.5), True, 180.0, 1.5, now, "화장실 가자")
    f.on_dialog_state("confirming", now + 1.0)
    return f


# --- judge 표 -----------------------------------------------------------------------------------------------

def test_judge_table():
    kw = dict(has_target=True, guiding=True, waiting=False)
    assert judge(0.60, True, 2.0, 0.0, ON, **kw)[:2] == ("pass", False)
    assert judge(0.52, True, 2.0, 0.0, ON, **kw)[:2] == ("fail", True)        # 비카야 문턱 0.55
    assert judge(0.52, False, 2.0, 0.0, ON, **kw)[:2] == ("pass", False)      # 발화만 문턱 0.50
    assert judge(0.10, False, 2.0, 200.0, ON, **kw)[:2] == ("pass", False)    # 손잡이 방향(180±60)
    assert judge(0.10, False, 2.0, 245.0, ON, **kw)[:2] == ("fail", True)     # 부채꼴 밖
    assert judge(0.10, False, 0.8, 0.0, ON, **kw)[:2] == ("short", False)     # D13
    assert judge(0.10, False, 2.0, 0.0, ON, has_target=True, guiding=False, waiting=False)[:2] == ("fail", False)


def test_doa_wraps_around_zero():
    cfg = SpkConfig(mode="on", doa_center=10.0, doa_width=30.0)
    assert judge(0.1, False, 2.0, 350.0, cfg, has_target=True, guiding=True, waiting=False)[0] == "pass"


def test_waiting_is_voice_only():
    kw = dict(has_target=True, guiding=True, waiting=True)
    assert judge(0.10, False, 0.6, 180.0, ON, **kw)[:2] == ("fail", False)    # 짧아도·손잡이 방향이어도
    assert judge(0.56, True, 2.0, None, ON, **kw)[:2] == ("pass", False)
    assert judge(None, True, 2.0, None, ON, **kw)[:2] == ("fail", False)


def test_no_target():
    assert judge(0.9, True, 2.0, None, ON, has_target=False, guiding=True, waiting=False)[:2] == ("no_target", False)


def test_log_mode_never_drops():
    cfg = SpkConfig(mode="log", doa_center=180.0, doa_width=60.0)
    assert judge(0.1, False, 2.0, 0.0, cfg, has_target=True, guiding=True, waiting=False) [:2] == ("fail", False)


def test_score_none_while_guiding_is_not_dropped():
    assert judge(None, False, 2.0, None, ON, has_target=True, guiding=True, waiting=False)[:2] == ("short", False)


# --- 설정 ---------------------------------------------------------------------------------------------------

def test_config_from_env():
    cfg = spk_config_from_env({"VICA_SPK_FOLLOW": "log", "VICA_SPK_MODEL": "/m.onnx", "VICA_SPK_TH_WAKE": "0.6",
                               "VICA_SPK_TH": "0.45", "VICA_SPK_SHORT_SEC": "0.9",
                               "VICA_USER_DOA_CENTER": "180", "VICA_USER_DOA_WIDTH": "60"})
    assert (cfg.mode, cfg.model_path, cfg.th_wake, cfg.th, cfg.short_sec, cfg.doa_center, cfg.doa_width) \
        == ("log", "/m.onnx", 0.6, 0.45, 0.9, 180.0, 60.0)


def test_config_defaults_and_unknown_mode():
    cfg = spk_config_from_env({"VICA_SPK_FOLLOW": "yes"})
    assert (cfg.mode, cfg.th_wake, cfg.th, cfg.short_sec, cfg.doa_center) == ("off", 0.55, 0.50, 1.0, None)


# --- 세션 경계 ------------------------------------------------------------------------------------------------

def test_session_tracker():
    t = SessionTracker()
    seq = ("idle", "grip_wait", "confirming", "navigating", "waiting", "idle")
    assert [t.on_dialog_state(s) for s in seq] == [None, None, "start", None, None, "end"]


def test_session_tracker_flags():
    t = SessionTracker()
    t.on_dialog_state("confirming")
    assert t.active and not t.guiding
    t.on_dialog_state("navigating")
    t.on_dialog_state("waiting")
    assert t.guiding and t.waiting
    t.on_dialog_state("estopped")
    assert not t.active and not t.guiding


def test_repeated_state_messages_no_extra_events():
    t = SessionTracker()
    assert [t.on_dialog_state(s) for s in ("confirming", "confirming", "navigating", "navigating")] \
        == ["start", None, None, None]


# --- SpeakerFollow ------------------------------------------------------------------------------------------

def test_enroll_from_last_utterance_on_start():
    f = started(follow()[0])
    assert f.has_target
    f.on_dialog_state("navigating", 12.0)
    assert f.check(voice(+1, 1.5), False, 0.0, 1.5, 13.0).verdict == "pass"
    v = f.check(voice(-1, 1.5), False, 0.0, 1.5, 14.0)
    assert (v.verdict, v.drop) == ("fail", True)


def test_start_without_recent_utterance():
    f, _ = follow()
    f.check(voice(+1, 1.5), False, None, 1.5, 0.0)
    f.on_dialog_state("confirming", 11.0)                 # 11 s 지난 발화뿐(앱으로 목적지)
    assert not f.has_target
    assert f.check(voice(-1, 1.5), False, None, 1.5, 12.0).verdict == "no_target"


def test_end_clears_target():
    f = started(follow()[0])
    f.on_dialog_state("estopped", 20.0)
    assert not f.has_target


def test_wake_audio_prepended_in_wake_window():
    f = started(follow()[0])
    f.note_wake_audio(voice(+1, 0.8), 30.0)
    assert f.check(voice(+1, 1.5), True, None, 1.5, 31.0).unit == "wake+utt"
    assert f.check(voice(+1, 1.5), False, None, 1.5, 32.0).unit == "utt"


def test_joined_uses_previous_within_20s():
    f = started(follow()[0])
    f.on_dialog_state("navigating", 11.0)
    f.on_dialog_state("waiting", 12.0)                                      # 되묻기는 대기 중에만 쓴다(I4)
    assert f.check(voice(+1, 1.5), False, None, 1.5, 20.0).joined > 0.9     # 시작 발화(10 s)와 이어 붙임
    assert f.check(voice(+1, 1.5), False, None, 1.5, 45.0).joined == -1.0    # 25 s 지남


def test_joined_not_computed_while_guiding():
    """I4: 안내 중에는 이어 붙인 점수를 계산하지 않는다 — 임베딩 1 회로 감시 스레드 지연을 줄인다."""
    f, emb = follow()
    started(f)
    f.on_dialog_state("navigating", 12.0)
    before = emb.calls
    v = f.check(voice(+1, 1.5), False, None, 1.5, 13.0)
    assert (v.joined, emb.calls - before) == (-1.0, 1)


def test_embed_none():
    f = started(follow()[0])
    f.on_dialog_state("navigating", 12.0)
    v = f.check(voice(-1, 0.05), False, None, 1.2, 13.0)                    # 임베딩 None, 안내 중
    assert (v.verdict, v.drop) == ("short", False)
    f.on_dialog_state("waiting", 14.0)
    assert f.check(voice(-1, 0.05), True, None, 1.2, 15.0).verdict == "fail"


def test_off_mode_does_not_embed():
    f, emb = follow(SpkConfig(mode="off"))
    v = f.check(voice(+1, 1.5), True, None, 1.5, 0.0)
    assert (v.verdict, v.drop, emb.calls) == ("no_target", False, 0)


def test_verdict_json_keys():
    v = Verdict("pass", 0.61, -1.0, "wake+utt", 1.5, 182.0, False, "목소리", "화장실 가자")
    import json
    d = json.loads(v.to_json())
    assert d == {"verdict": "pass", "score": 0.61, "joined": -1.0, "unit": "wake+utt", "speech_sec": 1.5,
                 "doa": 182.0, "reason": "목소리", "text": "화장실 가자"}
