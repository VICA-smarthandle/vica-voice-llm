"""화자 판정을 의도에 싣는 연결(계획 작업 6) — 판정 보관함·메시지 변환·대기 중 미션이 말하기(B5)."""
from __future__ import annotations

import json

from src.mission_question import quiet_for_mission
from src.ros_convert import intent_to_msg
from src.schema import VicaIntent
from src.speaker_link import SpeakerLink


def _j(text: str, verdict: str = "pass", score: float = 0.6, joined: float = -1.0) -> str:
    return json.dumps({"verdict": verdict, "score": score, "joined": joined, "unit": "utt",
                       "speech_sec": 1.2, "doa": None, "reason": "목소리", "text": text}, ensure_ascii=False)


def test_take_matches_text_and_consumes():
    link = SpeakerLink()
    link.put(_j("화장실 가자", "fail", 0.42, 0.61), 0.0)
    got = link.take("화장실 가자", 1.0)
    assert (got["verdict"], got["score"], got["joined"]) == ("fail", 0.42, 0.61)
    assert link.take("화장실 가자", 1.5) is None


def test_take_latest_of_same_text():
    link = SpeakerLink()
    link.put(_j("취소해", "fail"), 0.0)
    link.put(_j("취소해", "pass"), 2.0)
    assert link.take("취소해", 3.0)["verdict"] == "pass"


def test_take_expires_after_ttl():
    link = SpeakerLink(ttl_sec=15.0)
    link.put(_j("취소해"), 0.0)
    assert link.take("취소해", 16.0) is None


def test_take_none_when_absent_or_bad_json():
    link = SpeakerLink()
    link.put("{깨진", 0.0)
    assert link.take("취소해", 0.1) is None


def test_intent_to_msg_carries_speaker_fields():
    intent = VicaIntent(intent="cancel", speaker_verdict="fail", speaker_score=0.42, speaker_joined=0.61)
    msg = intent_to_msg(intent)
    assert (msg.speaker_verdict, round(msg.speaker_score, 2), round(msg.speaker_score_joined, 2)) == ("fail", 0.42, 0.61)


def test_intent_defaults_empty_speaker():
    msg = intent_to_msg(VicaIntent(intent="question"))
    assert (msg.speaker_verdict, msg.speaker_score, msg.speaker_score_joined) == ("", -1.0, -1.0)


def _quiet(kind: str, state: str, spk_on: bool):
    intent = VicaIntent(intent=kind, reply="화장실로 안내해드릴까요?", need_confirm=True)
    return quiet_for_mission(intent, state, "", 99.0, spk_on=spk_on).reply


def test_quiet_for_mission_waiting_gated_kinds():
    for kind in ("navigate", "cancel", "finish", "resume", "wait"):
        assert _quiet(kind, "waiting", True) == "", kind
        assert _quiet(kind, "waiting_release", True) == "", kind
    assert _quiet("question", "waiting", True) != ""


def test_quiet_for_mission_unchanged_when_off_or_not_waiting():
    assert _quiet("navigate", "waiting", False) != ""
    assert _quiet("navigate", "navigating", True) != ""


# --- 최종 검토 수정(I2·I3·C2) ----------------------------------------------------------------------------------

def test_quiet_waiting_resume_is_sent_without_confirm():
    """I2: 대기 중 '다시 가자' — 확인 전 보류되면 말도 미션 전달도 없다. 확정으로 보내 미션이 어디로 갈지 묻는다."""
    intent = VicaIntent(intent="resume", reply="다시 출발할까요?", need_confirm=True)
    out = quiet_for_mission(intent, "waiting", "", 99.0, spk_on=True)
    assert (out.reply, out.need_confirm) == ("", False)


def test_intent_to_msg_tolerates_old_interface(monkeypatch):
    """I3: 옛 vica_interfaces 설치본(__slots__ 에 새 필드 없음)에서도 의도 발행이 깨지지 않는다."""
    import src.ros_convert as rc

    class OldMsg:
        __slots__ = ("intent", "destination_candidate", "matched_destination_id", "confidence", "need_confirm",
                     "reply", "safety_flag", "wait_minutes")

    monkeypatch.setattr(rc, "VicaIntentMsg", OldMsg)
    msg = rc.intent_to_msg(VicaIntent(intent="cancel", speaker_verdict="pass"))
    assert msg.intent == "cancel"


def test_verdict_label_round_trip():
    """C2: 판정을 /vica/user_audio 의 layout.dim 라벨에 실어 소리와 원자적으로 짝짓는다."""
    from src.speaker_link import verdict_from_labels, verdict_label

    label = verdict_label(_j("", "fail", 0.31, -1.0))
    assert label.startswith("spk:")
    got = verdict_from_labels(["pcm16_mono_16000", label])
    assert (got["verdict"], got["score"]) == ("fail", 0.31)
    assert verdict_from_labels(["pcm16_mono_16000"]) is None
    assert verdict_from_labels(["pcm16_mono_16000", "spk:{깨진"]) is None
