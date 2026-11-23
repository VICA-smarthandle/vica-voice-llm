"""예약 문장(설계 2026-11-18 6절) — 글자가 사용자 확정본과 같아야 한다."""
from datetime import date

from src import reservation_phrases as p


def test_fixed_sentences_are_the_confirmed_text():
    assert p.ASK_NAME == "성함을 말씀해 주세요."
    assert p.DONE == "네, 예약이 완료되었습니다. 이제 어디로 갈까요?"
    assert p.TAKEN == "그 시간에는 이미 예약이 있습니다. 다른 시간을 말씀해 주세요."
    assert p.ASK_WHEN == "언제로 예약할까요?"
    assert p.NOT_RESERVED == "예약하지 않았습니다."
    assert p.ASK_WHERE == "어디로 예약할까요?"
    assert p.BAD_TIME == "그 시간에는 예약할 수 없습니다. 다른 시간을 말씀해 주세요."
    assert p.UNAVAILABLE == "지금은 예약을 받을 수 없습니다."
    assert p.CANCELLED == "예약이 취소되었습니다."          # ⑩ 11-19 사용자
    assert len(p.FIXED) == 9 and len(set(p.FIXED)) == 9


def test_confirm_sentence_is_the_demo_line():
    line = p.confirm_sentence("김민수", date(2026, 11, 19), "14:00", "16:00", "세미나실", date(2026, 11, 18))
    assert line == "김민수 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?"


def test_fixed_sentences_are_prewarmed_only_when_a_tool_switch_is_on(monkeypatch, tmp_path):
    """스위치가 둘 다 꺼져 있으면 미리 합성 목록이 지금과 같다(TTS 기동 시간 그대로)."""
    from src import mission_phrases as mp
    from src import tool_calls

    env_file = tmp_path / ".env"                             # 운영 .env 대신 — TTS 노드는 이 파일에서 스위치를 읽는다
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(tool_calls, "ENV_FILE", env_file)
    monkeypatch.delenv("VICA_TOOLS_TEXT", raising=False)
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    off = mp.standalone_prewarm([])
    assert not set(p.FIXED) & set(off)
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
    on = mp.standalone_prewarm([])
    assert on[:len(off)] == off and on[len(off):] == list(p.FIXED)
    monkeypatch.delenv("VICA_TOOLS_AUDIO")                   # 환경 변수 없이 .env 파일에만 켜 둔 경우(TTS 노드)
    env_file.write_text("VICA_TOOLS_TEXT=on\n", encoding="utf-8")
    assert mp.standalone_prewarm([]) == on
