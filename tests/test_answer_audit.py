"""실기 로그에서 로봇 질문·대답 행 뽑기 (설계 2026-11-14 5절, 계획 Task 1)."""
from __future__ import annotations

import pytest

from tools.answer_audit import extract_rows, kind_from_text, parse_log_line


@pytest.mark.parametrize("q,kind", [
    ("네?", "open"), ("어디로 모실까요?", "open"), ("잘 듣지 못했습니다. 어디로 가고 싶으신가요?", "open"),
    ("확인을 위해 한 번 더 말씀해 주세요", "open"),
    ("사회복지로 안내해드릴까요?", "confirm"),
    ("남자 화장실과 여자 화장실 중 어디로 갈까요?", "choice"),
    ("사회복지는 왼쪽에 있습니다. 여기서 대기할까요?", "yesno"), ("여기까지 안내를 마칠까요?", "yesno"),
    ("안녕하세요? 시각장애인 안내로봇 비카입니다. 안내를 받으시겠어요?", "yesno"),
    ("아까 화장실에 가자고 하신 사용자님이신가요?", "yesno"), ("직원의 도움이 필요하신가요?", "yesno"),
    ("안내가 필요 없으신가요?", "yesno_inverted"), ("몇 분쯤 걸리실까요?", "minutes"),
    ("어떤 서류를 내러 가시는지요?", "other"),
])
def test_kind_from_text(q, kind):
    assert kind_from_text(q) == kind


def test_parse_log_line():
    line = "[INFO] [1794399711.125038258] [vica_llm_intent_node]: 입력='네' -> intent=affirm matched= safety=normal"
    t, node, body = parse_log_line(line)
    assert (round(t, 2), node, body) == (1794399711.13, "vica_llm_intent_node",
                                         "입력='네' -> intent=affirm matched= safety=normal")
    assert parse_log_line("그냥 글자") is None


def test_rows_pair_question_answer_reaction_and_map():
    lines = [
        (100.0, "vica_mission_manager", "목적지 9개 로드: /x/vica_map_0630/destinations.yaml"),
        (110.0, "vica_tts_node", "재생[response]: 여기서 대기할까요?"),
        (115.0, "vica_llm_intent_node", "입력='아니' -> intent=deny matched= safety=normal"),
        (115.1, "vica_mission_manager",
         "음성 요청 intent=deny dest=- confirm=False: asking_next -> asking_next 말=['여기까지 안내를 마칠까요?']"),
    ]
    [row] = extract_rows(lines)
    assert (row["kind"], row["answer_text"], row["intent"], row["dialog_state"]) == ("yesno", "아니", "deny", "asking_next")
    assert row["dest_yaml"] == "/x/vica_map_0630/destinations.yaml"


def test_duplicate_overlapping_lines_kept_once():
    q = (10.0, "vica_tts_node", "재생[response]: 여기서 대기할까요?")
    a = (14.0, "vica_llm_intent_node", "입력='네' -> intent=affirm matched= safety=normal")
    assert len(extract_rows([q, a, (14.3, *a[1:])])) == 1


def test_ambient_lines_are_not_questions():
    lines = [(10.0, "vica_tts_node", "재생[ambient]: 비카가 대기 중입니다."),
             (12.0, "vica_llm_intent_node", "입력='네' -> intent=affirm matched= safety=normal")]
    assert extract_rows(lines) == []


def test_same_text_from_mission_or_voice_has_same_kind():
    """Review Focus 5: kind 는 글자로만, q_by 는 기록만."""
    by_mission = [(10.0, "vica_mission_manager", "TTS[response]: 몇 분쯤 걸리실까요?"),
                  (10.2, "vica_tts_node", "재생[response]: 몇 분쯤 걸리실까요?"),
                  (14.0, "vica_llm_intent_node", "입력='십 분' -> intent=wait matched= safety=normal")]
    by_voice = [(10.2, "vica_tts_node", "재생[response]: 몇 분쯤 걸리실까요?"),
                (14.0, "vica_llm_intent_node", "입력='십 분' -> intent=wait matched= safety=normal")]
    [m], [v] = extract_rows(by_mission), extract_rows(by_voice)
    assert (m["kind"], v["kind"]) == ("minutes", "minutes")
    assert (m["q_by"], v["q_by"]) == ("mission", "voice")


def test_rt_line_attaches_reply_and_path():
    lines = [(10.0, "vica_tts_node", "재생[response]: 네?"),
             (14.0, "vica_llm_intent_node",
              "[RT] heard='화장실' intent=navigate dest=d2 nc=True conf=0.90 reply='화장실로 안내해드릴까요?' src=llm dt=1.2s"),
             (14.01, "vica_llm_intent_node", "입력='화장실' -> intent=navigate matched=d2 safety=normal"),
             (14.02, "vica_llm_intent_node", "[A/B] audio=navigate/d2 1.2s | text=꺼짐 | heard='화장실' | whisper='화장실.'")]
    [row] = extract_rows(lines)
    assert (row["path"], row["nc"], row["llm_reply"], row["stt"], row["dest_id"]) == (
        "audio", "True", "화장실로 안내해드릴까요?", "화장실.", "d2")
