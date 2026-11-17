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


def test_voice_node_map_line_wins_over_mission():
    """검토 I1: 해석기가 실제로 쓴 목록은 음성 노드의 로드 줄이다."""
    lines = [
        (100.0, "vica_mission_manager", "목적지 9개 로드: /x/old/destinations.yaml"),
        (105.0, "vica_llm_intent_node", "목적지 6개 로드(비공개 1개는 접근 불가로): /y/new/destinations.yaml"),
        (110.0, "vica_tts_node", "재생[response]: 네?"),
        (115.0, "vica_llm_intent_node", "입력='408호로 가자' -> intent=navigate matched=x safety=normal"),
    ]
    [row] = extract_rows(lines)
    assert row["dest_yaml"] == "/y/new/destinations.yaml"


def test_no_map_line_leaves_dest_yaml_empty():
    lines = [(110.0, "vica_tts_node", "재생[response]: 네?"),
             (115.0, "vica_llm_intent_node", "입력='화장실' -> intent=navigate matched=x safety=normal")]
    assert extract_rows(lines)[0]["dest_yaml"] == ""


# ---- 질문 기록 로그(설계 4.1, 2단계 — 계획 2026-11-17 Task 6) --------------------------------
from tools.answer_audit import COLUMNS, question_log_summary  # noqa: E402


def _qa(extra):
    return [(110.0, "vica_tts_node", "재생[response]: 화장실로 안내해드릴까요?"),
            (115.0, "vica_llm_intent_node", "입력='그래' -> intent=affirm matched= safety=normal"), *extra]


def test_row_takes_question_id_from_voice_line():
    [row] = extract_rows(_qa([(115.01, "vica_llm_intent_node", "[질문] id=q1794-2 kind=confirm 나이=3.0s")]))
    assert (row["q_id"], row["q_kind"], row["q_age_s"]) == ("q1794-2", "confirm", "3.0")


def test_row_without_record_has_empty_fields():
    [row] = extract_rows(_qa([(115.01, "vica_llm_intent_node", "[질문] 없음")]))
    assert (row["q_id"], row["q_kind"], row["q_age_s"]) == ("", "", "")


def test_old_logs_without_question_lines_still_parse():
    [row] = extract_rows(_qa([]))
    assert (row["q_id"], row["q_kind"], row["q_age_s"]) == ("", "", "") and row["kind"] == "confirm"


def test_question_line_of_a_later_utterance_is_not_borrowed():
    [row] = extract_rows(_qa([(118.0, "vica_llm_intent_node", "[질문] id=q1794-3 kind=open 나이=0.5s")]))
    assert row["q_id"] == ""


def test_new_columns_go_at_the_end():
    assert COLUMNS[-3:] == ["q_id", "q_kind", "q_age_s"]


def test_question_log_summary_counts_opens_and_close_reasons():
    lines = [
        (100.0, "vica_mission_manager", "[질문] 열림 id=q1-1 kind=open 후보=0 요청=n 글자=네?"),
        (104.0, "vica_mission_manager", "[질문] 닫힘 id=q1-1 사유=answer 열린 시간=4.0s"),
        (105.0, "vica_mission_manager", "[질문] 열림 id=q1-2 kind=confirm 후보=1 요청=y 글자=화장실로 안내해드릴까요?"),
        (105.0, "vica_mission_manager", "질문 발화 — 재청취 요청 발행"),
        (111.0, "vica_mission_manager", "[질문] 닫힘 id=q1-2 사유=silence 열린 시간=6.0s"),
        (120.0, "vica_mission_manager", "[질문] 열림 id=q1-3 kind=yesno 후보=0 요청=y 글자=여기서 대기할까요?"),
        (120.0, "vica_mission_manager", "질문 발화 — 재청취 요청 발행"),
        (151.0, "vica_mission_manager", "[질문] 닫힘 id=q1-3 사유=fault 열린 시간=31.0s"),
    ]
    # 호출 응답 "네?"는 듣기 요청 없이 열린다(요청=n) — 듣기 요청 수와 맞춰 볼 것은 요청=y 인 열림이다.
    assert question_log_summary(lines) == {"opened": 3, "opened_with_request": 2, "listen_requests": 2,
                                           "closed": {"answer": 1, "silence": 1, "fault": 1}, "max_open_s": 31.0}
