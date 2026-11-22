"""소리→의도 직행 경로 시험 (Realtime 은 가짜 클라이언트)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _shortcut_intent, parse_intent, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData

DEST = DestinationData(id="starlight_1f_restroom", name="별빛관 1층 화장실",
                       confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?")
PCM = b"\x00\x00" * 1600


class FakeRT:
    def __init__(self, draft, heard, fail=None):
        self.draft, self.heard, self.fail = draft, heard, fail
        self.calls = []

    def ask(self, pcm, history, instructions):
        self.calls.append((pcm, list(history or []), instructions))
        if self.fail:
            raise self.fail
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7,
                              usage={"audio_tokens": 20})


@pytest.fixture
def rt(monkeypatch):
    holder = {}

    def use(draft, heard, fail=None):
        holder["c"] = FakeRT(draft, heard, fail)
        monkeypatch.setattr(parser, "get_realtime_client", lambda: holder["c"])
        return holder["c"]

    return use


class TestShortcutExtraction:
    def test_text_path_unchanged_for_wake_word(self):
        assert parse_intent("비카야", [DEST]).intent == "unknown"

    def test_shortcut_returns_none_for_normal_sentence(self):
        assert _shortcut_intent("화장실로 안내해줘", None, [DEST]) is None

    def test_shortcut_pending_affirm(self):
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        out = _shortcut_intent("그래", hist, [DEST])
        assert out.intent == "navigate" and out.matched_destination_id == DEST.id


class TestAudioPath:
    """소리 모드 = 모델 전결. 코드는 id 매핑·접근 불가·계약상 빈 reply 만 한다."""

    def test_new_destination_uses_model_decision(self, rt):
        c = rt({"intent": "navigate", "destination_candidate": "별빛관 1층 화장실",
                "need_confirm": True, "reply": DEST.confirm_prompt, "confidence": 0.9}, "화장실로 가줘")
        intent, heard, dt, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is True and intent.reply == DEST.confirm_prompt
        assert intent.confidence == pytest.approx(0.9)
        assert heard == "화장실로 가줘" and dt == pytest.approx(0.7)
        assert info["src"] == "llm" and info["usage"] == {"audio_tokens": 20}
        # 지시문: 전결 규칙과 목록의 확인 문구가 들어간다
        assert "need_confirm" in c.calls[0][2] and DEST.confirm_prompt in c.calls[0][2]

    def test_confirmed_navigate_is_silent_for_mission(self, rt):
        rt({"intent": "navigate", "destination_candidate": DEST.name, "need_confirm": False,
            "reply": "출발합니다"}, "네")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is False and intent.reply == ""   # 계약: 출발 안내는 미션이 말한다

    def test_correction_with_negative_word_is_not_deny(self, rt):
        # 09-20 실기 #19: 텍스트 경로는 부정어 규칙으로 '취소'했다. 전결 모드는 모델 결정 그대로.
        rt({"intent": "navigate", "destination_candidate": DEST.name, "need_confirm": True,
            "reply": ""}, "아니 화장실로 가자")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.need_confirm is True
        assert intent.reply == DEST.confirm_prompt      # 모델이 비우면 목록의 확인 문구

    def test_wake_word_is_no_longer_a_shortcut(self, rt):
        rt({"intent": "unknown", "reply": ""}, "비카야")
        intent, _, _, info = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "unknown" and intent.reply == "" and info["src"] == "llm"

    def test_pending_affirm_is_model_business(self, rt):
        # 확인 대기 판정을 코드가 하지 않는다 — 모델이 affirm 이라 하면 affirm 그대로.
        rt({"intent": "affirm", "reply": "네"}, "넵")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _, _ = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "affirm" and intent.reply == "" and intent.need_confirm is False

    def test_wait_uses_model_minutes(self, rt):
        rt({"intent": "wait", "reply": "알겠습니다", "wait_minutes": 15}, "한 십 분에서 십오 분")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "wait" and intent.wait_minutes == 15
        assert intent.reply == "" and intent.need_confirm is False

    def test_wait_without_minutes_is_minus_one(self, rt):
        rt({"intent": "wait", "reply": ""}, "좀 있다 올게")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.wait_minutes == -1

    @pytest.mark.parametrize("name", ["affirm", "deny", "finish"])
    def test_mission_spoken_intents_have_blank_reply(self, rt, name):
        rt({"intent": name, "reply": "네 알겠습니다", "need_confirm": True}, "응")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == name and intent.reply == "" and intent.need_confirm is False

    def test_unknown_destination_becomes_clarify(self, rt):
        rt({"intent": "navigate", "destination_candidate": "옥상", "need_confirm": True, "reply": ""}, "옥상 가자")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "clarify" and intent.reply == parser.ASK_DESTINATION
        assert intent.need_confirm is False and not intent.matched_destination_id

    def test_unapproachable_destination_keeps_reason(self, rt):
        blocked = DestinationData(id="mech", name="기계실", is_approachable=False,
                                  unavailable_reason="기계실은 안내할 수 없습니다.")
        rt({"intent": "navigate", "destination_candidate": "기계실", "need_confirm": True, "reply": ""}, "기계실")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST, blocked])
        assert intent.intent == "navigate" and intent.matched_destination_id == "mech"
        assert intent.reply == "기계실은 안내할 수 없습니다." and intent.need_confirm is False

    def test_cancel_first_asks_then_confirmed_is_silent(self, rt):
        rt({"intent": "cancel", "need_confirm": True, "reply": ""}, "취소해")
        first, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert first.need_confirm is True and first.reply == parser.CANCEL_CONFIRM
        rt({"intent": "cancel", "need_confirm": False, "reply": "취소했습니다"}, "응")
        second, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert second.need_confirm is False and second.reply == ""

    def test_pause_gets_default_ack(self, rt):
        rt({"intent": "pause", "reply": ""}, "잠깐만")
        intent, _, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "pause" and intent.reply == parser.PAUSE_ACK and intent.need_confirm is False

    def test_history_is_handed_to_model_untouched(self, rt):
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt), AIMessage("몇 분쯤 걸리실까요?")]
        c = rt({"intent": "wait", "reply": "", "wait_minutes": 5}, "오 분")
        parse_intent_audio(PCM, [DEST], history=hist)
        assert c.calls[0][1] == hist

    def test_realtime_failure_propagates(self, rt):
        rt({}, "", fail=TimeoutError("8초"))
        with pytest.raises(TimeoutError):
            parse_intent_audio(PCM, [DEST])

    def test_bad_draft_raises_value_error(self, rt):
        rt({"intent": "not-an-intent", "reply": ""}, "x")
        with pytest.raises(ValueError):
            parse_intent_audio(PCM, [DEST])


class TestAudioPrompt:
    def test_prompt_lists_confirm_phrase_and_blocked(self):
        blocked = DestinationData(id="mech", name="기계실", is_approachable=False)
        text = parser.build_audio_prompt([DEST, blocked])
        assert DEST.confirm_prompt in text and "기계실" in text and "접근 불가" in text
        assert "이력에 있는 말을 베껴 적지 마라" in text

    def test_tool_schema_has_need_confirm(self):
        from src.realtime_intent import build_intent_tool
        assert "need_confirm" in build_intent_tool()["parameters"]["properties"]

    def test_prompt_lists_building_and_floor(self):
        d = DestinationData(id="r407", name="407호", aliases=["407호", "윤지영"], building="로봇관", floor=4,
                            confirm_prompt="407호로 안내해드릴까요?")
        text = parser.build_audio_prompt([d])
        assert "위치: 로봇관 4층" in text and "윤지영" in text
        assert "기억이 없다고 하지 마라" in text

    def test_prompt_includes_situation_block(self):
        text = parser.build_audio_prompt([DEST], situation="\n[지금 상황]\n- 마지막 도착: 407호 (3분 전)\n")
        assert "마지막 도착: 407호" in text

    def test_parse_audio_passes_situation(self, rt):
        c = rt({"intent": "question", "reply": "407호예요."}, "아까 어디 갔었지?")
        parse_intent_audio(PCM, [DEST], situation="\n[지금 상황]\n- 마지막 도착: 407호 (방금)\n")
        assert "마지막 도착: 407호" in c.calls[0][2]

    def test_dynamic_blocks_come_last_for_prompt_cache(self):
        # 앞부분(역할·규칙·목적지)이 호출마다 같아야 Realtime 캐시가 먹는다 — 바뀌는 블록은 맨 뒤.
        from src.schema import RobotState
        a = parser.build_audio_prompt([DEST], RobotState(), situation="\n[지금 상황]\n- 안내 중: 없음\n")
        b = parser.build_audio_prompt([DEST], RobotState(is_moving=True), situation="\n[지금 상황]\n- 안내 중: 화장실\n")
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        assert prefix > len(a) * 0.8            # 8할 이상이 공통 접두
        assert a.index("[현재 로봇 상태]") > a.index("[말투]")
        assert a.rindex("[지금 상황]") > a.index("[현재 로봇 상태]")  # rindex: 실제 블록 헤더 (규칙 텍스트의 참조 아님)

    def test_ledger_dynamic_blocks_come_last_for_prompt_cache(self):
        # 대장(render_ledger) 경로도 같은 캐시 접두 계약을 지켜야 한다 — [현재 로봇 상태]는 안 나온다.
        from src.ledger_view import render_ledger
        from src.schema import RobotState
        st_a = RobotState(current_floor=4, current_building="로봇관", dialog_state="idle", place_here="407호 앞")
        st_b = RobotState(current_floor=4, current_building="로봇관", dialog_state="navigating", place_here="복도")
        a = parser.build_audio_prompt([DEST], st_a,
                                      situation=render_ledger(st_a, awaiting_answer=False, now_text="09:01"))
        b = parser.build_audio_prompt([DEST], st_b,
                                      situation=render_ledger(st_b, awaiting_answer=False, now_text="09:01"))
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        assert prefix > len(a) * 0.8
        assert "[현재 로봇 상태]" not in a
        assert a.rindex("[지금 상황]") > a.index("[말투]")

    def test_state_block_is_skipped_when_situation_has_ledger(self):
        from src.schema import RobotState
        st = RobotState(current_floor=4, current_building="로봇관", dialog_state="idle")
        with_ledger = parser.build_audio_prompt([DEST], st, situation="\n[지금 상황]\n- 건물/층: 로봇관 4층\n")
        assert "[현재 로봇 상태]" not in with_ledger and "[지금 상황]" in with_ledger
        without = parser.build_audio_prompt([DEST], st, situation="")
        assert "[현재 로봇 상태]" in without

    def test_directory_block_and_elevator_rule(self):
        text = parser.build_audio_prompt([DEST], directory_block="\n[다른 층 장소]\n- 세미나실: 로봇관 3층\n")
        assert "[다른 층 장소]" in text and "엘리베이터" in text

    def test_question_rule_names_ledger_lines(self):
        text = parser.build_audio_prompt([DEST])
        for key in ("몇 층", "지금 어디", "어디 가려고", "몇 시", "배터리"):
            assert key in text, key
        assert "물어보라는 식의 조언" in text   # 조언 금지 문구는 유지
        # 금지 문구에 장소 이름을 쓰지 않는다 — 모델이 되묻기에 "안내데스크"를 지어냈다(10-08 11:33).
        assert "안내 데스크" not in text


class TestAudioPromptStyle:
    """대답을 짧고 차분하게, 장소는 목록 안에서만 (2026-10-09 사용자: 길고 자유분방하고 목록에 없는
    곳까지 말한다. 최근 2주 LLM 대답 128개 중 50자 넘음 11개·최대 88자)."""

    def test_reply_is_about_30_chars(self):
        text = parser.build_audio_prompt([DEST])
        assert "30자 안팎" in text
        assert "60자" not in text and "50자 안팎" not in text

    def test_one_question_at_a_time(self):
        assert "질문은 하나만" in parser.build_audio_prompt([DEST])

    def test_one_sentence_by_default(self):
        """문장마다 TTS 가 새로 만들어 문장 사이에 빈틈이 생긴다 — 가급적 한 문장(2026-10-10 사용자)."""
        text = parser.build_audio_prompt([DEST], directory_block="\n[다른 층 장소]\n- 세미나실: 로봇관 3층\n")
        assert "가급적 한 문장" in text and "빈틈" in text
        assert "두 문장" not in text
        # 규칙 속 예시 답도 한 문장이어야 모델이 따라 한다.
        for two in ("4층이에요. 안내해드릴까요?", "목록에 없어요. 학과사무실", "잘 못 들었어요. 다시"):
            assert two not in text, two

    def test_text_path_one_sentence_too(self):
        text = parser._build_system_prompt([DEST])
        assert "가급적 한 문장" in text and "빈틈" in text

    def test_calm_tone_without_variation(self):
        text = parser.build_audio_prompt([DEST])
        assert "차분" in text
        assert "밝고 친근한" not in text and "조금씩 다르게" not in text

    def test_place_names_only_from_the_list(self):
        text = parser.build_audio_prompt([DEST])
        assert "목록에 없는 장소" in text and "지어내지 않는다" in text

    def test_rule_examples_are_not_places_to_say(self):
        assert "설명용" in parser.build_audio_prompt([DEST])

    def test_reask_without_listing_examples(self):
        assert "예시를 늘어놓지 않는다" in parser.build_audio_prompt([DEST])


def test_reserve_tool_result_is_handed_back(monkeypatch):
    """Realtime 이 reserve_facility 를 부르면 의도 배달(deliver_audio_draft)을 거치지 않고 ReserveCall 을 돌려준다."""
    import types

    from src.reservation_flow import ReserveCall
    result = RealtimeResult(draft={"facility": "세미나실", "date": "내일", "start_time": "14:00"},
                            heard_text="내일 오후 두 시에 세미나실 예약해 줘", latency_sec=1.0, tool="reserve_facility")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: types.SimpleNamespace(ask=lambda *a, **k: result))
    out, heard, dt, info = parser.parse_intent_audio(b"\x00\x00", [DEST])
    assert out == ReserveCall(facility="세미나실", date="내일", start_time="14:00")
    assert heard == "내일 오후 두 시에 세미나실 예약해 줘" and info["src"] == "llm" and dt == 1.0


def test_text_replay_goes_through_the_same_delivery(monkeypatch):
    """재생 측정용 글자 입력 — Realtime 에 글자를 넣고, 그 뒤 처리는 소리 경로와 같다."""
    class Client:
        def ask_text(self, text, history, instructions):
            self.seen = (text, list(history or []), instructions)
            return RealtimeResult(draft={"intent": "navigate", "destination_candidate": DEST.name,
                                         "need_confirm": True, "confidence": 0.9},
                                  heard_text=text, latency_sec=0.4)

    client = Client()
    monkeypatch.setattr(parser, "get_realtime_client", lambda: client)
    out, heard, dt, info = parser.parse_intent_realtime_text("화장실로 가자", [DEST], situation="[진행 중인 예약] 가\n")
    assert out.intent == "navigate" and out.matched_destination_id == DEST.id
    assert heard == "화장실로 가자" and dt == 0.4 and info["src"] == "llm"
    assert "[진행 중인 예약] 가" in client.seen[2]


def _minutes_q():
    from src.question_record import OpenQuestion
    return OpenQuestion(id="q7", kind="minutes", text="몇 분쯤 걸리실까요?", option_ids=(), opened_at=0.0)


def test_realtime_answer_is_filled(monkeypatch):
    from src.realtime_intent import RealtimeResult
    monkeypatch.setenv("VICA_ANSWER", "on")
    result = RealtimeResult(draft={"intent": "wait", "wait_minutes": 15, "answer": "minutes"},
                            heard_text="삼십분", latency_sec=1.0)
    intent, heard, _dt, info = parser._deliver_realtime_result(result, [], None, question=_minutes_q())
    # Task 3.5b(규칙 3): Realtime 이 들은 글자 "삼십분"의 규칙 대답이 값을 바로잡는다(Realtime 도 시간으로 읽었다).
    # 의도가 wait 면 그대로 — wait_minutes 는 Realtime 의 15, answer_value 만 규칙의 30(규칙 2).
    assert (intent.question_id, intent.answer, intent.answer_value, heard) == ("q7", "minutes", "30", "삼십분")
    # 수정 1차: 의도가 wait 면 의도는 그대로, wait_minutes 도 규칙의 30 — answer_value 와 갈리지 않는다.
    assert (intent.intent, intent.wait_minutes, info["src"]) == ("wait", 30, "llm+대답규칙")
    clamped = RealtimeResult(draft={"intent": "unknown", "answer": "yes"}, heard_text="네", latency_sec=1.0)
    assert parser._deliver_realtime_result(clamped, [], None, question=_minutes_q())[0].answer == "unclear"


def test_realtime_answer_is_empty_when_off():
    from src.realtime_intent import RealtimeResult
    result = RealtimeResult(draft={"intent": "wait", "wait_minutes": 15, "answer": "minutes"},
                            heard_text="삼십분", latency_sec=1.0)
    assert parser._deliver_realtime_result(result, [], None, question=_minutes_q())[0].answer == ""


# ----- Task 3.5b — 소리 경로도 짧은 대답 규칙을 쓴다(원칙 6·7) -------------------------------------------------------
OTHER = DestinationData(id="info", name="안내소", confirm_prompt="안내소로 안내해드릴까요?")


def _confirm_q():
    from src.question_record import OpenQuestion
    return OpenQuestion(id="q3", kind="confirm", text=DEST.confirm_prompt, option_ids=(DEST.id,), opened_at=0.0)


def _other_proposal(heard):
    # Realtime 이 "네"를 다른 목적지 제안으로 읽은 결과
    return RealtimeResult(draft={"intent": "navigate", "destination_candidate": OTHER.name, "need_confirm": True,
                                 "reply": OTHER.confirm_prompt, "confidence": 0.6, "answer": "destination"},
                          heard_text=heard, latency_sec=0.9)


def test_p6_sound_path_uses_the_heard_words_for_a_whole_yes(monkeypatch, capsys):
    """확인 질문 + heard "네" + Realtime affirm·answer 빈 값 → 대답 칸 yes(의도 그대로) — 같은 읽기일 때만 채운다(수정 2차 ③)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    said = RealtimeResult(draft={"intent": "affirm", "reply": "", "confidence": 0.8, "answer": ""}, heard_text="네.",
                          latency_sec=0.9)
    intent, heard, dt, info = parser._deliver_realtime_result(said, [DEST, OTHER], None, question=_confirm_q())
    assert (intent.intent, intent.answer, intent.answer_value, intent.question_id) == ("affirm", "yes", "", "q3")
    assert intent.confidence == pytest.approx(0.8) and (heard, dt) == ("네.", 0.9)
    assert info["src"] == "llm+대답규칙"
    log = capsys.readouterr().err
    assert "[대답규칙] 규칙 전(짧은답규칙 뒤) intent=affirm answer=-/-" in log      # 검토 M4 — 어느 시점의 값인지 글자로


def test_p6_sound_path_whole_yes_does_not_overturn_a_destination_reading(monkeypatch, capsys):
    """확인 질문 + heard "네" + Realtime 다른 곳 제안·answer destination → 그대로(수정 2차 ④ — 규칙이 덮지 않는다)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    intent, heard, dt, info = parser._deliver_realtime_result(_other_proposal("네."), [DEST, OTHER], None,
                                                              question=_confirm_q())
    assert (intent.intent, intent.answer, intent.answer_value, intent.question_id) == ("navigate", "destination", "info", "q3")
    assert (intent.need_confirm, intent.matched_destination_id, info["src"]) == (True, OTHER.id, "llm")
    assert "[대답규칙]" not in capsys.readouterr().err


def test_p7_sound_path_yes_from_not_for_robot_is_asked_again(monkeypatch):
    """확인 질문 + heard "네" + Realtime not_for_robot → unknown·unclear·reply 빈 값(affirm 아님) — 옆 사람의 "네"가 출발이 되지
    않게 하고, 버리지도 않아 미션이 다시 묻는다(수정 2차 ② · 규칙 3)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    monkeypatch.setenv("VICA_RECUT", "on")
    for llm in ("", "yes"):
        aside = RealtimeResult(draft={"intent": "not_for_robot", "reply": "", "answer": llm}, heard_text="네",
                               latency_sec=1.0)
        intent, _h, _dt, info = parser._deliver_realtime_result(aside, [DEST], None, question=_confirm_q())
        assert (intent.intent, intent.answer, intent.reply, intent.need_confirm) == ("unknown", "unclear", "", False), llm
        assert info["src"] == "llm+대답규칙"


# 검토자 재현(Critical) — "가자"·"가줘"는 AFFIRMATIVES 에 있지만 확인 질문 밖에서는 뜻이 반대다. Realtime 이 맞게 읽은 것을
# 규칙이 승낙으로 뒤집으면 안 된다 — 의도는 BASE 4cdc717 그대로 나간다(수정 2차 ①). 사용자 결정(11-22): 이 두 낱말은 목적지
# 확인 질문에서만 규칙의 "네"다 — 그 밖의 질문에서는 규칙 대답이 없어 Realtime 의 의도·대답 칸(no·other)이 그대로 나간다
# (그 전에는 규칙 yes 와 Realtime no 가 반대라 대답 칸이 unclear 가 되어 미션이 같은 질문을 되풀이했다, 최종 검토 I1 ②).
REVIEW_SCENES = [
    ("yesno", "안내를 취소할까요?", "가자", {"intent": "deny", "answer": "no"}, "navigating", "deny", "no"),
    ("yesno", "안내를 취소할까요?", "가자", {"intent": "resume", "answer": "other"}, "navigating", "resume", "other"),
    ("yesno", "다녀오시는 동안 여기서 기다릴까요?", "가자", {"intent": "resume", "answer": "no"}, "asking_next", "resume", "no"),
    ("yesno_inverted", "안내가 필요 없으신가요?", "가자", {"intent": "resume", "answer": "no"}, "waiting", "resume", "no"),
    ("yesno", "직원의 도움이 필요하신가요?", "가줘", {"intent": "deny", "answer": "no"}, "navigating", "deny", "no"),
    ("yesno", "여기까지 안내를 마칠까요?", "가자", {"intent": "resume", "answer": "other"}, "asking_next", "resume", "other"),
]


@pytest.mark.parametrize("kind,text,heard,draft,state,want_intent,want_answer", REVIEW_SCENES)
def test_p6_sound_path_rule_does_not_overturn_the_review_scenes(monkeypatch, kind, text, heard, draft, state,
                                                               want_intent, want_answer):
    from src.question_record import OpenQuestion
    from src.schema import RobotState
    monkeypatch.setenv("VICA_ANSWER", "on")
    question = OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)
    result = RealtimeResult(draft=draft, heard_text=heard, latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(result, [DEST, OTHER], RobotState(dialog_state=state),
                                                            question=question)
    assert (intent.intent, intent.answer, info["src"]) == (want_intent, want_answer, "llm")


def test_p6_sound_path_whole_yes_against_realtime_no_is_asked_again(monkeypatch):
    """최종 검토 I1 ②: 들은 글자가 통째 "네"(규칙 yes)인데 Realtime 이 no 로 읽으면 어느 쪽도 따르지 않는다 — 대답 칸 unclear,
    미션이 같은 질문을 다시 묻는다. 목적지 확인 질문의 "가자"도 같다(그 질문에서는 규칙의 "네"다)."""
    from src.question_record import OpenQuestion
    from src.schema import RobotState
    monkeypatch.setenv("VICA_ANSWER", "on")
    for kind, text, heard, state in (("yesno", "안내를 취소할까요?", "네", "navigating"),
                                     ("confirm", "화장실로 안내해드릴까요?", "가자", "confirming")):
        question = OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)
        result = RealtimeResult(draft={"intent": "deny", "answer": "no"}, heard_text=heard, latency_sec=1.0)
        intent, _h, _dt, info = parser._deliver_realtime_result(result, [DEST, OTHER], RobotState(dialog_state=state),
                                                                question=question)
        assert (intent.intent, intent.answer, info["src"]) == ("deny", "unclear", "llm+대답규칙"), (kind, heard)


@pytest.mark.parametrize("state,heard,draft,want_intent,want_answer", [
    ("confirming", "음", {"intent": "unknown", "reply": "", "answer": "unclear"}, "affirm", "unclear"),
    ("confirming", "어어", {"intent": "unknown", "reply": "", "answer": "unclear"}, "affirm", "unclear"),
    ("confirming", "음", {"intent": "affirm", "reply": "", "answer": "unclear"}, "affirm", "unclear"),
    ("confirming", "거기가 어디야", {"intent": "question", "reply": "1층에 있어요.", "answer": "unclear"}, "question", "other"),
])
def test_p3_i2_sound_path_soft_yes_with_unclear_is_asked_again(monkeypatch, state, heard, draft, want_intent, want_answer):
    """원칙 3·6(최종 검토 I2, 재현 v5): 확인 질문 중 '음' + Realtime unknown·unclear → _audio_short_answer 가 affirm 으로 바꾼 뒤에도
    대답 칸은 unclear 그대로(미션이 다시 묻는다) — 예전에는 other 가 돼 미션이 의도(affirm)대로 출발했다. 질문("거기가 어디야")은
    지금처럼 other(ca81b88)."""
    from src.schema import RobotState
    monkeypatch.setenv("VICA_ANSWER", "on")
    monkeypatch.setenv("VICA_RECUT", "on")
    result = RealtimeResult(draft=draft, heard_text=heard, latency_sec=1.0)
    intent, _h, _dt, _info = parser._deliver_realtime_result(result, [DEST, OTHER], RobotState(dialog_state=state),
                                                             question=_confirm_q())
    assert (intent.intent, intent.answer, intent.question_id) == (want_intent, want_answer, "q3")


def test_sound_path_is_unchanged_when_off(monkeypatch):
    monkeypatch.delenv("VICA_ANSWER", raising=False)
    intent, heard, _dt, info = parser._deliver_realtime_result(_other_proposal("네."), [DEST, OTHER], None,
                                                               question=_confirm_q())
    assert (intent.intent, intent.matched_destination_id, intent.need_confirm, intent.reply) == (
        "navigate", OTHER.id, True, OTHER.confirm_prompt)
    assert (intent.answer, intent.question_id, info["src"]) == ("", "", "llm")


def test_p6_sound_path_whole_no_on_a_proposal(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    intent, _h, _dt, info = parser._deliver_realtime_result(_other_proposal("아니요"), [DEST, OTHER], None,
                                                            question=_confirm_q())
    # 수정 2차: Realtime 이 다른 곳 제안으로 읽었으면 규칙 no 가 뒤집지 않는다.
    assert (intent.intent, intent.answer, intent.matched_destination_id, info["src"]) == (
        "navigate", "destination", OTHER.id, "llm")
    refused = RealtimeResult(draft={"intent": "deny", "reply": "", "answer": ""}, heard_text="아니요", latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(refused, [DEST, OTHER], None, question=_confirm_q())
    assert (intent.intent, intent.answer, info["src"]) == ("deny", "no", "llm+대답규칙")     # 같은 읽기 — 대답 칸만 채움


@pytest.mark.parametrize("asked,need_confirm,answer", [
    ("confirm", True, "other"),                    # 확인 질문 중 "취소" — 안내 취소 제안
    ("cancel_q", False, "yes"),                    # "안내를 취소할까요?"에 "취소" — 승낙(확정 취소)
])
def test_p6_sound_path_cancel_word_keeps_the_realtime_cancel(monkeypatch, capsys, asked, need_confirm, answer):
    """수정 1차: heard "취소"는 네/아니요 규칙 대답이 아니다 — Realtime 의 cancel 그대로, src 에 꼬리표 없음."""
    from src.question_record import OpenQuestion
    monkeypatch.setenv("VICA_ANSWER", "on")
    question = (_confirm_q() if asked == "confirm" else
                OpenQuestion(id="q9", kind="yesno", text=parser.CANCEL_CONFIRM, option_ids=(), opened_at=0.0))
    said = RealtimeResult(draft={"intent": "cancel", "need_confirm": need_confirm, "reply": "", "answer": answer},
                          heard_text="취소.", latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(said, [DEST], None, question=question)
    assert (intent.intent, intent.need_confirm, intent.answer, info["src"]) == ("cancel", need_confirm, answer, "llm")
    assert "[대답규칙]" not in capsys.readouterr().err


def test_p6_sound_path_minutes_rule_yields_to_a_refusal(monkeypatch, capsys):
    monkeypatch.setenv("VICA_ANSWER", "on")
    refused = RealtimeResult(draft={"intent": "finish", "answer": "no"}, heard_text="십 분 말고 그냥 안내 끝내 줘",
                             latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(refused, [DEST], None, question=_minutes_q())
    assert (intent.intent, intent.answer, intent.answer_value, intent.wait_minutes) == ("finish", "no", "", -1)
    assert info["src"] == "llm" and "[대답규칙]" not in capsys.readouterr().err


@pytest.mark.parametrize("heard,llm", [("십 분 정도요", ""), ("오 분만 있다가 가자", "unclear")])
def test_p7_sound_path_answer_is_not_dropped_as_not_for_robot(monkeypatch, heard, llm):
    """수정 3차: not_for_robot 인 분 대답은 버리지 않되 wait 로 살리지도 않는다 — unknown·unclear(미션이 다시 묻는다).
    "오 분만 있다가 가자"는 옆 사람에게 한 말일 수 있다(검토 재현) — 규칙이 대기로 바꾸면 안 된다."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    monkeypatch.setenv("VICA_RECUT", "on")
    dropped = RealtimeResult(draft={"intent": "not_for_robot", "answer": llm}, heard_text=heard, latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(dropped, [DEST], None, question=_minutes_q())
    assert (intent.intent, intent.wait_minutes, intent.reply, intent.need_confirm) == ("unknown", -1, "", False)
    assert (intent.answer, intent.answer_value, info["src"]) == ("unclear", "", "llm+대답규칙")
    monkeypatch.delenv("VICA_ANSWER")                  # 대답 칸이 꺼졌으면 지금처럼 not_for_robot 그대로
    off, _h, _dt, info = parser._deliver_realtime_result(dropped, [DEST], None, question=_minutes_q())
    assert (off.intent, off.answer, info["src"]) == ("not_for_robot", "", "llm")


def test_p6_sound_path_minutes_rule_does_not_turn_a_non_reading_into_wait(monkeypatch, capsys):
    """"아까 한 시간 기다렸잖아"(Realtime unknown) — 분 규칙이 60분 대기로 바꾸지 않는다(수정 3차, 검토 재현)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    said = RealtimeResult(draft={"intent": "unknown", "reply": "", "answer": ""}, heard_text="아까 한 시간 기다렸잖아",
                          latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(said, [DEST], None, question=_minutes_q())
    assert (intent.intent, intent.wait_minutes, intent.answer, info["src"]) == ("unknown", -1, "", "llm")
    assert "[대답규칙]" not in capsys.readouterr().err


def test_p6_sound_path_marks_nothing_when_the_rule_agrees(monkeypatch, capsys):
    monkeypatch.setenv("VICA_ANSWER", "on")
    agreed = RealtimeResult(draft={"intent": "affirm", "answer": "yes"}, heard_text="네", latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(agreed, [DEST], None, question=_confirm_q())
    assert (intent.intent, intent.answer, info["src"]) == ("affirm", "yes", "llm")
    assert "[대답규칙]" not in capsys.readouterr().err


def test_p6_sound_path_without_an_open_question_is_unchanged(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    intent, _h, _dt, info = parser._deliver_realtime_result(_other_proposal("네"), [DEST, OTHER], None, question=None)
    assert (intent.intent, intent.matched_destination_id, intent.answer, info["src"]) == ("navigate", OTHER.id, "", "llm")


def test_p6_sound_path_reserve_call_is_untouched(monkeypatch):
    from src.reservation_flow import ReserveCall
    monkeypatch.setenv("VICA_ANSWER", "on")
    call = RealtimeResult(draft={"facility": "세미나실"}, heard_text="네", latency_sec=1.0, tool="reserve_facility")
    out, _h, _dt, info = parser._deliver_realtime_result(call, [DEST], None, question=_confirm_q())
    assert out == ReserveCall(facility="세미나실") and info["src"] == "llm"


def test_p6_sound_path_short_answer_rule_then_answer_rule(monkeypatch, capsys):
    """대답을 기다리는 단계의 짧은 대답 규칙(_audio_short_answer)이 먼저 의도를 바꾸고, 대답 칸은 질문 기록의 규칙이 채운다."""
    from src.schema import RobotState
    monkeypatch.setenv("VICA_ANSWER", "on")
    asked = RealtimeResult(draft={"intent": "question", "reply": "네, 말씀하세요.", "answer": ""}, heard_text="그래",
                           latency_sec=1.0)
    intent, _h, _dt, info = parser._deliver_realtime_result(asked, [DEST], RobotState(dialog_state="confirming"),
                                                            question=_confirm_q())
    assert (intent.intent, intent.answer, intent.reply) == ("affirm", "yes", "")
    assert info["src"] == "llm+짧은답규칙+대답규칙"
    # 검토 M4: 로그의 '규칙 전' 은 _audio_short_answer 가 지나간 뒤의 값(affirm)이다 — Realtime 원래 의도(question)가 아니다.
    assert "규칙 전(짧은답규칙 뒤) intent=affirm" in capsys.readouterr().err


def test_sound_path_does_not_wait_for_the_local_transcript():
    """규칙에 대는 글자는 Realtime 이 들은 글자(heard_text)다 — 로컬 받아쓰기(/user_text)를 받는 인자가 없다."""
    import inspect
    assert list(inspect.signature(parser._deliver_realtime_result).parameters) == [
        "result", "destinations", "robot_state", "question"]
    assert "answer_field.short_answer(heard, question)" in inspect.getsource(parser._deliver_realtime_result)


def test_audio_prompt_carries_the_question_line(monkeypatch):
    from src import answer_field as af
    from src.realtime_intent import RealtimeResult
    seen = {}

    class Client:
        def ask(self, pcm, history, instructions):
            seen["instructions"] = instructions
            return RealtimeResult(draft={"intent": "unknown"}, heard_text="", latency_sec=0.1)

    monkeypatch.setattr(parser, "get_realtime_client", lambda: Client())
    parser.parse_intent_audio(b"", [], question=_minutes_q())
    assert "로봇이 물은 질문" not in seen["instructions"]
    monkeypatch.setenv("VICA_ANSWER", "on")
    parser.parse_intent_audio(b"", [], question=_minutes_q())
    assert seen["instructions"].endswith(af.with_question("", _minutes_q()))


def test_audio_prompt_gets_reservation_rules_only_when_the_reserve_tool_is_on(monkeypatch, tmp_path):
    """11-19 원인 실험 V2: 소리 지시문에 예약 대화 규칙이 없어 값 바꾸기를 목적지 정정으로 읽었다(0/4 → 4/4).
    스위치가 꺼져 있으면 지시문은 글자 하나까지 그대로다."""
    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    off = parser.build_audio_prompt([DEST], None, "[지금 상황] 끝\n")
    assert "set_intent 함수 한 번으로 정한다" in off and "[시설 예약]" not in off
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
    on = parser.build_audio_prompt([DEST], None, "[지금 상황] 끝\n")
    assert "set_intent 함수 한 번으로" not in on and "reserve_facility" in on.split("\n")[1]
    assert on.endswith("[지금 상황] 끝\n")                     # 바뀌는 상황은 맨 뒤 그대로(캐시)
    rules = parser.AUDIO_RESERVE_RULES
    assert rules in on and "affirm" in rules and "destination_candidate" in rules
    assert on.replace(rules, "").replace(parser.AUDIO_FIRST_LINE_WITH_RESERVE, parser.AUDIO_FIRST_LINE) == off


def test_parse_intent_audio_passes_the_trail_to_realtime(rt):
    client = rt({"intent": "affirm"}, "그래")
    trail = [{"args": {"name": "김민수"}, "heard": "김민수", "asked": "확인?"}]
    seen = {}
    original = client.ask

    def ask(pcm, history, instructions, **kw):
        seen.update(kw)
        return original(pcm, history, instructions)

    client.ask = ask
    parse_intent_audio(PCM, [DEST], tool_trail=trail)
    assert seen == {"tool_trail": trail}
    seen.clear()
    parse_intent_audio(PCM, [DEST])
    assert seen == {}                                          # 기록이 없으면 예전과 같은 호출
