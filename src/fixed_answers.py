"""정해진 대답(2026-11-21 voice-principles-intent-recut 4단계, 설계 6.4) — 순수 함수만 둔다.

"다시 말해줘"(repeat, 원칙 28)·"어디 갈 수 있어?"(places, 원칙 29)는 LLM 이 값을 고르기만 하고 문장은 코드가 만든다.
비카 자신에 대한 물음·모르는 것·할 수 없는 일(question + about, 원칙 25·22, 2026-11-22 Task N3)도 같다 — 문장은
vica_facts.ABOUT_SENTENCES 의 정해진 글이다.
`.env` VICA_TALK=on 이고 재단(VICA_RECUT=on)·대답 칸(VICA_ANSWER=on)이 함께 켜졌을 때만 움직인다(기본 꺼짐 = 지금과 같다).
로컬(Ollama)이 답하는 호출에는 싣지 않는다(원칙 16) — 지시문 두 줄은 재단을 싣는 지시문에만 붙고, 로컬이 답한 결과는 이 뒤처리를
지나지 않는다. ROS·LLM 을 모른다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional, Sequence

from . import answer_field, intent_recut, vica_facts
from .mission_phrases import WAIT_BEACON
from .question_record import OpenQuestion
from .replies import RETRY_PROMPT, WAKE_GREETING, expects_answer, places_sentence
from .schema import DestinationData, VicaIntent

REPEAT = "repeat"
PLACES = "places"
# 고르기 질문 — 미션은 스스로 묻지 않는다(음성이 넘긴 것뿐). CONFIRMING 중에 넘긴 것이면 미션은 repeat 에 자기 확인 질문을 다시
# 말하므로(_open_question_name), 이 종류만 음성이 그 글을 다시 넘긴다(hand_over_question 이 choice·보기 id 로).
_VOICE_HANDED_KINDS = ("choice",)

# 지시문 두 벌(글자·소리)이 재단의 의도 정의(intent_recut.PROMPT_INTENTS) 바로 뒤에 싣는 두 줄. 켰을 때만 들어간다.
PROMPT_LINES = """- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어", "방금 뭐라고 했어?"). reply 는 빈 문자열 — 다시 할 말은 시스템이 고른다(직접 옮겨 말하지 않는다).
- places: 어디로 갈 수 있는지 묻는다("어디 갈 수 있어?", "갈 수 있는 곳 알려줘"). 이 물음은 question 으로 답하지 않고 반드시 places 다 — reply 는 빈 문자열, 문장은 시스템이 만든다(목적지 이름을 직접 늘어놓거나 "알려드릴게요"라고 하지 않는다). 무엇을 할 수 있는지 물으면 places 가 아니라 question 이고 about 은 abilities 다."""


def on() -> bool:
    """스위치. `.env` VICA_TALK=on 이고 재단(intent_recut.on())·대답 칸(answer_field.on())이 켜졌을 때만 참 — 4단계는 2·3단계
    위에서만 켜진다(대답 칸이 꺼지면 "다시 출발할까요?"를 음성이 쥐어 미션이 repeat 에 말이 없다). 매번 읽는다."""
    return (os.environ.get("VICA_TALK", "").strip().lower() == "on"
            and intent_recut.on() and answer_field.on())


# 최종 검토 I4 — 음성 두 스위치와 미션 파라미터 셋은 한 묶음으로 켜고 끈다. 경고 글 끝에 붙인다.
_BUNDLE_NOTE = "다섯 개(VICA_RECUT·VICA_ANSWER + 미션 answer_field·reask_ladder·wake_revive)는 한 묶음으로 켠다"


def switch_warning() -> str:
    """음성 스위치 조합이 어긋났으면 경고 한 줄, 맞으면 빈 글자(최종 검토 I4). 동작은 막지 않는다 — 의도 노드가 기동할 때 로그로만 낸다.

    어긋남: VICA_ANSWER=on 인데 VICA_RECUT 꺼짐(재단 없는 대답 칸) / VICA_TALK=on 인데 VICA_RECUT·VICA_ANSWER 중 하나라도
    꺼짐(그러면 on() 이 거짓이라 VICA_TALK 은 아무것도 하지 않는다). 미션 쪽 셋은 이 노드가 모른다 — 미션 노드가 따로 본다."""
    recut, answer = intent_recut.on(), answer_field.on()
    talk = os.environ.get("VICA_TALK", "").strip().lower() == "on"
    wrong = []
    if answer and not recut:
        wrong.append("VICA_ANSWER=on 인데 VICA_RECUT 꺼짐")
    if talk and not (recut and answer):
        off = "·".join(key for key, value in (("VICA_RECUT", recut), ("VICA_ANSWER", answer)) if not value)
        wrong.append(f"VICA_TALK=on 인데 {off} 꺼짐(VICA_TALK 은 움직이지 않는다)")
    if not wrong:
        return ""
    return f"[스위치 조합] {' · '.join(wrong)} — {_BUNDLE_NOTE}"


def places_names(destinations: Sequence[DestinationData]) -> list[str]:
    """갈 수 있는 곳(is_approachable)의 이름, 목록 순서(원칙 18 — 갈 수 없는 곳은 말하지 않는다)."""
    return [d.name for d in destinations if d.name and d.is_approachable]


def _said_lines(robot_recent: Sequence[tuple[float, str]]) -> Iterator[str]:
    """최근 로봇 말(재생이 끝난 문장, /vica/tts_done), 새것부터. 호출 응답 "네?"와 대기 알림(M3)은 뺀다 — 노드도 마지막 로봇 말·
    대화 기록에서 대기 알림을 뺀다(_on_tts_done_text)."""
    for _, text in reversed(list(robot_recent)):
        line = (text or "").strip()
        if line and line not in (WAKE_GREETING, WAIT_BEACON):
            yield line


def last_said(robot_recent: Sequence[tuple[float, str]]) -> str:
    """로봇의 마지막 말. "네?"·대기 알림은 건너뛴다. 없으면 빈 글자."""
    return next(_said_lines(robot_recent), "")


def _to_mission(ask_text: str = "", ask_kind: str = "", option_ids: Sequence[str] = ()) -> VicaIntent:
    """미션에 보내는 repeat — 말은 미션이 한다: 쥐거나 접어 둔 질문이 있으면 그것을, 없으면 실려 간 글(ask_text)을 그 종류로."""
    return VicaIntent(intent=REPEAT, reply="", need_confirm=False, ask_text=ask_text, ask_kind=ask_kind,
                      ask_option_ids=list(option_ids))


def _voice_says(text: str) -> VicaIntent:
    """음성이 다시 말한다 — 지금 question 길(질문 꼴이면 hand_over_question 이 미션에 넘긴다)."""
    return VicaIntent(intent="question", reply=text, need_confirm=False)


def settle_repeat(question: Optional[OpenQuestion], robot_recent: Sequence[tuple[float, str]]) -> VicaIntent:
    """repeat 를 정한다(원칙 28 · 9, 수정 1·2차). 누가 쥔 질문인지는 미션이 가른다 — 음성은 종류로 짐작하지 않는다.

    1·2. 열린 질문이 없거나 "네?"다: "네?"·대기 알림을 건너뛴 마지막 말(last_said) 하나만 본다 — 더 앞의 질문을 찾아 올라가지
         않는다(이미 끝난 질문을 다시 묻게 된다, 재검토 N2). 질문 꼴이면 그 글을 실어 repeat 를 미션에 보낸다(미션이 쥐거나 접어 둔
         질문이 있으면 그것을, 없으면 그 글을 말한다 — 음성이 넘긴 말이 재개 질문을 닫지 않게, M1). 평서문이면 음성이 다시 말한다.
         말이 없으면: 기록 없음은 unknown(되묻는 말), "네?"는 글 없이 미션에(쥐거나 접어 둔 오래된 질문이 있으면 미션이 말한다).
    3. 그 밖의 열린 질문: 그 질문의 글·종류·보기를 실어 repeat 를 미션에 보낸다. 고르기 질문(choice)만 음성이 그 글을 다시 넘긴다.
    대답 칸은 싣지 않는다 — 미션이 repeat 를 다른 의도로 바꿔 읽지 않게.
    """
    if question is None or (question.text or "").strip() == WAKE_GREETING:
        said = last_said(robot_recent)
        if said:
            return _to_mission(said, "open") if expects_answer(said) else _voice_says(said)
        if question is None:
            return VicaIntent(intent="unknown", reply=RETRY_PROMPT, need_confirm=False)
        return _to_mission()
    if question.kind in _VOICE_HANDED_KINDS:
        return _voice_says(question.text)
    return _to_mission(question.text, question.kind, question.option_ids)


def early(draft, destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """뒤처리 맨 앞(원칙 28·29·25): repeat 는 문장·목적지 없이 그대로(노드가 settle_repeat 로 정한다), places 는 갈 수 있는 곳을
    알리는 question 으로, about 이 여덟 주제 중 하나인 question 은 그 정해진 문장(vica_facts.ABOUT_SENTENCES)으로 돌려보낸다 —
    LLM 이 reply 에 무엇을 적었든 버린다(Task N3). about 은 question 일 때만 본다.
    꺼졌거나 다른 값이거나 갈 수 있는 곳이 없으면 None(as_known 이 지금 길로 바꾼다)."""
    if not on():
        return None
    kind = getattr(draft, "intent", "")
    confidence = getattr(draft, "confidence", None) or 0.0
    if kind == REPEAT:
        return VicaIntent(intent=REPEAT, reply="", need_confirm=False, confidence=confidence)
    if kind == PLACES:
        sentence = places_sentence(places_names(destinations))
        if sentence:
            return VicaIntent(intent="question", reply=sentence, need_confirm=False, confidence=confidence)
    if kind == "question":
        about = getattr(draft, "about", "")
        # 원칙 25·22: 목록 밖의 값(대소문자·공백이 다른 것 포함)은 무시한다 — 지금의 question 길(LLM 의 reply)로 간다.
        sentence = vica_facts.ABOUT_SENTENCES.get(about) if isinstance(about, str) else None
        if sentence:
            return VicaIntent(intent="question", reply=sentence, need_confirm=False, confidence=confidence)
    return None


def as_known(draft):
    """갈 수 있는 곳이 하나도 없어 early 가 문장을 만들지 못한 places 를 지금 길의 값으로 읽는다(VicaIntent 에 places 는 없다).

    LLM 이 reply 를 썼으면 question(그 말 그대로), 지시대로 비웠으면 unknown — 재단 뒤처리가 되묻는 말(RETRY_PROMPT)로 둔다.
    question 의 빈 reply 는 글자 경로에선 RETRY_PROMPT, 소리 경로에선 침묵이 돼 두 경로가 갈린다. 다른 값은 그대로 돌려준다.
    """
    if getattr(draft, "intent", "") != PLACES:
        return draft
    return draft.model_copy(update={"intent": "question" if (draft.reply or "").strip() else "unknown"})
