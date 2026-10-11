"""취소에는 미션만 답한다(2026-10-11 사용자 결정 — 10-10 실기 LLM·미션 겹쳐 말하기).

10-10 21:26:52 대기 중 "안내를 취소해줘요" → LLM "안내를 취소할까요?"를 음성이 말하고, 미션도 "안내가 필요
없으신가요?"를 물어 두 목소리가 났다. 미션은 모든 상태에서 취소에 답하므로 음성은 LLM 답을 지운다.
요청 자체(취소 의도)는 그대로 미션에 간다.
"""
from __future__ import annotations

import pytest

from src.mission_question import quiet_for_mission
from src.replies import CANCEL_CONFIRM
from src.schema import VicaIntent, should_forward_intent
from src.tts_queue import request_for_intent


def _cancel(need_confirm=True, reply=CANCEL_CONFIRM):
    return VicaIntent(intent="cancel", reply=reply, need_confirm=need_confirm, safety_flag="normal")


@pytest.mark.parametrize("state", ["navigating", "paused", "waiting", "idle", "confirming",
                                   "asking_next", "returning", "awaiting_user"])
def test_voice_does_not_speak_the_cancel_question(state):
    out = quiet_for_mission(_cancel(), state, "", 9.0)
    assert out.reply == "" and request_for_intent(out) is None
    # 미션에는 그대로 간다 — 미션이 상태에 맞는 말로 답한다
    assert out.intent == "cancel" and out.need_confirm is True and should_forward_intent(out)


def test_confirmed_cancel_still_goes_to_the_mission():
    out = quiet_for_mission(_cancel(need_confirm=False, reply=""), "waiting", "안내가 필요 없으신가요?", 1.0)
    assert out.intent == "cancel" and out.need_confirm is False and should_forward_intent(out)


def test_other_intents_keep_their_reply():
    nav = VicaIntent(intent="navigate", reply="407호로 안내해드릴까요?", need_confirm=True,
                     destination_candidate="407호", safety_flag="normal")
    assert quiet_for_mission(nav, "idle", "", 9.0).reply == "407호로 안내해드릴까요?"
