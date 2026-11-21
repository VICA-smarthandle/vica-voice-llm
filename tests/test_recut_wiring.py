"""의도 노드 배선(원칙 7·19) — ros_node 소스 구조로 확인하고, 단서 붙이기는 순수 함수로 본다."""
from __future__ import annotations

import ast

from tests.test_one_voice_wiring import _method


def method_source(name: str) -> str:
    return ast.unparse(_method(name))


def test_p7_node_drops_not_for_robot_before_anything_is_published_or_said():
    body = method_source("_publish_intent")
    drop = body.index("NOT_FOR_ROBOT")
    assert drop < body.index("identity_answer") < body.index("_intent_pub.publish")
    assert body.index("speaker_verdict") < drop     # 화자 판정을 붙인 뒤 — 로그에 남는다
    assert "[끼어들지 않음]" in body
    # 버리는 분기는 스위치를 보지 않는다 — 꺼져 있으면 not_for_robot 이 만들어지지 않는다.
    assert "intent_recut.on()" not in body[drop - 80:drop + 400]


def test_p19_audio_path_passes_the_speaker_clue():
    body = method_source("_on_user_audio")
    assert body.index("with_speaker") < body.index("parse_intent_audio(")


def test_p19_clue_reads_the_verdict_without_clearing_it():
    body = method_source("_on_user_audio")
    clue = body[:body.index("parse_intent_audio(")]
    assert "self._audio_spk = None" not in clue[clue.index("with_speaker"):]
