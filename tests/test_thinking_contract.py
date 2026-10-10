"""/vica/thinking 은 미션과의 팀 계약이다 (2026-10-10 사용자 결정).

원래 TTS 의 '생각 중' 운율용 음성 내부 신호였는데(2026-09-01), 미션이 이 신호가 켜진 동안 대답 대기 시계를
잡게 됐다(vica_ros2_ws 645acd7) — 10-10 12:13 Realtime 이 9초 걸린 사이 미션이 다시 묻고 곧바로 수락했다.
그래서 LLM 노드가 판단 전후로 반드시 켜고 끄는지, 계약 문서에 올라 있는지를 고정한다.
rclpy 없이 소스 글자로 본다(test_approach_question_voice 방식).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _node_src() -> str:
    return (ROOT / "src" / "ros_node.py").read_text(encoding="utf-8")


def _section(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end)]


def test_llm_node_publishes_thinking_as_bool():
    assert 'self._thinking_pub = self.create_publisher(Bool, "/vica/thinking", 10)' in _node_src()


def test_audio_path_turns_thinking_on_and_always_off():
    body = _section(_node_src(), "    def _on_user_audio", "    def _shadow_text")
    on = body.index("self._thinking_pub.publish(Bool(data=True))")
    call = body.index("parse_intent_audio(")
    off = body.index("self._thinking_pub.publish(Bool(data=False))")
    assert on < call < off
    assert "finally:" in body[call:off]               # 실패해도 끈다 — 켠 채 남으면 미션이 상한까지 기다린다


def test_text_path_turns_thinking_off_in_finally():
    body = _section(_node_src(), "    def _on_user_text", "    def _on_listen_state_for_resume")
    off = body.index("self._thinking_pub.publish(Bool(data=False))")
    assert "finally:" in body[:off][-400:]


def test_contract_doc_lists_thinking_for_the_mission():
    doc = (ROOT / "docs" / "ros2-interface.md").read_text(encoding="utf-8")
    contract = doc[doc.index("## 2. Topic"):doc.index("음성 저장소 내부 토픽")]
    row = next(line for line in contract.splitlines() if "`/vica/thinking`" in line)
    assert "Mission Manager" in row and "std_msgs/msg/Bool" in row
