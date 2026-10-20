"""비공개(private) 목적지를 LLM 에 '접근 불가'로 보여 준다(2026-10-10 사용자 결정 A)."""
from pathlib import Path

import pytest

from src import langchain_intent_parser as parser
from src.destination_loader import load_destinations, load_voice_destinations
from src.langchain_intent_parser import build_audio_prompt, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.tts_queue import request_for_intent

ROOT = Path(__file__).resolve().parent.parent
PCM = b"\x00\x00" * 1600

YAML = """schema_version: 1
destinations:
- id: room407
  name: 407호
  aliases: ['407호', '407', '사무실']
  authorization: public
  is_approachable: true
- id: room408
  name: 408호
  aliases: ['408호', '408', '사무실']
  authorization: private
  is_approachable: true
  contact_phone: '01000000000'
"""


@pytest.fixture
def yaml_path(tmp_path):
    p = tmp_path / "destinations.yaml"
    p.write_text(YAML, encoding="utf-8")
    return p


class FakeRT:
    def __init__(self, draft, heard):
        self.draft, self.heard = draft, heard

    def ask(self, pcm, history, instructions):
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.5,
                              usage={"audio_tokens": 20})


def test_voice_catalog_keeps_private_as_not_approachable(yaml_path):
    dests = {d.id: d for d in load_voice_destinations(yaml_path)}
    assert set(dests) == {"room407", "room408"}
    assert dests["room408"].is_approachable is False
    assert dests["room407"].is_approachable is True


def test_other_readers_still_get_public_only(yaml_path):
    assert [d.id for d in load_destinations(yaml_path)] == ["room407"]


def test_audio_prompt_shows_private_as_not_approachable(yaml_path):
    prompt = build_audio_prompt(load_voice_destinations(yaml_path))
    line = next(l for l in prompt.splitlines() if l.startswith("- 408호"))
    assert line.endswith("접근 불가")
    assert "접근 불가인 곳" in prompt and "비슷한 다른 곳으로 바꾸지 않고" in prompt


def test_private_request_goes_to_mission_without_confirm(yaml_path, monkeypatch):
    fake = FakeRT({"intent": "navigate", "destination_candidate": "408호", "need_confirm": True,
                   "reply": "408호로 안내해드릴까요?", "confidence": 0.9}, "408호로 가자")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: fake)
    intent, _, _, _ = parse_intent_audio(PCM, load_voice_destinations(yaml_path))
    assert intent.intent == "navigate" and intent.matched_destination_id == "room408"
    assert intent.need_confirm is False
    assert request_for_intent(intent) is None


def test_llm_node_loads_the_voice_catalog():
    src = (ROOT / "src" / "ros_node.py").read_text(encoding="utf-8")
    body = src[src.index("    def _reload_destinations_if_changed"):src.index("def main(")]
    assert "load_voice_destinations(self._destinations_path)" in body
    assert 'authorization == "public"' not in body
