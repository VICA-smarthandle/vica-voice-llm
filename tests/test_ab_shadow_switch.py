"""[A/B] 그림자 비교 스위치 VICA_AB_SHADOW (2026-10-10 사용자 결정 — 기본 꺼짐).

audio 모드에서는 Realtime 이 판단하고, 같은 발화의 whisper 글자를 글자 경로(GPT-5.4-mini)에도 보내
결과를 [A/B] 로그로만 남겼다. 로봇 동작에는 쓰지 않는데 말할 때마다 OpenAI 호출이 하나 더 나가고,
그 호출이 한 번 실패하면 백엔드가 로컬(믿음)로 대피해 그동안 비교용으로 젯슨 GPU 에서 로컬 모델이 돌았다.
그래서 기본은 끄고, 분석이 필요할 때만 켠다. 꺼져 있어도 소리 경로 결과와 받아쓰기는 한 줄 남긴다.
Realtime 이 실패했을 때 글자 경로가 대신 판단하는 예비 길은 이 스위치와 무관하다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.realtime_intent import ab_shadow_enabled

ROOT = Path(__file__).resolve().parents[1]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VICA_AB_SHADOW", raising=False)
    assert ab_shadow_enabled() is False


@pytest.mark.parametrize("raw", ["on", "1", "true", "YES", " on "])
def test_on_values(monkeypatch, raw):
    monkeypatch.setenv("VICA_AB_SHADOW", raw)
    assert ab_shadow_enabled() is True


@pytest.mark.parametrize("raw", ["off", "0", "", "false", "no"])
def test_off_values(monkeypatch, raw):
    monkeypatch.setenv("VICA_AB_SHADOW", raw)
    assert ab_shadow_enabled() is False


# ---- 노드 배선 (rclpy 없이 소스 글자로, test_approach_question_voice 방식) -------------------

def _node_src() -> str:
    return (ROOT / "src" / "ros_node.py").read_text(encoding="utf-8")


def _shadow_body() -> str:
    src = _node_src()
    return src[src.index("    def _shadow_text"):src.index("    def _reload_destinations_if_changed")]


def test_node_reads_the_switch_once_and_logs_it():
    src = _node_src()
    assert "self._ab_shadow = ab_shadow_enabled()" in src
    assert "비교 기록(GPT)" in src


def test_shadow_skips_the_gpt_call_when_off():
    body = _shadow_body()
    off = body.index("if not self._ab_shadow:")
    assert off < body.index("parse_intent(")
    assert "return" in body[off:body.index("def work")]


def test_shadow_still_logs_audio_and_whisper_when_off():
    body = _shadow_body()
    off_branch = body[body.index("if not self._ab_shadow:"):body.index("def work")]
    assert "[A/B] audio=" in off_branch and "text=꺼짐" in off_branch and "whisper=" in off_branch


def test_env_example_documents_the_switch():
    assert "VICA_AB_SHADOW" in (ROOT / ".env.example").read_text(encoding="utf-8")
