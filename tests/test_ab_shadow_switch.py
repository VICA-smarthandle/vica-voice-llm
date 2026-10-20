"""[A/B] 그림자 비교 스위치 VICA_AB_SHADOW (2026-10-10 사용자 결정 — 기본 꺼짐)."""
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
