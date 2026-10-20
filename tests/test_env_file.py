"""마이크 점검기의 .env 편집 — 백업 먼저, 표시 남기기, 관계없는 줄 보존, 중복 키 거절, 되돌리기(설계 7절)."""
from __future__ import annotations

from datetime import datetime

import pytest
from dotenv import dotenv_values

from src.env_file import (EnvEditError, change_line, format_value, latest_backup, plan_changes, read_env,
                          restore_latest, values_from, write_changes)
from src.mic_check_rules import MicValues
from src.wakeword_monitor import (emergency_single_frame_from_env, listen_thresholds_from_env,
                                  wake_threshold_from_env)

SAMPLE = """# 맨 위 주석
VICA_LLM_PROVIDER=openai
VICA_MIC_AGC_DESIRED=0.007
# VICA_WAKE_THRESHOLD=0.6
VICA_MIC_GAMMA_ETAIL=1.5
VICA_MIC_NLATTEN=1
VICA_AUDIO_PROFILE=off
"""
NOW = datetime(2026, 10, 27, 14, 32, 5)
NOTE = "10-27 시연장 점검에서 바꿈"


@pytest.fixture
def env(tmp_path):
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_values_from_defaults_match_node():
    assert values_from({}) == MicValues(0.008, 0.02, 0.70, 0.50, 0.010, 1.0, 0)


def test_values_from_sample(env):
    assert values_from(read_env(env)) == MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)


@pytest.mark.parametrize("raw", ["0.65", "abc", "1.2", "", "0"])
def test_values_from_parses_like_node(monkeypatch, raw):
    for key in ("VICA_WAKE_THRESHOLD", "VICA_EMERGENCY_SINGLE_FRAME", "VICA_LISTEN_MIN_RMS",
                "VICA_SHORT_ANSWER_MIN_RMS"):
        monkeypatch.setenv(key, raw)
    mapping = {k: raw for k in ("VICA_WAKE_THRESHOLD", "VICA_EMERGENCY_SINGLE_FRAME",
                                "VICA_LISTEN_MIN_RMS", "VICA_SHORT_ANSWER_MIN_RMS")}
    v = values_from(mapping)
    assert v.wake_threshold == wake_threshold_from_env()
    assert v.emergency_single == emergency_single_frame_from_env()
    assert (v.listen_min_rms, v.short_answer_min_rms) == listen_thresholds_from_env()


def test_off_values_refused():
    with pytest.raises(EnvEditError):
        values_from({"VICA_EMERGENCY_SINGLE_FRAME": "off"})
    with pytest.raises(EnvEditError):
        values_from({"VICA_MIC_AGC_DESIRED": "off"})


def test_read_env_inline_comment_and_last_wins(tmp_path):
    p = tmp_path / ".env"
    p.write_text("A=0.010  # 메모\nB=1\nB=3\n# C=9\nexport D=4\n", encoding="utf-8")
    assert read_env(p) == {"A": "0.010", "B": "3", "D": "4"}


def test_format_value():
    assert format_value("listen_min_rms", 0.008) == "0.008"
    assert format_value("listen_min_rms", 0.0085) == "0.0085"
    assert format_value("short_answer_min_rms", 0.02) == "0.02"
    assert format_value("wake_threshold", 0.7) == "0.70"
    assert format_value("agc", 0.01) == "0.010"
    assert format_value("echo_etail", 1.0) == "1.0"
    assert format_value("echo_nlatten", 0) == "0"


def test_plan_changes_only_differences():
    a = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
    assert plan_changes(a, a) == []
    b = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.0, 1)
    changes = plan_changes(a, b)
    assert [(c.key, c.old, c.new) for c in changes] == [
        ("VICA_LISTEN_MIN_RMS", "0.008", "0.01"),
        ("VICA_WAKE_THRESHOLD", "0.70", "0.65"),
        ("VICA_MIC_GAMMA_ETAIL", "1.5", "1.0"),
    ]
    assert change_line(changes[1], NOTE) == "VICA_WAKE_THRESHOLD=0.65  # 10-27 시연장 점검에서 바꿈 (전 0.70)"


def test_write_replaces_existing_and_appends_missing(env, tmp_path):
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.010, 1.5, 1)
    backup = write_changes(env, plan_changes(before, after), NOTE, tmp_path / "bk", NOW)
    assert backup.name == ".env.20261027-143205" and backup.read_text(encoding="utf-8") == SAMPLE
    text = env.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[2] == "VICA_MIC_AGC_DESIRED=0.010  # 10-27 시연장 점검에서 바꿈 (전 0.007)"
    assert "# VICA_WAKE_THRESHOLD=0.6" in lines
    assert "# 마이크 점검기에서 추가 (2026-10-12 설계)" in lines
    assert lines[-2:] == ["VICA_LISTEN_MIN_RMS=0.01  # 10-27 시연장 점검에서 바꿈 (전 0.008)",
                          "VICA_WAKE_THRESHOLD=0.65  # 10-27 시연장 점검에서 바꿈 (전 0.70)"]
    assert lines[:2] == ["# 맨 위 주석", "VICA_LLM_PROVIDER=openai"]
    parsed = dotenv_values(env)
    assert parsed["VICA_WAKE_THRESHOLD"] == "0.65" and parsed["VICA_MIC_AGC_DESIRED"] == "0.010"
    assert values_from(read_env(env)) == after


def test_duplicate_key_refused_and_nothing_written(tmp_path):
    p = tmp_path / ".env"
    p.write_text("VICA_WAKE_THRESHOLD=0.7\nVICA_WAKE_THRESHOLD=0.6\n", encoding="utf-8")
    a = values_from(read_env(p))
    b = MicValues(a.listen_min_rms, a.short_answer_min_rms, 0.65, a.emergency_single, a.agc,
                  a.echo_etail, a.echo_nlatten)
    with pytest.raises(EnvEditError, match="2줄"):
        write_changes(p, plan_changes(a, b), NOTE, tmp_path / "bk", NOW)
    assert p.read_text(encoding="utf-8") == "VICA_WAKE_THRESHOLD=0.7\nVICA_WAKE_THRESHOLD=0.6\n"
    assert not (tmp_path / "bk").exists()


def test_no_changes_refused(env, tmp_path):
    with pytest.raises(EnvEditError):
        write_changes(env, [], NOTE, tmp_path / "bk", NOW)


def test_restore_latest_and_before_restore(env, tmp_path):
    bk = tmp_path / "bk"
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    write_changes(env, plan_changes(before, after), NOTE, bk, NOW)
    restored_from = restore_latest(env, bk, datetime(2026, 10, 27, 18, 0, 0))
    assert restored_from.name == ".env.20261027-143205"
    assert values_from(read_env(env)) == before
    kept_lines = [l for l in env.read_text(encoding="utf-8").splitlines() if l.strip() and "원래대로" not in l and "점검기에서" not in l
                  and not l.startswith("VICA_LISTEN_MIN_RMS") and not l.startswith("VICA_WAKE_THRESHOLD")
                  and not l.startswith("VICA_MIC_AGC") and not l.startswith("VICA_MIC_GAMMA")
                  and not l.startswith("VICA_MIC_NLATTEN")]
    assert kept_lines == ["# 맨 위 주석", "VICA_LLM_PROVIDER=openai", "# VICA_WAKE_THRESHOLD=0.6", "VICA_AUDIO_PROFILE=off"]
    kept = bk / ".env.20261027-180000.before-restore"
    assert kept.exists() and "VICA_WAKE_THRESHOLD=0.65" in kept.read_text(encoding="utf-8")
    assert latest_backup(bk).name == ".env.20261027-143205"


def test_restore_touches_only_mic_keys(env, tmp_path):
    bk = tmp_path / "bk"
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    write_changes(env, plan_changes(before, after), NOTE, bk, NOW)
    with env.open("a", encoding="utf-8") as f:
        f.write("VICA_DESTINATIONS_YAML=/new\n")
    now_text = env.read_text(encoding="utf-8")
    restore_latest(env, bk, datetime(2026, 10, 27, 18, 0, 0))
    text = env.read_text(encoding="utf-8")
    assert values_from(read_env(env)) == before
    assert "VICA_DESTINATIONS_YAML=/new" in text.splitlines()
    assert "VICA_LLM_PROVIDER=openai" in text.splitlines() and "VICA_AUDIO_PROFILE=off" in text.splitlines()
    assert "원래대로 (백업 .env.20261027-143205)" in text
    kept = bk / ".env.20261027-180000.before-restore"
    assert kept.read_text(encoding="utf-8") == now_text


def test_restore_removes_keys_absent_in_backup(env, tmp_path):
    bk = tmp_path / "bk"
    before = values_from(read_env(env))
    after = MicValues(0.008, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    write_changes(env, plan_changes(before, after), NOTE, bk, NOW)
    assert "VICA_WAKE_THRESHOLD" in read_env(env)
    restore_latest(env, bk, datetime(2026, 10, 27, 18, 0, 0))
    assert "VICA_WAKE_THRESHOLD" not in read_env(env)
    assert values_from(read_env(env)) == before


def test_restore_duplicate_key_refused(env, tmp_path):
    bk = tmp_path / "bk"
    before = values_from(read_env(env))
    write_changes(env, plan_changes(before, MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)), NOTE, bk, NOW)
    with env.open("a", encoding="utf-8") as f:
        f.write("VICA_MIC_AGC_DESIRED=0.02\n")
    text = env.read_text(encoding="utf-8")
    with pytest.raises(EnvEditError, match="2줄"):
        restore_latest(env, bk, datetime(2026, 10, 27, 18, 0, 0))
    assert env.read_text(encoding="utf-8") == text


def test_restore_without_backup(env, tmp_path):
    with pytest.raises(EnvEditError):
        restore_latest(env, tmp_path / "없는폴더", NOW)


def test_backup_same_second_gets_suffix(env, tmp_path):
    bk = tmp_path / "bk"
    a = values_from(read_env(env))
    b = MicValues(0.010, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
    first = write_changes(env, plan_changes(a, b), NOTE, bk, NOW)
    c = MicValues(0.012, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
    second = write_changes(env, plan_changes(b, c), NOTE, bk, NOW)
    assert first.name == ".env.20261027-143205" and second.name == ".env.20261027-143205-1"
    assert latest_backup(bk) == second


def _tmp_leftovers(folder):
    return [p.name for p in folder.iterdir() if p.name.startswith(".env.tmp-")]


def test_write_keeps_file_mode_and_leaves_no_temp(env, tmp_path):
    env.chmod(0o600)
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    write_changes(env, plan_changes(before, after), NOTE, tmp_path / "bk", NOW)
    assert env.stat().st_mode & 0o777 == 0o600
    assert _tmp_leftovers(env.parent) == []


def test_restore_keeps_file_mode(env, tmp_path):
    env.chmod(0o600)
    bk = tmp_path / "bk"
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    write_changes(env, plan_changes(before, after), NOTE, bk, NOW)
    restore_latest(env, bk, datetime(2026, 10, 27, 18, 0, 0))
    assert env.stat().st_mode & 0o777 == 0o600
    assert values_from(read_env(env)) == before


def test_failed_write_leaves_original(env, tmp_path, monkeypatch):
    import src.env_file as env_file_mod

    def boom(src, dst):
        raise OSError("디스크 가득 참 (시험)")

    monkeypatch.setattr(env_file_mod.os, "replace", boom)
    before = values_from(read_env(env))
    after = MicValues(0.010, 0.02, 0.65, 0.50, 0.007, 1.5, 1)
    with pytest.raises(OSError):
        write_changes(env, plan_changes(before, after), NOTE, tmp_path / "bk", NOW)
    assert env.read_text(encoding="utf-8") == SAMPLE
    assert _tmp_leftovers(env.parent) == []
