"""화자 검증 녹음기의 순수한 부분 — 문장 순서, 저장 형식(평가기가 그대로 읽는지), 사람 이름 검사."""
from __future__ import annotations

import numpy as np
import pytest

from tools.spk_verify_eval import load_takes, read_wav
from tools.spk_verify_record import prompts, safe_person, save_take


def test_commands_are_interleaved_never_same_twice_in_a_row():
    cmds = [p["text"] for p in prompts() if p["kind"] == "cmd"]
    assert all(a != b for a, b in zip(cmds, cmds[1:]))
    assert {t: cmds.count(t) for t in set(cmds)} == {"화장실로 가자": 3, "안내 끝내줘": 3, "취소해": 3}


def test_saved_takes_are_read_back_by_the_evaluator(tmp_path):
    six = np.zeros((1600, 6), dtype=np.int16)
    six[:, 1] = 1000
    save_take(tmp_path, "p1", {"kind": "enroll", "idx": 1, "text": "그래"}, six)
    save_take(tmp_path, "p1", {"kind": "wake", "idx": 7, "text": "비카야"}, six)
    save_take(tmp_path, "p1", {"kind": "cmd", "idx": 9, "text": "취소해"}, six)
    save_take(tmp_path, "p1", {"kind": "cmd", "idx": 8, "text": "안내 끝내줘"}, six)

    people = load_takes(tmp_path)
    assert people["p1"]["enroll"] == ["p1/enroll_01.wav"]
    assert people["p1"]["wake"] == "p1/wake_07.wav"
    assert people["p1"]["cmd"] == ["p1/cmd_08.wav", "p1/cmd_09.wav"]
    audio = read_wav(tmp_path / "p1/cmd_09.wav")
    assert audio.shape == (1600, 6)
    assert audio[0, 1] == pytest.approx(1000 / 32768) and audio[0, 0] == 0.0


def test_re_recording_replaces_the_take(tmp_path):
    a = np.zeros((1600, 6), dtype=np.int16)
    b = a.copy()
    b[:, 0] = 500
    prompt = {"kind": "cmd", "idx": 8, "text": "취소해"}
    save_take(tmp_path, "p1", prompt, a)
    save_take(tmp_path, "p1", prompt, b)
    assert load_takes(tmp_path)["p1"]["cmd"] == ["p1/cmd_08.wav"]
    assert read_wav(tmp_path / "p1/cmd_08.wav")[0, 0] == pytest.approx(500 / 32768)


@pytest.mark.parametrize("name", ["", "../x", "a/b", " "])
def test_person_name_cannot_escape_the_folder(name):
    with pytest.raises(ValueError):
        safe_person(name)


def test_person_name_is_trimmed():
    assert safe_person(" 민준 ") == "민준"
