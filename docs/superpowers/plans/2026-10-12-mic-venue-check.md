# 마이크 현장 점검기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시연장에서 ⑫를 끄고 한 번 돌리면, 휴대폰 화면에 실시간 소리·잡음·호출/멈춰 점수와 합격 판정을 보여 주고, 사람이 값을 고쳐 다시 채점하고 `.env` 에 저장·되돌릴 수 있는 점검기를 만든다.

**Architecture:** 판정은 ⑫와 같은 `WakewordMonitor` 를 그대로 쓴다(문턱 인자·실행 중 바꾸기·조각 알림만 더함). 순수 로직(`mic_check_rules` 판정, `env_file` `.env` 편집, `mic_check_replay` 녹음 재채점, `mic_check_session` 진행)은 `src/` 에 두고 가짜 모델로 시험한다. 하드웨어(마이크·칩·스피커)와 휴대폰 중계(websockets)는 `tools/` 에 둔다.

**Tech Stack:** Python 3.10, numpy, openWakeWord·faster-whisper(기존 모델), sounddevice, pyusb(기존 `dsp_state`), websockets 15.0.1(설치됨), pytest. 페이지는 의존성 없는 HTML+JS 한 파일.

**Spec:** `docs/superpowers/specs/2026-10-12-mic-venue-check-design.md` (화면 목업: https://claude.ai/artifact/2dZ8TTfMTNnLoDAv9Kom5Q)

## Global Constraints

- 저장소 `vica-voice-llm`, 브랜치 `after_feedback_waiting`. 새 브랜치를 만들지 않는다.
- **커밋·푸시는 사용자가 말할 때만 한다.** 각 작업 끝의 커밋 단계는 사용자 승인 뒤에 실행한다. 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- 새 의존성 없음. `websockets` 15.0.1 은 이미 `.venv` 에 있다.
- 점검기는 rclpy 를 쓰지 않고 ROS 에 아무것도 발행하지 않는다(진짜 비상정지가 걸리면 안 된다).
- ⑫ 동작 불변: `.env` 에 새 줄(`VICA_LISTEN_MIN_RMS`, `VICA_SHORT_ANSWER_MIN_RMS`)이 없으면 지금과 똑같다. 기존 시험 902개가 그대로 통과해야 한다.
- 기본 포트 8770. 기록은 `~/vica_data/mic_checks/<YYYYMMDD-HHMM>_<장소>/`, `.env` 백업은 `~/vica_data/mic_checks/env_backups/`.
- 합격 기준 숫자는 `src/mic_check_rules.py` 위쪽 상수 한 곳에만 둔다(초기값, 실기로 다듬는다).
- 마이크·스피커·칩·실제 모델을 쓰는 코드는 자동 시험하지 않는다(저장소 CLAUDE.md Testing Rules). 순수 로직만 단위 시험.
- 시험 명령은 저장소 뿌리에서 `.venv/bin/python -m pytest ...`. 로봇이 주행 중이면 전체 시험은 미룬다(주행 중 무거운 작업 금지).
- 주석·문서·화면 글자·커밋은 한국어, 기존 문체(주석은 "~한다").

## Review Focus

1. 휴대폰 화면이 꺼졌다 켜져 다시 붙을 때 — 지난 상태·결과가 곧바로 다시 보여야 한다(빈 화면 금지). → Task 5 `test_kept_message_sent_on_connect_and_commands_queued`
2. "녹음 시작"을 두 번 빨리 누를 때 — 두 번째는 무시하고 안내만 띄워야 한다(녹음이 둘로 갈라지면 안 됨). → Task 6 `test_double_record_is_ignored`
3. `.env` 에 같은 키가 주석으로만 있을 때(`# VICA_WAKE_THRESHOLD=0.6`) — 주석은 건드리지 않고 새 줄을 붙여야 한다. → Task 3 `test_write_replaces_existing_and_appends_missing`
4. 페이지가 범위 밖·이상한 값을 보낼 때(`{"cmd":"set","key":"wake_threshold","value":0.95}`, 글자 값, 모르는 키) — 값은 그대로, 안내만. → Task 6 `test_set_value_out_of_range_ignored`
5. 6단계 멘트 wav 가 하나 빠졌을 때 — 그 파일만 건너뛰고 나머지로 재야 한다(도구가 죽으면 안 됨). → Task 7 `test_load_clips_skips_missing`

---

## File Structure

| 파일 | 새로·수정 | 책임 |
|---|---|---|
| `src/wakeword_monitor.py` | 수정 | 문턱 두 개를 인자·`.env` 로, `set_thresholds()`, `state`, `on_frame`, 대화·질문 창 전사기 인자 |
| `src/ros_wakeword_node.py` | 수정 | 시작 로그에 수음 최소·짧은 답 최소 |
| `.env.example` | 수정 | 새 두 줄 설명 |
| `src/mic_check_rules.py` | 새로 | `MicValues`, 측정 자료형, 합격 기준 상수, `judge()`, `check_values()`, `noise_floor()` |
| `src/env_file.py` | 새로 | `.env` 읽기(⑫ 와 같은 해석)·바꾸기·백업·되돌리기 |
| `src/mic_check_replay.py` | 새로 | `Recording`, `Models`, `replay_call/answer/stop` |
| `src/mic_check_session.py` | 새로 | 단계 진행, 실시간 귀, 녹음, 재채점, 값 고치기, 저장, 기록, 메시지 |
| `tools/mic_check_web.py` | 새로 | `LiveServer` — `/` 페이지, `/ws` 실시간 통로 |
| `tools/mic_venue_check.py` | 새로 | 진입점 — ⑫·⑬ 확인, `.env`, 칩, 마이크, 모델, 멘트, `--restore`, `--demo` |
| `tools/mic_check_page.html` | 새로 | 휴대폰 화면 |
| `docs/mic-venue-check.md` | 새로 | 사용법 |
| `tests/test_wakeword_thresholds.py` 외 6개 | 새로 | 단위 시험 |

설계서 9절의 파일 목록에 `src/mic_check_session.py` 하나를 더 나눴다. 진행 로직을 하드웨어 없이 시험하려는 것이다.

---

### Task 1: 귀(WakewordMonitor)에 문턱 인자·실행 중 바꾸기·조각 알림

**Files:**
- Modify: `src/wakeword_monitor.py` (상수 87줄 아래, `__init__` 374~537, `process_frame` 799~826, `_listen_step` 1200~1202)
- Modify: `src/ros_wakeword_node.py:204-207` (시작 로그)
- Modify: `.env.example:40` 아래
- Test: `tests/test_wakeword_thresholds.py`

**Interfaces:**
- Produces:
  - `listen_thresholds_from_env() -> tuple[float, float]`
  - `WakewordMonitor(..., listen_min_rms: Optional[float] = None, short_answer_min_rms: Optional[float] = None, transcribe_listen: Optional[Callable] = None, transcribe_confirm: Optional[Callable] = None, on_frame: Optional[Callable[[dict], None]] = None)`
  - 속성 `monitor.listen_min_rms: float`, `monitor.short_answer_min_rms: float`, `monitor.state -> str` ("idle"|"postroll"|"listen")
  - `monitor.set_thresholds(*, wake=None, emergency_single=None, listen_min_rms=None, short_answer_min_rms=None) -> None`
  - `on_frame` 이 받는 dict: `{"rms": float, "a": float, "b": float, "state": str, "speaking": bool, "muted": bool, "t": float}` — `state` 는 그 조각을 처리하기 **전** 상태

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_wakeword_thresholds.py`:

```python
"""수음 최소·짧은 답 최소를 .env·인자로, 실행 중 문턱 바꾸기, 조각 점수 알림(2026-10-12 마이크 점검기).

설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md 9절. 새 줄이 없으면 지금과 똑같아야 한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.wakeword_monitor import (LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS, WakewordMonitor,
                                  listen_thresholds_from_env)
from tests.test_wakeword_monitor import LOUD, QUIET, Fake, run_frames

# 조각 rms ≈ 0.0128. 말 10조각 + 조용한 11조각인 창 전체 rms ≈ 0.0088 — 0.008 과 0.010 사이.
VOICE = np.full(1280, 420, dtype=np.int16)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("VICA_LISTEN_MIN_RMS", raising=False)
    monkeypatch.delenv("VICA_SHORT_ANSWER_MIN_RMS", raising=False)


def make(fake=None, texts=None, **kw):
    fake = fake or Fake(scores=[])
    return WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=(texts.append if texts is not None else (lambda t: None)),
        predict=fake.predict, transcribe=fake.transcribe,
        transcribe_confirm=lambda a: "네", **kw)


def answer(m, voiced, frame):
    """노드처럼 질문 예약 → 말 끝으로 질문 창을 열고, 말 조각 뒤 조용한 조각을 흘린다."""
    m.arm_followup(now=0.0)
    m.set_speaking(False, now=0.0)
    out = run_frames(m, voiced, frame, t0=0.08, vad=True)
    out += run_frames(m, 20, QUIET, t0=0.08 * (voiced + 1), vad=False)
    return out


def test_defaults_without_env():
    assert listen_thresholds_from_env() == (LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS) == (0.008, 0.02)
    m = make()
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.008, 0.02)


def test_env_values(monkeypatch):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", "0.012")
    monkeypatch.setenv("VICA_SHORT_ANSWER_MIN_RMS", "0.018")
    m = make()
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.012, 0.018)


@pytest.mark.parametrize("raw", ["", "abc", "0", "-0.01", "1", "1.5"])
def test_bad_env_falls_back(monkeypatch, raw):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", raw)
    monkeypatch.setenv("VICA_SHORT_ANSWER_MIN_RMS", raw)
    assert listen_thresholds_from_env() == (0.008, 0.02)


def test_args_beat_env(monkeypatch):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", "0.012")
    m = make(listen_min_rms=0.006, short_answer_min_rms=0.015)
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.006, 0.015)


def test_listen_min_rms_decides_acceptance():
    texts = []
    assert "user_text" in answer(make(texts=texts, listen_min_rms=0.008), 10, VOICE)
    assert texts == ["네"]
    texts2 = []
    assert "wake_silent" in answer(make(texts=texts2, listen_min_rms=0.010), 10, VOICE)
    assert texts2 == []


def test_short_answer_min_rms_decides_short_rescue():
    # 한 조각(0.08 s)짜리 "네" — 0.16 s 아래라 짧은 답 구제(창 rms ≥ 짧은 답 최소)로만 산다. 창 rms ≈ 0.026.
    texts = []
    assert "user_text" in answer(make(texts=texts, short_answer_min_rms=0.02), 1, LOUD)
    texts2 = []
    assert "wake_silent" in answer(make(texts=texts2, short_answer_min_rms=0.03), 1, LOUD)


def test_set_thresholds_listen_values():
    texts = []
    m = make(texts=texts)
    m.set_thresholds(listen_min_rms=0.010)
    assert "wake_silent" in answer(m, 10, VOICE) and texts == []


def test_set_thresholds_wake():
    wakes = []
    fake = Fake(scores=[(0.8, 0), (0.8, 0)])
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        on_wake=lambda: wakes.append(1), predict=fake.predict, transcribe=fake.transcribe)
    m.set_thresholds(wake=0.85)
    run_frames(m, 2, LOUD)
    assert wakes == []
    assert m.gate_a.threshold == m.gate_a_listen.threshold == 0.85


def test_set_thresholds_emergency_single():
    fake = Fake(scores=[(0, 0.9)] + [(0, 0)] * 10, text="멈춰")
    events = []
    m = WakewordMonitor(on_emergency=events.append, on_user_text=lambda t: None,
                        predict=fake.predict, transcribe=fake.transcribe)
    m.set_thresholds(emergency_single=0.95)
    run_frames(m, 6, LOUD)
    assert events == [] and fake.stt_calls == 0


def test_on_frame_reports_each_frame_without_changing_results():
    seq = [(0.9, 0), (0.9, 0)] + [(0, 0)] * 6
    plain = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                            predict=Fake(seq).predict, transcribe=lambda a: "")
    infos = []
    watched = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                              predict=Fake(seq).predict, transcribe=lambda a: "", on_frame=infos.append)
    assert run_frames(plain, 8, LOUD) == run_frames(watched, 8, LOUD)
    assert len(infos) == 8
    assert infos[0]["a"] == 0.9 and infos[0]["b"] == 0 and infos[0]["state"] == "idle"
    assert infos[2]["state"] == "listen"      # 두 번째 조각에서 호출 확정 → 세 번째 조각은 창 안
    assert abs(infos[0]["rms"] - 3000 / 32768) < 1e-6
    assert infos[0]["muted"] is False and infos[0]["speaking"] is False


def test_on_frame_while_muted():
    infos = []
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        predict=Fake([]).predict, transcribe=lambda a: "", on_frame=infos.append)
    m.set_muted(True, now=0.0)
    run_frames(m, 1, LOUD, t0=0.01)
    assert infos[0]["muted"] is True and infos[0]["a"] == 0.0 and infos[0]["b"] == 0.0


def test_state_property():
    m = make(Fake(scores=[(0.9, 0), (0.9, 0)]))
    assert m.state == "idle"
    run_frames(m, 2, LOUD)
    assert m.state == "listen"


def test_injected_listen_transcriber_is_used_in_free_window():
    texts = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)], text="긴급용")
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=texts.append,
                        predict=fake.predict, transcribe=fake.transcribe,
                        transcribe_listen=lambda a: "화장실로 가자")
    run_frames(m, 2, LOUD)                               # 호출 → 자유 창
    run_frames(m, 10, LOUD, t0=0.16, vad=True)          # 말
    run_frames(m, 40, QUIET, t0=0.96, vad=False)        # 말끝 0.8 s·최소 개방 2.5 s 지나 닫힘
    assert texts == ["화장실로 가자"]
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_wakeword_thresholds.py -q`
Expected: FAIL — `ImportError: cannot import name 'listen_thresholds_from_env'`

- [ ] **Step 3: `listen_thresholds_from_env` 를 더한다**

`src/wakeword_monitor.py` 에서 `SHORT_ANSWER_MIN_RMS = 0.02` 줄 바로 아래에 넣는다:

```python


def listen_thresholds_from_env() -> tuple[float, float]:
    """(수음 최소, 짧은 답 최소) — VICA_LISTEN_MIN_RMS·VICA_SHORT_ANSWER_MIN_RMS (2026-10-12 마이크 점검기).

    줄이 없거나 이상한 값(숫자 아님·0 이하·1 이상)이면 위의 코드 기본값이다 — 그래서 `.env` 에 줄이
    없는 지금은 동작이 그대로다. 점검기가 시연장에서 값을 고쳐 저장하면 이 줄이 생긴다.
    """
    def one(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if 0.0 < value < 1.0 else default

    return (one("VICA_LISTEN_MIN_RMS", LISTEN_MIN_RMS),
            one("VICA_SHORT_ANSWER_MIN_RMS", SHORT_ANSWER_MIN_RMS))
```

- [ ] **Step 4: 생성자 인자와 속성을 더한다**

`__init__` 인자 목록 끝(`single_b: object = "env",` 다음 줄)에 넣는다:

```python
        listen_min_rms: Optional[float] = None,
        short_answer_min_rms: Optional[float] = None,
        transcribe_listen: Optional[Callable[[np.ndarray], str]] = None,
        transcribe_confirm: Optional[Callable[[np.ndarray], str]] = None,
        on_frame: Optional[Callable[[dict], None]] = None,
```

`__init__` 본문에서 두 줄을 바꾼다:

```python
        self._transcribe_listen: Optional[Callable[[np.ndarray], str]] = transcribe_listen
```
```python
        self._transcribe_confirm: Optional[Callable[[np.ndarray], str]] = transcribe_confirm
```

(원래는 둘 다 `= None` 이었다. 점검기·재채점이 한 번 올린 모델을 나눠 쓰려고 받는다. 주지 않으면 `_load_real()` 이 지금처럼 채운다.)

`self._gate_b_base = gate_b   # set_speaking 이 재생 중 완화/복원한다` 줄 바로 아래에 넣는다:

```python
        # 청취 판정의 두 소리 문턱(2026-10-12 마이크 점검기) — 인자 > .env > 코드 기본. 점검기가 실행 중
        # set_thresholds 로 바꾸고, ⑫ 노드는 .env 에서 읽는다(줄이 없으면 0.008·0.02 그대로).
        env_listen, env_short = listen_thresholds_from_env()
        self.listen_min_rms = env_listen if listen_min_rms is None else float(listen_min_rms)
        self.short_answer_min_rms = env_short if short_answer_min_rms is None else float(short_answer_min_rms)
        # 조각마다 점수 알림(점검기 화면용). 없으면 아무것도 하지 않는다 — ⑫ 노드는 주지 않는다.
        self._on_frame = on_frame
```

`self._wake_seq = 0                           # 호출 번호 — 판정과 짝을 맞춘다` 줄 바로 아래(`__init__` 끝)에 메서드 두 개와 속성을 넣는다:

```python

    @property
    def state(self) -> str:
        """idle / postroll / listen — 점검기·재채점이 칩 사람 말 표시를 넣을지 정할 때 쓴다(run() 과 같은 규칙)."""
        return self._state

    def set_thresholds(self, *, wake: Optional[float] = None, emergency_single: Optional[float] = None,
                       listen_min_rms: Optional[float] = None,
                       short_answer_min_rms: Optional[float] = None) -> None:
        """실행 중 문턱 바꾸기(2026-10-12 마이크 점검기). None 인 것은 그대로 둔다. ⑫ 노드는 부르지 않는다.

        호출 문턱은 창 밖 관문과 창 안 관찰 관문에 함께 쓴다. 긴급은 한 조각 문턱만 바꾼다 — 두 조각 연속
        문턱(0.5, 로봇이 말하는 동안 0.35)은 그대로다.
        """
        if wake is not None:
            self.gate_a.threshold = float(wake)
            self.gate_a_listen.threshold = float(wake)
        if emergency_single is not None:
            self.gate_b.single = float(emergency_single)
        if listen_min_rms is not None:
            self.listen_min_rms = float(listen_min_rms)
        if short_answer_min_rms is not None:
            self.short_answer_min_rms = float(short_answer_min_rms)

    def _notify_frame(self, frame: np.ndarray, now: float, scores: Optional[dict]) -> None:
        if self._on_frame is None:
            return
        try:
            self._on_frame({
                "rms": _frame_rms(frame),
                "a": 0.0 if scores is None else float(scores["a"]),
                "b": 0.0 if scores is None else float(scores["b"]),
                "state": self._state, "speaking": self._speaking,
                "muted": scores is None, "t": now,
            })
        except Exception:
            pass   # 화면 알림이 실패해도 감시는 계속 간다
```

- [ ] **Step 5: `process_frame` 에서 알린다**

`process_frame` 의 뮤트 분기를 바꾼다:

```python
        if self._is_muted(now):
            self.gate_a.reset()
            self.gate_b.reset()
            self._notify_frame(frame, now, None)
            return None

        scores = self._predict(frame)
        self._notify_frame(frame, now, scores)
        fire_b = self._feed_emergency(float(scores["b"]), now)
```

(원래 코드의 `scores = self._predict(frame)` 와 `fire_b = ...` 두 줄을 위 세 줄로 바꾼 것이다.)

- [ ] **Step 6: `_listen_step` 이 인스턴스 값을 쓰게 한다**

`_listen_step` 안 두 곳:

```python
        short_rescue = (short and self._listen_is_followup
                        and rms >= self.short_answer_min_rms)
        if rms < self.listen_min_rms or (short and not short_rescue):
```

- [ ] **Step 7: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_wakeword_thresholds.py -q`
Expected: PASS (18 passed)

- [ ] **Step 8: ⑫ 시작 로그와 `.env.example`**

`src/ros_wakeword_node.py` 의 첫 시작 로그를 바꾼다:

```python
        self.get_logger().info(
            f"호출 문턱 {self._monitor.gate_a.threshold:.2f} · "
            f"말 시작 기다림 {LISTEN_START_SEC:.1f}s · "
            f"수음 최소 {self._monitor.listen_min_rms:.4f} · "
            f"짧은 답 최소 {self._monitor.short_answer_min_rms:.4f} · "
            f"긴급 소리 저장 {self._estop_clip_dir or '꺼짐'}")
```

`.env.example` 의 `# VICA_EMERGENCY_SINGLE_FRAME=0.5 ...` 줄 바로 아래에 넣는다:

```
# VICA_LISTEN_MIN_RMS=0.008  # 수음 최소 크기(청취 창 전체 rms). 마이크 점검기가 시연장에서 바꿔 적는다(2026-10-12)
# VICA_SHORT_ANSWER_MIN_RMS=0.02  # 질문 창 짧은 답("네") 구제 최소 크기. 수음 최소의 1.5배 이상(2026-10-12)
```

- [ ] **Step 9: 전체 시험**

Run: `.venv/bin/python -m pytest -q`
Expected: 기존 902 + 새 18 모두 PASS

- [ ] **Step 10: 커밋 (사용자 승인 뒤)**

```bash
git add src/wakeword_monitor.py src/ros_wakeword_node.py .env.example tests/test_wakeword_thresholds.py
git commit -m "feat(voice): 귀의 수음 최소·짧은 답 최소를 .env·인자로, 실행 중 문턱 바꾸기·조각 알림 — 마이크 점검기 준비(줄이 없으면 동작 그대로)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 합격 기준과 판정 (`src/mic_check_rules.py`)

**Files:**
- Create: `src/mic_check_rules.py`
- Test: `tests/test_mic_check_rules.py`

**Interfaces:**
- Produces:
  - `MicValues(listen_min_rms, short_answer_min_rms, wake_threshold, emergency_single, agc, echo_etail, echo_nlatten)` frozen; `.chip() -> tuple`
  - `CallTry(woke, captured, peak, text="", emergency=False)`, `AnswerTry(expected, accepted, correct, clip_rms, text="", emergency=False)`, `StopTry(confirmed, gate, peak, text="", woke=False)`, `RobotRun(self_reactions, level)`
  - `Measurements(noise=[], calls=[], answers=[], stops=[], robot=None, chips={}, robot_thresholds=None)` — `chips` 키: `"noise"|"call"|"answer"|"stop"|"robot"` → 녹음 때 `MicValues.chip()`; `robot_thresholds` = 6단계 때 `(wake_threshold, emergency_single)`
  - `Row(key, name, grade, detail, todo="")`, `Verdict(rows, overall)` + `.title`
  - `judge(m: Measurements, v: MicValues, previous_robot_level: Optional[float] = None) -> Verdict`
  - `check_values(v: MicValues) -> list[str]`, `noise_floor(values) -> Optional[float]`, `echo_name(etail, nlatten) -> Optional[str]`
  - 상수 `RANGES`, `NAMES`, `AGC_CHOICES`, `ECHO_CHOICES`, `ECHO_LABELS`, `CALL_REPS=3`, `ANSWER_REPS=4`, `STOP_REPS=5`, 등급 `PASS WARN FAIL REDO WAIT`

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_check_rules.py`:

```python
"""마이크 점검기 합격 기준(설계 6절 초기값)·저장 검사(7절)."""
from __future__ import annotations

import pytest

from src.mic_check_rules import (FAIL, PASS, REDO, WAIT, WARN, AnswerTry, CallTry, Measurements, MicValues,
                                 RobotRun, StopTry, check_values, echo_name, judge, noise_floor)

V = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
KEYS = ("noise", "call", "answer", "stop", "robot")


def good() -> Measurements:
    return Measurements(
        noise=[0.0030, 0.0032, 0.0035],
        calls=[CallTry(True, True, 0.93, "화장실로 가자")] * 3,
        answers=[AnswerTry("yes", True, True, 0.045, "네")] * 2 + [AnswerTry("no", True, True, 0.040, "아니요")] * 2,
        stops=[StopTry(True, True, 0.86, "멈춰")] * 5,
        robot=RobotRun(0, 0.031),
        chips={k: V.chip() for k in KEYS},
        robot_thresholds=(0.70, 0.50),
    )


def row(m, key, v=V, prev=None):
    return next(r for r in judge(m, v, prev).rows if r.key == key)


def test_all_good_is_pass():
    verdict = judge(good(), V)
    assert verdict.overall == PASS and verdict.title == "시연 가능"


def test_empty_is_wait():
    verdict = judge(Measurements(), V)
    assert verdict.overall == WAIT and verdict.title == "점검 중"


@pytest.mark.parametrize("floor,expected", [(0.0040, PASS), (0.00401, WARN), (0.0061, WARN), (0.0062, FAIL)])
def test_noise_ratio_boundaries(floor, expected):
    m = good()
    m.noise = [floor * 0.9, floor * 0.95, floor]
    assert row(m, "noise").grade == expected


def test_noise_spread_warns():
    m = good()
    m.noise = [0.0010, 0.0020, 0.0030]      # 세 측정이 3배 차이
    r = row(m, "noise")
    assert r.grade == WARN and "세 번 차이" in r.detail


def test_call_margin_and_miss():
    m = good()
    m.calls = [CallTry(True, True, 0.79)] * 3
    assert row(m, "call").grade == WARN      # 0.79 < 0.70 + 0.1
    m.calls = [CallTry(True, True, 0.80)] * 3
    assert row(m, "call").grade == PASS
    m.calls = [CallTry(True, True, 0.9)] * 2 + [CallTry(False, False, 0.6)]
    assert row(m, "call").grade == FAIL
    m.calls = [CallTry(True, True, 0.9)] * 2 + [CallTry(True, False, 0.9)]
    r = row(m, "call")
    assert r.grade == FAIL and "말을 못 받음" in r.detail


def test_answer_margin_and_wrong_word():
    m = good()
    m.answers = [AnswerTry("yes", True, True, 0.029)] * 4
    assert row(m, "answer").grade == WARN    # 0.029 < 0.02 × 1.5
    m.answers = [AnswerTry("yes", True, True, 0.030)] * 4
    assert row(m, "answer").grade == PASS
    m.answers = [AnswerTry("yes", True, True, 0.04)] * 3 + [AnswerTry("no", True, False, 0.04, "네")]
    r = row(m, "answer")
    assert r.grade == FAIL and "'네'" in r.detail
    m.answers = [AnswerTry("yes", True, True, 0.04)] * 3 + [AnswerTry("no", False, False, None)]
    assert row(m, "answer").grade == FAIL


@pytest.mark.parametrize("ok,expected", [(5, PASS), (4, WARN), (3, FAIL)])
def test_stop_counts(ok, expected):
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰")] * ok + [StopTry(False, False, 0.44)] * (5 - ok)
    assert row(m, "stop").grade == expected


def test_stop_miss_reasons():
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰")] * 3 + [StopTry(False, False, 0.44), StopTry(False, True, 0.8, "멈추라")]
    r = row(m, "stop")
    assert "점수 0.44" in r.detail and "확인 기각 '멈추라'" in r.detail
    assert "잡음" in r.todo and "말투" in r.todo


def test_robot_rows():
    m = good()
    m.robot = RobotRun(1, 0.03)
    assert row(m, "robot").grade == FAIL
    m.robot = RobotRun(0, 0.046)
    assert row(m, "robot", prev=0.030).grade == WARN     # 0.046 > 0.030 × 1.5
    assert row(m, "robot", prev=0.031).grade == PASS     # 0.046 < 0.0465
    assert row(m, "robot", prev=None).grade == PASS


def test_robot_redo_when_thresholds_changed():
    m = good()
    v = MicValues(0.008, 0.02, 0.70, 0.43, 0.007, 1.5, 1)   # 멈춰 문턱만 바꿈
    r = row(m, "robot", v=v)
    assert r.grade == REDO and "6단계" in r.todo


def test_cross_trigger_fails():
    m = good()
    m.answers = list(m.answers)
    m.answers[3] = AnswerTry("no", True, True, 0.04, "아니요", emergency=True)
    assert row(m, "cross").grade == FAIL
    m = good()
    m.stops = [StopTry(True, True, 0.9, "멈춰", woke=True)] + [StopTry(True, True, 0.9, "멈춰")] * 4
    assert row(m, "cross").grade == FAIL


def test_agc_change_needs_rerecord_echo_change_only_robot():
    m = good()
    agc = MicValues(0.008, 0.02, 0.70, 0.50, 0.010, 1.5, 1)
    v = judge(m, agc)
    assert {r.key: r.grade for r in v.rows} == {
        "noise": REDO, "call": REDO, "answer": REDO, "stop": REDO, "robot": REDO, "cross": REDO}
    assert v.overall == REDO and v.title == "다시 녹음 필요"
    echo = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.0, 1)
    v = judge(m, echo)
    grades = {r.key: r.grade for r in v.rows}
    assert grades["robot"] == REDO
    assert all(g == PASS for k, g in grades.items() if k != "robot")


def test_overall_takes_worst():
    m = good()
    m.robot = RobotRun(2, 0.03)
    m.stops = [StopTry(True, True, 0.9)] * 4 + [StopTry(False, False, 0.4)]
    assert judge(m, V).overall == FAIL


def test_check_values():
    assert check_values(V) == []
    bad = MicValues(0.014, 0.02, 0.70, 0.50, 0.007, 1.5, 1)
    assert any("1.5배" in p for p in check_values(bad))
    out = MicValues(0.008, 0.02, 0.95, 0.30, 0.007, 1.5, 1)
    problems = check_values(out)
    assert any("호출 문턱" in p for p in problems) and any("멈춰" in p for p in problems)


def test_noise_floor_and_echo_name():
    assert noise_floor([]) is None
    assert noise_floor([0.001 * i for i in range(1, 11)]) == pytest.approx(0.010)
    assert echo_name(1.5, 1) == "strong" and echo_name(1.0, 0) == "factory" and echo_name(2.0, 1) is None
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mic_check_rules'`

- [ ] **Step 3: 구현한다**

`src/mic_check_rules.py`:

```python
"""마이크 현장 점검기 — 값·측정·합격 기준·판정(순수 함수, 마이크·모델 없음).

설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md 6·7절. 숫자는 모두 초기값이다(사용자 10-12:
"수치만 보고서는 모른다, 계속 돌려봐야 안다"). 기준을 다듬을 때는 이 파일 위쪽 상수만 고친다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------- 합격 기준 (초기값)
NOISE_PASS_RATIO = 2.0         # 수음 최소 ÷ 잡음 바닥. 10-06 규칙 "수음 최소 = 잡음 × 2" 와 같은 여유
NOISE_WARN_RATIO = 1.3         # 이 아래면 사람이 말하지 않아도 잡음이 말로 잡힐 수 있다
NOISE_SPREAD_WARN = 2.0        # 2단계 세 측정의 최대 ÷ 최소 — 10-06 의 들쭉날쭉을 드러낸다
WAKE_MARGIN = 0.10             # 가장 낮은 호출 점수가 호출 문턱 + 이것 아래면 아슬아슬
SHORT_MARGIN = 1.5             # 가장 작은 짧은 답 소리가 짧은 답 최소 × 이것 아래면 아슬아슬
STOP_PASS, STOP_WARN = 5, 4    # "멈춰" 5번 중 확정 수
ROBOT_LEVEL_WARN = 1.5         # 로봇 목소리가 지난 점검의 이 배를 넘으면 볼륨이 올라가 있다
SHORT_OVER_LISTEN = 1.5        # 저장 검사: 짧은 답 최소 ≥ 수음 최소 × 이것 (10-06 규칙)
NOISE_PERCENTILE = 0.9         # 잡음 바닥 = 조각 rms 의 상위 10 %

CALL_REPS, ANSWER_REPS, STOP_REPS = 3, 4, 5

# ---------------------------------------------------------------- 값 고치기 허용 범위 (설계 7절)
RANGES = {   # 필드: (최소, 최대, 눈금)
    "listen_min_rms": (0.005, 0.015, 0.0005),
    "short_answer_min_rms": (0.012, 0.025, 0.001),
    "wake_threshold": (0.50, 0.90, 0.01),
    "emergency_single": (0.40, 0.70, 0.01),
}
NAMES = {
    "listen_min_rms": "수음 최소", "short_answer_min_rms": "짧은 답 최소",
    "wake_threshold": "호출 문턱", "emergency_single": "“멈춰” 한 조각 문턱",
    "agc": "AGC 목표", "echo": "에코 지우기",
}
AGC_CHOICES = (0.005, 0.007, 0.010)                                          # 이 로봇에서 써 본 값만
ECHO_CHOICES = {"strong": (1.5, 1), "mid": (1.0, 1), "factory": (1.0, 0)}   # 이름: (GAMMA_ETAIL, NLATTENONOFF)
ECHO_LABELS = {"strong": "센 (1.5·켜짐)", "mid": "중간 (1.0·켜짐)", "factory": "공장값 (1.0·꺼짐)"}

PASS, WARN, FAIL, REDO, WAIT = "pass", "warn", "fail", "redo", "wait"
_RANK = {PASS: 0, WARN: 1, WAIT: 2, REDO: 3, FAIL: 4}
TITLES = {PASS: "시연 가능", WARN: "시연 가능 · 확인 필요", WAIT: "점검 중",
          REDO: "다시 녹음 필요", FAIL: "고친 뒤 다시"}


@dataclass(frozen=True)
class MicValues:
    """점검기가 다루는 값 일곱 개 — `.env` 와 1:1 (src/env_file.KEYS)."""
    listen_min_rms: float
    short_answer_min_rms: float
    wake_threshold: float
    emergency_single: float
    agc: float
    echo_etail: float
    echo_nlatten: int

    def chip(self) -> tuple:
        """칩에 쓰는 셋 — 이것이 바뀌면 녹음이 무효다(칩이 소리 자체를 바꾼다)."""
        return (round(float(self.agc), 6), round(float(self.echo_etail), 6), int(self.echo_nlatten))

    def thresholds(self) -> tuple:
        """6단계(실시간으로 잰 자기 반응)에 영향을 주는 문턱 둘."""
        return (round(float(self.wake_threshold), 6), round(float(self.emergency_single), 6))


def echo_name(etail: float, nlatten: int) -> Optional[str]:
    """에코 단계 이름. 사다리 밖 값이면 None."""
    for name, (e, n) in ECHO_CHOICES.items():
        if abs(e - float(etail)) < 1e-6 and int(n) == int(nlatten):
            return name
    return None


def check_values(v: MicValues) -> list:
    """저장을 막을 문제(빈 목록이면 저장 가능): 허용 범위 밖, 짧은 답 최소 < 수음 최소 × 1.5."""
    problems = []
    for key, (lo, hi, _step) in RANGES.items():
        x = float(getattr(v, key))
        if not lo - 1e-9 <= x <= hi + 1e-9:
            problems.append(f"{NAMES[key]} {x:g} 은(는) 허용 범위 {lo:g} ~ {hi:g} 밖입니다")
    if v.short_answer_min_rms < v.listen_min_rms * SHORT_OVER_LISTEN - 1e-9:
        problems.append(f"짧은 답 최소는 수음 최소의 {SHORT_OVER_LISTEN:g}배 이상이어야 합니다 "
                        f"(지금 {v.short_answer_min_rms / v.listen_min_rms:.2f}배)")
    return problems


def noise_floor(values: Sequence[float]) -> Optional[float]:
    """조각 rms 의 상위 10 % 값. 비어 있으면 None."""
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(len(xs) * NOISE_PERCENTILE))]


# ---------------------------------------------------------------- 측정
@dataclass(frozen=True)
class CallTry:
    """'비카야, 화장실로 가자' 한 번의 재채점 결과."""
    woke: bool
    captured: bool            # 청취 창이 이어진 말을 받았나(받아쓰기가 나왔나)
    peak: float               # 호출 점수 최고
    text: str = ""
    emergency: bool = False   # 이 말에 멈춰가 확정됐나(오발동)


@dataclass(frozen=True)
class AnswerTry:
    expected: str             # "yes" | "no"
    accepted: bool
    correct: bool             # 긍정·부정이 맞나
    clip_rms: Optional[float]
    text: str = ""
    emergency: bool = False


@dataclass(frozen=True)
class StopTry:
    confirmed: bool
    gate: bool                # 관문(한 조각 또는 두 조각)을 넘었나 — 넘었으면 확인 STT 를 탔다
    peak: float               # 멈춰 점수 최고
    text: str = ""            # 확인 STT 글자(관문을 넘었을 때)
    woke: bool = False        # 이 말에 호출이 확정됐나(오발동)


@dataclass(frozen=True)
class RobotRun:
    self_reactions: int
    level: float              # 재생 중 조각 rms 평균(에코 지우기 뒤 소리)


@dataclass
class Measurements:
    noise: list = field(default_factory=list)       # 2단계 세 측정
    calls: list = field(default_factory=list)       # CallTry
    answers: list = field(default_factory=list)     # AnswerTry
    stops: list = field(default_factory=list)       # StopTry
    robot: Optional[RobotRun] = None
    chips: dict = field(default_factory=dict)       # 줄 키 → 녹음할 때 MicValues.chip()
    robot_thresholds: Optional[tuple] = None        # 6단계 때 MicValues.thresholds()


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    grade: str
    detail: str
    todo: str = ""


@dataclass(frozen=True)
class Verdict:
    rows: tuple
    overall: str

    @property
    def title(self) -> str:
        return TITLES[self.overall]


# ---------------------------------------------------------------- 판정
def judge(m: Measurements, v: MicValues, previous_robot_level: Optional[float] = None) -> Verdict:
    """설계 6절 표 그대로. 가장 나쁜 줄이 종합이다(불합격 > 다시 녹음 > 점검 중 > 주의 > 합격)."""
    rows = [
        _noise_row(m, v),
        _call_row(m, v),
        _answer_row(m, v),
        _stop_row(m, v),
        _robot_row(m, v, previous_robot_level),
        _cross_row(m, v),
    ]
    overall = max((r.grade for r in rows), key=_RANK.__getitem__)
    return Verdict(tuple(rows), overall)


_AGC_REDO = "AGC 가 바뀌어 다시 녹음해야 합니다"


def _agc_stale(m: Measurements, key: str, v: MicValues) -> bool:
    chip = m.chips.get(key)
    return chip is not None and chip[0] != v.chip()[0]


def _noise_row(m: Measurements, v: MicValues) -> Row:
    name = "잡음 바닥"
    if not m.noise:
        return Row("noise", name, WAIT, "2단계 전")
    if _agc_stale(m, "noise", v):
        return Row("noise", name, REDO, _AGC_REDO, "2단계를 다시 하세요")
    floor, low = max(m.noise), min(m.noise)
    ratio = v.listen_min_rms / floor if floor > 0 else math.inf
    spread = floor / low if low > 0 else math.inf
    detail = f"{floor:.4f} · 수음 최소까지 {ratio:.1f}배"
    if spread > NOISE_SPREAD_WARN:
        detail += f" · 세 번 차이 {spread:.1f}배"
    if ratio < NOISE_WARN_RATIO:
        return Row("noise", name, FAIL, detail,
                   "에어컨·환풍기 같은 소음원을 찾거나 로봇 자리를 옮기세요. 또는 수음 최소를 올리세요.")
    if ratio < NOISE_PASS_RATIO:
        return Row("noise", name, WARN, detail,
                   "잡음이 수음 최소에 가깝습니다. 더 시끄러워지면 말하지 않아도 말로 잡힐 수 있습니다.")
    if spread > NOISE_SPREAD_WARN:
        return Row("noise", name, WARN, detail, "잡음이 일정하지 않습니다. 조용할 때 2단계를 다시 해 보세요.")
    return Row("noise", name, PASS, detail)


def _call_row(m: Measurements, v: MicValues) -> Row:
    name = "“비카야, 화장실로 가자” ×3"
    tries = m.calls
    if tries and _agc_stale(m, "call", v):
        return Row("call", name, REDO, _AGC_REDO, "3단계를 다시 하세요")
    if len(tries) < CALL_REPS:
        return Row("call", name, WAIT, f"{len(tries)}/{CALL_REPS}")
    ok = sum(t.woke and t.captured for t in tries)
    low = min(t.peak for t in tries)
    detail = f"{ok}/{CALL_REPS} · 가장 낮은 점수 {low:.2f}"
    if ok < CALL_REPS:
        missed = [t for t in tries if not (t.woke and t.captured)]
        why = "호출 놓침" if any(not t.woke for t in missed) else "호출 뒤 말을 못 받음"
        return Row("call", name, FAIL, f"{detail} · {why}",
                   "손잡이 쪽을 향해 또렷하게 다시 해 보세요. 그래도 놓치면 호출 문턱을 보세요.")
    if low < v.wake_threshold + WAKE_MARGIN - 1e-9:
        return Row("call", name, WARN, f"{detail} (문턱 + {WAKE_MARGIN:.1f} 아래)", "호출이 아슬아슬하게 걸립니다.")
    return Row("call", name, PASS, detail)


def _answer_row(m: Measurements, v: MicValues) -> Row:
    name = "“네”·“아니요” ×4"
    tries = m.answers
    if tries and _agc_stale(m, "answer", v):
        return Row("answer", name, REDO, _AGC_REDO, "4단계를 다시 하세요")
    if len(tries) < ANSWER_REPS:
        return Row("answer", name, WAIT, f"{len(tries)}/{ANSWER_REPS}")
    ok = sum(t.accepted and t.correct for t in tries)
    sizes = [t.clip_rms for t in tries if t.clip_rms is not None]
    low = min(sizes) if sizes else 0.0
    detail = f"{ok}/{ANSWER_REPS} · 가장 작은 소리 {low:.3f} (짧은 답 최소의 {low / v.short_answer_min_rms:.1f}배)"
    if ok < ANSWER_REPS:
        wrong = [t for t in tries if t.accepted and not t.correct]
        if wrong:
            return Row("answer", name, FAIL, f"{detail} · 받아쓰기 '{wrong[0].text}'",
                       "긍정·부정을 다르게 알아들었습니다. 받아쓴 글자를 보고 다시 해 보세요.")
        return Row("answer", name, FAIL, detail, "소리가 작았으면 짧은 답 최소를 낮추세요.")
    if low < v.short_answer_min_rms * SHORT_MARGIN - 1e-9:
        return Row("answer", name, WARN, detail, "짧은 답이 아슬아슬하게 들어갑니다.")
    return Row("answer", name, PASS, detail)


def _stop_row(m: Measurements, v: MicValues) -> Row:
    name = "“멈춰” ×5"
    tries = m.stops
    if tries and _agc_stale(m, "stop", v):
        return Row("stop", name, REDO, _AGC_REDO, "5단계를 다시 하세요")
    if len(tries) < STOP_REPS:
        return Row("stop", name, WAIT, f"{len(tries)}/{STOP_REPS}")
    ok = sum(t.confirmed for t in tries)
    misses = [t for t in tries if not t.confirmed]
    parts = [f"확인 기각 '{t.text}'" if t.gate else f"점수 {t.peak:.2f}" for t in misses]
    detail = f"{ok}/{STOP_REPS}" + (" · 놓침: " + ", ".join(parts) if parts else "")
    if ok >= STOP_PASS:
        return Row("stop", name, PASS, detail)
    todo = []
    if any(not t.gate for t in misses):
        todo.append("점수가 낮았습니다. 잡음을 확인하세요. 멈춰 문턱을 낮출 수 있지만 오발동이 늘어납니다.")
    if any(t.gate for t in misses):
        todo.append("받아쓴 글자가 '멈춰'가 아니었습니다. 말투를 확인하세요.")
    return Row("stop", name, WARN if ok >= STOP_WARN else FAIL, detail, " ".join(todo))


def _robot_row(m: Measurements, v: MicValues, previous: Optional[float]) -> Row:
    name = "로봇 혼자 말하기"
    r = m.robot
    if r is None:
        return Row("robot", name, WAIT, "6단계 전")
    chip = m.chips.get("robot")
    if (chip is not None and chip != v.chip()) or (
            m.robot_thresholds is not None and tuple(m.robot_thresholds) != v.thresholds()):
        return Row("robot", name, REDO, "AGC·에코·문턱이 바뀌어 다시 재야 합니다", "6단계를 다시 하세요")
    detail = f"자기 반응 {r.self_reactions}회 · 로봇 목소리 {r.level:.3f}"
    if previous:
        detail += f" (지난 점검 {previous:.3f})"
    if r.self_reactions > 0:
        return Row("robot", name, FAIL, detail, "스피커 볼륨을 낮추고 6단계만 다시 하세요.")
    if previous and r.level > previous * ROBOT_LEVEL_WARN:
        return Row("robot", name, WARN, detail, "로봇 목소리가 지난 점검보다 큽니다. 스피커 볼륨을 확인하세요.")
    return Row("robot", name, PASS, detail)


def _cross_row(m: Measurements, v: MicValues) -> Row:
    name = "오발동"
    if any(_agc_stale(m, k, v) for k in ("call", "answer", "stop")) and (m.calls or m.answers or m.stops):
        return Row("cross", name, REDO, _AGC_REDO, "3~5단계를 다시 하세요")
    n = (sum(t.emergency for t in m.calls) + sum(t.emergency for t in m.answers)
         + sum(t.woke for t in m.stops))
    if n:
        return Row("cross", name, FAIL, f"{n}회",
                   "다른 말에 멈춰나 비카야가 잘못 걸렸습니다. 문턱을 내렸다면 되돌리세요.")
    done = len(m.calls) >= CALL_REPS and len(m.answers) >= ANSWER_REPS and len(m.stops) >= STOP_REPS
    return Row("cross", name, PASS if done else WAIT, "0회")
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_rules.py -q`
Expected: PASS

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add src/mic_check_rules.py tests/test_mic_check_rules.py
git commit -m "feat(voice): 마이크 점검기 합격 기준·판정 — 기준 숫자는 한 곳에 초기값으로(설계 6·7절)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `.env` 읽기·바꾸기·백업·되돌리기 (`src/env_file.py`)

**Files:**
- Create: `src/env_file.py`
- Test: `tests/test_env_file.py`

**Interfaces:**
- Consumes: `MicValues` (Task 2); `agc_desired_from_env`, `echo_tuning_from_env` (`src/dsp_state.py`); `LISTEN_MIN_RMS`, `SHORT_ANSWER_MIN_RMS`, `WAKE_THRESHOLD_DEFAULT`, `EMERGENCY_SINGLE_FRAME_DEFAULT` (`src/wakeword_monitor.py`)
- Produces:
  - `KEYS: dict[str, str]` (MicValues 필드 → .env 키), `EnvEditError(Exception)`, `Change(field, key, old, new)`
  - `read_env(path: Path) -> dict`, `values_from(mapping: Mapping) -> MicValues`
  - `format_value(field: str, value) -> str`, `plan_changes(before: MicValues, after: MicValues) -> list[Change]`, `change_line(change: Change, note: str) -> str`
  - `write_changes(path, changes, note, backup_dir, now=None) -> Path`(백업 경로), `backup_file(path, backup_dir, now=None, suffix="") -> Path`, `latest_backup(backup_dir) -> Optional[Path]`, `restore_latest(path, backup_dir, now=None) -> Path`(되돌린 백업 경로)

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_env_file.py`:

```python
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
    assert lines[2] == "VICA_MIC_AGC_DESIRED=0.010  # 10-27 시연장 점검에서 바꿈 (전 0.007)"   # 제자리
    assert "# VICA_WAKE_THRESHOLD=0.6" in lines                       # 주석은 그대로
    assert "# 마이크 점검기에서 추가 (2026-10-12 설계)" in lines
    assert lines[-2:] == ["VICA_LISTEN_MIN_RMS=0.01  # 10-27 시연장 점검에서 바꿈 (전 0.008)",
                          "VICA_WAKE_THRESHOLD=0.65  # 10-27 시연장 점검에서 바꿈 (전 0.70)"]
    assert lines[:2] == ["# 맨 위 주석", "VICA_LLM_PROVIDER=openai"]
    # ⑫ 가 쓰는 python-dotenv 로 읽어도 새 값이 나온다
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
    assert env.read_text(encoding="utf-8") == SAMPLE
    kept = bk / ".env.20261027-180000.before-restore"
    assert kept.exists() and "VICA_WAKE_THRESHOLD=0.65" in kept.read_text(encoding="utf-8")
    assert latest_backup(bk).name == ".env.20261027-143205"        # before-restore 는 되돌릴 후보가 아니다


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
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_env_file.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.env_file'`

- [ ] **Step 3: 구현한다**

`src/env_file.py`:

```python
"""`.env` 의 마이크 값 읽기·바꾸기·백업·되돌리기 — 마이크 현장 점검기(2026-10-12) 전용.

설계 7절. 바꾸기 전에 늘 백업하고, 바꾼 줄 끝에 언제·어디서·전 값을 남긴다. 관계없는 줄·주석·순서는 건드리지
않는다. 같은 키가 주석 아닌 줄로 두 번 이상 있으면 어느 줄이 쓰일지 헷갈리므로 쓰지 않는다.
읽는 규칙은 ⑫ 노드와 같다(wakeword_monitor·dsp_state 의 해석 함수와 같은 결과).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from .dsp_state import agc_desired_from_env, echo_tuning_from_env
from .mic_check_rules import MicValues
from .wakeword_monitor import (EMERGENCY_SINGLE_FRAME_DEFAULT, LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS,
                               WAKE_THRESHOLD_DEFAULT)

KEYS = {   # MicValues 필드 → .env 키
    "listen_min_rms": "VICA_LISTEN_MIN_RMS",
    "short_answer_min_rms": "VICA_SHORT_ANSWER_MIN_RMS",
    "wake_threshold": "VICA_WAKE_THRESHOLD",
    "emergency_single": "VICA_EMERGENCY_SINGLE_FRAME",
    "agc": "VICA_MIC_AGC_DESIRED",
    "echo_etail": "VICA_MIC_GAMMA_ETAIL",
    "echo_nlatten": "VICA_MIC_NLATTEN",
}
APPEND_HEADER = "# 마이크 점검기에서 추가 (2026-10-12 설계)"
_BACKUP_RE = re.compile(r"^\.env\.\d{8}-\d{6}(-\d+)?$")
_OFF = ("off", "none", "false", "no")


class EnvEditError(Exception):
    """쓰면 안 되는 상태 — 화면에 그대로 보여 줄 문장을 담는다."""


@dataclass(frozen=True)
class Change:
    field: str
    key: str
    old: str    # 바꾸기 전 ⑫ 가 쓰던 값(줄이 없으면 코드 기본값)
    new: str


def _line_key(line: str) -> Optional[str]:
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        return None
    left = s.split("=", 1)[0].strip()
    if left.startswith("export "):
        left = left[len("export "):].strip()
    return left or None


def _line_value(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
        return value[1:value.index(value[0], 1)]
    return value.split(" #", 1)[0].split("\t#", 1)[0].strip()


def read_env(path: Path) -> dict:
    """주석 아닌 KEY=VALUE 를 읽는다(같은 키가 여럿이면 마지막 — python-dotenv 와 같다)."""
    out: dict = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key = _line_key(line)
            if key:
                out[key] = _line_value(line)
    return out


def _float(mapping: Mapping, key: str, default: float) -> float:
    raw = str(mapping.get(key, "")).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 < value < 1.0 else default


def values_from(mapping: Mapping) -> MicValues:
    """⑫ 노드가 같은 환경에서 쓰게 될 값. 노드와 같은 해석(빈 값·이상한 값은 기본값)."""
    if str(mapping.get(KEYS["emergency_single"], "")).strip().lower() in _OFF:
        raise EnvEditError("VICA_EMERGENCY_SINGLE_FRAME 이 off 입니다. 점검기는 한 조각 규칙이 켜진 상태를 "
                           "전제로 합니다 — 줄을 지우거나 숫자로 바꾼 뒤 다시 실행하세요.")
    agc = agc_desired_from_env(str(mapping.get(KEYS["agc"], "0.010")))   # ⑫ 의 기본도 "0.010"
    if agc is None:
        raise EnvEditError("VICA_MIC_AGC_DESIRED 가 꺼져 있습니다(off·0). 점검기는 AGC 목표가 있는 상태를 전제로 합니다.")
    etail, nlatten = echo_tuning_from_env(str(mapping.get(KEYS["echo_etail"], "")),
                                          str(mapping.get(KEYS["echo_nlatten"], "")))
    return MicValues(
        listen_min_rms=_float(mapping, KEYS["listen_min_rms"], LISTEN_MIN_RMS),
        short_answer_min_rms=_float(mapping, KEYS["short_answer_min_rms"], SHORT_ANSWER_MIN_RMS),
        wake_threshold=_float(mapping, KEYS["wake_threshold"], WAKE_THRESHOLD_DEFAULT),
        emergency_single=_float(mapping, KEYS["emergency_single"], EMERGENCY_SINGLE_FRAME_DEFAULT),
        agc=agc,
        # 줄이 없으면 ⑫ 는 칩에 쓰지 않는다 — 칩 공장값(1.0·꺼짐)으로 본다.
        echo_etail=1.0 if etail is None else etail,
        echo_nlatten=0 if nlatten is None else nlatten,
    )


def format_value(field: str, value) -> str:
    if field in ("listen_min_rms", "short_answer_min_rms"):
        text = f"{float(value):.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if field in ("wake_threshold", "emergency_single"):
        return f"{float(value):.2f}"
    if field == "agc":
        return f"{float(value):.3f}"
    if field == "echo_etail":
        return f"{float(value):.1f}"
    return str(int(value))


def plan_changes(before: MicValues, after: MicValues) -> list:
    """before(지금 ⑫ 가 쓰는 값) → after 에서 달라진 값마다 Change. KEYS 순서."""
    changes = []
    for field, key in KEYS.items():
        old, new = format_value(field, getattr(before, field)), format_value(field, getattr(after, field))
        if old != new:
            changes.append(Change(field, key, old, new))
    return changes


def change_line(change: Change, note: str) -> str:
    return f"{change.key}={change.new}  # {note} (전 {change.old})"


def backup_file(path: Path, backup_dir: Path, now: Optional[datetime] = None, suffix: str = "") -> Path:
    path, backup_dir = Path(path), Path(backup_dir)
    if not path.exists():
        raise EnvEditError(f"{path} 가 없습니다.")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f".env.{stamp}{suffix}"
    n = 1
    while target.exists():
        target = backup_dir / f".env.{stamp}-{n}{suffix}"
        n += 1
    shutil.copy2(path, target)
    return target


def write_changes(path: Path, changes: list, note: str, backup_dir: Path,
                  now: Optional[datetime] = None) -> Path:
    """백업한 뒤 바꾼다. 백업 경로를 돌려준다. 바꿀 것이 없거나 같은 키가 두 줄 이상이면 쓰지 않는다."""
    path = Path(path)
    if not changes:
        raise EnvEditError(".env 와 다른 값이 없습니다.")
    if not path.exists():
        raise EnvEditError(f"{path} 가 없습니다.")
    lines = path.read_text(encoding="utf-8").splitlines()
    for c in changes:
        hits = [i for i, line in enumerate(lines) if _line_key(line) == c.key]
        if len(hits) > 1:
            raise EnvEditError(f"{c.key} 가 .env 에 {len(hits)}줄 있어 어느 줄이 쓰일지 알 수 없습니다. "
                               "하나만 남긴 뒤 다시 저장하세요.")
    backup = backup_file(path, backup_dir, now)
    appended = []
    for c in changes:
        hits = [i for i, line in enumerate(lines) if _line_key(line) == c.key]
        if hits:
            lines[hits[0]] = change_line(c, note)
        else:
            appended.append(change_line(c, note))
    if appended:
        if APPEND_HEADER not in lines:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(APPEND_HEADER)
        lines.extend(appended)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup


def latest_backup(backup_dir: Path) -> Optional[Path]:
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        return None
    found = sorted(p for p in backup_dir.iterdir() if _BACKUP_RE.match(p.name))
    return found[-1] if found else None


def restore_latest(path: Path, backup_dir: Path, now: Optional[datetime] = None) -> Path:
    """가장 최근 백업(마지막 저장 직전 상태)으로 되돌린다. 되돌리기 전 지금 파일을 .before-restore 로 남긴다."""
    latest = latest_backup(backup_dir)
    if latest is None:
        raise EnvEditError(f"되돌릴 백업이 없습니다 ({backup_dir}).")
    backup_file(path, backup_dir, now, suffix=".before-restore")
    shutil.copy2(latest, path)
    return latest
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_env_file.py -q`
Expected: PASS

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add src/env_file.py tests/test_env_file.py
git commit -m "feat(voice): 마이크 점검기 .env 편집 — 백업 먼저·바꾼 줄 표시·중복 키 거절·되돌리기, 읽기는 ⑫ 와 같은 해석

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 녹음 재채점 (`src/mic_check_replay.py`)

**Files:**
- Create: `src/mic_check_replay.py`
- Test: `tests/test_mic_check_replay.py`

**Interfaces:**
- Consumes: `WakewordMonitor` 새 인자(Task 1), `MicValues`, `CallTry`, `AnswerTry`, `StopTry`(Task 2)
- Produces:
  - `DT = 0.08`, `Recording(frames: tuple, vads: tuple)` + `.seconds`
  - `Models(predict, transcribe, transcribe_listen=None, transcribe_confirm=None, listen_hint=None)` + `Models.from_loaded(monitor)`
  - `replay_call(models, rec, quiet: Optional[Recording], values) -> CallTry`
  - `replay_answer(models, rec, quiet, values, expected: str) -> AnswerTry`
  - `replay_stop(models, rec, quiet, values) -> StopTry`
  - `is_affirmative(text) -> bool`, `is_negative(text) -> bool`

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_check_replay.py`:

```python
"""녹음 재채점 — 같은 녹음·같은 값이면 같은 결과, 문턱을 바꾸면 결과가 따라 바뀐다(설계 5절)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.mic_check_replay import Models, Recording, replay_answer, replay_call, replay_stop
from src.mic_check_rules import MicValues

WAKE = np.full(1280, 1111, dtype=np.int16)    # 가짜 모델: 호출 점수 0.9
STOP = np.full(1280, 2222, dtype=np.int16)    # 가짜 모델: 멈춰 점수 0.9
VOICE = np.full(1280, 3000, dtype=np.int16)   # 말소리(점수 0)
QUIET = np.zeros(1280, dtype=np.int16)


def predict(frame):
    v = int(frame[0])
    return {"a": 0.9 if v == 1111 else 0.0, "b": 0.9 if v == 2222 else 0.0}


def models(verify="멈춰", listen="화장실로 가자", confirm="네"):
    return Models(predict=predict, transcribe=lambda a: verify,
                  transcribe_listen=lambda a: listen, transcribe_confirm=lambda a: confirm)


V = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)


def rec(*parts):
    frames, vads = [], []
    for frame, n, vad in parts:
        frames += [frame] * n
        vads += [vad] * n
    return Recording(tuple(frames), tuple(vads))


QUIET_REC = rec((QUIET, 10, False))
CALL = rec((QUIET, 3, False), (WAKE, 2, False), (VOICE, 10, True), (QUIET, 5, False))
ANSWER = rec((QUIET, 3, False), (VOICE, 10, True), (QUIET, 5, False))
STOP_REC = rec((QUIET, 3, False), (STOP, 1, False), (QUIET, 8, False))


def test_call_woke_and_captured():
    r = replay_call(models(), CALL, QUIET_REC, V)
    assert r.woke and r.captured and r.text == "화장실로 가자"
    assert r.peak == pytest.approx(0.9) and not r.emergency


def test_call_missed_when_wake_threshold_above_score():
    r = replay_call(models(), CALL, QUIET_REC, replace(V, wake_threshold=0.95))
    assert not r.woke and not r.captured


def test_answer_yes_no():
    yes = replay_answer(models(confirm="네."), ANSWER, QUIET_REC, V, "yes")
    assert yes.accepted and yes.correct and yes.clip_rms is not None
    no = replay_answer(models(confirm="네."), ANSWER, QUIET_REC, V, "no")
    assert no.accepted and not no.correct


def test_answer_rejected_when_listen_min_above_clip():
    r = replay_answer(models(), ANSWER, QUIET_REC, replace(V, listen_min_rms=0.2), "yes")
    assert not r.accepted and r.clip_rms is not None and r.clip_rms < 0.2


def test_stop_confirmed():
    r = replay_stop(models(), STOP_REC, QUIET_REC, V)
    assert r.confirmed and r.gate and r.text == "멈춰" and r.peak == pytest.approx(0.9) and not r.woke


def test_stop_rejected_by_verify():
    r = replay_stop(models(verify="멈추라"), STOP_REC, QUIET_REC, V)
    assert not r.confirmed and r.gate and r.text == "멈추라"


def test_stop_missed_below_single_threshold():
    r = replay_stop(models(), STOP_REC, QUIET_REC, replace(V, emergency_single=0.95))
    assert not r.confirmed and not r.gate and r.peak == pytest.approx(0.9)


def test_same_recording_same_result():
    assert replay_stop(models(), STOP_REC, QUIET_REC, V) == replay_stop(models(), STOP_REC, QUIET_REC, V)
    assert replay_call(models(), CALL, QUIET_REC, V) == replay_call(models(), CALL, QUIET_REC, V)


def test_cross_trigger_seen_in_answer():
    mixed = rec((QUIET, 3, False), (STOP, 1, False), (QUIET, 8, False))
    r = replay_answer(models(verify="멈춰"), mixed, QUIET_REC, V, "yes")
    assert r.emergency


def test_without_quiet_recording():
    assert replay_stop(models(), STOP_REC, None, V).confirmed
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mic_check_replay'`

- [ ] **Step 3: 구현한다**

`src/mic_check_replay.py`:

```python
"""녹음을 ⑫ 와 같은 귀(WakewordMonitor)에 다시 흘려 채점한다 — 마이크 현장 점검기(2026-10-12).

10-06 src/venue_replay.py(feat/midm-fallback) 를 지금 귀에 맞게 옮겼다: 대화 STT·긴급 확인 STT 가 나뉘었고
(10-12), "멈춰" 한 조각 규칙이 생겼다. 녹음 하나마다 새 귀를 만든다 — 같은 녹음·같은 값이면 늘 같은 결과다.
칩 사람 말 표시(vad)는 노드처럼 청취 중일 때만 넣는다(WakewordMonitor.run 과 같은 규칙).
설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md 5절
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .handle_mode import AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply
from .mic_check_rules import AnswerTry, CallTry, MicValues, StopTry
from .wakeword_monitor import FRAME, SAMPLE_RATE, WakewordMonitor

DT = FRAME / SAMPLE_RATE      # 0.08 s
FLUSH_FRAMES = 25             # 호출 모델의 직전 2초 기억을 비우는 조용한 조각
PAD_FRAMES = 50               # 녹음 뒤 조용한 조각(4 s) — 청취 창과 멈춰 확인이 닫힐 시간


@dataclass(frozen=True)
class Recording:
    frames: tuple             # int16 1280 조각들
    vads: tuple               # 조각마다 칩 사람 말 표시(Optional[bool])

    @property
    def seconds(self) -> float:
        return len(self.frames) * DT


@dataclass(frozen=True)
class Models:
    """한 번 올린 모델을 나눠 쓴다(재채점마다 다시 올리지 않는다)."""
    predict: Callable[[np.ndarray], dict]
    transcribe: Callable[[np.ndarray], str]                            # 긴급 확인(small)
    transcribe_listen: Optional[Callable[[np.ndarray], str]] = None    # 대화(turbo, 필터판)
    transcribe_confirm: Optional[Callable[[np.ndarray], str]] = None   # 질문 창(귀띔판)
    listen_hint: Optional[str] = None

    @classmethod
    def from_loaded(cls, monitor: WakewordMonitor) -> "Models":
        """_load_real() 을 마친 귀의 모델을 꺼낸다."""
        return cls(monitor._predict, monitor._transcribe, monitor._transcribe_listen,
                   monitor._transcribe_confirm, monitor._listen_hint)


class _Run:
    """재채점 한 번의 귀와 그 결과."""

    def __init__(self, models: Models, values: MicValues):
        self.texts: list = []
        self.emergencies: list = []
        self.counting = False
        self.peak_a = 0.0
        self.peak_b = 0.0
        self._predict = models.predict
        self.monitor = WakewordMonitor(
            on_emergency=self.emergencies.append,
            on_user_text=self.texts.append,
            predict=self._tracked,
            transcribe=models.transcribe,
            transcribe_listen=models.transcribe_listen,
            transcribe_confirm=models.transcribe_confirm,
            listen_hint=models.listen_hint,
            voice_barge_in=False,
            gate_a=values.wake_threshold,
            single_b=values.emergency_single,
            listen_min_rms=values.listen_min_rms,
            short_answer_min_rms=values.short_answer_min_rms,
        )

    def _tracked(self, frame: np.ndarray) -> dict:
        scores = self._predict(frame)
        if self.counting:
            self.peak_a = max(self.peak_a, float(scores["a"]))
            self.peak_b = max(self.peak_b, float(scores["b"]))
        return scores

    def feed(self, frames: Sequence, vads: Sequence, t: float) -> tuple:
        out = []
        for frame, vad in zip(frames, vads):
            out.append(self.monitor.process_frame(
                frame, now=t, vad=(vad if self.monitor.state == "listen" else None)))
            t += DT
        return t, out


def _cycle(quiet: Optional[Recording], n: int) -> list:
    src = list(quiet.frames) if quiet is not None and quiet.frames else [np.zeros(FRAME, dtype=np.int16)]
    return [src[i % len(src)] for i in range(n)]


def _play(models: Models, values: MicValues, rec: Recording, quiet: Optional[Recording],
          followup: bool) -> tuple:
    """조용한 조각 → (질문 창 열기) → 녹음 → 조용한 조각. (_Run, 녹음 이후 결과들) 을 돌려준다."""
    run = _Run(models, values)
    t, _ = run.feed(_cycle(quiet, FLUSH_FRAMES), [False] * FLUSH_FRAMES, 0.0)
    if followup:
        run.monitor.arm_followup(now=t)          # 노드처럼: 질문을 말했다는 예약
        run.monitor.set_speaking(False, now=t)   # → 질문이 끝난 순간 질문 창이 열린다
    run.counting = True
    t, out = run.feed(list(rec.frames), list(rec.vads), t)
    run.counting = False
    _, tail = run.feed(_cycle(quiet, PAD_FRAMES), [False] * PAD_FRAMES, t)
    return run, out + tail


def is_affirmative(text: str) -> bool:
    return normalize_short_reply(text) in (AFFIRMATIVES | SOFT_AFFIRMATIVES)


def is_negative(text: str) -> bool:
    return normalize_short_reply(text) in NEGATIVES


def replay_call(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues) -> CallTry:
    """'비카야, 화장실로 가자' — 대기 상태에서 흘려 호출과 자유 창 받아쓰기를 본다."""
    run, out = _play(models, values, rec, quiet, followup=False)
    return CallTry(woke="wake" in out, captured=bool(run.texts), peak=run.peak_a,
                   text=run.texts[0] if run.texts else "", emergency=bool(run.emergencies))


def replay_answer(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues,
                  expected: str) -> AnswerTry:
    """'네'·'아니요' — 질문 창을 열고 흘려 받아들여지는지와 긍정·부정을 본다."""
    run, _ = _play(models, values, rec, quiet, followup=True)
    text = run.texts[0] if run.texts else ""
    stats = run.monitor.last_listen_stats
    correct = bool(text) and (is_affirmative(text) if expected == "yes" else is_negative(text))
    return AnswerTry(expected=expected, accepted=bool(run.texts), correct=correct,
                     clip_rms=float(stats["rms"]) if stats else None, text=text,
                     emergency=bool(run.emergencies))


def replay_stop(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues) -> StopTry:
    """'멈춰' — 관문(한 조각·두 조각) → 확인 STT → 정확 매칭까지 ⑫ 와 같게."""
    run, out = _play(models, values, rec, quiet, followup=False)
    check = run.monitor.last_emergency_check
    return StopTry(confirmed=bool(run.emergencies), gate=check is not None, peak=run.peak_b,
                   text=(check or {}).get("text", "") or "", woke="wake" in out)
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_replay.py -q`
Expected: PASS

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add src/mic_check_replay.py tests/test_mic_check_replay.py
git commit -m "feat(voice): 마이크 점검기 녹음 재채점 — 10-06 venue_replay 를 지금 귀(확인 STT 분리·한 조각 규칙)에 맞게, 멈춰 확인 추가

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 휴대폰 중계 (`tools/mic_check_web.py`)

**Files:**
- Create: `tools/mic_check_web.py`
- Test: `tests/test_mic_check_web.py`

**Interfaces:**
- Produces:
  - `DEFAULT_PORT = 8770`
  - `LiveServer(page_html: str, host="0.0.0.0", port=DEFAULT_PORT)`; `.start(timeout=5.0)`, `.stop()`, `.port: int`(실제 포트), `.commands: queue.Queue[dict]`, `.publish(message: dict, keep: Optional[str] = None)`
  - 페이지는 `/`, 실시간 통로는 `/ws`. 페이지 → 도구 메시지는 `{"cmd": str, ...}` 만 받는다.

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_check_web.py`:

```python
"""휴대폰 중계 — 페이지, 다시 붙을 때 마지막 상태, 명령 줄 세우기(설계 4·9절)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from websockets.sync.client import connect

from tools.mic_check_web import LiveServer


@pytest.fixture
def server():
    s = LiveServer("<html><body>점검</body></html>", host="127.0.0.1", port=0)
    s.start()
    yield s
    s.stop()


def test_page_and_404(server):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as r:
        assert r.status == 200 and "text/html" in r.headers["Content-Type"]
        assert "점검" in r.read().decode("utf-8")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/nope")
    assert err.value.code == 404


def test_kept_message_sent_on_connect_and_commands_queued(server):
    server.publish({"type": "status", "n": 1}, keep="status")
    server.publish({"type": "status", "n": 2}, keep="status")     # 같은 이름은 마지막 것만
    with connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
        assert json.loads(ws.recv(timeout=2)) == {"type": "status", "n": 2}
        ws.send(json.dumps({"cmd": "record"}))
        ws.send("이상한 글자")               # 무시
        ws.send(json.dumps({"no": "cmd"}))   # 무시
        ws.send(json.dumps({"cmd": "rejudge"}))
        assert server.commands.get(timeout=2) == {"cmd": "record"}
        assert server.commands.get(timeout=2) == {"cmd": "rejudge"}


def test_publish_reaches_connected_client(server):
    with connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
        ws.send(json.dumps({"cmd": "ping"}))                     # 붙은 것을 확인하는 왕복
        assert server.commands.get(timeout=2) == {"cmd": "ping"}
        server.publish({"type": "frame", "rms": 0.01})
        assert json.loads(ws.recv(timeout=2)) == {"type": "frame", "rms": 0.01}
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_web.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.mic_check_web'`

- [ ] **Step 3: 구현한다**

`tools/mic_check_web.py`:

```python
"""마이크 점검기 휴대폰 중계 — 페이지 하나와 실시간 통로 하나(websockets 15, 새 의존성 없음).

같은 와이파이의 휴대폰이 http://<젯슨>:8770/ 을 열면 페이지를 받고, 페이지가 /ws 로 붙어 실시간 값을 받고
버튼 명령을 보낸다. 점검기가 도는 동안에만 열린다. keep 이름을 붙인 메시지(상태·결과)는 기억해 두었다가
새로 붙는 휴대폰에 먼저 보낸다 — 화면이 꺼졌다 켜져도 빈 화면이 되지 않게. 설계 4·9절.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Optional

from websockets.asyncio.server import broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

DEFAULT_PORT = 8770


class LiveServer:
    def __init__(self, page_html: str, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self._page = page_html.encode("utf-8")
        self._host, self._want_port = host, port
        self.port: Optional[int] = None
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self._clients: set = set()
        self._kept: dict = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Future] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="mic-check-web", daemon=True)

    def start(self, timeout: float = 5.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("휴대폰 중계를 열지 못했습니다(시간 초과)")
        if self._error is not None:
            raise RuntimeError(f"휴대폰 중계를 열지 못했습니다: {self._error}")

    def stop(self) -> None:
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(lambda: stop.done() or stop.set_result(None))
        self._thread.join(timeout=5.0)

    def publish(self, message: dict, keep: Optional[str] = None) -> None:
        """어느 스레드에서든 부른다. keep 이름을 주면 새로 붙는 휴대폰에도 이 메시지를 먼저 보낸다."""
        text = json.dumps(message, ensure_ascii=False)
        if keep:
            with self._lock:
                self._kept[keep] = text
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._send_all, text)

    def _send_all(self, text: str) -> None:
        broadcast(self._clients, text)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 — 시작 실패(포트 사용 중 등)를 start() 로 넘긴다
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = self._loop.create_future()
        async with serve(self._handler, self._host, self._want_port,
                         process_request=self._process_request) as server:
            self.port = list(server.sockets)[0].getsockname()[1]
            self._ready.set()
            await self._stop

    def _process_request(self, connection, request):
        if request.path == "/ws":
            return None
        if request.path in ("/", "/index.html"):
            headers = Headers([("Content-Type", "text/html; charset=utf-8"),
                               ("Content-Length", str(len(self._page))),
                               ("Cache-Control", "no-store")])
            return Response(200, "OK", headers, self._page)
        return connection.respond(404, "없는 주소입니다\n")

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            with self._lock:
                kept = list(self._kept.values())
            for text in kept:
                await ws.send(text)
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(cmd, dict) and isinstance(cmd.get("cmd"), str):
                    self.commands.put(cmd)
        finally:
            self._clients.discard(ws)
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_web.py -q`
Expected: PASS

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add tools/mic_check_web.py tests/test_mic_check_web.py
git commit -m "feat(voice): 마이크 점검기 휴대폰 중계 — 페이지·실시간 통로 하나(websockets 15), 다시 붙으면 마지막 상태부터

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 점검 진행 (`src/mic_check_session.py`)

**Files:**
- Create: `src/mic_check_session.py`
- Test: `tests/test_mic_check_session.py`

**Interfaces:**
- Consumes: Task 1~4 전부(`WakewordMonitor` 새 인자·`set_thresholds`·`state`, `mic_check_rules.*`, `env_file.*`, `mic_check_replay.*`)
- Produces:
  - `Clip(name: str, text: str, pcm: np.ndarray, rate: int)`
  - `Hardware(chip_apply: Callable[[MicValues], list], read_vad: Callable[[], Optional[bool]], play: Callable[[np.ndarray, int], None], sleep=time.sleep, spawn=<스레드로 실행>)`
  - `Session(*, models, hw, publish: Callable[[dict, Optional[str]], None], env_values, env_path, place, record_dir, backup_dir, clips=(), previous_robot_level=None, chip_problems=(), clock=datetime.now)`
  - `session.on_frame(frame)`, `session.handle(cmd: dict)`, `session.close()`, `session.done: bool`, `session.values`, `session.env_values`, `session.meas`, `session.live`, `session.step`
  - 데모용 읽기 속성: `recording_kind: Optional[str]`, `recording_pos: int`, `recording_expected: Optional[str]`, `robot_running: bool`
  - `previous_robot_level(root: Path, exclude: Optional[Path] = None) -> Optional[float]`
  - 메시지(→ 페이지): `status`(keep), `results`(keep), `frame`, `event`, `toast`, `save_preview`, `busy`(`{"text": "채점 중"|""}`). 명령(← 페이지): `record`, `robot`, `redo`(+`step`), `set`(+`key`,`value`), `rejudge`, `save_preview`, `save`, `restore`, `finish`, `ping`

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_check_session.py`:

```python
"""점검 진행 — 가짜 마이크·칩·스피커로 단계 진행, 두 번 누르기, 값 바꾸기, 다시 채점, 저장·되돌리기(설계 3·5·7절)."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from src.env_file import read_env, values_from
from src.mic_check_replay import Models
from src.mic_check_rules import PASS, REDO
from src.mic_check_session import Clip, Hardware, Session, previous_robot_level

NOISE = np.full(1280, 100, dtype=np.int16)    # rms ≈ 0.003
WAKE = np.full(1280, 1111, dtype=np.int16)
STOP = np.full(1280, 2222, dtype=np.int16)
STOP44 = np.full(1280, 2244, dtype=np.int16)  # 멈춰 점수 0.44 — 0.5 아래
YES = np.full(1280, 3000, dtype=np.int16)
NO = np.full(1280, 3001, dtype=np.int16)

SCORES = {1111: (0.9, 0.0), 2222: (0.0, 0.9), 2244: (0.0, 0.44)}


def predict(frame):
    a, b = SCORES.get(int(frame[0]), (0.0, 0.0))
    return {"a": a, "b": b}


MODELS = Models(predict=predict, transcribe=lambda a: "멈춰", transcribe_listen=lambda a: "화장실로 가자",
                transcribe_confirm=lambda a: "아니요" if np.any(a == 3001) else "네")


class FakeHw:
    def __init__(self):
        self.applied, self.vad, self.session = [], False, None

    def chip_apply(self, v):
        self.applied.append(v)
        return []

    def read_vad(self):
        return self.vad

    def play(self, pcm, rate):
        for _ in range(5):                    # 재생하는 동안 마이크 조각이 들어온다
            self.session.on_frame(NOISE)


def make(tmp_path, prev=None):
    msgs = []
    hw = FakeHw()
    env = tmp_path / ".env"
    env.write_text("VICA_MIC_AGC_DESIRED=0.007\nVICA_MIC_GAMMA_ETAIL=1.5\nVICA_MIC_NLATTEN=1\n", encoding="utf-8")
    s = Session(models=MODELS,
                hw=Hardware(hw.chip_apply, hw.read_vad, hw.play, sleep=lambda sec: None, spawn=lambda fn: fn()),
                publish=lambda m, keep=None: msgs.append(m),
                env_values=values_from(read_env(env)), env_path=env, place="연구실",
                record_dir=tmp_path / "rec", backup_dir=tmp_path / "bk",
                clips=[Clip("a.wav", "안녕하세요", np.zeros(1600, np.float32), 16000)],
                previous_robot_level=prev, clock=lambda: datetime(2026, 10, 27, 14, 32, 5))
    hw.session = s
    return s, hw, msgs


def feed(s, hw, frame, n, vad=False):
    hw.vad = vad
    for _ in range(n):
        s.on_frame(frame)


def record(s, hw, parts, total=50):
    s.handle({"cmd": "record"})
    used = 0
    for frame, n, vad in parts:
        feed(s, hw, frame, n, vad)
        used += n
    feed(s, hw, NOISE, total - used)


def last(msgs, kind):
    return [m for m in msgs if m["type"] == kind][-1]


def run_all(s, hw, stop_frames=(STOP,) * 5):
    record(s, hw, [], total=188)                                              # 2단계 조용히 15 s
    for _ in range(3):
        record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    for frame in (YES, YES, NO, NO):
        record(s, hw, [(NOISE, 3, False), (frame, 10, True)])
    for frame in stop_frames:
        record(s, hw, [(NOISE, 3, False), (frame, 1, False)])
    s.handle({"cmd": "robot"})
    feed(s, hw, NOISE, 1)                                                     # 재생 끝을 받아 마무리


def test_full_run_passes_and_records(tmp_path):
    s, hw, msgs = make(tmp_path)
    run_all(s, hw)
    assert s.step.key == "result"
    res = last(msgs, "results")
    assert res["overall"] == PASS, res["rows"]
    assert len(s.meas.noise) == 3 and s.meas.robot.self_reactions == 0
    data = json.loads((tmp_path / "rec" / "result.json").read_text(encoding="utf-8"))
    assert data["verdict"]["overall"] == PASS and data["place"] == "연구실"
    assert (tmp_path / "rec" / "quiet.wav").exists() and (tmp_path / "rec" / "stop_5.wav").exists()


def test_double_record_is_ignored(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "record"})
    s.handle({"cmd": "record"})
    assert last(msgs, "toast")["text"].startswith("지금은")
    feed(s, hw, NOISE, 188)
    assert len(s.meas.noise) == 3 and s.step.key == "call"


def test_set_value_out_of_range_ignored(tmp_path):
    s, hw, msgs = make(tmp_path)
    before = s.values
    for cmd in ({"cmd": "set", "key": "wake_threshold", "value": 0.95},
                {"cmd": "set", "key": "listen_min_rms", "value": "abc"},
                {"cmd": "set", "key": "모르는값", "value": 1},
                {"cmd": "set", "key": "agc", "value": 0.02},
                {"cmd": "set", "key": "echo", "value": "엄청 셈"}):
        s.handle(cmd)
        assert s.values == before
    assert len([m for m in msgs if m["type"] == "toast"]) == 5
    assert hw.applied == []


def test_set_value_updates_live_and_chip(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.653})
    assert s.values.wake_threshold == 0.65 and s.live.gate_a.threshold == 0.65    # 눈금(0.01)에 맞춤
    assert hw.applied == []
    s.handle({"cmd": "set", "key": "agc", "value": 0.010})
    assert hw.applied[-1].agc == 0.010
    s.handle({"cmd": "set", "key": "echo", "value": "mid"})
    assert (hw.applied[-1].echo_etail, hw.applied[-1].echo_nlatten) == (1.0, 1)
    assert last(msgs, "status")["values"]["echo"] == "mid"


def test_rejudge_with_lower_stop_threshold(tmp_path):
    s, hw, msgs = make(tmp_path)
    run_all(s, hw, stop_frames=(STOP,) * 4 + (STOP44,))
    stop_row = next(r for r in last(msgs, "results")["rows"] if r["key"] == "stop")
    assert stop_row["grade"] == "warn" and "점수 0.44" in stop_row["detail"]
    s.handle({"cmd": "set", "key": "emergency_single", "value": 0.43})
    assert last(msgs, "results")["stale"] is True
    s.handle({"cmd": "rejudge"})
    res = last(msgs, "results")
    rows = {r["key"]: r for r in res["rows"]}
    assert rows["stop"]["grade"] == PASS
    assert rows["robot"]["grade"] == REDO                     # 문턱이 바뀌어 6단계는 다시
    assert res["stale"] is False
    first = {r["key"]: r for r in res["first"]}
    assert first["stop"]["grade"] == "warn"


def test_agc_change_then_record_restarts_that_step(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    assert len(s.meas.calls) == 1
    s.handle({"cmd": "set", "key": "agc", "value": 0.010})
    rows = {r["key"]: r["grade"] for r in last(msgs, "results")["rows"]}
    assert rows["noise"] == REDO and rows["call"] == REDO
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    assert len(s.meas.calls) == 1 and "처음부터" in [m for m in msgs if m["type"] == "toast"][-1]["text"]


def test_redo_clears_step(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    s.handle({"cmd": "redo", "step": "quiet"})
    assert s.meas.noise == [] and s.step.key == "quiet" and len(s.meas.calls) == 1


def test_rejudge_without_recordings(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "rejudge"})
    assert "녹음이 없습니다" in last(msgs, "toast")["text"]


def test_save_and_restore(tmp_path):
    s, hw, msgs = make(tmp_path)
    original = (tmp_path / ".env").read_text(encoding="utf-8")
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.65})
    s.handle({"cmd": "save_preview"})
    assert last(msgs, "save_preview")["lines"] == [
        "VICA_WAKE_THRESHOLD=0.65  # 10-27 연구실 점검에서 바꿈 (전 0.70)"]
    s.handle({"cmd": "save"})
    assert "VICA_WAKE_THRESHOLD=0.65" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert s.env_values.wake_threshold == 0.65 and (tmp_path / "bk").is_dir()
    s.handle({"cmd": "restore"})
    assert (tmp_path / ".env").read_text(encoding="utf-8") == original
    assert s.values.wake_threshold == 0.70 and s.env_values.wake_threshold == 0.70
    assert s.live.gate_a.threshold == 0.70


def test_save_blocked_by_rule(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "listen_min_rms", "value": 0.014})     # 0.02 < 0.014 × 1.5
    s.handle({"cmd": "save_preview"})
    assert "1.5배" in last(msgs, "toast")["text"]
    assert not [m for m in msgs if m["type"] == "save_preview"]


def test_previous_robot_level(tmp_path):
    root = tmp_path / "mic_checks"
    for name, level in (("20261020-1000_연구실", 0.020), ("20261026-0900_연구실", 0.029)):
        d = root / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"measurements": {"robot": {"level": level}}}), encoding="utf-8")
    now_dir = root / "20261027-1430_시연장"
    now_dir.mkdir()
    (root / "env_backups").mkdir()
    assert previous_robot_level(root, exclude=now_dir) == 0.029
    assert previous_robot_level(tmp_path / "없음") is None


def test_finish(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "finish"})
    assert s.done
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mic_check_session'`

- [ ] **Step 3: 구현한다**

`src/mic_check_session.py`:

```python
"""마이크 현장 점검기의 진행 — 단계·녹음·실시간 귀·재채점·값 고치기·저장·기록. 하드웨어와 웹은 밖에서 넣는다.

도구(tools/mic_venue_check.py)가 마이크·칩·스피커·휴대폰 중계를 붙이고, 시험은 가짜를 붙인다. 메서드는 모두
한 스레드(처리 스레드)에서 부른다. 예외는 6단계 재생 루틴 하나로, 그것만 다른 스레드에서 귀의
note_robot_speech·set_speaking 을 부른다(⑫ 노드도 ROS 스레드가 같은 일을 한다). 시각은 벽시계가 아니라
조각 시계(0.08 s 씩)다 — 재채점과 실시간이 같은 시간 규칙을 쓰게.
설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md
"""
from __future__ import annotations

import json
import threading
import time
import wave
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .env_file import (KEYS, EnvEditError, change_line, plan_changes, read_env, restore_latest, values_from,
                       write_changes)
from .mic_check_replay import DT, Models, Recording, replay_answer, replay_call, replay_stop
from .mic_check_rules import (AGC_CHOICES, ANSWER_REPS, CALL_REPS, ECHO_CHOICES, ECHO_LABELS, NAMES, RANGES,
                              STOP_REPS, Measurements, MicValues, RobotRun, check_values, echo_name, judge,
                              noise_floor)
from .wakeword_monitor import SAMPLE_RATE, WakewordMonitor

NOISE_WINDOW = 125         # 실시간 잡음 바닥 = 최근 10 s
QUIET_SEGMENTS = 3         # 2단계 15 s 를 5 s 씩 세 번 따로 잰다
TTS_TAIL_SEC = 0.4         # src/ros_tts_node.py TAIL_SEC 와 같다 — 끝난 뒤에도 0.4 s 더 '말하는 중'
AFTER_LINE_SEC = 1.0       # 문장 사이
ROBOT_LEAD_SEC = 1.0       # 잡음 바닥에서 시작
ROBOT_SETTLE_SEC = 2.0     # 마지막 멘트 뒤 긴급 확인이 끝날 시간
STEP_NO_OFFSET = 2         # 화면 단계 번호: 1 = 준비(도구), quiet = 2 … result = 7
TOTAL_STEPS = 7
REC_KINDS = ("quiet", "call", "answer", "stop")
ROW_OF = {"quiet": "noise", "call": "call", "answer": "answer", "stop": "stop"}
ANSWER_EXPECTED = ("yes", "yes", "no", "no")


@dataclass(frozen=True)
class Step:
    key: str
    name: str
    prompts: tuple           # 차례마다 화면 큰 글자
    seconds: float           # 녹음 길이(녹음 없는 단계는 0)


STEPS = (
    Step("quiet", "조용히", ("15초 동안 말하지 마세요",), 15.0),
    Step("call", "호출", ("“비카야, 화장실로 가자”",) * CALL_REPS, 4.0),
    Step("answer", "짧은 답", ("“네”", "“네”", "“아니요”", "“아니요”"), 4.0),
    Step("stop", "멈춰", ("“멈춰”",) * STOP_REPS, 4.0),
    Step("robot", "로봇 혼자 말하기", ("30초 동안 조용히 계세요",), 0.0),
    Step("result", "결과표", ("점검이 끝났습니다",), 0.0),
)
NEEDS = {"quiet": 1, "call": CALL_REPS, "answer": ANSWER_REPS, "stop": STOP_REPS, "robot": 1}


@dataclass(frozen=True)
class Clip:
    name: str
    text: str
    pcm: np.ndarray
    rate: int


def _spawn_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="mic-check-robot", daemon=True).start()


@dataclass
class Hardware:
    chip_apply: Callable[[MicValues], list]       # 칩에 쓰고 문제 목록(빈 목록 = 성공)
    read_vad: Callable[[], Optional[bool]]        # 칩 사람 말 표시(SPEECHDETECTED)
    play: Callable[[np.ndarray, int], None]       # 6단계 멘트 재생(끝날 때까지 막힘)
    sleep: Callable[[float], None] = time.sleep
    spawn: Callable[[Callable[[], None]], None] = _spawn_thread


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))


def _noise_segments(rec: Recording) -> list:
    rms = [_rms(f) for f in rec.frames]
    k = max(1, len(rms) // QUIET_SEGMENTS)
    return [noise_floor(rms[i * k:(i + 1) * k]) for i in range(QUIET_SEGMENTS) if rms[i * k:(i + 1) * k]]


def _values_dict(v: MicValues) -> dict:
    return {"listen_min_rms": v.listen_min_rms, "short_answer_min_rms": v.short_answer_min_rms,
            "wake_threshold": v.wake_threshold, "emergency_single": v.emergency_single,
            "agc": v.agc, "echo": echo_name(v.echo_etail, v.echo_nlatten) or "start"}


def previous_robot_level(root: Path, exclude: Optional[Path] = None) -> Optional[float]:
    """가장 최근 점검 기록(result.json)의 로봇 목소리 크기. 없으면 None."""
    root = Path(root)
    if not root.is_dir():
        return None
    for d in sorted((p for p in root.iterdir() if p.is_dir() and p != exclude), reverse=True):
        try:
            data = json.loads((d / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        level = ((data.get("measurements") or {}).get("robot") or {}).get("level")
        if isinstance(level, (int, float)) and level > 0:
            return float(level)
    return None


class _Recorder:
    def __init__(self, kind: str, need: int, expected: Optional[str]):
        self.kind, self.need, self.expected = kind, need, expected
        self.frames: list = []
        self.vads: list = []

    def add(self, frame: np.ndarray, vad: Optional[bool]) -> None:
        self.frames.append(frame)
        self.vads.append(vad)

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.need

    def left(self) -> float:
        return max(0.0, (self.need - len(self.frames)) * DT)

    def recording(self) -> Recording:
        return Recording(tuple(self.frames), tuple(self.vads))


class Session:
    def __init__(self, *, models: Models, hw: Hardware, publish: Callable[..., None], env_values: MicValues,
                 env_path: Path, place: str, record_dir: Path, backup_dir: Path, clips: Sequence[Clip] = (),
                 previous_robot_level: Optional[float] = None, chip_problems: Sequence[str] = (),
                 clock: Callable[[], datetime] = datetime.now):
        self.models, self.hw, self._publish_fn = models, hw, publish
        self.env_path, self.place = Path(env_path), place
        self.record_dir, self.backup_dir = Path(record_dir), Path(backup_dir)
        self.clips = list(clips)
        self.previous_robot_level = previous_robot_level
        self.chip_problems = list(chip_problems)
        self.clock = clock
        self.started = clock()
        self.env_values = env_values      # 지금 .env 가 말하는 값
        self.start_values = env_values    # 점검을 시작할 때 값("처음 값" 비교)
        self.values = env_values          # 화면에서 고치는 값
        self.judged_values = env_values   # 결과표를 낸 값(문턱 넷) — 다시 채점하면 values 로 맞춘다
        self.done = False
        self.step_index = 0
        self.meas = Measurements()
        self._chips: dict = self.meas.chips
        self._quiet: Optional[Recording] = None
        self._calls: list = []
        self._answers: list = []           # (Recording, expected)
        self._stops: list = []
        self._first: Optional[Measurements] = None
        self._rec: Optional[_Recorder] = None
        self._t = 0.0
        self._frame_vad: Optional[bool] = None
        self._noise_window: deque = deque(maxlen=NOISE_WINDOW)
        self._busy_text = ""
        self._robot_running = False
        self._robot_done = threading.Event()
        self._robot_rms: list = []
        self._robot_frames: list = []
        self._self_reactions = 0
        self._robot_thresholds: Optional[tuple] = None
        self._robot_error = ""
        self._files: dict = {}
        self._history: list = []
        self._saves: list = []
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.live = WakewordMonitor(
            on_emergency=self._live_emergency,
            on_user_text=self._live_text,
            on_reject=self._live_reject,
            on_listen_state=self._live_listen_state,
            on_wake_score=self._live_wake_score,
            on_emergency_score=self._live_emergency_near,
            on_frame=self._live_frame,
            predict=models.predict, transcribe=models.transcribe,
            transcribe_listen=models.transcribe_listen, transcribe_confirm=models.transcribe_confirm,
            listen_hint=models.listen_hint, voice_barge_in=False,
            gate_a=env_values.wake_threshold, single_b=env_values.emergency_single,
            listen_min_rms=env_values.listen_min_rms, short_answer_min_rms=env_values.short_answer_min_rms,
        )
        self._publish_status()
        self._publish_results()

    # ------------------------------------------------------------------ 읽기 속성
    @property
    def step(self) -> Step:
        return STEPS[self.step_index]

    @property
    def recording_kind(self) -> Optional[str]:
        return None if self._rec is None else self._rec.kind

    @property
    def recording_pos(self) -> int:
        return 0 if self._rec is None else len(self._rec.frames)

    @property
    def recording_expected(self) -> Optional[str]:
        return None if self._rec is None else self._rec.expected

    @property
    def robot_running(self) -> bool:
        return self._robot_running

    # ------------------------------------------------------------------ 조각
    def on_frame(self, frame: np.ndarray) -> None:
        self._t += DT
        recording = self._rec is not None
        listening = self.live.state == "listen"
        vad = self.hw.read_vad() if (recording or listening) else None
        self._frame_vad = vad
        self.live.process_frame(frame, now=self._t, vad=(vad if listening else None))
        rms = _rms(frame)
        if self._robot_running:
            self._robot_rms.append(rms)
            self._robot_frames.append(frame)
        elif not recording:
            self._noise_window.append(rms)
        if recording:
            self._rec.add(frame, vad)
            if self._rec.full:
                self._finish_recording()
        if self._robot_done.is_set():
            self._robot_done.clear()
            self._finish_robot()

    # ------------------------------------------------------------------ 명령
    def handle(self, cmd: dict) -> None:
        name = cmd.get("cmd")
        if name == "finish":
            self.done = True
            return
        if name == "ping":
            return
        actions = {
            "record": self._start_recording, "robot": self._start_robot,
            "redo": lambda: self._redo(cmd.get("step")),
            "set": lambda: self._set_value(cmd.get("key"), cmd.get("value")),
            "rejudge": self._rejudge, "save_preview": self._save_preview, "save": self._save,
            "restore": self._restore,
        }
        action = actions.get(name)
        if action is None:
            return
        if self._busy():
            what = "로봇이 말하는" if self._robot_running else "녹음"
            self._toast(f"지금은 {what} 중입니다. 끝난 뒤 다시 누르세요.")
            return
        action()

    def close(self) -> None:
        self._write_record()

    # ------------------------------------------------------------------ 녹음
    def _busy(self) -> bool:
        return self._rec is not None or self._robot_running

    def _attempts(self, key: str) -> int:
        return {"quiet": int(self._quiet is not None), "call": len(self._calls), "answer": len(self._answers),
                "stop": len(self._stops), "robot": int(self.meas.robot is not None)}.get(key, 0)

    def _start_recording(self) -> None:
        kind = self.step.key
        if kind not in REC_KINDS:
            self._toast("이 단계에는 녹음이 없습니다.")
            return
        row = ROW_OF[kind]
        if row in self._chips and self._chips[row][0] != self.values.chip()[0]:
            self._clear(kind)
            self._toast("AGC 가 바뀌어 이 단계를 처음부터 다시 녹음합니다.")
        expected = ANSWER_EXPECTED[len(self._answers)] if kind == "answer" else None
        if kind == "answer":
            # 실시간 화면에서도 질문 창으로 받는지 보이게 — 노드처럼 질문 예약 → 말 끝
            self.live.arm_followup(now=self._t)
            self.live.set_speaking(False, now=self._t)
        # "멈춰" 사이 2 s(⑫ 쿨다운)는 4 s 녹음이라 늘 지켜진다 — 따로 막지 않는다.
        self._rec = _Recorder(kind, int(round(self.step.seconds / DT)), expected)
        self._publish_status()

    def _finish_recording(self) -> None:
        rec_obj, self._rec = self._rec, None
        kind, rec = rec_obj.kind, rec_obj.recording()
        self._busy_note("채점 중")
        chip = self.values.chip()
        if kind == "quiet":
            self._quiet = rec
            self.meas.noise = _noise_segments(rec)
        elif kind == "call":
            self._calls.append(rec)
            self.meas.calls.append(replay_call(self.models, rec, self._quiet, self.judged_values))
        elif kind == "answer":
            self._answers.append((rec, rec_obj.expected))
            self.meas.answers.append(
                replay_answer(self.models, rec, self._quiet, self.judged_values, rec_obj.expected))
        else:
            self._stops.append(rec)
            self.meas.stops.append(replay_stop(self.models, rec, self._quiet, self.judged_values))
        self._chips.setdefault(ROW_OF[kind], chip)
        self._first = None
        self._save_wav(kind, rec, rec_obj.expected)
        self._busy_note("")
        self._advance()
        self._publish_status()
        self._publish_results()

    def _clear(self, kind: str) -> None:
        if kind == "quiet":
            self._quiet, self.meas.noise = None, []
        elif kind == "call":
            self._calls, self.meas.calls = [], []
        elif kind == "answer":
            self._answers, self.meas.answers = [], []
        elif kind == "stop":
            self._stops, self.meas.stops = [], []
        elif kind == "robot":
            self.meas.robot, self.meas.robot_thresholds = None, None
        self._chips.pop("noise" if kind == "quiet" else kind, None)
        self._first = None

    def _redo(self, key: Optional[str]) -> None:
        keys = [s.key for s in STEPS]
        if key not in NEEDS:
            self._toast("다시 할 단계를 고르세요.")
            return
        self._clear(key)
        self.step_index = keys.index(key)
        self._publish_status()
        self._publish_results()

    def _advance(self) -> None:
        while self.step.key in NEEDS and self._attempts(self.step.key) >= NEEDS[self.step.key]:
            self.step_index += 1

    # ------------------------------------------------------------------ 6단계
    def _start_robot(self) -> None:
        if self.step.key != "robot":
            self._toast("로봇 혼자 말하기 단계가 아닙니다.")
            return
        if not self.clips:
            self._toast("틀 멘트가 없습니다 — assets/baked 를 확인하세요.")
            return
        self._robot_running = True
        self._robot_rms, self._robot_frames, self._self_reactions = [], [], 0
        self._robot_thresholds = self.values.thresholds()
        self._robot_error = ""
        self._publish_status()
        self.hw.spawn(self._robot_routine)

    def _robot_routine(self) -> None:
        """⑫ 가 로봇이 말할 때 하는 그대로: 문장 소식(자기 말 무시·마이크 닫기 규칙) → 말하는 중 → 재생 → 꼬리 → 끝."""
        try:
            self.hw.sleep(ROBOT_LEAD_SEC)
            for clip in self.clips:
                self.live.note_robot_speech(clip.text, now=self._t)
                self.live.set_speaking(True, now=self._t)
                self.hw.play(clip.pcm, clip.rate)
                self.hw.sleep(TTS_TAIL_SEC)
                self.live.set_speaking(False, now=self._t)
                self.hw.sleep(AFTER_LINE_SEC)
            self.hw.sleep(ROBOT_SETTLE_SEC)
        except Exception as exc:  # noqa: BLE001 — 재생 실패를 화면으로 알린다
            self._robot_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._robot_done.set()

    def _finish_robot(self) -> None:
        self._robot_running = False
        level = float(np.mean(self._robot_rms)) if self._robot_rms else 0.0
        self.meas.robot = RobotRun(self._self_reactions, level)
        self.meas.robot_thresholds = self._robot_thresholds
        self._chips["robot"] = self.values.chip()
        self._save_wav("robot", Recording(tuple(self._robot_frames), ()), None)
        if self._robot_error:
            self._toast(f"재생 중 문제가 있었습니다: {self._robot_error}")
        self._advance()
        self._publish_status()
        self._publish_results()

    # ------------------------------------------------------------------ 값 고치기
    def _with_chip(self, v: MicValues) -> MicValues:
        """칩 값은 다시 채점할 수 없다(다시 녹음해야 한다) — 판정에는 늘 지금 칩 값을 쓴다."""
        return replace(v, agc=self.values.agc, echo_etail=self.values.echo_etail,
                       echo_nlatten=self.values.echo_nlatten)

    def _new_values(self, key, value) -> MicValues:
        if key in RANGES:
            lo, hi, step = RANGES[key]
            try:
                x = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{NAMES[key]} 값이 숫자가 아닙니다.") from None
            if not lo - 1e-9 <= x <= hi + 1e-9:
                raise ValueError(f"{NAMES[key]} 은(는) {lo:g} ~ {hi:g} 사이여야 합니다.")
            x = round(round((x - lo) / step) * step + lo, 6)
            return replace(self.values, **{key: x})
        if key == "agc":
            allowed = set(AGC_CHOICES) | {self.start_values.agc}
            try:
                x = float(value)
            except (TypeError, ValueError):
                raise ValueError("AGC 목표 값이 숫자가 아닙니다.") from None
            if not any(abs(x - a) < 1e-9 for a in allowed):
                raise ValueError("AGC 목표는 " + " · ".join(f"{a:.3f}" for a in sorted(allowed)) + " 중에서 고르세요.")
            return replace(self.values, agc=x)
        if key == "echo":
            choices = self._echo_choices()
            if value not in choices:
                raise ValueError("에코 지우기 단계를 고르세요.")
            etail, nlatten = choices[value]
            return replace(self.values, echo_etail=etail, echo_nlatten=nlatten)
        raise ValueError("모르는 값입니다.")

    def _echo_choices(self) -> dict:
        choices = dict(ECHO_CHOICES)
        if echo_name(self.start_values.echo_etail, self.start_values.echo_nlatten) is None:
            choices["start"] = (self.start_values.echo_etail, self.start_values.echo_nlatten)
        return choices

    def _set_value(self, key, value) -> None:
        try:
            new = self._new_values(key, value)
        except ValueError as exc:
            self._toast(str(exc))
            return
        old = self.values
        if new.chip() != old.chip():
            problems = self.hw.chip_apply(new)
            if problems:
                self.hw.chip_apply(old)
                self._toast("칩에 쓰지 못해 그대로 둡니다: " + "; ".join(problems))
                return
        self.values = new
        self._apply_thresholds(new)
        self._publish_status()
        self._publish_results()

    def _apply_thresholds(self, v: MicValues) -> None:
        self.live.set_thresholds(wake=v.wake_threshold, emergency_single=v.emergency_single,
                                 listen_min_rms=v.listen_min_rms, short_answer_min_rms=v.short_answer_min_rms)

    def _measure_all(self, values: MicValues) -> Measurements:
        m = Measurements(noise=list(self.meas.noise), chips=self._chips,
                         robot=self.meas.robot, robot_thresholds=self.meas.robot_thresholds)
        m.calls = [replay_call(self.models, r, self._quiet, values) for r in self._calls]
        m.answers = [replay_answer(self.models, r, self._quiet, values, e) for r, e in self._answers]
        m.stops = [replay_stop(self.models, r, self._quiet, values) for r in self._stops]
        return m

    def _rejudge(self) -> None:
        n = len(self._calls) + len(self._answers) + len(self._stops)
        if not n:
            self._toast("다시 채점할 녹음이 없습니다.")
            return
        self._busy_note("다시 채점 중")
        self.judged_values = self.values
        self.meas = self._measure_all(self.values)
        if self.values != self.start_values and self._first is None:
            self._first = self._measure_all(self.start_values)
        self._busy_note("")
        self._history.append({"time": self.clock().isoformat(timespec="seconds"),
                              "values": asdict(self.values), "overall": self._verdict().overall})
        self._toast(f"녹음 {n}개를 새 값으로 다시 채점했습니다.")
        self._publish_status()
        self._publish_results()

    # ------------------------------------------------------------------ 저장·되돌리기
    def _note(self) -> str:
        return f"{self.clock():%m-%d} {self.place} 점검에서 바꿈"

    def _save_preview(self) -> None:
        problems = check_values(self.values)
        if problems:
            self._toast(" / ".join(problems))
            return
        changes = plan_changes(self.env_values, self.values)
        if not changes:
            self._toast(".env 와 다른 값이 없습니다.")
            return
        note = self._note()
        self._publish({"type": "save_preview", "lines": [change_line(c, note) for c in changes],
                       "backup_dir": str(self.backup_dir)})

    def _save(self) -> None:
        problems = check_values(self.values)
        if problems:
            self._toast(" / ".join(problems))
            return
        changes = plan_changes(self.env_values, self.values)
        try:
            backup = write_changes(self.env_path, changes, self._note(), self.backup_dir, self.clock())
        except EnvEditError as exc:
            self._toast(str(exc))
            return
        self.env_values = self.values
        self._saves.append({"time": self.clock().isoformat(timespec="seconds"), "backup": str(backup),
                            "lines": [change_line(c, self._note()) for c in changes]})
        self._toast("저장했습니다. ⑫를 다시 켜야 적용됩니다.")
        self._publish_status()
        self._write_record()

    def _restore(self) -> None:
        try:
            src = restore_latest(self.env_path, self.backup_dir, self.clock())
            restored = values_from(read_env(self.env_path))
        except EnvEditError as exc:
            self._toast(str(exc))
            return
        if restored.chip() != self.values.chip():
            problems = self.hw.chip_apply(restored)
            if problems:
                self._toast("칩에 쓰지 못했습니다: " + "; ".join(problems))
        self.env_values = self.values = restored
        self._apply_thresholds(restored)
        self._saves.append({"time": self.clock().isoformat(timespec="seconds"), "restored_from": str(src)})
        self._toast(f"{src.name} 으로 되돌렸습니다. ⑫를 다시 켜야 적용됩니다.")
        self._publish_status()
        self._publish_results()
        self._write_record()

    # ------------------------------------------------------------------ 실시간 귀 알림
    def _event(self, grade: str, title: str, detail: str = "") -> None:
        msg = {"type": "event", "grade": grade, "title": title, "detail": detail,
               "time": self.clock().strftime("%H:%M:%S")}
        self._publish(msg)
        try:
            with open(self.record_dir / "events.log", "a", encoding="utf-8") as f:
                f.write(f"{msg['time']} [{grade}] {title} {detail}\n")
        except OSError:
            pass

    def _live_frame(self, info: dict) -> None:
        nf = noise_floor(self._noise_window)
        self._publish({
            "type": "frame", "rms": round(info["rms"], 5), "wake": round(info["a"], 3),
            "stop": round(info["b"], 3), "vad": self._frame_vad, "robot": self._robot_running,
            "muted": bool(info.get("muted")), "state": info["state"],
            "noise": None if nf is None else round(nf, 5),
            "rec_left": None if self._rec is None else round(self._rec.left(), 1),
            "busy": self._busy_text,
        })

    def _live_wake_score(self, kind: str, peak: float) -> None:
        if kind == "fire":
            if self._robot_running:
                self._self_reactions += 1
                self._event("fail", "자기 반응 · 호출", f"점수 {peak:.2f}")
            else:
                self._event("pass", "호출 확정", f"점수 {peak:.2f}")
        elif kind == "near" and not self._robot_running:
            self._event("info", "호출 놓침", f"점수 {peak:.2f}")
        elif kind == "self":
            self._event("info", "자기 목소리 호출 무시", f"점수 {peak:.2f}")

    def _live_emergency(self, event) -> None:
        c = self.live.last_emergency_check or {}
        took = (c.get("postroll") or 0.0) + (c.get("stt") or 0.0)
        detail = f"점수 {c.get('score') or 0:.2f} · 확인 '{c.get('text', '')}' · 판정 {took:.2f}초"
        if self._robot_running:
            self._self_reactions += 1
            self._event("fail", "자기 반응 · 멈춰", detail)
        else:
            self._event("pass", "멈춰 확정", detail)

    def _live_reject(self, text: str) -> None:
        self._event("fail", "멈춰 기각", f"받아쓰기 '{text}'")

    def _live_emergency_near(self, info: dict) -> None:
        detail = f"점수 {info.get('peak', 0):.2f}" + (" · 쉬는 시간" if info.get("cooldown") else "")
        self._event("fail", "멈춰 놓침", detail)

    def _live_text(self, text: str) -> None:
        self._event("pass", "말 받아들임", f"받아쓰기 '{text}'")

    def _live_listen_state(self, state: str) -> None:
        if state.startswith("empty"):
            self._event("info", "청취 창 빈손", state)

    # ------------------------------------------------------------------ 메시지·기록
    def _publish(self, message: dict, keep: Optional[str] = None) -> None:
        try:
            self._publish_fn(message, keep)
        except Exception:  # noqa: BLE001 — 화면이 없어도 점검은 계속 간다
            pass

    def _toast(self, text: str) -> None:
        self._publish({"type": "toast", "text": text})

    def _busy_note(self, text: str) -> None:
        """채점 중 표시 — 처리 스레드가 재채점으로 잠깐 막혀 조각 메시지가 끊기는 동안 화면에 띄운다."""
        self._busy_text = text
        self._publish({"type": "busy", "text": text})

    def _progress(self) -> dict:
        return {
            "quiet": ["pass"] if self._quiet is not None else [],
            "call": ["pass" if t.woke and t.captured else "fail" for t in self.meas.calls],
            "answer": ["pass" if t.accepted and t.correct else "fail" for t in self.meas.answers],
            "stop": ["pass" if t.confirmed else "fail" for t in self.meas.stops],
            "robot": [] if self.meas.robot is None else
                     ["pass" if self.meas.robot.self_reactions == 0 else "fail"],
        }

    def _publish_status(self) -> None:
        st = self.step
        n = self._attempts(st.key)
        recordable = st.key in REC_KINDS
        agc_choices = sorted(set(AGC_CHOICES) | {self.start_values.agc})
        echo_labels = dict(ECHO_LABELS)
        if "start" in self._echo_choices():
            echo_labels["start"] = f"시작 값 ({self.start_values.echo_etail:g}·{self.start_values.echo_nlatten})"
        self._publish({
            "type": "status", "place": self.place,
            "step": {"key": st.key, "no": self.step_index + STEP_NO_OFFSET, "of": TOTAL_STEPS, "name": st.name,
                     "say": st.prompts[min(n, len(st.prompts) - 1)],
                     "rep": min(n + 1, len(st.prompts)) if recordable else 0,
                     "reps": len(st.prompts) if recordable else 0,
                     "seconds": st.seconds, "recording": self._rec is not None, "robot": self._robot_running},
            "steps": [{"key": s.key, "name": s.name, "need": NEEDS[s.key]} for s in STEPS if s.key in NEEDS],
            "progress": self._progress(),
            "values": _values_dict(self.values), "env_values": _values_dict(self.env_values),
            "ranges": {k: list(v) for k, v in RANGES.items()}, "names": NAMES,
            "agc_choices": agc_choices, "echo_choices": echo_labels,
            "env_keys": {**{k: KEYS[k] for k in RANGES}, "agc": KEYS["agc"],
                         "echo": f"{KEYS['echo_etail']} · {KEYS['echo_nlatten']}"},
            "chip_ok": not self.chip_problems, "chip_problems": self.chip_problems,
            "noise_at_check": max(self.meas.noise) if self.meas.noise else None,
        }, "status")

    def _verdict(self):
        return judge(self.meas, self._with_chip(self.judged_values), self.previous_robot_level)

    def _publish_results(self) -> None:
        v = self._verdict()
        first = None
        if self._first is not None and self.judged_values != self.start_values:
            first = [asdict(r) for r in judge(self._first, self._with_chip(self.start_values),
                                              self.previous_robot_level).rows]
        self._publish({"type": "results", "title": v.title, "overall": v.overall,
                       "rows": [asdict(r) for r in v.rows], "first": first,
                       "stale": self._thresholds_differ()}, "results")
        self._write_record()

    def _thresholds_differ(self) -> bool:
        a, b = self.values, self.judged_values
        return (a.listen_min_rms, a.short_answer_min_rms, a.wake_threshold, a.emergency_single) != (
            b.listen_min_rms, b.short_answer_min_rms, b.wake_threshold, b.emergency_single)

    def _save_wav(self, kind: str, rec: Recording, expected: Optional[str]) -> None:
        if not rec.frames:
            return
        n = self._files[kind] = self._files.get(kind, 0) + 1
        name = "quiet" if kind == "quiet" else ("robot" if kind == "robot" else f"{kind}_{n}")
        if expected:
            name += f"_{expected}"
        try:
            with wave.open(str(self.record_dir / f"{name}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(np.concatenate(rec.frames).astype("<i2").tobytes())
            if rec.vads:
                (self.record_dir / f"{name}.vad.json").write_text(
                    json.dumps([None if x is None else bool(x) for x in rec.vads]), encoding="utf-8")
        except OSError as exc:
            self._toast(f"녹음 파일을 쓰지 못했습니다: {exc}")

    def _write_record(self) -> None:
        v = self._verdict()
        data = {
            "place": self.place, "started": self.started.isoformat(timespec="seconds"),
            "values_start": asdict(self.start_values), "values_now": asdict(self.values),
            "env_values": asdict(self.env_values), "judged_values": asdict(self.judged_values),
            "measurements": asdict(self.meas), "previous_robot_level": self.previous_robot_level,
            "verdict": {"overall": v.overall, "title": v.title, "rows": [asdict(r) for r in v.rows]},
            "rejudges": self._history, "saves": self._saves, "chip_problems": self.chip_problems,
        }
        try:
            (self.record_dir / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        except OSError:
            pass
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_session.py -q`
Expected: PASS

실패하면 먼저 확인할 것: `test_full_run_passes_and_records` 의 `res["rows"]` 를 출력해 어느 줄이 합격이 아닌지 본다. 잡음 줄은 NOISE(rms≈0.003)라 0.008/0.003 ≈ 2.6배여야 한다.

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add src/mic_check_session.py tests/test_mic_check_session.py
git commit -m "feat(voice): 마이크 점검기 진행 — 단계·녹음·실시간 귀·재채점·값 고치기·저장/되돌리기·기록(하드웨어는 밖에서 주입)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 진입점 (`tools/mic_venue_check.py`)

**Files:**
- Create: `tools/mic_venue_check.py`
- Test: `tests/test_mic_venue_check_tool.py`

**Interfaces:**
- Consumes: `Session`, `Hardware`, `Clip`, `previous_robot_level`(Task 6), `LiveServer`, `DEFAULT_PORT`(Task 5), `Models`(Task 4), `values_from`, `read_env`, `restore_latest`, `latest_backup`, `EnvEditError`(Task 3), `dsp_state.DspState`·`apply_agc_desired_level`·`apply_echo_tuning`, `audio_out.play`·`reset_device_cache`
- Produces: `voice_stack_running(cmdlines) -> list[str]`, `load_clips(baked_dir, names=ROBOT_CLIPS) -> tuple[list[Clip], list[str]]`, `lan_ip() -> str`, `write_chip(values) -> list[str]`, `main(argv=None) -> int`

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_venue_check_tool.py`:

```python
"""점검기 진입점의 순수한 부분 — ⑫·⑬ 찾기, 멘트 읽기(빠진 파일 건너뜀). 마이크·칩·모델은 실기로 본다."""
from __future__ import annotations

import json
import wave

import numpy as np

from tools.mic_venue_check import ROBOT_CLIPS, lan_ip, load_clips, voice_stack_running


def test_voice_stack_running():
    lines = [
        "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch launch/vica_voice.launch.py map_id:=x",
        "/home/ji_w/VICA-smarthandle/vica-voice-llm/.venv/bin/python -m src.ros_wakeword_node",
        "/home/ji_w/VICA-smarthandle/vica-voice-llm/.venv/bin/python -m tools.mic_venue_check --name 연구실",
        "bash",
    ]
    assert voice_stack_running(lines) == lines[:2]


def test_robot_clips_are_the_eight_in_spec():
    assert len(ROBOT_CLIPS) == 8 and ROBOT_CLIPS[0] == "mission_msg_approach_onboarding.wav"


def test_load_clips_skips_missing(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    (tmp_path / "manifest.json").write_text(json.dumps({"a.wav": "안녕하세요"}), encoding="utf-8")
    clips, missing = load_clips(tmp_path, ("a.wav", "b.wav"))
    assert [c.name for c in clips] == ["a.wav"] and clips[0].text == "안녕하세요" and clips[0].rate == 16000
    assert clips[0].pcm.dtype == np.float32 and missing == ["b.wav"]


def test_lan_ip_is_text():
    assert isinstance(lan_ip(), str) and lan_ip().count(".") == 3
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_venue_check_tool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.mic_venue_check'`

- [ ] **Step 3: 구현한다**

`tools/mic_venue_check.py`:

```python
"""마이크 현장 점검기 — 시연장에서 한 번 돌려 '이 장소에서 비카가 제대로 듣는가'를 휴대폰으로 본다.

설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md · 사용법: docs/mic-venue-check.md

    cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.mic_venue_check --name 시연장
    .venv/bin/python -m tools.mic_venue_check --restore     # .env 를 가장 최근 백업으로
    .venv/bin/python -m tools.mic_venue_check --demo        # 마이크·모델 없이 화면만(가짜 소리)

전제: ⑫·⑬ 꺼짐 · 로봇 정지 · reSpeaker USB · 스피커는 reSpeaker 출력. 도구는 ROS 에 아무것도 보내지 않는다.
가져오기(import)만으로는 .env 를 읽지 않는다 — 시험이 운영 설정에 흔들리지 않게 프로그램으로 실행할 때만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import socket
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⑫ 와 같게 .env 를 src 가져오기보다 먼저 읽는다 — wakeword_monitor 가 가져오는 순간 환경에서
    # 문턱·청취 창 시간을 읽는다(노드의 09-01 결함과 같은 것).
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

from src.env_file import EnvEditError, latest_backup, restore_latest, values_from  # noqa: E402
from src.mic_check_replay import Models  # noqa: E402
from src.mic_check_rules import MicValues  # noqa: E402
from src.mic_check_session import Clip, Hardware, Session, previous_robot_level  # noqa: E402
from src.wakeword_monitor import FRAME, SAMPLE_RATE, WakewordMonitor  # noqa: E402
from tools.mic_check_web import DEFAULT_PORT, LiveServer  # noqa: E402

ENV_PATH = ROOT / ".env"
DATA = Path.home() / "vica_data" / "mic_checks"
BACKUP_DIR = DATA / "env_backups"
PAGE = Path(__file__).with_name("mic_check_page.html")
CHANNELS = 6              # reSpeaker 6채널 펌웨어 — ch0 이 칩 처리음(⑫ 가 듣는 소리)
SETTLE_SEC = 2.0          # 칩에 쓴 뒤 AGC·에코가 자리 잡을 시간(잡음은 그 뒤에 잰다)
VOICE_PATTERNS = ("vica_voice.launch.py", "-m src.ros_")   # ⑫·⑬ 중복 검사와 같은 무늬
ROBOT_CLIPS = (           # 설계 3절 — 호출어·긴급어와 닮은 말이 든 실제 멘트, 합 29.3 s
    "mission_msg_approach_onboarding.wav",
    "mission_msg_approach_question.wav",
    "mission_msg_wait_spot_left_default.wav",
    "mission_msg_estopped.wav",
    "mission_msg_paused.wav",
    "mission_msg_obstacle_stop.wav",
    "mission_msg_estop_reject.wav",
    "mission_msg_obstacle_avoid.wav",
)


def voice_stack_running(cmdlines: Sequence[str]) -> list:
    """떠 있는 음성 칸(⑫·⑬) 명령줄들. 이 도구가 마이크·스피커·칩을 혼자 써야 한다."""
    return [line for line in cmdlines if any(p in line for p in VOICE_PATTERNS)]


def _proc_cmdlines() -> list:
    me = os.getpid()
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        out.append(raw.replace(b"\0", b" ").decode("utf-8", "replace").strip())
    return out


def load_clips(baked_dir: Path, names: Sequence[str] = ROBOT_CLIPS) -> tuple:
    """(Clip 목록, 빠진 파일 이름 목록). 글자는 manifest.json 에서 — ⑫ 의 자기 말 무시 규칙에 쓴다."""
    baked_dir = Path(baked_dir)
    try:
        manifest = json.loads((baked_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    clips, missing = [], []
    for name in names:
        try:
            with wave.open(str(baked_dir / name), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                rate = w.getframerate()
        except (OSError, wave.Error, EOFError):
            missing.append(name)
            continue
        clips.append(Clip(name, str(manifest.get(name, "")), pcm, rate))
    return clips, missing


def lan_ip() -> str:
    """휴대폰이 열 주소. 바깥으로 아무것도 보내지 않고 나가는 쪽 주소만 고른다."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def write_chip(values: MicValues) -> list:
    """칩에 AGC·에코를 쓰고 읽어서 확인한다. 문제 목록(빈 목록 = 성공). 마이크 스트림이 닫혀 있을 때만 부를 것."""
    from src import audio_out
    from src.dsp_state import apply_agc_desired_level, apply_echo_tuning

    audio_out.reset_device_cache()   # 재생 스트림이 열려 있으면 칩이 쓰기를 거부한다 — 먼저 닫는다
    problems = []
    if not apply_agc_desired_level(values.agc):
        problems.append(f"AGC {values.agc:.3f} 쓰기 실패")
    wrote = apply_echo_tuning(values.echo_etail, values.echo_nlatten)
    etail = wrote.get("GAMMA_ETAIL")
    if etail is None or abs(float(etail) - values.echo_etail) > 1e-3:
        problems.append(f"에코 꼬리 {values.echo_etail} 쓰기 실패(읽은 값 {etail})")
    nl = wrote.get("NLATTENONOFF")
    if nl is None or int(nl) != int(values.echo_nlatten):
        problems.append(f"에코 메아리 {values.echo_nlatten} 쓰기 실패(읽은 값 {nl})")
    return problems


def _find_input() -> int:
    import sounddevice as sd

    for i, dev in enumerate(sd.query_devices()):
        if "respeaker" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            return i
    raise SystemExit("reSpeaker 6채널 입력을 찾지 못했습니다 — USB 연결을 확인하세요(다른 마이크로 대신하지 않습니다).")


class Mic:
    """reSpeaker ch0 스트림. 칩 값은 스트림이 닫혀 있을 때만 써지므로 열고 닫기를 여기서 한다."""

    def __init__(self, device: int):
        self.device = device
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            block = np.frombuffer(indata, dtype=np.int16).reshape(-1, CHANNELS)
            self.frames.put(block[:, 0].copy())

        self._stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                                         dtype="int16", device=self.device, callback=callback)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def _chip_apply_for(mic: Mic):
    def apply(values: MicValues) -> list:
        was_open = mic.is_open
        mic.close()
        problems = write_chip(values)
        time.sleep(SETTLE_SEC)
        if was_open:
            mic.open()
        return problems
    return apply


def _place_hint(path: str) -> Optional[str]:
    if not path:
        print("  (장소 귀띔 없음 — --destinations 를 주면 ⑫ 와 같은 귀띔으로 받아씁니다)")
        return None
    from src.destination_loader import build_place_hint, load_destinations
    try:
        return build_place_hint(load_destinations(path))
    except Exception as exc:  # noqa: BLE001
        print(f"  (장소 귀띔 생략: {exc})")
        return None


def _load_models(listen_hint: Optional[str]) -> Models:
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None, listen_hint=listen_hint)
    m._load_real()
    return Models.from_loaded(m)


def _drain(session: Session, server: LiveServer) -> None:
    while True:
        try:
            cmd = server.commands.get_nowait()
        except queue.Empty:
            return
        session.handle(cmd)


def _restore_cli() -> int:
    latest = latest_backup(BACKUP_DIR)
    if latest is None:
        print(f"되돌릴 백업이 없습니다 ({BACKUP_DIR})")
        return 1
    answer = input(f".env 를 {latest.name} 으로 되돌릴까요? 지금 .env 는 .before-restore 로 남깁니다 [y/N] ")
    if answer.strip().lower() != "y":
        print("그대로 둡니다.")
        return 0
    restore_latest(ENV_PATH, BACKUP_DIR)
    print("되돌렸습니다. ⑫를 다시 켜야 적용됩니다.")
    return 0


class _Demo:
    """--demo: 마이크·모델·칩 없이 가짜 소리로 화면을 움직인다(휴대폰 화면 확인용). 진짜 .env 는 건드리지 않는다."""
    WAKE, STOP, YES, NO, VOICE = 20001, 20002, 20003, 20004, 20005

    def __init__(self):
        self.vad = False
        self.rng = np.random.default_rng(0)

    def predict(self, frame: np.ndarray) -> dict:
        v = int(frame[0])
        return {"a": 0.92 if v == self.WAKE else 0.02, "b": 0.88 if v == self.STOP else 0.03}

    def models(self) -> Models:
        def confirm(audio: np.ndarray) -> str:
            return "아니요" if np.any(audio == self.NO) else "네"
        return Models(self.predict, lambda a: "멈춰", lambda a: "화장실로 가자", confirm)

    def frame(self, session: Session) -> np.ndarray:
        kind, pos = session.recording_kind, session.recording_pos
        self.vad = False
        if kind == "call" and pos in (3, 4):
            return np.full(FRAME, self.WAKE, np.int16)
        if kind == "call" and 5 <= pos < 15:
            self.vad = True
            return np.full(FRAME, self.VOICE, np.int16)
        if kind == "answer" and 3 <= pos < 11:
            self.vad = True
            return np.full(FRAME, self.NO if session.recording_expected == "no" else self.YES, np.int16)
        if kind == "stop" and pos == 3:
            return np.full(FRAME, self.STOP, np.int16)
        scale = 900 if session.robot_running else 100
        return (self.rng.standard_normal(FRAME) * scale).astype(np.int16)


def _run_demo(server: LiveServer, env_values: MicValues, record_dir: Path, clips: list,
              prev: Optional[float], place: str) -> int:
    demo = _Demo()
    demo_env = record_dir / "demo.env"
    shutil.copy(ENV_PATH, demo_env)
    session = Session(models=demo.models(),
                      hw=Hardware(lambda v: [], lambda: demo.vad, lambda pcm, rate: time.sleep(len(pcm) / rate)),
                      publish=server.publish, env_values=env_values, env_path=demo_env, place=place,
                      record_dir=record_dir, backup_dir=record_dir / "env_backups", clips=clips,
                      previous_robot_level=prev)
    print("  [데모] 가짜 소리로 돕니다. 저장·되돌리기는 기록 폴더의 demo.env 에만 씁니다.")
    next_t = time.monotonic()
    try:
        while not session.done:
            _drain(session, server)
            session.on_frame(demo.frame(session))
            next_t += FRAME / SAMPLE_RATE
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        session.close()
        server.stop()
        print(f"  기록: {record_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="마이크 현장 점검기")
    ap.add_argument("--name", default="점검", help="장소 이름(기록 폴더·.env 표시에 쓴다)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--destinations", default=os.environ.get("VICA_DESTINATIONS_YAML", ""),
                    help="장소 귀띔용 destinations.yaml(⑫ 와 같은 파일). 없으면 귀띔 없이 받아쓴다")
    ap.add_argument("--restore", action="store_true", help=".env 를 가장 최근 백업으로 되돌리고 끝낸다")
    ap.add_argument("--demo", action="store_true", help="마이크·모델 없이 가짜 소리로 화면만 본다")
    args = ap.parse_args(argv)
    if args.restore:
        return _restore_cli()
    os.chdir(ROOT)   # 호출 모델 경로(models/...)가 저장소 기준이다
    if not args.demo:
        running = voice_stack_running(_proc_cmdlines())
        if running:
            print("⑫·⑬ 음성 칸이 켜져 있습니다. 그 칸을 Ctrl+C 로 끈 뒤 다시 실행하세요:")
            for line in running[:5]:
                print("   ", line[:120])
            return 1
    try:
        env_values = values_from(os.environ)
    except EnvEditError as exc:
        print(exc)
        return 1
    safe = re.sub(r"[^\w가-힣-]+", "_", args.name).strip("_") or "점검"
    record_dir = DATA / f"{datetime.now():%Y%m%d-%H%M}_{safe}"
    record_dir.mkdir(parents=True, exist_ok=True)
    prev = previous_robot_level(DATA, exclude=record_dir)
    clips, missing = load_clips(ROOT / "assets" / "baked")
    for name in missing:
        print(f"  (멘트 {name} 없음 — 건너뜀)")
    server = LiveServer(PAGE.read_text(encoding="utf-8"), port=args.port)
    server.start()
    print(f"\n  휴대폰으로 여세요:  http://{lan_ip()}:{server.port}\n")
    server.publish({"type": "prep", "text": "젯슨이 준비하는 중입니다(모델 올리기 1~2분)…"}, keep="status")
    if args.demo:
        return _run_demo(server, env_values, record_dir, clips, prev, args.name)

    from src import audio_out
    from src.dsp_state import DspState

    dsp = DspState()
    if not dsp.available:
        print("reSpeaker 상태 레지스터(사람 말 표시)를 읽을 수 없습니다 — udev·USB 를 확인하세요.")
        server.stop()
        return 1
    mic = Mic(_find_input())
    chip_apply = _chip_apply_for(mic)
    print("  칩에 .env 값을 씁니다(⑫ 가 켤 때와 같은 일)…")
    problems = chip_apply(env_values)
    for p in problems:
        print(f"  ⚠ {p}")
    print("  모델을 올립니다(1~2분)…")
    models = _load_models(_place_hint(args.destinations))
    session = Session(models=models,
                      hw=Hardware(chip_apply, dsp.speech_detected,
                                  lambda pcm, rate: audio_out.play(pcm, rate, blocking=True)),
                      publish=server.publish, env_values=env_values, env_path=ENV_PATH, place=args.name,
                      record_dir=record_dir, backup_dir=BACKUP_DIR, clips=clips, previous_robot_level=prev,
                      chip_problems=problems)
    mic.open()
    print("  준비 끝 — 휴대폰 화면을 따라 하세요. 끝낼 때는 화면의 '점검 끝내기' 또는 Ctrl+C.")
    try:
        while not session.done:
            _drain(session, server)
            try:
                frame = mic.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            session.on_frame(frame)
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        mic.close()
        back = write_chip(session.env_values)   # 칩을 .env 값으로(저장했다면 새 값)
        if back:
            print("  ⚠ 칩 되돌리기 문제: " + "; ".join(back))
        session.close()
        server.stop()
        print(f"  기록: {record_dir}")
        print("  이제 ⑫를 켜고 시작 로그의 값을 확인한 뒤 시연 대화를 한 번 해 보세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_venue_check_tool.py -q`
Expected: PASS

- [ ] **Step 5: 커밋 (사용자 승인 뒤)**

```bash
git add tools/mic_venue_check.py tests/test_mic_venue_check_tool.py
git commit -m "feat(voice): 마이크 현장 점검기 진입점 — ⑫·⑬ 확인·칩 쓰기/되돌리기·멘트 8개·--restore·--demo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 휴대폰 화면 (`tools/mic_check_page.html`)

**Files:**
- Create: `tools/mic_check_page.html`
- Test: `tests/test_mic_check_page.py`

**Interfaces:**
- Consumes: Task 6 메시지(`prep`, `status`, `frame`, `event`, `results`, `toast`, `save_preview`, `busy`)와 명령(`record`, `robot`, `redo`+`step`, `set`+`key`+`value`, `rejudge`, `save_preview`, `save`, `restore`, `finish`). 값 키: `listen_min_rms`, `short_answer_min_rms`, `wake_threshold`, `emergency_single`, `agc`, `echo`.

- [ ] **Step 1: 실패하는 시험을 쓴다**

`tests/test_mic_check_page.py`:

```python
"""휴대폰 화면 정적 검사 — 스크립트가 찾는 고정 id 가 모두 있고, id 가 겹치지 않는다."""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "tools" / "mic_check_page.html"
BUILT = ("ctl-", "val-", "chg-", "warn-", "lbl-")      # 값 고치기 칸은 스크립트가 만든다


def test_ids_used_by_script_exist_once():
    html = PAGE.read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script))
    used = {u for u in used if not u.startswith(BUILT)}
    ids = re.findall(r'\sid="([A-Za-z0-9_-]+)"', html.split("<script>", 1)[0])
    assert len(ids) == len(set(ids)), "id 가 겹친다"
    missing = sorted(used - set(ids))
    assert not missing, f"화면에 없는 id: {missing}"


def test_no_external_resources():
    html = PAGE.read_text(encoding="utf-8")
    assert "http://" not in html.replace("ws://", "") and "https://" not in html   # 시연장 인터넷 없이도
```

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_page.py -q`
Expected: FAIL — `FileNotFoundError`

- [ ] **Step 3: 페이지를 쓴다**

`tools/mic_check_page.html`:

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>비카 마이크 점검</title>
<style>
/* 휴대폰 한 줄 화면. 계측기 느낌: 눈금·기준선·최근 최고값. 외부 글꼴·라이브러리 없음(시연장 인터넷 없이도) */
:root {
  --bg: #E9EDF0; --panel: #FFFFFF; --ink: #16212B; --muted: #5B6975; --line: #D3DAE0; --track: #E3E8EC;
  --rms: #12807F; --wake: #2C63C9; --stop: #B23A78; --pass: #1D8650; --warn: #B98100; --fail: #C23C2A;
  --idle: #9AA5AE; --btn: #16212B; --on-btn: #FFFFFF;
  --mono: ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0D1318; --panel: #151E25; --ink: #E3E9ED; --muted: #93A1AC; --line: #26313A; --track: #222C34;
    --rms: #47C4C2; --wake: #7DA6F2; --stop: #E57BB3; --pass: #4CC98A; --warn: #F0BE4A; --fail: #F0745F;
    --idle: #5E6B75; --btn: #E3E9ED; --on-btn: #10171D;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif; }
main { max-width: 440px; margin: 0 auto; padding: 12px 12px calc(56px + env(safe-area-inset-bottom, 0px)); display: grid; gap: 12px; }
section, header.top { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px; display: grid; gap: 10px; min-width: 0; }
h2 { margin: 0; font-size: 12px; color: var(--muted); font-weight: 700; letter-spacing: .04em; }
button { font: inherit; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--wake); outline-offset: 2px; }
.name { font-weight: 700; font-size: 16px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 12px; font-weight: 600; border-radius: 99px; padding: 1px 9px; border: 1px solid currentColor; white-space: nowrap; color: var(--muted); }
.chip.ok { color: var(--pass); } .chip.warn { color: var(--warn); } .chip.bad { color: var(--fail); }
.chip.robot { color: var(--warn); } .chip.vad.on { color: var(--pass); }
.prep { font-size: 13px; color: var(--warn); }
.row1 { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: var(--muted); }
.row1 b { color: var(--ink); }
.say { font-size: 22px; font-weight: 700; line-height: 1.35; }
.rep { font-family: var(--mono); }
.bigbtn { font-weight: 600; font-size: 16px; background: var(--btn); color: var(--on-btn); border: 0; border-radius: 12px; padding: 13px; width: 100%; cursor: pointer; }
.ghost { flex: 1; font-weight: 600; font-size: 14px; padding: 10px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
.mini { font-size: 12px; padding: 2px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
.bigbtn:disabled, .ghost:disabled { opacity: .45; cursor: not-allowed; }
.recbar { height: 10px; border-radius: 99px; background: var(--track); overflow: hidden; }
.recbar i { display: block; height: 100%; background: var(--fail); }
.recbar i.robot { background: var(--warn); }
.rectext { font-size: 13px; display: flex; justify-content: space-between; gap: 8px; font-variant-numeric: tabular-nums; }
.rectext .dot { color: var(--fail); font-weight: 700; }
.mhead { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.meter { display: grid; gap: 4px; }
.mlabel { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; font-size: 13px; }
.val { font-family: var(--mono); font-weight: 600; font-variant-numeric: tabular-nums; }
.state { font-size: 12px; font-weight: 600; }
.bar { position: relative; height: 22px; background: var(--track); border-radius: 6px; }
.bar .fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 6px; background: var(--idle); }
.bar .band { position: absolute; left: 0; top: 0; bottom: 0; width: 0; background: repeating-linear-gradient(135deg, var(--ink) 0 1.5px, transparent 1.5px 5px); opacity: .28; border-radius: 6px 0 0 6px; }
.bar .peak { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink); left: 0; }
.bar .th { position: absolute; top: -5px; bottom: -5px; width: 0; border-left: 2px solid var(--ink); }
.bar .th.dash { border-left-style: dashed; opacity: .8; }
.ticks { position: relative; height: 16px; font: 11px/16px var(--mono); color: var(--muted); }
.ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.ticks .edge-l { transform: none; } .ticks .edge-r { transform: translateX(-100%); }
.ticks .th-l { color: var(--ink); font-weight: 600; }
.noise { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; font-size: 13px; }
.noise .why { flex-basis: 100%; font-size: 11px; color: var(--muted); }
canvas { display: block; width: 100%; height: 190px; }
.events { display: grid; gap: 6px; }
.ev { display: grid; grid-template-columns: 6px 1fr auto; gap: 10px; border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
.ev .stripe { border-radius: 3px; background: var(--idle); }
.ev.pass .stripe { background: var(--pass); } .ev.fail .stripe { background: var(--fail); }
.ev .t { font-weight: 700; font-size: 14px; } .ev .s { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
.ev time { font: 12px var(--mono); color: var(--muted); }
.empty { font-size: 13px; color: var(--muted); }
.prow { display: grid; grid-template-columns: 7em 1fr auto; gap: 8px; align-items: center; font-size: 13px; }
.dots { display: flex; gap: 4px; flex-wrap: wrap; }
.dots i { width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--idle); }
.dots i.pass { background: var(--pass); border-color: var(--pass); } .dots i.fail { background: var(--fail); border-color: var(--fail); }
.pnote { font: 12px var(--mono); color: var(--muted); }
.rrow { border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; display: grid; gap: 2px; }
.rrow .top { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; font-size: 14px; }
.rrow .txt { font: 12px var(--mono); color: var(--muted); overflow-wrap: anywhere; }
.rrow .todo { font-size: 12px; color: var(--ink); }
.rrow .was { font-size: 11px; color: var(--muted); border-top: 1px dashed var(--line); padding-top: 3px; }
#rows { display: grid; gap: 6px; }
.g { font-size: 11px; font-weight: 700; border-radius: 99px; padding: 1px 8px; border: 1px solid currentColor; white-space: nowrap; }
.g-pass { color: var(--pass); } .g-warn { color: var(--warn); } .g-fail { color: var(--fail); } .g-redo, .g-wait { color: var(--idle); }
.hint { font-size: 12px; color: var(--warn); }
.ctl { display: grid; gap: 4px; padding-block: 8px; border-bottom: 1px solid var(--line); }
.ctl:last-child { border-bottom: 0; }
.ctl .row { display: flex; justify-content: space-between; gap: 8px; font-size: 14px; font-weight: 600; }
.ctl .v { font-family: var(--mono); }
.ctl .chg { font-size: 11px; color: var(--warn); }
.ctl input[type="range"] { width: 100%; accent-color: var(--wake); }
.ctl .h { font-size: 12px; color: var(--muted); }
.ctl .env { font: 11px var(--mono); color: var(--muted); overflow-wrap: anywhere; }
.ctl .w, .err { font-size: 12px; color: var(--fail); font-weight: 600; }
.seg { display: flex; gap: 4px; }
.seg button { flex: 1; font-weight: 600; font-size: 13px; padding: 7px 4px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--btn); color: var(--on-btn); border-color: var(--btn); }
.btns { display: grid; gap: 8px; } .btnrow { display: flex; gap: 8px; }
.sheet { border: 2px solid var(--ink); border-radius: 14px; padding: 12px; display: grid; gap: 8px; }
.sheet .t { font-weight: 700; } .sheet .s { font-size: 12px; color: var(--muted); }
.sheet pre { margin: 0; font: 11.5px/1.6 var(--mono); background: var(--track); border-radius: 8px; padding: 8px; overflow-x: auto; }
ol.wrap { margin: 0; padding-left: 1.3em; font-size: 14px; display: grid; gap: 4px; }
.toast { position: fixed; left: 12px; right: 12px; bottom: calc(12px + env(safe-area-inset-bottom, 0px)); max-width: 416px; margin: 0 auto; background: var(--ink); color: var(--panel); border-radius: 10px; padding: 10px 12px; font-size: 14px; }
.busy { position: fixed; top: calc(8px + env(safe-area-inset-top, 0px)); left: 50%; transform: translateX(-50%); background: var(--warn); color: #10171D; border-radius: 99px; padding: 4px 14px; font-weight: 700; font-size: 13px; }
</style>
</head>
<body>
<main>
  <header class="top">
    <div class="name">마이크 점검 · <span id="place">…</span></div>
    <div class="chips"><span class="chip ok">⑫·⑬ 꺼짐</span><span class="chip" id="chipEnv">칩 확인 중</span><span class="chip bad" id="chipConn">연결 중</span></div>
    <div class="prep" id="prep" hidden></div>
  </header>

  <section aria-live="polite">
    <div class="row1"><span><b id="stepNo">단계 – / 7</b> · <span id="stepName">준비</span></span><span class="rep" id="stepRep"></span></div>
    <div class="say" id="stepSay">젯슨이 준비하는 중입니다</div>
    <div id="stepAction"></div>
  </section>

  <section>
    <div class="mhead"><h2>실시간 · 지금 값으로 통과하는지</h2>
      <span class="chips"><span class="chip robot" id="robotChip" hidden>로봇 말하는 중 · 멈춰 0.35</span><span class="chip" id="mutedChip" hidden>마이크 닫힘</span><span class="chip vad" id="vadChip">칩 사람 말</span></span></div>
    <div class="meter">
      <div class="mlabel"><span>소리 크기</span><span><span class="state" id="rmsState">–</span> · <span class="val" id="rmsVal">–</span></span></div>
      <div class="bar" id="rmsBar"><div class="fill"></div><div class="band"></div><div class="th" id="rmsL1"></div><div class="th" id="rmsL2"></div><div class="peak"></div></div>
      <div class="ticks"><span class="edge-l" style="left:0">0.001</span><span class="th-l" id="rmsT1"></span><span class="th-l" id="rmsT2"></span><span style="left:66.7%">0.1</span><span class="edge-r" style="left:100%">1</span></div>
    </div>
    <div class="noise"><span>잡음 바닥</span><span class="val" id="noiseVal">재는 중</span><span class="chip" id="noiseChip">10초 모으는 중</span><span class="why">사람·로봇이 말하지 않을 때 최근 10초 소리의 상위 10 %</span></div>
    <div class="meter">
      <div class="mlabel"><span>“비카야”</span><span><span class="state" id="wakeState">–</span> · <span class="val" id="wakeVal">–</span></span></div>
      <div class="bar" id="wakeBar"><div class="fill"></div><div class="th" id="wakeL"></div><div class="peak"></div></div>
      <div class="ticks"><span class="edge-l" style="left:0">0</span><span class="th-l" id="wakeT"></span><span class="edge-r" style="left:100%">1</span></div>
    </div>
    <div class="meter">
      <div class="mlabel"><span>“멈춰”</span><span><span class="state" id="stopState">–</span> · <span class="val" id="stopVal">–</span></span></div>
      <div class="bar" id="stopBar"><div class="fill"></div><div class="th" id="stopL"></div><div class="th dash" id="stopSpkL" hidden></div><div class="peak"></div></div>
      <div class="ticks"><span class="edge-l" style="left:0">0</span><span id="stopSpkT" hidden></span><span class="th-l" id="stopT"></span><span class="edge-r" style="left:100%">1</span></div>
    </div>
  </section>

  <section><h2>최근 10초</h2><canvas id="graph" aria-label="최근 10초 흐름 그래프"></canvas></section>
  <section><h2>판정 (실시간, ⑫와 같은 귀)</h2><div class="events" id="events"></div></section>
  <section><h2>진행</h2><div id="progress"></div></section>
  <section>
    <div class="mhead"><h2>결과표 (녹음 재채점)</h2><span class="chip" id="verdict">점검 중</span></div>
    <div id="rows"></div>
    <div class="hint" id="staleHint" hidden>값이 바뀌었습니다. <b>다시 채점</b>을 누르면 남긴 녹음으로 다시 봅니다.</div>
  </section>
  <section>
    <h2>값 고치기</h2>
    <div id="controls"></div>
    <div class="err" id="ctlErr" hidden></div>
    <div class="btns">
      <button class="bigbtn" type="button" id="rejudgeBtn">다시 채점</button>
      <div class="btnrow"><button type="button" class="ghost" id="saveBtn">저장…</button><button type="button" class="ghost" id="restoreBtn">원래대로</button></div>
    </div>
    <div class="sheet" id="sheet" hidden>
      <div class="t">.env에 이렇게 씁니다</div><div id="sheetBody"></div><div class="s" id="sheetNote"></div>
      <div class="btnrow"><button type="button" class="ghost" id="cancelSave">취소</button><button type="button" class="bigbtn" id="confirmSave">확인하고 저장</button></div>
    </div>
    <div class="sheet" id="restoreSheet" hidden>
      <div class="t">.env 를 가장 최근 백업으로 되돌릴까요?</div><div class="s">지금 .env 는 .before-restore 로 남깁니다. ⑫를 다시 켜야 적용됩니다.</div>
      <div class="btnrow"><button type="button" class="ghost" id="cancelRestore">취소</button><button type="button" class="bigbtn" id="confirmRestore">되돌리기</button></div>
    </div>
  </section>
  <section id="wrapup">
    <h2>마무리</h2>
    <ol class="wrap">
      <li><b>점검 끝내기</b>를 누르면 칩을 .env 값으로 되돌리고 도구가 끝납니다.</li>
      <li>⑫를 켜고 시작 로그의 수음 최소·짧은 답 최소·AGC·에코 값을 확인합니다.</li>
      <li>시연 대화를 한 번 해 봅니다: “비카야” → 목적지 → 확인 질문에 “네”.</li>
      <li>값을 저장했다면 시연이 끝난 뒤 <b>원래대로</b>(또는 --restore)를 잊지 마세요.</li>
    </ol>
    <button type="button" class="ghost" id="finishBtn">점검 끝내기</button>
    <div class="sheet" id="finishSheet" hidden>
      <div class="t">점검을 끝낼까요?</div><div class="s">기록은 젯슨의 ~/vica_data/mic_checks 에 남습니다.</div>
      <div class="btnrow"><button type="button" class="ghost" id="cancelFinish">취소</button><button type="button" class="bigbtn" id="confirmFinish">끝내기</button></div>
    </div>
  </section>
</main>
<div class="busy" id="busy" hidden></div>
<div class="toast" id="toast" role="status" hidden></div>
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const logPos = (v) => Math.max(0, Math.min(1, (Math.log10(Math.max(v, 1e-4)) + 3) / 3));
  const linPos = (v) => Math.max(0, Math.min(1, v));
  const GN = { pass: "합격", warn: "주의", fail: "불합격", redo: "다시 녹음", wait: "점검 중" };
  const SPEAKING_STOP = 0.35, HIST = 125;
  const CTL = ["listen_min_rms", "short_answer_min_rms", "wake_threshold", "emergency_single"];
  const HINT = {
    listen_min_rms: "올리면 잡음을 덜 받고, 작은 목소리를 놓칩니다.",
    short_answer_min_rms: "낮추면 작은 “네”도 받고, 잡음도 받습니다.",
    wake_threshold: "낮추면 잘 걸리고, 엉뚱한 말에도 걸립니다.",
    emergency_single: "낮추면 잘 걸리고, 오발동이 늘어납니다.",
    agc: "칩이 소리를 바꾸므로, 바꾸면 2~6단계를 다시 녹음합니다.",
    echo: "약하게 하면 로봇이 자기 말에 반응할 수 있습니다. 바꾸면 6단계를 다시 합니다.",
  };
  let S = null, R = null, local = null, dragging = null, ws = null, last = null, built = false, toastTimer = null;
  const hist = [], events = [];
  const peaks = { rms: { v: 0, t: 0 }, wake: { v: 0, t: 0 }, stop: { v: 0, t: 0 } };

  const cur = () => Object.assign({}, S ? S.values : {}, local || {});
  const fmtTh = (v) => (v < 0.1 ? v.toFixed(4).replace(/0$/, "") : v.toFixed(2));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const diff = (a, b) => (typeof a === "number" ? Math.abs(a - b) > 1e-9 : a !== b);

  // ---- 연결
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
    ws.onmessage = (e) => { let m; try { m = JSON.parse(e.data); } catch (_) { return; } onMessage(m); };
  }
  function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
  function setConn(ok) {
    const c = $("chipConn");
    c.textContent = ok ? "연결됨" : "연결 끊김 — 다시 붙는 중";
    c.className = "chip " + (ok ? "ok" : "bad");
  }
  function onMessage(m) {
    if (m.type === "prep") { $("prep").hidden = false; $("prep").textContent = m.text; }
    else if (m.type === "status") { S = m; $("prep").hidden = true; if (!dragging) local = null; renderStatus(); }
    else if (m.type === "frame") onFrame(m);
    else if (m.type === "event") {
      events.unshift(m); if (events.length > 6) events.length = 6;
      if (hist.length) hist[hist.length - 1].mark = m.grade;
      renderEvents();
    }
    else if (m.type === "results") { R = m; renderResults(); }
    else if (m.type === "toast") toast(m.text);
    else if (m.type === "save_preview") showSave(m);
    else if (m.type === "busy") { $("busy").hidden = !m.text; $("busy").textContent = m.text || ""; }
  }

  // ---- 상태
  function renderStatus() {
    $("place").textContent = S.place;
    const env = $("chipEnv");
    env.textContent = S.chip_ok ? "칩 = .env" : "칩 쓰기 문제";
    env.className = "chip " + (S.chip_ok ? "ok" : "bad");
    env.title = (S.chip_problems || []).join("\n");
    const st = S.step;
    $("stepNo").textContent = `단계 ${st.no} / ${st.of}`;
    $("stepName").textContent = st.name;
    $("stepRep").textContent = st.reps ? `${st.rep} / ${st.reps}회` : "";
    $("stepSay").textContent = st.say;
    renderAction(null);
    placeLines();
    renderProgress();
    syncControls();
  }
  function renderAction(f) {
    if (!S) return;
    const st = S.step, box = $("stepAction");
    let html;
    if (f && f.rec_left != null) {
      const frac = Math.max(0, Math.min(1, 1 - f.rec_left / st.seconds));
      html = `<div class="rectext"><span class="dot">● 녹음 중 · 지금 말하세요</span><span>${f.rec_left.toFixed(1)}초 남음</span></div><div class="recbar"><i style="width:${(frac * 100).toFixed(0)}%"></i></div>`;
    } else if (st.robot) {
      html = `<div class="rectext"><span>로봇이 멘트를 말하는 중입니다. 조용히 계세요.</span></div><div class="recbar"><i class="robot" style="width:100%"></i></div>`;
    } else if (st.key === "robot") {
      html = `<button class="bigbtn" type="button" data-cmd="robot">로봇 말하기 시작 (30초)</button>`;
    } else if (st.key === "result") {
      html = `<div class="rectext"><span>결과표와 값 고치기는 아래에 있습니다.</span></div>`;
    } else {
      html = `<button class="bigbtn" type="button" data-cmd="record">녹음 시작 (${st.seconds}초)</button>`;
    }
    if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
  }

  // ---- 조각
  function paint(id, v, peak, pos, color) {
    const bar = $(id);
    bar.querySelector(".fill").style.width = (pos(v) * 100) + "%";
    bar.querySelector(".fill").style.background = color;
    bar.querySelector(".peak").style.left = `calc(${pos(peak) * 100}% - 1px)`;
  }
  function setState(id, over) {
    const e = $(id);
    e.textContent = over ? "넘음" : "아래";
    e.style.color = over ? "var(--pass)" : "var(--muted)";
  }
  function onFrame(f) {
    if (!S) return;
    const now = performance.now() / 1000;
    hist.push(f); if (hist.length > HIST) hist.shift();
    for (const [k, v] of [["rms", f.rms], ["wake", f.wake], ["stop", f.stop]]) {
      const p = peaks[k];
      if (v >= p.v || now - p.t > 1.5) { p.v = v; p.t = now; }
    }
    last = f;
    const v = cur();
    let rc = "var(--idle)", rs = "잡음으로 버림";
    if (f.rms >= v.short_answer_min_rms) { rc = "var(--pass)"; rs = "짧은 답도 받음"; }
    else if (f.rms >= v.listen_min_rms) { rc = "var(--warn)"; rs = "긴 말만 받음"; }
    paint("rmsBar", f.rms, peaks.rms.v, logPos, rc);
    $("rmsVal").textContent = f.rms.toFixed(4);
    $("rmsState").textContent = rs; $("rmsState").style.color = rc;
    $("rmsBar").querySelector(".band").style.width = f.noise == null ? "0" : (logPos(f.noise) * 100) + "%";
    if (f.noise == null) {
      $("noiseVal").textContent = "재는 중"; $("noiseChip").textContent = "10초 모으는 중"; $("noiseChip").className = "chip";
    } else {
      const r = v.listen_min_rms / f.noise;
      $("noiseVal").textContent = f.noise.toFixed(4);
      $("noiseChip").textContent = `수음 최소까지 ${r.toFixed(1)}배`;
      $("noiseChip").className = "chip " + (r >= 2 ? "ok" : r >= 1.3 ? "warn" : "bad");
    }
    paint("wakeBar", f.wake, peaks.wake.v, linPos, f.wake >= v.wake_threshold ? "var(--pass)" : "var(--wake)");
    $("wakeVal").textContent = f.wake.toFixed(2);
    setState("wakeState", peaks.wake.v >= v.wake_threshold);
    const over = (x) => x >= v.emergency_single || (f.robot && x >= SPEAKING_STOP);
    paint("stopBar", f.stop, peaks.stop.v, linPos, over(f.stop) ? "var(--pass)" : "var(--stop)");
    $("stopVal").textContent = f.stop.toFixed(2);
    setState("stopState", over(peaks.stop.v));
    $("stopSpkL").hidden = !f.robot; $("stopSpkT").hidden = !f.robot;
    $("robotChip").hidden = !f.robot;
    $("mutedChip").hidden = !f.muted;
    $("vadChip").classList.toggle("on", !!f.vad);
    $("busy").hidden = !f.busy; $("busy").textContent = f.busy || "";
    renderAction(f);
    drawGraph();
  }
  function placeLines() {
    const v = cur();
    if (v.listen_min_rms == null) return;
    const set = (id, p) => { $(id).style.left = (p * 100) + "%"; };
    const lab = (id, p, t) => { const e = $(id); e.style.left = (p * 100) + "%"; e.textContent = t; };
    set("rmsL1", logPos(v.listen_min_rms)); set("rmsL2", logPos(v.short_answer_min_rms));
    lab("rmsT1", logPos(v.listen_min_rms), fmtTh(v.listen_min_rms));
    lab("rmsT2", logPos(v.short_answer_min_rms), fmtTh(v.short_answer_min_rms));
    set("wakeL", linPos(v.wake_threshold)); lab("wakeT", linPos(v.wake_threshold), v.wake_threshold.toFixed(2));
    set("stopL", linPos(v.emergency_single)); lab("stopT", linPos(v.emergency_single), v.emergency_single.toFixed(2));
    set("stopSpkL", linPos(SPEAKING_STOP)); lab("stopSpkT", linPos(SPEAKING_STOP), "0.35");
  }

  // ---- 그래프: 세 줄(소리·비카야·멈춰), 기준선 점선, 잡음 빗금, 로봇 말한 구간, 판정 표시
  const cv = $("graph"), ctx = cv.getContext("2d");
  function drawGraph() {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const v = cur();
    if (v.listen_min_rms == null) return;
    const padL = 48, padB = 14, gap = 6, laneH = (h - padB - gap * 2) / 3, plotW = w - padL - 2;
    const lanes = [
      { name: "소리", color: css("--rms"), key: "rms", pos: logPos, ths: [v.listen_min_rms, v.short_answer_min_rms] },
      { name: "비카야", color: css("--wake"), key: "wake", pos: linPos, ths: [v.wake_threshold] },
      { name: "멈춰", color: css("--stop"), key: "stop", pos: linPos, ths: [v.emergency_single] },
    ];
    const ink = css("--ink"), muted = css("--muted"), track = css("--track"), warn = css("--warn");
    const off = HIST - hist.length, step = plotW / (HIST - 1);
    const xAt = (i) => padL + step * i;
    ctx.font = "11px " + css("--mono");
    lanes.forEach((ln, li) => {
      const top = li * (laneH + gap), bot = top + laneH;
      ctx.fillStyle = track; ctx.fillRect(padL, top, plotW, laneH);
      ctx.fillStyle = ink; ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.fillText(ln.name, 0, top + laneH / 2);
      hist.forEach((f, i) => {
        if (!f.robot) return;
        ctx.fillStyle = warn; ctx.globalAlpha = 0.12; ctx.fillRect(xAt(i + off) - step / 2, top, step + 0.5, laneH); ctx.globalAlpha = 1;
      });
      if (ln.key === "rms" && last && last.noise != null) {
        const ny = bot - ln.pos(last.noise) * laneH;
        ctx.save(); ctx.beginPath(); ctx.rect(padL, ny, plotW, bot - ny); ctx.clip();
        ctx.strokeStyle = ink; ctx.globalAlpha = 0.22; ctx.lineWidth = 1;
        for (let x = padL - laneH; x < w; x += 6) { ctx.beginPath(); ctx.moveTo(x, bot); ctx.lineTo(x + laneH, bot - laneH); ctx.stroke(); }
        ctx.restore();
      }
      ln.ths.forEach((th) => {
        const y = bot - ln.pos(th) * laneH;
        ctx.strokeStyle = ink; ctx.globalAlpha = 0.55; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w, y); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      });
      if (hist.length > 1) {
        const path = () => { ctx.beginPath(); hist.forEach((f, i) => { const x = xAt(i + off), y = bot - ln.pos(f[ln.key]) * laneH; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); };
        path(); ctx.lineTo(xAt(HIST - 1), bot); ctx.lineTo(xAt(off), bot); ctx.closePath();
        ctx.fillStyle = ln.color; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
        path(); ctx.strokeStyle = ln.color; ctx.lineWidth = 1.6; ctx.stroke();
      }
    });
    hist.forEach((f, i) => {
      if (!f.mark || f.mark === "info") return;
      ctx.strokeStyle = css(f.mark === "pass" ? "--pass" : "--fail"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(xAt(i + off), 0); ctx.lineTo(xAt(i + off), h - padB); ctx.stroke();
    });
    ctx.fillStyle = muted; ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left"; ctx.fillText("−10초", padL, h - 2);
    ctx.textAlign = "right"; ctx.fillText("지금", w - 2, h - 2);
  }

  // ---- 판정·진행·결과표
  function renderEvents() {
    $("events").innerHTML = events.length
      ? events.map((e) => `<div class="ev ${esc(e.grade)}"><span class="stripe"></span><div><div class="t">${esc(e.title)}</div><div class="s">${esc(e.detail || "")}</div></div><time>${esc(e.time || "")}</time></div>`).join("")
      : `<div class="empty">아직 판정이 없습니다. 말하면 여기에 쌓입니다.</div>`;
  }
  function renderProgress() {
    const P = S.progress || {};
    let html = (S.steps || []).map((s) => {
      const got = P[s.key] || [];
      let dots = "";
      for (let i = 0; i < Math.max(s.need, got.length); i++) dots += `<i class="${got[i] || ""}"></i>`;
      const redo = got.length ? `<button type="button" class="mini" data-cmd="redo" data-step="${s.key}">다시</button>` : "<span></span>";
      return `<div class="prow"><span>${esc(s.name)}</span><span class="dots">${dots}</span>${redo}</div>`;
    }).join("");
    if (S.noise_at_check != null) {
      html += `<div class="pnote">잡음: 점검 때 ${S.noise_at_check.toFixed(4)}${last && last.noise != null ? " · 지금 " + last.noise.toFixed(4) : ""}</div>`;
    }
    $("progress").innerHTML = html;
  }
  function renderResults() {
    if (!R) return;
    const vd = $("verdict");
    vd.textContent = R.title; vd.className = "chip g-" + R.overall;
    const first = {};
    (R.first || []).forEach((r) => { first[r.key] = r; });
    $("rows").innerHTML = R.rows.map((r) => {
      const f = first[r.key];
      const was = f && (f.grade !== r.grade || f.detail !== r.detail) ? `<div class="was">처음 값: ${GN[f.grade]} · ${esc(f.detail)}</div>` : "";
      const todo = r.todo ? `<div class="todo">${esc(r.todo)}</div>` : "";
      return `<div class="rrow"><div class="top"><span>${esc(r.name)}</span><span class="g g-${r.grade}">${GN[r.grade]}</span></div><div class="txt">${esc(r.detail)}</div>${todo}${was}</div>`;
    }).join("");
    $("staleHint").hidden = !R.stale;
  }

  // ---- 값 고치기
  function seg(k, opts) {
    return `<div class="ctl"><div class="row"><span id="lbl-${k}">${esc(S.names[k])}</span><span><span class="chg" id="chg-${k}"></span> <span class="v" id="val-${k}"></span></span></div>
      <div class="seg" role="group" aria-labelledby="lbl-${k}">${opts.map(([val, label]) => `<button type="button" data-k="${k}" data-v="${esc(val)}">${esc(label)}</button>`).join("")}</div>
      <div class="h">${HINT[k]}</div><div class="env">${esc(S.env_keys[k])}</div></div>`;
  }
  function buildControls() {
    let html = "";
    CTL.forEach((k) => {
      const [lo, hi, step] = S.ranges[k];
      html += `<div class="ctl"><div class="row"><label for="ctl-${k}">${esc(S.names[k])}</label><span><span class="chg" id="chg-${k}"></span> <span class="v" id="val-${k}"></span></span></div>
        <input type="range" id="ctl-${k}" min="${lo}" max="${hi}" step="${step}">
        <div class="h">${HINT[k]}</div><div class="w" id="warn-${k}" hidden></div><div class="env">${esc(S.env_keys[k])}</div></div>`;
    });
    html += seg("agc", S.agc_choices.map((a) => [String(a), a.toFixed(3)]));
    html += seg("echo", Object.entries(S.echo_choices));
    $("controls").innerHTML = html;
    CTL.forEach((k) => {
      const inp = $("ctl-" + k);
      inp.addEventListener("input", () => { dragging = k; local = Object.assign(local || {}, { [k]: parseFloat(inp.value) }); syncControls(); placeLines(); });
      inp.addEventListener("change", () => { dragging = null; send({ cmd: "set", key: k, value: parseFloat(inp.value) }); });
    });
    $("controls").querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () =>
      send({ cmd: "set", key: b.dataset.k, value: b.dataset.k === "agc" ? parseFloat(b.dataset.v) : b.dataset.v })));
    built = true;
  }
  function syncControls() {
    if (!S) return;
    if (!built) buildControls();
    const v = cur(), env = S.env_values;
    CTL.forEach((k) => {
      const inp = $("ctl-" + k), show = k.endsWith("rms") ? fmtTh : (x) => x.toFixed(2);
      if (dragging !== k && diff(parseFloat(inp.value), v[k])) inp.value = v[k];
      $("val-" + k).textContent = show(v[k]);
      $("chg-" + k).textContent = diff(v[k], env[k]) ? "바뀜 · .env " + show(env[k]) : "";
    });
    document.querySelectorAll('.seg button[data-k="agc"]').forEach((b) => b.setAttribute("aria-pressed", String(!diff(parseFloat(b.dataset.v), v.agc))));
    document.querySelectorAll('.seg button[data-k="echo"]').forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v.echo)));
    $("val-agc").textContent = v.agc.toFixed(3);
    $("chg-agc").textContent = diff(v.agc, env.agc) ? "바뀜" : "";
    $("val-echo").textContent = S.echo_choices[v.echo] || v.echo;
    $("chg-echo").textContent = v.echo !== env.echo ? "바뀜" : "";
    const w = $("warn-emergency_single");
    w.hidden = !(v.emergency_single < 0.5 - 1e-9);
    w.textContent = "0.50 아래입니다. 다른 말에 멈춰가 잘못 걸릴 수 있으니 결과표의 오발동 줄을 꼭 보세요.";
    const bad = v.short_answer_min_rms < v.listen_min_rms * 1.5 - 1e-9;
    $("ctlErr").hidden = !bad;
    $("ctlErr").textContent = `짧은 답 최소는 수음 최소의 1.5배 이상이어야 합니다(지금 ${(v.short_answer_min_rms / v.listen_min_rms).toFixed(2)}배). 이대로는 저장할 수 없습니다.`;
    $("saveBtn").disabled = bad;
  }

  // ---- 확인 창·안내
  function showSave(m) {
    const pre = document.createElement("pre");
    pre.textContent = m.lines.join("\n");
    $("sheetBody").replaceChildren(pre);
    $("sheetNote").textContent = `쓰기 전에 지금 .env 를 ${m.backup_dir} 에 백업합니다. ⑫를 다시 켜야 적용됩니다.`;
    $("sheet").hidden = false;
  }
  function toast(t) {
    const el = $("toast");
    el.textContent = t; el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 5000);
  }
  $("saveBtn").onclick = () => send({ cmd: "save_preview" });
  $("cancelSave").onclick = () => { $("sheet").hidden = true; };
  $("confirmSave").onclick = () => { $("sheet").hidden = true; send({ cmd: "save" }); };
  $("restoreBtn").onclick = () => { $("restoreSheet").hidden = false; };
  $("cancelRestore").onclick = () => { $("restoreSheet").hidden = true; };
  $("confirmRestore").onclick = () => { $("restoreSheet").hidden = true; send({ cmd: "restore" }); };
  $("rejudgeBtn").onclick = () => send({ cmd: "rejudge" });
  $("finishBtn").onclick = () => { $("finishSheet").hidden = false; };
  $("cancelFinish").onclick = () => { $("finishSheet").hidden = true; };
  $("confirmFinish").onclick = () => { $("finishSheet").hidden = true; send({ cmd: "finish" }); toast("점검을 끝냈습니다. 젯슨 터미널의 안내를 따르세요."); };
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-cmd]");
    if (b) send({ cmd: b.dataset.cmd, step: b.dataset.step });
  });
  renderEvents();
  connect();
})();
</script>
</body>
</html>
```

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `.venv/bin/python -m pytest tests/test_mic_check_page.py -q`
Expected: PASS

- [ ] **Step 5: 데모로 화면을 눈으로 본다(노트북이나 젯슨, ⑫ 켜져 있어도 됨)**

Run: `.venv/bin/python -m tools.mic_venue_check --demo --name 데모 --port 8771`
Expected: 터미널에 `http://<주소>:8771` 이 나온다. 브라우저로 열어 확인한다.
- 막대·잡음·그래프가 움직인다.
- "녹음 시작"을 누르면 남은 초가 줄고, 4초 뒤 진행 점과 결과표가 바뀐다.
- 값 막대를 움직이면 기준선이 따라 움직인다.
- "저장…" → 줄 미리보기 → "확인하고 저장"을 하면 기록 폴더의 `demo.env` 만 바뀐다(진짜 `.env` 는 그대로).
- 끝은 "점검 끝내기" 또는 Ctrl+C.

- [ ] **Step 6: 커밋 (사용자 승인 뒤)**

```bash
git add tools/mic_check_page.html tests/test_mic_check_page.py
git commit -m "feat(voice): 마이크 점검기 휴대폰 화면 — 실시간 막대·잡음 빗금·10초 그래프·판정·진행·결과표·값 고치기(목업 그대로, 외부 자원 없음)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 사용법 문서와 전체 확인

**Files:**
- Create: `docs/mic-venue-check.md`
- Modify: `docs/superpowers/specs/2026-10-12-mic-venue-check-design.md` (구현 중 바뀐 두 곳 반영)

- [ ] **Step 1: 사용법을 쓴다**

`docs/mic-venue-check.md`:

````markdown
# 마이크 현장 점검기 사용법

시연장에 도착하면 한 번 돌려 "이 장소에서 비카가 제대로 듣는가"를 휴대폰으로 확인한다. 기본은 확인만 한다.
값을 고치고 싶으면 화면에서 바꿔 다시 채점하고, 원하면 `.env` 에 저장한다(백업·되돌리기 있음).
설계: `docs/superpowers/specs/2026-10-12-mic-venue-check-design.md`

## 준비

1. 로봇은 멈춰 있어야 한다(주행 중 금지).
2. 터미네이터의 ⑫(또는 ⑬) 칸을 Ctrl+C 로 끈다 — 점검기가 마이크·스피커·칩을 혼자 써야 한다.
3. 휴대폰을 로봇과 같은 와이파이에 붙인다.

## 실행

```
cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.mic_venue_check --name 시연장
```

터미널에 나온 `http://<젯슨 IP>:8770` 을 휴대폰으로 연다. 모델을 올리는 1~2분 동안은 "준비하는 중"이 보인다.
⑫ 와 같은 장소 귀띔을 쓰려면 `--destinations ~/vica_data/destinations/<지도>/destinations.yaml` 을 붙인다.

## 순서 (약 8~10분, 손잡이 자리에 서서)

| 단계 | 할 일 |
|---|---|
| 2 조용히 | "녹음 시작"을 누르고 15초 말하지 않는다 |
| 3 호출 | 누를 때마다 "비카야, 화장실로 가자" — 3번 |
| 4 짧은 답 | "네" 2번, "아니요" 2번 |
| 5 멈춰 | "멈춰" 5번(진짜 비상정지는 걸리지 않는다) |
| 6 로봇 혼자 말하기 | "로봇 말하기 시작"을 누르고 30초 조용히 |
| 7 결과표 | 읽고, 필요하면 값을 고친다 |

단계는 진행 칸의 "다시"로 따로 다시 할 수 있다.

## 결과 읽기

- **시연 가능**: 그대로 시연한다.
- **시연 가능 · 확인 필요**: 주의 줄의 안내를 읽는다(아슬아슬한 곳).
- **다시 녹음 필요**: AGC·에코·문턱을 바꿔 그 단계 녹음이 무효다. 해당 단계를 다시 한다.
- **고친 뒤 다시**: 불합격 줄의 할 일을 하고 그 단계를 다시 한다. 로봇이 자기 말에 반응했으면 먼저 스피커 볼륨을 낮춘다.

결과표는 남긴 녹음을 ⑫ 와 같은 귀로 다시 채점한 것이다. 위쪽 "판정" 칸은 실시간으로 바로 보이는 것이고, 드물게 다를 수 있다.

## 값 고치기

- 막대·단추로 바꾸면 화면의 기준선이 바로 움직인다. 그 자리에서 말해 보며 본다.
- **다시 채점**: 남긴 녹음을 새 값으로 다시 본다(말할 필요 없음). AGC·에코를 바꿨으면 해당 단계를 다시 녹음해야 한다.
- **저장…**: `.env` 에 쓸 줄을 먼저 보여 주고, "확인하고 저장"을 눌러야 쓴다. 쓰기 전에 `~/vica_data/mic_checks/env_backups/` 에 백업한다.
  바꾼 줄 끝에 `# MM-DD <장소> 점검에서 바꿈 (전 <옛 값>)` 이 남는다. **⑫를 다시 켜야 적용된다.**
- **원래대로**: 가장 최근 백업(마지막 저장 직전)으로 되돌린다. 터미널에서는 `.venv/bin/python -m tools.mic_venue_check --restore`.

## 끝낸 뒤

1. 화면의 "점검 끝내기"(또는 Ctrl+C). 칩은 `.env` 값으로 되돌아간다.
2. ⑫를 켜고 시작 로그의 `수음 최소 … · 짧은 답 최소 …`, `AGC 목표 레벨 설정`, `에코 억제 설정` 값을 확인한다.
3. 시연 대화를 한 번 해 본다: "비카야" → 목적지 → 확인 질문에 "네".
4. 저장했다면 시연이 끝난 뒤 되돌리기를 잊지 않는다.

## 기록

`~/vica_data/mic_checks/<날짜-시각>_<장소>/` — `result.json`(값·측정·판정·재채점·저장 이력), 녹음 wav 와 조각별
칩 사람 말 표시(`*.vad.json`), `events.log`. 여러 번 돌린 결과로 `src/mic_check_rules.py` 위쪽의 기준 숫자를 다듬는다.
"지난 점검" 로봇 목소리 크기 비교는 이 폴더의 가장 최근 기록을 쓴다 — 시연 전에 연구실에서 한 번 돌려 둔다.

## 화면만 미리 보기

`.venv/bin/python -m tools.mic_venue_check --demo --port 8771` — 마이크·모델 없이 가짜 소리로 움직인다.
저장·되돌리기는 기록 폴더의 `demo.env` 에만 쓴다.

## 문제 해결

| 증상 | 할 일 |
|---|---|
| "⑫·⑬ 음성 칸이 켜져 있습니다" | 그 칸을 Ctrl+C 로 끄고 다시 실행 |
| "reSpeaker 상태 레지스터를 읽을 수 없습니다" | USB 를 다시 꽂고 udev 규칙 확인 |
| 상태 줄 "칩 쓰기 문제" | 칩 값이 `.env` 와 다를 수 있다. 터미널의 ⚠ 줄을 보고 다시 실행 |
| 휴대폰이 못 연다 | 같은 와이파이인지, 주소·포트(8770)가 맞는지. 다른 프로그램이 8770 을 쓰면 `--port` |
| "VICA_EMERGENCY_SINGLE_FRAME 이 off" | `.env` 의 그 줄을 지우거나 숫자로 바꾼다 |
````

- [ ] **Step 2: 설계서를 구현에 맞춘다**

`docs/superpowers/specs/2026-10-12-mic-venue-check-design.md` 3절 표의 1단계 칸에서
"칩 설정을 읽어 `.env` 와 다르면 ⑫가 켤 때처럼 `.env` 값을 쓰고, 다시 읽어 같은지 확인한다" 를
"⑫가 켤 때처럼 칩에 `.env` 값을 쓰고 다시 읽어 같은지 확인한다(늘 쓴다 — 같은 값이면 바뀌는 것이 없다)" 로 바꾼다.

6절 표 "로봇 혼자 말하기" 줄 아래에 한 줄을 더한다:
"- 6단계는 실시간으로 재므로 다시 채점할 수 없다. AGC·에코나 호출·멈춰 문턱이 6단계 때와 다르면 그 줄은 **다시 녹음**이다."

9절 표 아래에 한 줄을 더한다:
"- 진행 로직은 `src/mic_check_session.py` 로 나눴다(하드웨어 없이 시험하려고). 단계별로 녹음할 때의 칩 값을 따로 기억해,
  AGC 를 바꾼 뒤 그 단계를 녹음하면 그 단계는 처음부터 다시 녹음한다."

- [ ] **Step 3: 전체 시험**

Run: `.venv/bin/python -m pytest -q`
Expected: 기존 902 + 새 시험 모두 PASS

- [ ] **Step 4: 커밋 (사용자 승인 뒤)**

```bash
git add docs/mic-venue-check.md docs/superpowers/specs/2026-10-12-mic-venue-check-design.md docs/superpowers/plans/2026-10-12-mic-venue-check.md
git commit -m "docs(voice): 마이크 현장 점검기 사용법·설계서 구현 반영·구현 계획

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: 연구실 실기 (사람이 함께, ⑫ 끈 상태, 로봇 정지) — 자동화하지 않는다**

설계서 10절 순서 그대로:
1. 휴대폰으로 접속해 막대·잡음·그래프가 움직인다.
2. 1~7단계를 마치고 결과표가 나온다. 걸린 시간을 적는다(목표 8~10분).
3. 값 하나(예: 호출 문턱 0.65)를 바꿔 다시 채점하고, 저장한 뒤 `git diff --no-index ~/vica_data/mic_checks/env_backups/.env.<시각> .env` 로 차이와 백업을 확인한다.
4. 도구를 끝내고 ⑫를 다시 켜 시작 로그에 새 값이 보이는지, 평소 대화 5턴이 정상인지 본다.
5. 점검기 `--restore` 로 되돌리고 ⑫ 시작 로그가 원래 값인지 본다.
6. Ctrl+C 로 도중에 끊어도 칩이 `.env` 값으로 돌아가는지(⑫ 시작 로그의 AGC·에코) 본다.
7. 이때의 로봇 목소리 크기가 이후 점검의 비교 기준이 된다 — 결과를 사용자에게 보고한다.
