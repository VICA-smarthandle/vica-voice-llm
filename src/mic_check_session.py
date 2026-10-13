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

from .env_file import (KEYS, change_line, plan_changes, read_env, restore_latest, values_from,
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
    stop_play: Callable[[], None] = lambda: None  # 재생 중인 멘트를 끊는다(끝내기·Ctrl+C 때 6단계를 멈추려고)


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
        robot = (data.get("measurements") or {}).get("robot") or {}
        level = robot.get("level")
        if robot.get("played_ok") is False:      # 재생이 실패한 점검의 조용한 level 은 기준으로 쓰지 않는다
            continue
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
        self._robot_abort = threading.Event()   # 끝내기·Ctrl+C 가 6단계 루틴에 멈추라고 알린다
        self._robot_idle = threading.Event()    # 루틴이 돌지 않을 때 켜져 있다
        self._robot_idle.set()
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
            self.shutdown()
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

    def shutdown(self, timeout: float = 3.0) -> None:
        """6단계 재생 루틴을 멈추고 끝날 때까지 기다린다 — 칩 되돌리기 전에 스피커 스트림이 닫혀 있게."""
        self._robot_abort.set()
        try:
            self.hw.stop_play()
        except Exception:  # noqa: BLE001 — 멈추기 실패가 끝내기를 막지 않는다
            pass
        self._robot_idle.wait(timeout)

    # ------------------------------------------------------------------ 녹음
    def _busy(self) -> bool:
        return self._rec is not None or self._robot_running

    def _attempts(self, key: str) -> int:
        return {"quiet": int(self._quiet is not None), "call": len(self._calls), "answer": len(self._answers),
                "stop": len(self._stops), "robot": int(self.meas.robot is not None)}.get(key, 0)

    def _chip_blocked(self) -> bool:
        if not self.chip_problems:
            return False
        self._toast("칩에 .env 값을 쓰지 못해 점검을 시작하지 않습니다: " + "; ".join(self.chip_problems)
                    + " — 도구를 다시 실행하세요.")
        return True

    def _start_recording(self) -> None:
        if self._chip_blocked():
            return
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
        try:
            if kind == "quiet":
                noise = _noise_segments(rec)
            elif kind == "call":
                tried = replay_call(self.models, rec, self._quiet, self.judged_values)
            elif kind == "answer":
                tried = replay_answer(self.models, rec, self._quiet, self.judged_values, rec_obj.expected)
            else:
                tried = replay_stop(self.models, rec, self._quiet, self.judged_values)
        except Exception as exc:  # noqa: BLE001 — 한 번의 채점 실패가 도구 전체를 끝내지 않게
            self._busy_note("")
            self._toast(f"채점하지 못했습니다({type(exc).__name__}: {exc}). 이 녹음은 버리고 다시 녹음하세요.")
            self._publish_status()
            return
        if kind == "quiet":
            self._quiet = rec
            self.meas.noise = noise
        elif kind == "call":
            self._calls.append(rec)
            self.meas.calls.append(tried)
        elif kind == "answer":
            self._answers.append((rec, rec_obj.expected))
            self.meas.answers.append(tried)
        else:
            self._stops.append(rec)
            self.meas.stops.append(tried)
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
        if self._chip_blocked():
            return
        if self.step.key != "robot":
            self._toast("로봇 혼자 말하기 단계가 아닙니다.")
            return
        if not self.clips:
            self._toast("틀 멘트가 없습니다 — assets/baked 를 확인하세요.")
            return
        if self._thresholds_differ():
            self._toast("값이 바뀌었습니다. 먼저 다시 채점을 누른 뒤 6단계를 하세요.")
            return
        self._robot_running = True
        self._robot_abort.clear()
        self._robot_idle.clear()
        self._robot_rms, self._robot_frames, self._self_reactions = [], [], 0
        self._robot_thresholds = self.values.thresholds()
        self._robot_error = ""
        self._publish_status()
        self.hw.spawn(self._robot_routine)

    def _robot_routine(self) -> None:
        """⑫ 가 로봇이 말할 때 하는 그대로: 문장 소식(자기 말 무시·마이크 닫기 규칙) → 말하는 중 → 재생 → 꼬리 → 끝."""
        try:
            self._robot_sleep(ROBOT_LEAD_SEC)
            for clip in self.clips:
                if self._robot_abort.is_set():
                    break
                self.live.note_robot_speech(clip.text, now=self._t)
                self.live.set_speaking(True, now=self._t)
                self.hw.play(clip.pcm, clip.rate)
                self._robot_sleep(TTS_TAIL_SEC)
                self.live.set_speaking(False, now=self._t)
                self._robot_sleep(AFTER_LINE_SEC)
            else:
                self._robot_sleep(ROBOT_SETTLE_SEC)
            if self._robot_abort.is_set():
                self._robot_error = "중단됨"
        except Exception as exc:  # noqa: BLE001 — 재생 실패를 화면으로 알린다
            self._robot_error = f"{type(exc).__name__}: {exc}"
        finally:
            if self._robot_error:
                self.live.set_speaking(False, now=self._t)    # 중간에 끊겼으면 '말하는 중' 을 풀어 둔다
            self._robot_done.set()
            self._robot_idle.set()

    def _robot_sleep(self, sec: float) -> None:
        if not self._robot_abort.is_set():
            self.hw.sleep(sec)

    def _finish_robot(self) -> None:
        self._robot_running = False
        level = float(np.mean(self._robot_rms)) if self._robot_rms else 0.0
        self.meas.robot = RobotRun(self._self_reactions, level, played_ok=not self._robot_error)
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
        try:
            meas = self._measure_all(self.values)
            first = None
            if self.values != self.start_values and self._first is None:
                first = self._measure_all(self.start_values)
        except Exception as exc:  # noqa: BLE001 — 실패해도 이전 결과를 그대로 둔다
            self._busy_note("")
            self._toast(f"다시 채점하지 못했습니다({type(exc).__name__}: {exc}). 이전 결과를 그대로 둡니다.")
            return
        self.judged_values = self.values
        self.meas = meas
        if first is not None:
            self._first = first
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
        except Exception as exc:  # noqa: BLE001 — EnvEditError 말고도(디스크·권한) 화면에 알린다
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
        except Exception as exc:  # noqa: BLE001
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
