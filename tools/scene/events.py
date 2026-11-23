"""노드 로그 줄 → 사건(Event). 장면 묶기·원인 판정이 이 모듈의 결과만 쓴다 (설계 2026-11-20 5.1).

읽는 곳(파일만 읽고 지우지·덮어쓰지 않는다. ROS 에 안 붙는다):
- 노드 로그 `~/.ros/log/python*.log` — 파일 mtime 으로 기간을 거른다 (field_report.read_logs 방식).
- relaunch 출력 `~/vica_data/relaunch/voice*_*.log` — 노드 로그와 같은 줄은 한 번만, 시각 없는 stderr 줄은 앞 줄 시각을 쓴다.
- journal `vica-llm` 유닛(2026-11-16~) — `[LLM] 호출 실패` 줄만 가져온다.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tools.answer_audit import (_INPUT, _MREQ, _MSAY, _PLAY, _QCLOSE, _QOPEN, _QVOICE, _RT,
                                parse_log_line)

NODES = {"vica_wakeword_node": "wake", "vica_llm_intent_node": "llm", "vica_llm": "llm",
         "vica_tts_node": "tts", "vica_mission_manager": "mis"}
DEFAULT_LOG_DIR = Path.home() / ".ros" / "log"
DEFAULT_RELAUNCH_DIR = Path.home() / "vica_data" / "relaunch"
JOURNAL_UNIT = "vica-llm"
JOURNAL_TIMEOUT_S = 20

# 미션 대화 상태 이름 (field_report.STATES 와 같다) — 상태 전이 줄만 state 로 센다.
STATES = {"idle", "confirming", "navigating", "arrived", "failed", "estopped", "paused", "approaching",
          "awaiting_user", "turning", "returning", "seeking", "asking_next", "asking_wait_time", "waiting",
          "waiting_release", "moving_to_wait_spot", "moving_back_to_dest", "grip_wait", "asking", "collecting"}

_NUM = r"(-?[\d.]+)"
_WAKE_SCORE = re.compile(rf"^🎯 호출 점수 (확정|놓침|자기 말 무시): {_NUM} \(문턱 {_NUM}\)")
_WAKE_OPEN = re.compile(r"^🙋 비카야 호출 #(\d+)")
_WAKE_RESCUE = re.compile(r"^🙋 창 안 호출 \(소리로 구제\): (.*)$")
_GHOST = re.compile(rf"^청취 기각: empty:ghost speech={_NUM}s rms={_NUM}")
_REJECT = re.compile(r"^청취 기각: (.*)$")
_USER_TEXT = re.compile(r"-> /vica/user_text: '(.*)'$")
_USER_AUDIO = re.compile(rf"^🎧 발화 소리 -> /vica/user_audio \({_NUM}s\)")
_QUALITY = re.compile(rf"^수음 품질: rms {_NUM} · peak {_NUM} · clip {_NUM}%")
_TIMING = re.compile(rf"^계측: 대기 {_NUM}s · 발화 {_NUM}s · 말끝판정 {_NUM}s · STT {_NUM}s")
_SPK = re.compile(rf"^\[화자\] (pass|fail|short|no_target) 점수 {_NUM}/{_NUM} (\S+) ([\d.]+)s 방향 (\S+) (.*?)( — 버림)?$")
_FORWARD = re.compile(r"^긴급 관문 발동 → STT 기각 \(전사: '(.*?)'\).*질문 답으로 넘김")
_RELISTEN = re.compile(r"^재청취 기각\(무의미\): '(.*)' intent=(\w+)$")
_RT_TAIL = re.compile(r"src=(\w+) dt=([\d.]+)s")
_MIS_WAKE = re.compile(r"^'비카야'\((들음|대답 안 함)\): (\w+) -> (\w+)")
_ARROW = re.compile(r"(?:^|[:)] )([a-z_]+) (?:->|→) ([a-z_]+)")
_ESTOP_STATE = re.compile(r"-> state=([a-z_]+)")
_STARTUP = re.compile(r"^(VICA 웨이크워드 감시 시작|VICA TTS node 시작|VICA LLM intent node 시작|"
                      r"vica_mission_manager 시작|호출 문턱 |긴급 관문 [\d.]+ |AGC 목표|AGC 최대|에코 억제 설정|"
                      r"의도 입력 모드|목적지 \d+개 로드|손잡이: 잡기|고정 문장 워밍업)")
_LLM_FAIL = re.compile(r"\[LLM\] 호출 실패")
_NODE_LINE = re.compile(r"^\[(\w+)\] \[(\d+\.\d+)\] \[([\w.]+)\]: (.*)$")
_STREAM_PREFIX = re.compile(r"^\[vica_\w+-\d+\] ")
_JOURNAL = re.compile(r"^(\S+) \S+ [^\s\[]+\[\d+\]: (.*)$")


@dataclass(frozen=True)
class Event:
    t: float          # epoch s (노드 로그 줄의 시각)
    node: str         # wake | llm | tts | mis
    kind: str
    data: dict = field(default_factory=dict, hash=False, compare=False)
    raw: str = ""     # 로그 줄의 메시지 부분 그대로


def _f(x: str) -> float:
    return float(x)


def classify(t: float, node_full: str, msg: str) -> Optional[Event]:
    """한 줄 → Event. 분석에 안 쓰는 줄·모르는 노드는 None."""
    node = NODES.get(node_full)
    if node is None:
        return None

    def ev(kind: str, /, **data) -> Event:
        return Event(t, node, kind, data, msg)

    if _LLM_FAIL.search(msg):
        return ev("llm_fail", text=msg)
    if node == "llm" and node_full == "vica_llm":
        return None                                  # stderr 계열은 호출 실패 줄만 쓴다
    if _STARTUP.match(msg):
        return ev("startup", text=msg)
    if msg.startswith("[USB]"):
        return ev("usb", text=msg)

    if node == "wake":
        m = _WAKE_SCORE.match(msg)
        if m:
            kind = {"확정": "wake_fire", "놓침": "wake_near", "자기 말 무시": "wake_self"}[m.group(1)]
            return ev(kind, peak=_f(m.group(2)), threshold=_f(m.group(3)))
        m = _WAKE_OPEN.match(msg)
        if m:
            return ev("wake_open", n=int(m.group(1)))
        m = _WAKE_RESCUE.match(msg)
        if m:
            return ev("wake_open", rescue=True, text=m.group(1))
        if msg.startswith("귀 감시가 멈췄습니다"):
            return ev("usb", text=msg, stopped=True)
        m = _GHOST.match(msg)
        if m:
            return ev("listen_ghost", speech=_f(m.group(1)), rms=_f(m.group(2)))
        if msg == "청취 기각: empty:wake-ignored":
            return ev("wake_ignored")
        m = _REJECT.match(msg)
        if m:
            return ev("listen_reject", state=m.group(1))
        if msg.startswith("청취 창 빈손 종료"):
            return ev("listen_empty_free")
        if msg.startswith("질문 예약 —"):
            return ev("listen_reserve")
        m = _USER_TEXT.search(msg)
        if m:
            return ev("user_text", text=m.group(1))
        m = _USER_AUDIO.match(msg)
        if m:
            return ev("user_audio", sec=_f(m.group(1)))
        m = _QUALITY.match(msg)
        if m:
            return ev("listen_quality", rms=_f(m.group(1)), peak=_f(m.group(2)), clip=_f(m.group(3)))
        m = _TIMING.match(msg)
        if m:
            return ev("listen_timing", wait=_f(m.group(1)), speech=_f(m.group(2)),
                      endpoint=_f(m.group(3)), stt=_f(m.group(4)))
        m = _SPK.match(msg)
        if m:
            return ev("spk", verdict=m.group(1), score=_f(m.group(2)), threshold=_f(m.group(3)),
                      label=m.group(4), sec=_f(m.group(5)), direction=m.group(6),
                      reason=m.group(7), dropped=bool(m.group(8)))
        m = _FORWARD.match(msg)
        if m:
            return ev("forward_answer", text=m.group(1))
        return None

    if node == "llm":
        m = _INPUT.match(msg)
        if m:
            return ev("llm_input", text=m.group(1), intent=m.group(2), matched=m.group(3), safety=m.group(4))
        m = _RT.match(msg)
        if m:
            data = dict(heard=m.group(1), intent=m.group(2), dest=m.group(3), nc=m.group(4) == "True",
                        reply=m.group(5))
            tail = _RT_TAIL.search(msg)
            if tail:
                data.update(src=tail.group(1), dt=_f(tail.group(2)))
            return ev("llm_rt", **data)
        m = _RELISTEN.match(msg)
        if m:
            return ev("relisten_drop", text=m.group(1), intent=m.group(2))
        m = _QVOICE.match(msg)
        if m:
            return ev("q_voice", id=m.group(1), kind=m.group(2), age=_f(m.group(3)))
        return None

    if node == "tts":
        m = _PLAY.match(msg)
        if m:
            return ev("tts_play", priority=m.group(1), text=m.group(2))
        return None

    # mis
    m = _MSAY.match(msg)
    if m:
        return ev("mis_say", priority=m.group(1), text=m.group(2))
    m = _MREQ.match(msg)
    if m:
        return ev("mis_req", intent=m.group(1), dest=m.group(2), confirm=m.group(3) == "True",
                  **{"from": m.group(4), "to": m.group(5)}, rest=m.group(6))
    m = _MIS_WAKE.match(msg)
    if m:
        return ev("mis_wake", heard=m.group(1) == "들음", **{"from": m.group(2), "to": m.group(3)})
    m = _QOPEN.match(msg)
    if m:
        return ev("q_open", id=m.group(1), kind=m.group(2), candidates=int(m.group(3)),
                  requested=m.group(4) == "y", text=m.group(5))
    m = _QCLOSE.match(msg)
    if m:
        return ev("q_close", id=m.group(1), reason=m.group(2), open_s=_f(m.group(3)))
    m = _ESTOP_STATE.search(msg)
    if m and msg.startswith("중앙 estop="):
        return ev("state", to=m.group(1), via="estop")
    m = _ARROW.search(msg)
    if m and m.group(1) in STATES and m.group(2) in STATES:
        via = "handle" if msg.startswith("손잡이:") else "mission"
        return ev("state", **{"from": m.group(1), "to": m.group(2)}, via=via)
    return None


# ── 읽기 ─────────────────────────────────────────────────────────────

def _journal_lines(since: float, until: float) -> list[str]:
    """journal 의 `[LLM] 호출 실패` 줄(시험에서 바꿔 끼운다). 못 읽으면 빈 목록."""
    fmt = "%Y-%m-%d %H:%M:%S"
    cmd = ["journalctl", "--user", "-u", JOURNAL_UNIT, "--since", dt.datetime.fromtimestamp(since).strftime(fmt),
           "--until", dt.datetime.fromtimestamp(until).strftime(fmt), "-o", "short-iso", "--no-pager",
           "-g", "LLM\\] 호출 실패"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=JOURNAL_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.splitlines()


def _parse_stream_line(line: str):
    """relaunch·journal 줄 → ('node', t, node_full, msg) | ('stderr', msg) | None."""
    line = _STREAM_PREFIX.sub("", line.rstrip("\n"))
    p = parse_log_line(line)
    if p:
        return "node", p[0], p[1], p[2]
    if _LLM_FAIL.search(line):
        return "stderr", re.sub(r"^\[vica_\w+-\d+\] ", "", line)
    return None


def _node_log_events(log_dir: Path, since: float):
    for path in sorted(glob.glob(str(log_dir / "python*.log"))):
        try:
            if os.path.getmtime(path) < since:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            p = parse_log_line(line)
            if p:
                yield p


def _relaunch_events(relaunch_dir: Path, since: float):
    for path in sorted(glob.glob(str(relaunch_dir / "voice*_*.log"))):
        try:
            mtime = os.path.getmtime(path)
            if mtime < since:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                parsed = [_parse_stream_line(l) for l in f]
        except OSError:
            continue
        times = [x[1] for x in parsed if x and x[0] == "node"]
        prev = None
        for i, x in enumerate(parsed):
            if not x:
                continue
            if x[0] == "node":
                prev = x[1]
                yield x[1], x[2], x[3]
            else:                                    # 시각 없는 stderr — 앞 줄, 없으면 뒤 줄, 없으면 파일 mtime
                nxt = next((y[1] for y in parsed[i + 1:] if y and y[0] == "node"), None)
                t = prev if prev is not None else (nxt if nxt is not None else mtime)
                yield t, "vica_llm", x[1]


def _journal_events(since: float, until: float):
    for line in _journal_lines(since, until):
        m = _JOURNAL.match(line)
        if not m:
            continue
        x = _parse_stream_line(m.group(2))
        if not x:
            continue
        if x[0] == "node":
            yield x[1], x[2], x[3]
            continue
        try:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S%z").timestamp()
        except ValueError:
            continue
        yield t, "vica_llm", x[1]


def read_events(since: float, until: float, log_dir: Path = DEFAULT_LOG_DIR,
                relaunch_dir: Path = DEFAULT_RELAUNCH_DIR, journal: bool = True) -> list[Event]:
    """since~until(epoch s) 의 사건을 시각순으로. 같은 줄이 여러 출처에 있으면 한 번만."""
    sources = [_node_log_events(Path(log_dir), since), _relaunch_events(Path(relaunch_dir), since)]
    if journal:
        sources.append(_journal_events(since, until))
    seen: set[tuple] = set()
    out: list[Event] = []
    for src in sources:
        nth: dict[tuple, int] = {}          # 한 출처 안의 같은 줄은 번호를 붙여 서로 다른 사건으로 센다
        for t, node_full, msg in src:
            if t < since or t > until:
                continue
            base = (t, NODES.get(node_full), msg)
            nth[base] = nth.get(base, 0) + 1
            key = base + (nth[base],)       # 출처 사이에서만 같은 번호끼리 합쳐진다
            if key in seen:
                continue
            seen.add(key)
            e = classify(t, node_full, msg)
            if e is not None:
                out.append(e)
    out.sort(key=lambda e: e.t)
    return out
