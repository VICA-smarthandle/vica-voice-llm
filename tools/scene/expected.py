"""설계상 대답 — 규칙 결과·예상 로봇 문장·LLM 다시 묻기 (설계 2026-11-20 5.2).

늘 "추정"이다: 같은 입력을 지금 코드에 다시 넣은 결과이고, 미션 로직을 돌리지 않고 의도 → 문장 표로 문장을 만든다.
- 규칙(지름길·로컬 규칙·미션 질문 대답)이 정하면 by="규칙". 소리(audio) 모드는 지름길을 안 쓰므로 None.
- LLM 을 거치는 말은 reask(사람이 버튼을 눌렀을 때만) → by="LLM 추정". 결과는 캐시한다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from tools.scene import phrases
from tools.scene.events import Event
from tools.scene.scenes import Scene, state_at

DEFAULT_CACHE_DIR = Path.home() / "vica_data" / "scene_viewer" / "reask"
LLM_NOTE = "다시 물은 결과 — 그때와 다를 수 있음"
PATHS = ("midm", "gpt", "realtime")
_MODE = re.compile(r"의도 입력 모드: (audio|text) \| 로컬 규칙: (켬|꺼짐)")
_MAP = re.compile(r"^목적지 \d+개 로드(?:\(.*?\))?: (\S+)")
_NO_REPLY_INTENTS = {"unknown", "clarify", "question", "chat", "general"}


@dataclass
class Expected:
    intent: Optional[str]
    by: str                      # 규칙 | LLM 추정 | 없음
    sentence: Optional[str]      # 예상 로봇 문장(추정)
    reply: Optional[str]         # 해석기가 낸 reply 그대로
    note: str = ""


# ── 장면에서 읽는 것 ─────────────────────────────────────────────────

def input_mode(events: list[Event], t: float) -> tuple[str, bool]:
    """시각 t 직전의 `의도 입력 모드: audio|text | 로컬 규칙: 켬|꺼짐` → (모드, 로컬 규칙). 없으면 ("", False)."""
    mode, local = "", False
    for e in sorted(events, key=lambda e: e.t):
        if e.t > t:
            break
        if e.kind == "startup":
            m = _MODE.search(e.raw or e.data.get("text", ""))
            if m:
                mode, local = m.group(1), m.group(2) == "켬"
    return mode, local


def path_for(events: list[Event], t: float) -> str:
    """그때 쓴 경로: audio → realtime, text+로컬 규칙 → midm, text+클라우드 → gpt. 모르면 ""."""
    mode, local = input_mode(events, t)
    if mode == "audio":
        return "realtime"
    if mode == "text":
        return "midm" if local else "gpt"
    return ""


def destinations_for_scene(events: list[Event], t: float, loader: Optional[Callable] = None) -> list:
    """시각 t 직전 `목적지 N개 로드…: <path>` 의 지도에서 목적지를 읽는다. 못 읽으면 []."""
    path = ""
    for e in sorted(events, key=lambda e: e.t):
        if e.t > t:
            break
        if e.kind == "startup":
            m = _MAP.match(e.raw or e.data.get("text", ""))
            if m:
                path = m.group(1)
    if not path or not Path(path).exists():
        return []
    try:
        if loader is None:
            from src.destination_loader import load_voice_destinations as loader
        return list(loader(path))
    except (OSError, ValueError):
        return []


def _user_text(scene: Scene) -> Optional[Event]:
    return next((e for e in scene.events if e.kind == "user_text"), None)


def _last_robot_say(scene: Scene, t: float) -> str:
    said = [e.data["text"] for e in scene.events
            if e.kind in ("tts_play", "mis_say") and e.t <= t and e.data.get("priority") != "ambient"]
    return said[-1] if said else ""


# ── 의도 → 문장 표 ───────────────────────────────────────────────────

def _find(destinations: list, dest_id: Optional[str]):
    return next((d for d in destinations if dest_id and d.id == dest_id), None)


def sentence_for(intent: str, dest_id: Optional[str], need_confirm: bool, state: str,
                 destinations: list) -> Optional[str]:
    """의도 → 예상 로봇 문장. 표에 없으면 None. LLM 이 문장을 짓는 의도(clarify·question 등)는 호출한 쪽이 reply 를 쓴다."""
    from src.destination_loader import _josa_euro

    dest = _find(destinations, dest_id)
    if intent == "navigate":
        if dest is None:
            return None
        if need_confirm:
            return dest.confirm_prompt or f"{dest.name}{_josa_euro(dest.name)} 안내해드릴까요?"
        return phrases.START.format(name=dest.name, josa=_josa_euro(dest.name))
    if intent == "cancel":
        return phrases.CANCEL_CONFIRM if need_confirm else phrases.CANCELED
    if intent == "pause":
        return phrases.PAUSED
    if intent == "resume" and need_confirm:
        return phrases.RESUME_CONFIRM
    if intent == "finish":
        return phrases.FINISH
    return None


def _from_intent(intent, by: str, note: str, state: str, destinations: list) -> Expected:
    kind = getattr(intent, "intent", None)
    reply = getattr(intent, "reply", "") or ""
    sentence = sentence_for(kind, getattr(intent, "matched_destination_id", None),
                            bool(getattr(intent, "need_confirm", False)), state, destinations)
    if sentence is None and reply and kind in _NO_REPLY_INTENTS:
        sentence = reply                     # LLM 이 문장을 짓는 의도 — reply 그대로
    return Expected(kind, by, sentence, reply or None, note)


# ── 규칙 ─────────────────────────────────────────────────────────────

def rule_expected(scene: Scene, destinations: list, events: Optional[list[Event]] = None,
                  state: Optional[str] = None) -> Optional[Expected]:
    """지름길·로컬 규칙·미션 질문 대답 규칙이 정하면 그 결과, 아니면(LLM 몫·소리 모드·글자 없음) None."""
    ut = _user_text(scene)
    if ut is None:
        return None
    evs = events if events is not None else scene.events
    mode, local = input_mode(evs, ut.t)
    if mode == "audio":
        return None                           # 소리 모드는 지름길을 쓰지 않는다
    if state is None:
        state = state_at(evs, ut.t)
    from langchain_core.messages import AIMessage

    from src import langchain_intent_parser as lip
    from src.schema import RobotState

    text = ut.data.get("text", "")
    last = _last_robot_say(scene, ut.t)
    history = [AIMessage(last)] if last else None
    robot_state = RobotState(dialog_state=state)
    hit = lip._shortcut_intent(text, history, destinations, dialog_state=state if local else "", local=local)
    if hit is None:
        hit = lip._mission_question_answer(text, history, robot_state, local)
    if hit is None:
        return None
    return _from_intent(hit, "규칙", "코드 규칙이 정함(LLM 안 거침) — 예상 문장은 추정", state, destinations)


# ── LLM 다시 묻기 ────────────────────────────────────────────────────

def _is_driving() -> bool:
    out = subprocess.run(["systemctl", "--user", "is-active", "vica-drive.target"],
                         capture_output=True, text=True, timeout=10)
    return out.stdout.strip() == "active"


def _default_interpret(scene: Scene, path: str, destinations: list, state: str):
    """answer_replay 와 같은 방식 — 해석기를 import 하기 전에 경로 환경값을 넣는다. realtime 은 저장 클립."""
    os.environ.update(__import__("tools.answer_replay", fromlist=["path_env"]).path_env(path))
    from tools import answer_replay as ar

    ut = _user_text(scene)
    from langchain_core.messages import AIMessage

    from src.schema import RobotState

    last = _last_robot_say(scene, ut.t)
    history = [AIMessage(last)] if last else []
    robot_state = RobotState(dialog_state=state)
    from src import langchain_intent_parser as lip

    if path != "realtime":
        return lip.parse_intent(ut.data.get("text", ""), destinations, history=history, robot_state=robot_state,
                                model=ar.MODELS[path])
    import wave

    clips_file = ar.CLIPS_DIR / "clips.jsonl"
    if not clips_file.exists():
        return None
    clips = [json.loads(line) for line in clips_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    wav = ar.clip_for({"ts": ut.t, "stt": ut.data.get("text", "")}, clips, ar.CLIPS_DIR, window_s=8.0)
    if wav is None or not Path(wav).exists():
        return None
    with wave.open(str(wav)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            return None
        pcm = w.readframes(w.getnframes())
    intent, _heard, _lat, _info = lip.parse_intent_audio(pcm, destinations, history=history, robot_state=robot_state)
    return intent


def reask(scene: Scene, path: str, interpret: Optional[Callable] = None, *,
          destinations: Optional[list] = None, events: Optional[list[Event]] = None,
          cache_dir: Optional[Path] = None, is_driving: Optional[Callable[[], bool]] = None) -> Expected:
    """같은 입력을 그 경로로 LLM 에 다시 묻는다. 캐시가 있으면 그대로. interpret(scene, path, destinations, state) → 의도|None."""
    if path not in PATHS:
        raise ValueError(f"모르는 경로: {path}")
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache = cdir / f"{scene.id}_{path}.json"
    if cache.exists():
        return Expected(**json.loads(cache.read_text(encoding="utf-8")))
    if path == "midm" and (is_driving or _is_driving)():
        raise RuntimeError("주행 중이라 로컬 다시 묻기 거부")
    ut = _user_text(scene)
    if ut is None:
        return Expected(None, "없음", None, None, "받아쓴 글자가 없어 다시 물을 수 없음")
    evs = events if events is not None else scene.events
    if destinations is None:
        destinations = destinations_for_scene(evs, ut.t)
    state = state_at(evs, ut.t)
    intent = (interpret or _default_interpret)(scene, path, destinations, state)
    if intent is None:
        result = Expected(None, "없음", None, None, "소리 없음 — 저장된 클립을 못 찾음")
    else:
        result = _from_intent(intent, "LLM 추정", f"{LLM_NOTE} (경로 {path})", state, destinations)
    cdir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=1), encoding="utf-8")
    return result
