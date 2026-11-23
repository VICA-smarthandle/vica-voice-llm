"""대화 장면 분석기 웹 서버 (설계 2026-11-20 5.5) — 파일만 읽고 ROS 에 안 붙는다. 표준 라이브러리만."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from tools.scene import expected as expected_mod
from tools.scene.causes import Cause, flagged
from tools.scene.events import Event
from tools.scene.scenes import Scene

REPO = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("page.html")
LISTEN_DIR = Path.home() / "vica_data" / "listen_clips"
WAKE_DIR = Path.home() / "vica_data" / "wake_clips"
CACHE_DIR = Path.home() / "vica_data" / "scene_viewer"
CLIP_WINDOW_S = 3.0
SILENT = "말소리 안 잡힘"
CHILD_TIMEOUT_S = {"rule": 90, "midm": 120, "gpt": 90, "realtime": 90}
_WAKE_CLIP = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_(near|wake)_([\d.]+)\.wav$")
_TIME_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d")


class ReaskRefused(Exception):
    """다시 묻기를 거부한 이유(주행 중 등) — 화면에 그 문구를 그대로 보인다."""


def parse_time(s: str) -> float:
    for fmt in _TIME_FMTS:
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return float(s)


def resolve_clip(name: str, dirs: Optional[tuple] = None) -> Optional[Path]:
    """이름(파일 이름만) → listen_clips·wake_clips 안의 실제 파일. 밖이거나 없으면 None."""
    if not name or "/" in name or "\\" in name or "\0" in name or name in (".", ".."):
        return None
    for d in dirs if dirs is not None else (LISTEN_DIR, WAKE_DIR):
        try:
            root = Path(d).resolve()
            p = (root / name).resolve()
        except (OSError, RuntimeError):
            continue
        if p.parent == root and p.is_file():
            return p
    return None


def _clips_in(t0: float, t1: float, dirs: tuple) -> list[dict]:
    out: list[dict] = []
    jl = Path(dirs[0]) / "clips.jsonl"
    if jl.exists():
        try:
            lines = jl.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                c = json.loads(line)
                ts = dt.datetime.fromisoformat(c["time"]).timestamp()
            except (ValueError, KeyError):
                continue
            if t0 <= ts <= t1 and (Path(dirs[0]) / c.get("file", "")).exists():
                out.append({"kind": "listen", "t": ts, "file": c["file"], "result": c.get("result"),
                            "text": c.get("text", ""), "rms": c.get("rms")})
    wd = Path(dirs[1])
    if wd.is_dir():
        for f in wd.iterdir():
            m = _WAKE_CLIP.match(f.name)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp() + int(m.group(3)) / 1000
            if t0 <= ts <= t1:
                out.append({"kind": "wake" if m.group(4) == "wake" else "near", "t": ts, "file": f.name,
                            "result": "호출 확정" if m.group(4) == "wake" else "호출 놓침", "text": "",
                            "score": float(m.group(5))})
    return sorted(out, key=lambda c: c["t"])


def ev_json(e: Event) -> dict:
    return {"t": e.t, "node": e.node, "kind": e.kind, "data": e.data, "raw": e.raw}


def _first(scene: Scene, *kinds):
    return next((e for e in scene.events if e.kind in kinds), None)


def _has(scene: Scene, *kinds) -> bool:
    return _first(scene, *kinds) is not None


def stages(scene: Scene, causes: list[Cause]) -> list[dict]:
    """8단계 띠 — 장면 사건에서 내가 끌어낸 것(pass 통과 · broken 끊김 · estimate 추정 · none 해당 없음/기록 없음)."""
    names = {c.name for c in causes if c.status != "정상"}
    ev = _first
    out = []

    def add(name, status, why=""):
        out.append({"name": name, "status": status, "why": why})

    if _has(scene, "wake_open", "wake_fire"):
        add("호출 점수", "pass", "호출 확정")
    elif _has(scene, "wake_near"):
        add("호출 점수", "broken", "점수만 나오고 확정 없음")
    else:
        add("호출 점수", "none", "질문으로 시작한 장면")
    if _has(scene, "user_text", "user_audio"):
        add("청취 창", "pass", "말이 잡힘")
    elif _has(scene, "listen_ghost", "listen_empty_free", "listen_reject") or SILENT in names:
        add("청취 창", "broken", SILENT if SILENT in names else "버려짐")
    elif _has(scene, "wake_ignored"):
        add("청취 창", "pass", "위치 알림(설계)")
    else:
        add("청취 창", "none", "기록 없음")
    if _has(scene, "user_text"):
        add("받아쓰기", "pass", ev(scene, "user_text").data.get("text", ""))
    elif _has(scene, "user_audio"):
        add("받아쓰기", "estimate", "소리만 있고 글자 줄 없음")
    else:
        add("받아쓰기", "none", "글자 없음")
    if "같은 말 두 번 전달" in names or _has(scene, "forward_answer"):
        add("긴급 관문", "broken", "질문 답으로 넘김")
    elif _has(scene, "spk") and any(e.kind == "spk" and e.data.get("dropped") for e in scene.events):
        add("긴급 관문", "broken", "화자 판정으로 버림")
    elif _has(scene, "user_text"):
        add("긴급 관문", "estimate", "걸렸다는 줄이 없음 → 통과로 추정")
    else:
        add("긴급 관문", "none", "")
    if "LLM 시간 초과" in names or _has(scene, "llm_fail"):
        add("LLM 판정", "broken", "시간 초과·호출 실패")
    elif _has(scene, "llm_input", "llm_rt"):
        e = ev(scene, "llm_input", "llm_rt")
        add("LLM 판정", "pass", f"intent={e.data.get('intent', '')}")
    elif _has(scene, "user_text"):
        add("LLM 판정", "broken", "받아쓰기 뒤 판정 줄 없음") if "원인 미분류" in names \
            else add("LLM 판정", "estimate", "판정 줄 없음(규칙·지름길일 수 있음)")
    else:
        add("LLM 판정", "none", "")
    if _has(scene, "relisten_drop") or "재청취 기각" in names:
        add("기각 여부", "broken", "재청취 기각(무의미)")
    elif _has(scene, "llm_input", "llm_rt"):
        add("기각 여부", "pass", "기각 없음")
    else:
        add("기각 여부", "none", "")
    if _has(scene, "tts_play"):
        broken = {"느린 응답", "재생 죽음", "USB 끊김"} & names
        add("TTS 대기열·재생", "broken" if broken else "pass", " · ".join(sorted(broken)) or "재생됨")
    elif _has(scene, "user_text", "mis_say"):
        add("TTS 대기열·재생", "broken", "재생 줄 없음")
    else:
        add("TTS 대기열·재생", "none", "")
    if _has(scene, "mis_req", "state", "mis_say", "mis_wake", "q_open"):
        add("미션 대화 단계", "broken" if "질문 겹침" in names else "pass", "질문 겹침" if "질문 겹침" in names else "")
    else:
        add("미션 대화 단계", "none", "")
    return out


def scene_row(s: Scene, causes: list[Cause]) -> dict:
    ut = _first(s, "user_text")
    return {"id": s.id, "start": s.start, "trigger": s.trigger, "flagged": flagged(causes),
            "user_text": ut.data.get("text", "") if ut else "",
            "causes": [{"name": c.name, "status": c.status} for c in causes]}


def _default_analyze(since: float, until: float):
    from tools.scene.tally import analyze
    return analyze(since, until)


class Service:
    def __init__(self, analyze_fn: Callable = None, reask_fn: Callable = None, rule_fn: Callable = None,
                 clip_dirs: tuple = None, cache_dir: Path = None):
        self.analyze_fn = analyze_fn or _default_analyze
        self.reask_fn = reask_fn or self._child_reask
        self.rule_fn = rule_fn or self._child_rule
        self.clip_dirs = clip_dirs or (LISTEN_DIR, WAKE_DIR)
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.lock = threading.Lock()
        self.periods: dict[tuple, tuple] = {}
        self.by_id: dict[str, tuple] = {}
        self.rules: dict[str, Optional[dict]] = {}

    def analyze(self, since: float, until: float):
        key = (round(since), round(until))
        with self.lock:
            hit = self.periods.get(key)
            if hit and (until < hit[0] - 60 or time.time() - hit[0] < 60):
                return hit[1:]
            res = self.analyze_fn(since, until)
            self.periods[key] = (time.time(), *res)
            for s in res[1]:
                self.by_id[s.id] = (s, res[2].get(s.id, []), res[0])
            return res

    def window(self, q: dict) -> tuple[float, float]:
        now = time.time()
        until = parse_time(q["until"][0]) if q.get("until", [""])[0] else now
        since = parse_time(q["since"][0]) if q.get("since", [""])[0] else until - 86400
        return since, until

    def scenes_json(self, since: float, until: float) -> dict:
        events, scenes, causes, tally = self.analyze(since, until)
        return {"since": since, "until": until, "tally": tally, "n_events": len(events),
                "scenes": [scene_row(s, causes.get(s.id, [])) for s in scenes],
                "cause_names": sorted({c.name for cs in causes.values() for c in cs})}

    def find(self, sid: str, since=None, until=None):
        if sid not in self.by_id and since is not None:
            self.analyze(since, until)
        return self.by_id.get(sid)

    def payload(self, scene: Scene, events: list[Event]) -> dict:
        """자식 프로세스에 넘기는 장면 — 장면 사건 + 그 앞의 기동·상태 줄(문턱·지도·dialog_state 를 읽는 데 필요)."""
        ctx = [e for e in events if e.kind in ("startup", "state") and e.t <= scene.end]
        return {"scene": {"id": scene.id, "start": scene.start, "end": scene.end, "trigger": scene.trigger,
                          "linked_by": scene.linked_by, "question_id": scene.question_id,
                          "events": [ev_json(e) for e in scene.events]},
                "events": [ev_json(e) for e in sorted(ctx + scene.events, key=lambda e: e.t)]}

    def _run_child(self, scene: Scene, events: list[Event], path: str):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=self.cache_dir, delete=False,
                                         encoding="utf-8") as f:
            json.dump(self.payload(scene, events), f, ensure_ascii=False)
            tmp = f.name
        try:
            p = subprocess.run([sys.executable, "-m", "tools.scene", "reask", "--scene-json", tmp, "--path", path],
                               cwd=str(REPO), env=os.environ.copy(), capture_output=True, text=True,
                               timeout=CHILD_TIMEOUT_S[path])
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{path} 다시 묻기 시간 초과") from exc
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(("{", "null"))), "")
        if p.returncode == 3:
            raise ReaskRefused(json.loads(line).get("error", "거부") if line else "거부")
        if p.returncode != 0 or not line:
            raise RuntimeError((p.stderr or "자식 프로세스 실패")[-300:])
        return json.loads(line)

    def _child_reask(self, scene, path, events):
        return self._run_child(scene, events, path)

    def _child_rule(self, scene, events):
        return self._run_child(scene, events, "rule")

    def rule(self, scene: Scene, events: list[Event]) -> Optional[dict]:
        if scene.id not in self.rules:
            try:
                r = self.rule_fn(scene, events)
                self.rules[scene.id] = asdict(r) if isinstance(r, expected_mod.Expected) else r
            except Exception as exc:
                return {"intent": None, "by": "없음", "sentence": None, "reply": None,
                        "note": f"규칙 계산 실패: {exc}"}
        return self.rules[scene.id]

    def cached_reasks(self, sid: str) -> dict:
        out = {}
        for p in expected_mod.PATHS:
            f = expected_mod.DEFAULT_CACHE_DIR / f"{sid}_{p}.json"
            if f.exists():
                try:
                    out[p] = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        return out

    def scene_json(self, sid: str, since=None, until=None) -> Optional[dict]:
        hit = self.find(sid, since, until)
        if hit is None:
            return None
        scene, causes, events = hit
        last_t = scene.events[-1].t if scene.events else scene.start
        robot = [{"t": e.t, "src": "미션" if e.kind == "mis_say" else "TTS", "priority": e.data.get("priority"),
                  "text": e.data.get("text", "")} for e in scene.events if e.kind in ("tts_play", "mis_say")]
        lane = {"wake": "호출", "llm": "LLM", "tts": "TTS", "mis": "미션"}
        names = [c.name for c in causes if c.status != "정상"]
        ut = _first(scene, "user_text")
        mode, local = expected_mod.input_mode(events, ut.t if ut else scene.start)
        return {
            "id": scene.id, "start": scene.start, "end": scene.end, "last_t": last_t, "trigger": scene.trigger,
            "linked_by": scene.linked_by, "question_id": scene.question_id,
            "user_text": ut.data.get("text", "") if ut else "",
            "causes": [asdict(c) for c in causes],
            "silent": SILENT if SILENT in names else None,
            "stages": stages(scene, causes),
            "expected": {"rule": self.rule(scene, events), "reasked": self.cached_reasks(scene.id),
                         "path_then": expected_mod.path_for(events, ut.t if ut else scene.start),
                         "mode": mode, "paths": list(expected_mod.PATHS)},
            "robot": robot,
            "events": [dict(ev_json(e), lane=lane.get(e.node, e.node)) for e in scene.events],
            "clips": _clips_in(scene.start - CLIP_WINDOW_S, last_t + CLIP_WINDOW_S, self.clip_dirs),
        }

    def reask(self, sid: str, path: str, since=None, until=None) -> dict:
        hit = self.find(sid, since, until)
        if hit is None:
            raise KeyError(sid)
        scene, _c, events = hit
        res = self.reask_fn(scene, path, events)
        return asdict(res) if isinstance(res, expected_mod.Expected) else res


CSRF_HEADER = "X-Scene-Viewer"


def make_handler(svc: Service, extra_hosts: tuple = ()):
    class H(BaseHTTPRequestHandler):
        def _allowed_hosts(self) -> set:
            port = self.server.server_address[1]
            return {f"{h}:{port}" for h in ("127.0.0.1", "localhost", *extra_hosts)}

        def _guard(self, method: str) -> bool:
            """DNS rebinding·교차 사이트 요청 차단 — 거부하면 403 을 보내고 False."""
            allowed = self._allowed_hosts()
            host = (self.headers.get("Host") or "").lower()
            origin = self.headers.get("Origin")
            why = None
            if host not in allowed:
                why = "허용되지 않은 Host"
            elif origin is not None and origin.lower() not in {f"http://{h}" for h in allowed}:
                why = "허용되지 않은 Origin"
            elif method == "POST" and self.headers.get(CSRF_HEADER) != "1":
                why = f"{CSRF_HEADER} 헤더 없음"
            if why:
                self._json(403, {"error": why})
                return False
            return True

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _route(self, method: str):
            if not self._guard(method):
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = [unquote(x) for x in u.path.strip("/").split("/")]
            try:
                if method == "GET" and u.path == "/":
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if method == "GET" and parts == ["api", "scenes"]:
                    return self._json(200, svc.scenes_json(*svc.window(q)))
                if method == "GET" and parts[:2] == ["api", "scene"] and len(parts) == 3:
                    since, until = svc.window(q) if q.get("since") else (None, None)
                    d = svc.scene_json(parts[2], since, until)
                    return self._json(200, d) if d else self._json(404, {"error": "없는 장면(기간을 먼저 여세요)"})
                if method == "GET" and parts == ["api", "clip"]:
                    name = q.get("file", [""])[0]
                    p = resolve_clip(name, svc.clip_dirs)
                    if p is None:
                        bad = (not name) or "/" in name or "\\" in name or ".." in name
                        return self._json(403 if bad else 404, {"error": "허용되지 않은 파일" if bad else "없는 파일"})
                    return self._send(200, p.read_bytes(), "audio/wav")
                if method == "POST" and parts[:2] == ["api", "reask"] and len(parts) == 3:
                    path = q.get("path", [""])[0]
                    if path not in expected_mod.PATHS:
                        return self._json(400, {"error": f"경로는 {'·'.join(expected_mod.PATHS)} 중 하나"})
                    since, until = svc.window(q) if q.get("since") else (None, None)
                    try:
                        return self._json(200, svc.reask(parts[2], path, since, until))
                    except KeyError:
                        return self._json(404, {"error": "없는 장면(기간을 먼저 여세요)"})
                    except ReaskRefused as exc:
                        return self._json(409, {"error": str(exc), "refused": True})
                    except Exception as exc:
                        return self._json(500, {"error": str(exc)})
                self._json(404, {"error": "없는 경로"})
            except ValueError as exc:
                self._json(400, {"error": f"요청 값 오류: {exc}"})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return H


def serve(host: str = "127.0.0.1", port: int = 8772, allow_hosts: tuple = ()) -> None:
    extra = tuple(h for h in (host, *allow_hosts) if h not in ("0.0.0.0", ""))
    srv = ThreadingHTTPServer((host, port), make_handler(Service(), extra))
    print(f"장면 분석기 http://{host}:{port}/  (끄기: Ctrl-C)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
