"""정답 묶음을 지금 구조의 해석 경로에 다시 넣어 기준 점수를 잰다 (설계 2026-11-14 5절).

    .venv/bin/python -m tools.answer_replay --path midm|gpt|realtime --labels .../labeled.csv [--rounds 3]

midm·gpt = 받아쓴 글자 → parse_intent(글자 경로). realtime = 저장 소리(~/vica_data/listen_clips) → parse_intent_audio.
결과: 정답 파일 옆 replay_<path>_<YYYYMMDD-HHMM>.json·.md. 주행 중에는 돌리지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.replies import LLM_UNAVAILABLE
from tools.answer_grade import answer_of, grade

STOP_AFTER_ERRORS = 5
CLIPS_DIR = Path.home() / "vica_data" / "listen_clips"


def _norm(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z가-힣]+", text or ""))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


MODELS = {"midm": "midm2-mini", "gpt": "gpt-5.4-mini"}


def path_env(path: str, local_rules: bool | None = None) -> dict:
    """경로별 환경 — 배포와 같은 스위치. midm = ⑬(오프라인, 로컬 규칙 켬), gpt = ⑫ 예비(규칙 끔)."""
    rules = (path == "midm") if local_rules is None else local_rules
    env = {"VICA_LLM_FALLBACK_MODEL": "", "VICA_LOCAL_RULES": "1" if rules else "0",
           "VICA_INTENT_INPUT": "audio" if path == "realtime" else "text"}
    if path == "midm":
        env.update(VICA_LLM_PROVIDER="ollama", OLLAMA_HOST="http://localhost:11434", OLLAMA_API_KEY="")
    elif path == "gpt":
        env["VICA_LLM_PROVIDER"] = "openai"
    return env


def text_for(row: dict) -> str:
    """글자 경로의 입력 = 받아쓴 글자(설계 5절). 없으면 로그의 들은 글자."""
    return (row.get("stt") or "").strip() or (row.get("answer_text") or "").strip()


def call_with_retry(fn: Callable, tries: int = 3, wait_s: float = 5.0, retry_on: str = "rate_limit"):
    """요청 한도 초과(오류 글자에 retry_on)만 wait_s 쉬고 다시. 다른 오류는 바로 올린다."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if retry_on not in str(exc) or attempt == tries - 1:
                raise
            time.sleep(wait_s * (attempt + 1))
    return None


def clip_for(row: dict, clips: list[dict], clips_dir: Path, window_s: float = 3.0) -> Optional[Path]:
    """행과 같은 말의 저장 소리. 시각이 window_s 안이고, 받아쓴 글자가 있으면 글자도 같아야 한다."""
    ts = float(row["ts"])
    want = _norm(row.get("stt", ""))
    best = None
    for c in clips:
        if c.get("result") != "text":
            continue
        gap = abs(dt.datetime.fromisoformat(c["time"]).timestamp() - ts)
        if gap > window_s or (want and _norm(c.get("text", "")) != want):
            continue
        if best is None or gap < best[0]:
            best = (gap, c)
    return Path(clips_dir) / best[1]["file"] if best else None


def replay(rows: list[dict], interpret: Callable[[dict], tuple], destinations_for: Callable[[str], Optional[Sequence]]) -> dict:
    """행마다 status(ok·reask·wrong·skip·error)·pred·latency 와 요약."""
    out_rows, errors_in_row, stopped = [], 0, False
    for row in rows:
        rec = {k: row.get(k, "") for k in ("ts", "date", "time", "kind", "question", "answer_text",
                                            "label_answer", "label_value")}
        dests = destinations_for(row.get("dest_yaml", ""))
        if dests is None:
            out_rows.append({**rec, "status": "skip", "pred": "", "pred_value": "", "latency": None, "why": "목적지 파일"})
            continue
        if not (row.get("answer_text") or "").strip():
            pred, pv, latency, why = "unclear", "", 0.0, "빈 글자"
        else:
            try:
                intent, latency = interpret(row)
            except Exception as exc:  # noqa: BLE001 — 경로 오류는 행 결과로 남긴다
                intent, latency, why = "error", 0.0, f"{type(exc).__name__}: {exc}"
            else:
                why = ""
            if intent is None:
                out_rows.append({**rec, "status": "skip", "pred": "", "pred_value": "", "latency": None, "why": "소리 없음"})
                continue
            if intent == "error" or getattr(intent, "reply", "") == LLM_UNAVAILABLE:
                errors_in_row += 1
                out_rows.append({**rec, "status": "error", "pred": "", "pred_value": "", "latency": latency,
                                 "why": why or "LLM 못 씀"})
                if errors_in_row >= STOP_AFTER_ERRORS:
                    stopped = True
                    break
                continue
            pred, pv = answer_of(intent, row["kind"], row["question"], dests)
            why = f"{intent.intent}/{intent.matched_destination_id or '-'}/nc={intent.need_confirm}"
        errors_in_row = 0
        status = grade(row["label_answer"], row.get("label_value", ""), pred, pv, row.get("kind", ""))
        out_rows.append({**rec, "status": status, "pred": pred, "pred_value": pv, "latency": latency, "why": why})
    by_kind: dict = {}
    for r in out_rows:
        by_kind.setdefault(r["kind"], Counter())[r["status"]] += 1
    lat = [r["latency"] for r in out_rows if r["status"] in ("ok", "reask", "wrong") and r["latency"]]
    return {"rows": out_rows, "counts": Counter(r["status"] for r in out_rows),
            "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "p50": _quantile(lat, 0.5), "p90": _quantile(lat, 0.9), "stopped": stopped}


def _report_md(path_name: str, runs: list[dict], setup: str = "") -> str:
    lines = [f"# 재생 결과 — {path_name} ({dt.datetime.now():%Y-%m-%d %H:%M}, {len(runs)}회)", "",
             f"설정: {setup}", "",
             "wrong = 정반대·묻지 않은 동작·조용히 버림(설계 5절). mismatch = 그 밖의 어긋남(따로 셈). "
             "한계: ros_node 후처리(재청취 기각·quiet_for_mission·identity_answer)와 미션 반응은 재현하지 않는다.", ""]
    for i, run in enumerate(runs, 1):
        c = run["counts"]
        lines.append(f"- {i}회차: ok {c.get('ok', 0)} · reask {c.get('reask', 0)} · **wrong {c.get('wrong', 0)}** · "
                     f"mismatch {c.get('mismatch', 0)} · "
                     f"skip {c.get('skip', 0)} · error {c.get('error', 0)} · p50 {run['p50']:.2f} s · p90 {run['p90']:.2f} s"
                     + (" · 중단" if run["stopped"] else ""))
    wrong_any = {(r["date"], r["time"], r["answer_text"]) for run in runs for r in run["rows"] if r["status"] == "wrong"}
    lines += ["", f"한 번이라도 wrong: {len(wrong_any)}행", "", "| kind | ok | reask | wrong | mismatch | skip | error |", "|---|---|---|---|---|---|---|"]
    for kind, c in sorted(runs[0]["by_kind"].items()):
        lines.append(f"| {kind} | {c.get('ok', 0)} | {c.get('reask', 0)} | {c.get('wrong', 0)} | {c.get('mismatch', 0)} | "
                     f"{c.get('skip', 0)} | {c.get('error', 0)} |")
    lines += ["", "## wrong 행 (1회차)", "", "| 시각 | 질문 | 대답 | 정답 | 예측 | 의도 |", "|---|---|---|---|---|---|"]
    for r in runs[0]["rows"]:
        if r["status"] == "wrong":
            lines.append(f"| {r['date']} {r['time']} | {r['question'][:30]} | {r['answer_text'][:30]} | "
                         f"{r['label_answer']} {r['label_value']} | {r['pred']} {r['pred_value']} | {r['why']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("midm", "gpt", "realtime"), required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--local-rules", choices=("auto", "on", "off"), default="auto")
    args = ap.parse_args()

    # src 의 parser 를 부르기 전에 경로를 고정한다 — import 때 .env 를 읽고 설정을 굳힌다(load_dotenv 는 있는 값을 안 덮음).
    env = path_env(args.path, None if args.local_rules == "auto" else args.local_rules == "on")
    os.environ.update(env)
    model = MODELS.get(args.path)
    setup = (f"경로 {args.path} · 모델 {model or os.environ.get('VICA_REALTIME_MODEL', 'Realtime 기본')} · "
             f"로컬 규칙 {env['VICA_LOCAL_RULES']} · 입력 {'저장 소리' if args.path == 'realtime' else '받아쓴 글자(stt, 없으면 들은 글자)'}")
    from langchain_core.messages import AIMessage

    from src.destination_loader import load_voice_destinations
    from src.langchain_intent_parser import parse_intent, parse_intent_audio
    from src.schema import RobotState

    labels_path = Path(os.path.expanduser(args.labels))
    rows = [r for r in csv.DictReader(open(labels_path, encoding="utf-8")) if r.get("label_answer")]
    cache: dict = {}

    def destinations_for(path: str):
        if path not in cache:
            try:
                cache[path] = load_voice_destinations(path) if path and Path(path).exists() else None
            except (OSError, ValueError):
                cache[path] = None
        return cache[path]

    def context(row: dict):
        dests = destinations_for(row.get("dest_yaml", ""))
        return dests, [AIMessage(row["question"])], RobotState(dialog_state=row.get("dialog_state", ""))

    def interpret_text(row: dict):
        dests, history, state = context(row)
        t0 = time.monotonic()
        intent = parse_intent(text_for(row), dests, history=history, robot_state=state, model=model)
        return intent, time.monotonic() - t0

    clips = []
    if args.path == "realtime":
        with open(CLIPS_DIR / "clips.jsonl", encoding="utf-8") as f:
            clips = [json.loads(line) for line in f if line.strip()]

    def interpret_audio(row: dict):
        wav_path = clip_for(row, clips, CLIPS_DIR, window_s=8.0)
        if wav_path is None or not wav_path.exists():
            return None, 0.0
        with wave.open(str(wav_path)) as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1:
                return None, 0.0
            pcm = w.readframes(w.getnframes())
        dests, history, state = context(row)
        intent, _heard, latency, _info = call_with_retry(
            lambda: parse_intent_audio(pcm, dests, history=history, robot_state=state))
        return intent, latency

    interpret = interpret_audio if args.path == "realtime" else interpret_text
    runs = []
    for i in range(args.rounds):
        run = replay(rows, interpret, destinations_for)
        c = run["counts"]
        print(f"[{args.path}] {i + 1}회차: ok {c.get('ok', 0)} reask {c.get('reask', 0)} wrong {c.get('wrong', 0)} "
              f"mismatch {c.get('mismatch', 0)} "
              f"skip {c.get('skip', 0)} error {c.get('error', 0)} p50 {run['p50']:.2f}s p90 {run['p90']:.2f}s", flush=True)
        runs.append(run)
        if run["stopped"]:
            print("오류가 5건 이어져 멈춤 — 경로(Ollama·인터넷)를 확인하세요", file=sys.stderr)
            break
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    base = labels_path.parent / f"replay_{args.path}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(runs, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    base.with_suffix(".md").write_text(_report_md(args.path, runs, setup), encoding="utf-8")
    print(f"→ {base}.json · .md")


if __name__ == "__main__":
    main()
