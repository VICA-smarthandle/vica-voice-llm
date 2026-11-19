"""예약 묶음(tools/reserve_cases.py)을 운영 해석 경로에 넣어 도구 출력의 켜는 기준을 잰다 (설계 2026-11-18 7절).

    .venv/bin/python -m tools.reserve_replay --path gpt|realtime

운영 지시문 그대로(gpt = parse_intent, realtime = 소리 경로 지시문에 글자 입력)에 [지금 상황]의 진행 중인 예약 한 줄과
대화 기록(넘긴 질문 = 로봇 줄)을 붙여 부르고, 결과를 음성 노드와 같은 갈림(handle_result)에 넣어 예약 흐름이 쥔 값·
말한 문장·저장 여부로 채점한다. realtime 은 소리가 아니라 글자를 넣는다 — 소리로 이름을 얼마나 듣는지는 실기로 본다.
합격: 첫 요청 15/15 · 그 밖에서 reserve 호출 0/13 · 이어진 대답 10/10.
결과: ~/vica_data/tool_call_test_20261118/reserve_replay_<경로>_<시각>.jsonl. 주행 중에는 돌리지 않는다.

채점용 시설은 운영 시간을 하루 종일로 둔다 — LLM 이 읽은 값만 본다(운영 시간 검사는 흐름의 단위 시험이 본다).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage

from src.ledger_view import render_ledger
from src.question_record import OpenQuestion
from src.replies import LLM_UNAVAILABLE
from src.reservation_flow import ReservationFlow, ReserveCall, handle_result, join_situation, start_gate
from src.reservation_store import Facility, ReservationStore
from src.schema import DestinationData, RobotState
from tools.reserve_cases import CASES, CONFIRM, NOW, Case

OUT_DIR = Path.home() / "vica_data" / "tool_call_test_20261118"
STOP_AFTER_ERRORS = 5
FACILITIES = [Facility("세미나실", ("세미나룸",), 2, "00:00", "23:59"), Facility("상담실", (), 2, "00:00", "23:59")]
# 11-18 측정과 같은 목적지 이름. 시설 이름(세미나실·상담실)이 목적지에도 있다 — "세미나실로 가자"는 이동이어야 한다.
DESTINATION_NAMES = ("입구", "화장실", "안내소", "사회복지", "휴게실", "407호", "작업실", "민원실", "식당", "세미나실", "상담실")
DESTINATIONS = [DestinationData(id=f"d{i}", name=name, confirm_prompt=f"{name}로 안내해드릴까요?")
                for i, name in enumerate(DESTINATION_NAMES)]
STATE = RobotState(dialog_state="idle")

# (글자, 대화 기록, [지금 상황]) → (VicaIntent | ReserveCall, 걸린 초[, 모델이 적은 들린 말])
Interpret = Callable[[str, list, str], tuple]


def _grade(case: Case, result, out, passed, flow: ReservationFlow) -> list[str]:
    """기대와 어긋난 것들. 비면 맞음."""
    want, snap, problems = case.expect, flow.snapshot(), []
    called = isinstance(result, ReserveCall)
    if "reserve" in want and called != want["reserve"]:
        problems.append("reserve 호출" if called else "reserve 안 부름")
    for key in ("facility", "date", "start", "end", "name"):
        if key in want and snap[key] != want[key]:
            problems.append(f"{key} {snap[key] or '-'} ≠ {want[key] or '-'}")
    if "say" in want:
        said = out.say if out is not None else ""
        if (out is None or out.kind != "yesno") if want["say"] == CONFIRM else said != want["say"]:
            problems.append(f"문장 '{said}'")
    if "saved" in want and len(flow.store.list()) != want["saved"]:
        problems.append(f"저장 {len(flow.store.list())}건")
    if "active" in want and flow.active != want["active"]:
        problems.append("진행 상태 남음" if flow.active else "진행 상태 버려짐")
    if "intent" in want:
        if passed is None or passed.intent not in want["intent"]:
            problems.append(f"의도 {getattr(passed, 'intent', '-')}")
        elif want.get("destination") and want["destination"] not in (passed.destination_candidate or ""):
            problems.append(f"목적지 {passed.destination_candidate or '-'}")
    return problems


def run_case(case: Case, interpret: Interpret) -> dict:
    """발화 하나: 앞 상황을 만들고 → 해석 → 음성 노드와 같은 갈림 → 채점."""
    with tempfile.TemporaryDirectory() as tmp:
        flow = ReservationFlow(ReservationStore(Path(tmp) / "reservations.json"), FACILITIES, clock=lambda: NOW)
        history, asked = [], None
        for call, said in case.setup:
            asked = flow.on_call(call, said, "ok")
            history += [HumanMessage(said), AIMessage(asked.say)]     # 글자 모드의 대화 기록 — 넘긴 질문이 로봇 줄
        question = (OpenQuestion(id="replay", kind=asked.kind, text=asked.say, option_ids=(), opened_at=0.0)
                    if asked is not None else None)
        situation = join_situation(render_ledger(STATE, awaiting_answer=bool(case.setup), now_text="14:03"),
                                   flow.situation_line())
        failed = {"id": case.id, "text": case.text, "heard": "", "wall_s": 0.0, "reserve_called": False,
                  "ok": False, "error": True, "result": "", "say": "", "passed": ""}
        try:
            result, wall, *rest = interpret(case.text, history, situation)
        except Exception as exc:  # noqa: BLE001 — 경로 오류는 행 결과로 남긴다
            return {**failed, "why": f"{type(exc).__name__}: {exc}"}
        heard = rest[0] if rest else case.text      # Realtime 은 모델이 적은 들린 말로 값의 근거를 본다(운영과 같다)
        row = {"id": case.id, "text": case.text, "heard": heard, "wall_s": round(wall, 2),
               "reserve_called": isinstance(result, ReserveCall), "error": False}
        if getattr(result, "reply", "") == LLM_UNAVAILABLE:
            return {**failed, "wall_s": round(wall, 2), "why": "LLM 못 씀"}
        gate = start_gate(STATE.dialog_state, question, flow.asked_text)
        out, passed = handle_result(flow, result, heard, gate=gate, tools_used=True)
        problems = _grade(case, result, out, passed, flow)
        shown = (json.dumps({k: v for k, v in vars(result).items() if v}, ensure_ascii=False)
                 if isinstance(result, ReserveCall)
                 else f"{result.intent}/{result.destination_candidate or '-'}/nc={result.need_confirm}")
        return {**row, "ok": not problems, "why": "; ".join(problems), "result": shown,
                "say": out.say if out is not None else "", "passed": passed.intent if passed is not None else "",
                "state": flow.snapshot()}


def summarize(rows: list[dict]) -> dict:
    first = [r for r in rows if r["id"][0] in "RE"]
    other = [r for r in rows if r["id"].startswith("N")]
    follow = [r for r in rows if r["id"].startswith("F")]
    timed = [r["wall_s"] for r in rows if r.get("wall_s")]
    return {"first_ok": sum(r["ok"] for r in first), "first_total": len(first),
            "stray_reserve": sum(r["reserve_called"] for r in other), "other_ok": sum(r["ok"] for r in other),
            "other_total": len(other),
            "follow_ok": sum(r["ok"] for r in follow), "follow_total": len(follow),
            "errors": sum(bool(r.get("error")) for r in rows),
            "avg_s": round(sum(timed) / len(timed), 2) if timed else 0.0}


def summary_line(s: dict) -> str:
    return (f"첫 요청 {s['first_ok']}/{s['first_total']} · 그 밖에서 reserve 호출 {s['stray_reserve']}/{s['other_total']}"
            f"(의도 맞음 {s['other_ok']}/{s['other_total']}) · 이어진 대답 {s['follow_ok']}/{s['follow_total']}"
            f" · 평균 응답 {s['avg_s']:.2f} s" + (f" · 오류 {s['errors']}" if s["errors"] else ""))


def _gpt_interpreter() -> Interpret:
    from src.langchain_intent_parser import DEFAULT_MODEL, parse_intent

    def interpret(text: str, history: list, situation: str):
        t0 = time.monotonic()
        result = parse_intent(text, DESTINATIONS, history=history, robot_state=STATE, model=DEFAULT_MODEL,
                              situation=situation)
        return result, time.monotonic() - t0
    return interpret


def _realtime_interpreter() -> Interpret:
    from src.langchain_intent_parser import parse_intent_realtime_text
    from tools.answer_replay import call_with_retry

    def interpret(text: str, history: list, situation: str):
        result, heard, latency, _info = call_with_retry(lambda: parse_intent_realtime_text(
            text, DESTINATIONS, history=history, robot_state=STATE, situation=situation))
        return result, latency, heard
    return interpret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("gpt", "realtime"), required=True)
    ap.add_argument("--only", default="", help="쉼표로 나눈 id 머리글자·id (예: F 또는 R1,E3)")
    args = ap.parse_args()

    # 해석기를 import 하기 전에 경로를 고정한다(import 때 .env 를 읽고 설정을 굳힌다) — 208행 재생과 같은 스위치.
    from tools.answer_replay import path_env, tools_env
    os.environ.update(path_env(args.path))
    os.environ.update(tools_env(True))
    scratch = tempfile.mkdtemp(prefix="reserve_replay_")
    os.environ["VICA_RESERVATION_DIR"] = scratch      # 시설 목록은 저장소 것, 운영 예약 파일은 건드리지 않는다
    interpret = _realtime_interpreter() if args.path == "realtime" else _gpt_interpreter()

    picks = [x.strip() for x in args.only.split(",") if x.strip()]
    cases = [c for c in CASES if not picks or c.id in picks or c.id[0] in picks]
    rows, errors_in_row = [], 0
    for case in cases:
        row = run_case(case, interpret)
        rows.append(row)
        mark = "오류" if row["error"] else ("맞음" if row["ok"] else "틀림")
        print(f"{case.id:>3} {mark} {row['wall_s']:5.2f}s | {case.text} → {row['result']}"
              + (f" | {row['why']}" if row["why"] else ""), flush=True)
        errors_in_row = errors_in_row + 1 if row["error"] else 0
        if errors_in_row >= STOP_AFTER_ERRORS:
            print("오류가 5건 이어져 멈춤 — 인터넷·API 키를 확인하세요", file=sys.stderr)
            break
    s = summarize(rows)
    print(f"[{args.path}] {summary_line(s)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"reserve_replay_{args.path}_{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": s, "path": args.path}, ensure_ascii=False) + "\n")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
