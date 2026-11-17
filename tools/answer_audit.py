"""실기 로그에서 "로봇 질문 → 사용자 대답 → 읽힌 의도 → 미션 반응" 행을 뽑는다 (설계 2026-11-14 5절).

입력: ~/.ros/log 의 vica_tts_node·vica_llm_intent_node·vica_mission_manager 로그.
질문 = 대답 앞 40 s 안의 마지막 재생 줄(ambient 제외)이 "?"·"말씀해 주세요"로 끝날 때.

    .venv/bin/python -m tools.answer_audit --since 2026-10-10 --out ~/vica_data/voice_answer_audit_20261113/qa_rows.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
from pathlib import Path
from typing import Iterable, Optional

QUESTION_WINDOW_S = 40.0
REACTION_WINDOW = (-0.5, 3.0)       # 대답 줄 기준 미션 반응 줄을 찾는 범위(s)
COLUMNS = ["ts", "date", "time", "question", "q_by", "kind", "answer_text", "stt", "path", "intent", "dest_id",
           "nc", "llm_reply", "dialog_state", "mission", "dest_yaml",
           "q_id", "q_kind", "q_age_s"]      # 미션 질문 기록(설계 4.1) — 그 말이 들어올 때 열려 있던 질문

_LINE = re.compile(r"^\[(\w+)\] \[(\d+\.\d+)\] \[([\w.]+)\]: (.*)$")
_PLAY = re.compile(r"^재생\[(\w+)\]: (.*)$")
_INPUT = re.compile(r"^입력='(.*)' -> intent=(\w+) matched=(\S*) safety=(\w+)")
_RT = re.compile(r"^\[RT\] heard='(.*?)' intent=(\w+) dest=(\S+) nc=(\w+) conf=\S+ reply='(.*?)' src=")
_AB = re.compile(r"^\[A/B\] .*whisper='(.*)'$")
_MSAY = re.compile(r"^TTS\[(\w+)\]: (.*)$")
_MREQ = re.compile(r"^음성 요청 intent=(\w+) dest=(\S+) confirm=(\w+): (\w+) -> (\w+) (.*)$")
_MAP = re.compile(r"^목적지 \d+개 로드: (\S+)")                        # 미션
_MAP_VOICE = re.compile(r"^목적지 \d+개 로드\(.*?\): (\S+)")          # 음성 노드 — 해석기가 실제로 쓴 목록
_PREFIX = re.compile(r"^.*?(도착했습니다|있습니다)\.\s*")
_QVOICE = re.compile(r"^\[질문\] id=(\S+) kind=(\w+) 나이=([\d.]+)s$")            # 음성 노드 — 입력= 줄 바로 뒤
_QOPEN = re.compile(r"^\[질문\] 열림 id=(\S+) kind=(\w+) 후보=(\d+) 글자=(.*)$")    # 미션
_QCLOSE = re.compile(r"^\[질문\] 닫힘 id=(\S+) 사유=(\w+) 열린 시간=([\d.]+)s$")   # 미션
Q_LINE_WINDOW_S = 1.0               # 입력= 줄과 [질문] 줄은 같은 콜백에서 찍힌다


def kind_from_text(question: str) -> str:
    """로봇 질문 글자 → 질문 kind(설계 4.1). 맞는 것이 없으면 other."""
    q = _PREFIX.sub("", (question or "").strip())
    if q == "네?" or "어디로 가고 싶으신가요" in q or "말씀해 주세요" in q or "어디로 모실까요" in q:
        return "open"
    if " 중 어디로" in q or " 중에 어디로" in q:
        return "choice"
    if q.endswith("안내해드릴까요?"):
        return "confirm" if not q.startswith("어디") else "open"
    if "필요 없으신가요" in q:
        return "yesno_inverted"
    if "몇 분" in q:
        return "minutes"
    if any(k in q for k in ("대기할까요", "기다릴까요", "마칠까요", "안내를 받으시겠어요", "사용자님이신가요",
                            "도움이 필요하신가요", "취소할까요", "다시 출발할까요")):
        return "yesno"
    return "other"


def parse_log_line(line: str) -> Optional[tuple[float, str, str]]:
    m = _LINE.match(line.rstrip("\n"))
    return (float(m.group(2)), m.group(3), m.group(4)) if m else None


def _is_question(text: str) -> bool:
    s = text.strip()
    return s.endswith("?") or s.endswith("말씀해 주세요") or s.endswith("말씀해 주세요.")


def extract_rows(lines: Iterable[tuple[float, str, str]]) -> list[dict]:
    """(시각, 노드, 본문) 줄들 → 질문 뒤 대답 행들."""
    ev = sorted(lines, key=lambda e: e[0])
    plays, msays, mreqs, users, maps, voice_maps = [], [], [], [], [], []
    last_rt = None
    for t, node, body in ev:
        if node == "vica_tts_node":
            p = _PLAY.match(body)
            if p and p.group(1) != "ambient":
                plays.append((t, p.group(2)))
        elif node == "vica_llm_intent_node":
            vm = _MAP_VOICE.match(body)
            if vm:
                voice_maps.append((t, vm.group(1)))
                continue
            r = _RT.match(body)
            if r:
                last_rt = (t, r)
                continue
            i = _INPUT.match(body)
            if i:
                u = {"t": t, "text": i.group(1), "intent": i.group(2), "dest": i.group(3), "path": "text",
                     "nc": "", "llm_reply": "", "stt": "", "q_id": "", "q_kind": "", "q_age_s": ""}
                if last_rt and abs(t - last_rt[0]) < 0.5:
                    u.update(path="audio", nc=last_rt[1].group(4), llm_reply=last_rt[1].group(5))
                users.append(u)
                continue
            a = _AB.match(body)
            if a and users and abs(t - users[-1]["t"]) < 0.5:
                users[-1]["stt"] = a.group(1)
                continue
            qv = _QVOICE.match(body)
            if qv and users and 0 <= t - users[-1]["t"] < Q_LINE_WINDOW_S:
                users[-1].update(q_id=qv.group(1), q_kind=qv.group(2), q_age_s=qv.group(3))
        elif node == "vica_mission_manager":
            m = _MAP.match(body)
            if m:
                maps.append((t, m.group(1)))
                continue
            s = _MSAY.match(body)
            if s:
                msays.append((t, s.group(2)))
                continue
            q = _MREQ.match(body)
            if q:
                mreqs.append((t, q))

    rows, last_seen = [], {}
    for u in users:
        t = u["t"]
        if t - last_seen.get(u["text"], -1e9) < 0.5:   # 겹친 로그의 같은 말
            continue
        last_seen[u["text"]] = t
        prev = [(pt, txt) for pt, txt in plays if t - QUESTION_WINDOW_S < pt < t - 0.3]
        if not prev or not _is_question(prev[-1][1]):
            continue
        qt, qtext = prev[-1]
        react = next((q for mt, q in mreqs
                      if t + REACTION_WINDOW[0] <= mt <= t + REACTION_WINDOW[1] and q.group(1) == u["intent"]), None)
        dest_yaml = (next((p for mt, p in reversed(voice_maps) if mt <= t), "")
                     or next((p for mt, p in reversed(maps) if mt <= t), ""))
        stamp = dt.datetime.fromtimestamp(t)
        rows.append({
            "ts": round(t, 3), "date": stamp.strftime("%m-%d"), "time": stamp.strftime("%H:%M:%S"),
            "question": qtext,
            "q_by": "mission" if any(abs(mt - qt) < 6 and mtxt == qtext for mt, mtxt in msays) else "voice",
            "kind": kind_from_text(qtext), "answer_text": u["text"], "stt": u["stt"], "path": u["path"],
            "intent": u["intent"], "dest_id": "" if u["dest"] in ("", "None", "-") else u["dest"],
            "nc": u["nc"], "llm_reply": u["llm_reply"],
            "dialog_state": react.group(4) if react else "",
            "mission": f"{react.group(4)}->{react.group(5)} {react.group(6)}" if react else "",
            "dest_yaml": dest_yaml,
            "q_id": u["q_id"], "q_kind": u["q_kind"], "q_age_s": u["q_age_s"],
        })
    return rows


def question_log_summary(lines: Iterable[tuple[float, str, str]]) -> dict:
    """미션의 질문 기록 로그 요약 — 연 횟수 · 듣기 요청 횟수 · 닫힌 사유별 횟수 · 가장 오래 열린 시간(s)."""
    opened = requests = 0
    closed: dict = {}
    longest = 0.0
    for _t, node, body in lines:
        if node != "vica_mission_manager":
            continue
        if _QOPEN.match(body):
            opened += 1
        elif body.startswith("질문 발화 — 재청취 요청 발행"):
            requests += 1
        else:
            c = _QCLOSE.match(body)
            if c:
                closed[c.group(2)] = closed.get(c.group(2), 0) + 1
                longest = max(longest, float(c.group(3)))
    return {"opened": opened, "listen_requests": requests, "closed": closed, "max_open_s": longest}


def _read_logs(log_dir: Path, since: float):
    for path in glob.glob(str(log_dir / "python*.log")):
        if os.path.getmtime(path) < since or os.path.getsize(path) < 1000:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed and parsed[0] >= since:
                    yield parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-10")
    ap.add_argument("--logs", default=str(Path.home() / ".ros/log"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    since = dt.datetime.fromisoformat(args.since).timestamp()
    lines = list(_read_logs(Path(args.logs), since))
    rows = extract_rows(lines)
    out = Path(os.path.expanduser(args.out))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    kinds: dict = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"행 {len(rows)} · kind {dict(sorted(kinds.items(), key=lambda x: -x[1]))} · "
          f"목적지 파일 빈 행 {sum(1 for r in rows if not r['dest_yaml'])} → {out}")
    have = [r for r in rows if r["q_id"]]
    same = sum(1 for r in have if r["q_kind"] == r["kind"])
    q = question_log_summary(lines)
    print(f"질문 기록 있음 {len(have)} · 종류 같음 {same} · 종류 다름 {len(have) - same} · 기록 없음 {len(rows) - len(have)}")
    print(f"미션 질문 열림 {q['opened']} · 듣기 요청 {q['listen_requests']} · 닫힘 {q['closed']} · 가장 오래 열린 {q['max_open_s']:.1f} s")


if __name__ == "__main__":
    main()
