"""정답 묶음 초안 — 의심 행 전부 + kind 별 표본, 규칙으로 정할 수 있는 정답만 채운다 (설계 2026-11-14 5절).

규칙으로 못 정한 행은 label_source=todo. 사람(Claude 초안 → 사용자 확인)이 채워 labeled.csv 로 저장한다.

    .venv/bin/python -m tools.answer_label --rows .../qa_rows.csv --flagged .../qa_flagged.csv --out .../labeled_draft.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Sequence

from src.handle_mode import AFFIRMATIVES, NEGATIVES
from src.langchain_intent_parser import parse_wait_minutes

WAKE_WORDS = ("비카야", "피카야", "지카야")
EXTRA_COLUMNS = ["label_answer", "label_value", "label_source", "note"]


def _words(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", text or "")


def _destination_id(text: str, destinations: Sequence) -> str:
    """글자에 이름·별칭이 하나의 목적지만 맞으면 그 id. 다른 이름 속 이름("화장실"⊂"남자 화장실")은 뺀다."""
    squashed = re.sub(r"\s+", "", text or "")
    hits = []   # (맞은 글자, id)
    for d in destinations:
        for name in [d.name, *(getattr(d, "aliases", []) or [])]:
            key = re.sub(r"\s+", "", name or "")
            if key and key in squashed:
                hits.append((key, d.id))
    hits = [(k, i) for k, i in hits if not any(k != k2 and k in k2 for k2, _ in hits)]
    ids = {i for _, i in hits}
    return ids.pop() if len(ids) == 1 else ""


def draft_label(row: dict, destinations: Sequence) -> tuple[str, str, str]:
    """(answer, value, source). source = rule · todo."""
    text = (row.get("answer_text") or "").strip()
    words = _words(text)
    if not words:
        return "unclear", "", "rule"
    if "".join(words) in WAKE_WORDS:
        return "repeat", "", "rule"
    dest = _destination_id(text, destinations)
    if row.get("kind") == "open" and not dest and (words[0] in NEGATIVES or all(w in AFFIRMATIVES for w in words)):
        return "unclear", "", "rule"   # 열린 질문에 네/아니요는 답이 아니다 — 되묻기가 맞음
    if words[0] in NEGATIVES:
        return ("destination", dest, "rule") if dest else ("no", "", "rule")
    if all(w in AFFIRMATIVES for w in words):
        return "yes", "", "rule"
    if row.get("kind") == "minutes":
        minutes = parse_wait_minutes(text, 1.0)
        if minutes:
            return "minutes", str(minutes), "rule"
    if dest:
        return ("choice" if row.get("kind") == "choice" else "destination"), dest, "rule"
    return "", "", "todo"


def select_rows(rows: list[dict], flagged_keys: set, per_kind: int = 30, seed: int = 20261114) -> list[dict]:
    """의심 행 전부 + 나머지에서 kind 별 per_kind 개(고정 seed). 시각 순."""
    flagged = [r for r in rows if (r["date"], r["time"]) in flagged_keys]
    rest: dict = {}
    for r in rows:
        if (r["date"], r["time"]) not in flagged_keys:
            rest.setdefault(r["kind"], []).append(r)
    rng = random.Random(seed)
    picked = []
    for kind in sorted(rest):
        group = rest[kind]
        picked.extend(rng.sample(group, min(per_kind, len(group))))
    return sorted(flagged + picked, key=lambda r: float(r["ts"]))


def main() -> None:
    from src.destination_loader import load_voice_destinations

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--flagged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=30)
    args = ap.parse_args()
    expand = lambda p: Path(os.path.expanduser(p))  # noqa: E731
    rows = list(csv.DictReader(open(expand(args.rows), encoding="utf-8")))
    flagged = {(r["date"], r["time"]) for r in csv.DictReader(open(expand(args.flagged), encoding="utf-8"))
               if r.get("flags")}
    cache: dict = {}

    def dests_for(path: str):
        if path not in cache:
            try:
                cache[path] = load_voice_destinations(path)
            except (OSError, ValueError):
                cache[path] = []
        return cache[path]

    out_rows = []
    for r in select_rows(rows, flagged, per_kind=args.per_kind):
        answer, value, source = draft_label(r, dests_for(r.get("dest_yaml", "")))
        out_rows.append({**r, "label_answer": answer, "label_value": value, "label_source": source, "note": ""})
    with open(expand(args.out), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + EXTRA_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    todo = sum(1 for r in out_rows if r["label_source"] == "todo")
    print(f"행 {len(out_rows)} (의심 {sum(1 for r in out_rows if (r['date'], r['time']) in flagged)}) · todo {todo} → {args.out}")


if __name__ == "__main__":
    main()
