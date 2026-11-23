"""저장된 재생 결과를 새 답지로 다시 채점한다 — LLM 을 다시 부르지 않는다 (설계 2026-11-20 8절, 계획 Task 1-4).

    .venv/bin/python -m tools.key_regrade --result .../replay_gpt_key-….json --key .../review2/answer_key.csv

답지가 바뀌었을 때(사용자가 남은 행을 확인함 · 판정용 새 행을 보탬) 쓴다. 결과: 같은 폴더 <이름>_regraded.json · .md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.answer_replay import _report_md, regrade
from tools.key_grade import load_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    src = Path(os.path.expanduser(args.result))
    key = load_key(os.path.expanduser(args.key))
    runs = [regrade(run, key) for run in json.loads(src.read_text(encoding="utf-8"))]
    base = src.with_name(src.stem + "_regraded")
    base.with_suffix(".json").write_text(json.dumps(runs, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    base.with_suffix(".md").write_text(_report_md(src.stem, runs, f"다시 채점 — 답지 {Path(args.key).name}"), encoding="utf-8")
    for i, run in enumerate(runs, 1):
        for set_, c in sorted(run["key"].items()):
            print(f"{i}회차 {set_}: ok {c.get('ok', 0)} reask {c.get('reask', 0)} wrong {c.get('wrong', 0)} "
                  f"mismatch {c.get('mismatch', 0)} · 버린 요청 {run['key_dropped'].get(set_, 0)} · "
                  f"확인 없이 출발 {run['key_go'].get(set_, 0)} · 미확인 {c.get('pending', 0)} · 버림 {c.get('drop', 0)}")
    print(f"→ {base}.json · .md")


if __name__ == "__main__":
    main()
