"""음성 지연 리포트 — 주행 세션 로그에서 발화별 STT·LLM·TTS 소요시간 표를 만든다.

사용법 (주행이 끝난 뒤 아무 때나):

    python scripts/voice_latency_report.py            # ~/.ros/log 의 최신 세션
    python scripts/voice_latency_report.py --log-dir <경로>

읽는 재료 (노드가 남기는 로그 줄):
- wakeword: "호출 발화 -> /vica/user_text: '…'" (전사 완료 시각)
            "계측: 대기 …s · 발화 …s · 말끝판정 …s · STT …s" (구간 분리 —
            2026-08-28 도입. 이 줄이 없는 옛 로그는 해당 칸이 '-')
- llm:      "입력='…' -> intent=…" (판정 완료 시각)
- tts:      "발화 대기[response]" / "재생[response]" (합성 대기·재생 시작)

각 구간의 뜻:
- 대기      청취 창 열림 → 사용자가 말을 시작 (사용자 반응 시간 — 시스템 잘못 아님)
- 발화      사용자가 말한 길이
- 말끝판정  말끝 → 침묵으로 확정 (규정 0.8초 근방이 정상)
- STT       whisper 전사 (주행 중 CPU 경합 의심 구간)
- LLM       전사 발행 → intent 판정 (확인 답 단축은 0초)
- TTS합성   대기열 → 재생 시작 (문장 캐시 적중은 0초)
- 말끝→응답 사람이 체감하는 총 대기 = 말끝판정+STT+LLM+미션+TTS합성

짝짓기 오류 방지: 상한(LLM 30초·TTS 10초/30초)을 넘는 짝은 '-' 로 비운다 —
대답이 없는 발화 유형(affirm 등)이 한참 뒤 응답과 짝지어지는 허상을 막는다.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import re
import statistics

LINE = re.compile(r"\[(?:INFO|WARN)\] \[(\d+)\.(\d+)\] \[([a-z_0-9]+)\]: (.*)")
TIMING = re.compile(
    r"계측: 대기 ([\d.]+)s · 발화 ([\d.]+)s · 말끝판정 ([\d.]+)s · STT ([\d.]+)s")

# 짝짓기 상한 (초) — 넘으면 그 발화의 해당 칸을 비운다
LLM_MATCH_MAX = 30.0
TTS_QUEUE_MATCH_MAX = 10.0
TTS_PLAY_MATCH_MAX = 30.0


def parse_log(path: str) -> list[tuple[float, str]]:
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.match(line.strip())
            if m:
                out.append((int(m.group(1)) + int(m.group(2)) / 1e9, m.group(4)))
    return out


def find_session_logs(log_dir: str) -> dict[str, str]:
    """노드별 최신 로그 파일을 내용으로 식별한다 (파일명은 pid 라 못 믿는다)."""
    want = {"vica_wakeword_node": "wakeword",
            "vica_llm_intent_node": "llm",
            "vica_tts_node": "tts"}
    found: dict[str, tuple[float, str]] = {}
    for path in glob.glob(os.path.join(log_dir, "python*_*.log")):
        try:
            with open(path, errors="replace") as f:
                head = f.read(4000)
        except OSError:
            continue
        for node, key in want.items():
            if f"[{node}]" in head:
                mtime = os.path.getmtime(path)
                if key not in found or mtime > found[key][0]:
                    found[key] = (mtime, path)
    return {k: v[1] for k, v in found.items()}


def build_rows(ww, llm, tts):
    rows = []
    for i, (t1, msg) in enumerate(ww):
        if "호출 발화" not in msg:
            continue
        text = re.search(r"'(.*)'", msg).group(1)
        # 계측 줄은 발화 직후 2줄 안에서 찾는다 (수음 품질 다음)
        timing = None
        for _, nmsg in ww[i + 1:i + 4]:
            tm = TIMING.search(nmsg)
            if tm:
                timing = tuple(float(x) for x in tm.groups())
                break
        t2 = next((t for t, m in llm
                   if t1 - 0.5 < t <= t1 + LLM_MATCH_MAX and f"입력='{text}'" in m),
                  None)
        t3 = t4 = None
        if t2 is not None:
            t3 = next((t for t, m in tts
                       if t2 - 0.5 < t <= t2 + TTS_QUEUE_MATCH_MAX
                       and "발화 대기[response]" in m), None)
            if t3 is not None:
                t4 = next((t for t, m in tts
                           if t3 <= t <= t3 + TTS_PLAY_MATCH_MAX
                           and "재생[response]" in m), None)
        # 단축 판정(0초대)은 프로세스 간 로그 순서가 뒤집혀 미세 음수가 된다
        llm_sec = max(0.0, t2 - t1) if t2 is not None else None
        tts_sec = (t4 - t3) if (t3 is not None and t4 is not None) else None
        # 말끝→응답: 말끝판정+STT(계측) + 전사발행→재생 (LLM·미션·TTS 포함)
        felt = None
        if timing is not None and t4 is not None:
            felt = timing[2] + timing[3] + (t4 - t1)
        rows.append((t1, text, timing, llm_sec, tts_sec, felt))
    return rows


def fmt(v, width=6):
    return f"{v:>{width}.2f}" if v is not None else " " * (width - 1) + "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default=os.path.expanduser("~/.ros/log"))
    args = ap.parse_args()

    logs = find_session_logs(args.log_dir)
    missing = {"wakeword", "llm", "tts"} - set(logs)
    if missing:
        raise SystemExit(f"로그를 못 찾았다: {sorted(missing)} (--log-dir 확인)")

    ww, llm, tts = (parse_log(logs[k]) for k in ("wakeword", "llm", "tts"))
    rows = build_rows(ww, llm, tts)
    if not rows:
        raise SystemExit("발화 기록이 없다 — 이 세션에서 음성을 쓰지 않았다.")

    day = datetime.datetime.fromtimestamp(rows[0][0]).strftime("%Y-%m-%d")
    print(f"음성 지연 리포트 — {day} · 발화 {len(rows)}건")
    for k in ("wakeword", "llm", "tts"):
        print(f"  {k}: {os.path.basename(logs[k])}")
    print()
    head = (f"{'시각':^8} {'발화':<16} {'대기':>6} {'발화':>6} {'말끝':>6}"
            f" {'STT':>6} {'LLM':>6} {'TTS':>6} {'말끝→응답':>8}")
    print(head)
    print("─" * 76)
    cols = {k: [] for k in ("wait", "speech", "tail", "stt", "llm", "tts", "felt")}
    for t1, text, timing, llm_sec, tts_sec, felt in rows:
        hm = datetime.datetime.fromtimestamp(t1).strftime("%H:%M:%S")
        w = s = tl = st = None
        if timing:
            w, s, tl, st = timing
            cols["wait"].append(w); cols["speech"].append(s)
            cols["tail"].append(tl); cols["stt"].append(st)
        if llm_sec is not None:
            cols["llm"].append(llm_sec)
        if tts_sec is not None:
            cols["tts"].append(tts_sec)
        if felt is not None:
            cols["felt"].append(felt)
        label = text[:15]
        print(f"{hm} {label:<16} {fmt(w)} {fmt(s)} {fmt(tl)}"
              f" {fmt(st)} {fmt(llm_sec)} {fmt(tts_sec)} {fmt(felt, 8)}")
    print("─" * 76)
    med = {k: statistics.median(v) if v else None for k, v in cols.items()}
    print(f"{'중앙값':<25} {fmt(med['wait'])} {fmt(med['speech'])}"
          f" {fmt(med['tail'])} {fmt(med['stt'])} {fmt(med['llm'])}"
          f" {fmt(med['tts'])} {fmt(med['felt'], 8)}")
    mx = {k: max(v) if v else None for k, v in cols.items()}
    print(f"{'최대':<25} {fmt(mx['wait'])} {fmt(mx['speech'])}"
          f" {fmt(mx['tail'])} {fmt(mx['stt'])} {fmt(mx['llm'])}"
          f" {fmt(mx['tts'])} {fmt(mx['felt'], 8)}")

    # 병목 판정 — 시스템 몫 구간만 비교한다 (대기·발화는 사용자 몫)
    system = {"말끝판정": med["tail"], "STT": med["stt"],
              "LLM": med["llm"], "TTS합성": med["tts"]}
    known = {k: v for k, v in system.items() if v is not None}
    if known:
        worst = max(known, key=known.get)
        print(f"\n병목(중앙값 기준): {worst} {known[worst]:.2f}s")
        if med["stt"] is None:
            print("※ 계측 줄이 없는 옛 로그다 — 새 빌드로 주행하면 STT 가 분리된다.")


if __name__ == "__main__":
    main()
