#!/usr/bin/env python3
"""귀 실기 시험의 서기 — 사람이 시험하고, 프로그램은 받아 적는다.

로그를 뒤져서 어느 사건이 몇 회차였는지 복원하는 작업이 실기 30회를
지웠다(2026-09-01). vica_mark.py 철학 그대로: 사람이 회차를 찍고,
서기는 그 회차 동안 파이프라인 토픽에서 벌어진 일을 자동으로 붙여 적는다.
시험을 대신 진행하는 기능은 없다 — 발행하는 토픽 0개, 구독만 한다.

    python3 scripts/ear_scribe.py 조기마감

엔터        = 다음 회차 시작 (이후 사건은 이 회차 소속)
글 + 엔터   = 방금 회차의 정답지(실제로 말한 내용) 기록
q + 엔터    = 종료 — CSV 원자료 + devlog 용 마크다운 표 출력

결과: ~/vica_data/ear_exams/<이름>_<시각>.csv / .md

주의: 반짝 무효화는 토픽 사건을 남기지 않는다(설계상 무이벤트).
창이 조기 마감되지 않고 6초를 채웠는지로 효과를 확인한다.
"""

import csv
import datetime
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from rcl_interfaces.msg import Log

try:  # 워크스페이스 미소싱이어도 서기는 돌아가게 한다 (의도 칸만 빈다)
    from vica_interfaces.msg import VicaIntent
except ImportError:
    VicaIntent = None

# 마감 상태 → 표에 쓸 짧은 한국어 (원문은 CSV 에 그대로 남는다)
_CLOSE_LABEL = {
    "closed": "통과",
    "empty": "빈손",
    "empty:ghost": "유령기각",
    "empty:reject": "환각기각",
    "empty:short-reject": "구제기각",
    "empty:echo": "에코기각",
}


def _close_label(state: str) -> str:
    for key, label in _CLOSE_LABEL.items():
        if state == key or state.startswith(key + " "):
            return label
    return state


class Scribe(Node):
    """구독 전용 서기. 사건을 (시각, 회차, 출처, 내용) 으로 쌓는다."""

    def __init__(self) -> None:
        super().__init__("vica_ear_scribe")
        self.lock = threading.Lock()
        self.events = []          # (epoch, trial, 출처, 내용)
        self.trial = 0            # 0 = 준비 구간 (표에서 제외, CSV 에는 남음)

        sub = self.create_subscription
        sub(String, "/vica/wake", lambda m: self._add("wake", m.data), 10)
        sub(String, "/vica/listen_state", lambda m: self._add("창", m.data), 10)
        sub(String, "/vica/user_text", lambda m: self._add("전사", m.data), 10)
        sub(String, "/vica/tts_request",
            lambda m: self._add("tts요청", m.data[:60]), 10)
        sub(String, "/vica/tts_done", lambda m: self._add("tts끝", m.data[:60]), 10)
        sub(Empty, "/vica/tts_stop", lambda m: self._add("tts중단", ""), 10)
        sub(Bool, "/vica/tts_state",
            lambda m: self._add("tts상태", "재생" if m.data else "정지"), 10)
        sub(Bool, "/vica/thinking",
            lambda m: self._add("생각", "시작" if m.data else "끝"), 10)
        sub(Bool, "/vica/listen_request",
            lambda m: self._add("청취요청", str(m.data)), 10)
        # 세부 사유(barge-in·에코 제거·긴급 관문·미션 상태 전이)는 로그로만
        # 나온다 — vica 노드의 로그를 골라 담는다.
        sub(Log, "/rosout", self._on_rosout, 10)
        if VicaIntent is not None:
            sub(VicaIntent, "/vica/intent", self._on_intent, 10)

    def _add(self, source: str, detail: str, echo: bool = True) -> None:
        with self.lock:
            self.events.append((time.time(), self.trial, source, detail))
        if echo:
            short = detail if len(detail) <= 48 else detail[:48] + "…"
            print(f"    · {source} {short}")

    def _on_intent(self, msg) -> None:
        dest = msg.destination_candidate or msg.matched_destination_id
        text = f"{msg.intent}"
        if dest:
            text += f"→{dest}"
        text += f" ({msg.confidence:.2f})"
        self._add("의도", text)

    def _on_rosout(self, msg: Log) -> None:
        if "vica" not in msg.name or msg.name == "vica_ear_scribe":
            return
        # 로그는 양이 많다 — CSV 에는 다 남기고 화면에는 띄우지 않는다.
        self._add(f"log:{msg.name}", msg.msg, echo=False)


def _windows(trial_events):
    """listen_state 사건열에서 창(개방→마감) 목록을 복원한다."""
    out, opened = [], None
    for t, _, source, detail in trial_events:
        if source != "창":
            continue
        if detail == "open":
            opened = t
        elif detail != "speech" and opened is not None:
            out.append(f"{t - opened:.1f}s→{_close_label(detail)}")
            opened = None
    if opened is not None:
        out.append("열린 채(미마감)")
    return out


def _summary_row(idx, start, answer, trial_events):
    heard = [d for _, _, s, d in trial_events if s == "전사"]
    if not heard:  # 전사가 없으면 기각 사유가 "들은 것"의 답이다
        heard = [_close_label(d) + (" " + d.split(" ", 1)[1] if " " in d else "")
                 for _, _, s, d in trial_events
                 if s == "창" and d.startswith("empty")]
    intents = [d for _, _, s, d in trial_events if s == "의도"]
    stops = sum(1 for _, _, s, _ in trial_events if s == "tts중단")
    barge = sum(1 for _, _, s, d in trial_events
                if s.startswith("log:") and "barge" in d)
    extra = []
    if stops:
        extra.append(f"stop×{stops}")
    if barge:
        extra.append(f"barge-in×{barge}")
    stamp = datetime.datetime.fromtimestamp(start).strftime("%H:%M:%S")
    cell = lambda items: " / ".join(items) if items else "—"
    return (f"| {idx} | {stamp} | {answer or '—'} | {cell(heard)} "
            f"| {cell(_windows(trial_events))} | {cell(intents)} "
            f"| {cell(extra)} |")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "시험"
    out_dir = os.path.expanduser("~/vica_data/ear_exams")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(
        out_dir, f"{name}_{datetime.datetime.now():%m%d_%H%M}")

    rclpy.init()
    scribe = Scribe()
    spin = threading.Thread(target=rclpy.spin, args=(scribe,), daemon=True)
    spin.start()

    trial_starts = {}   # 회차 -> 시작 epoch
    answers = {}        # 회차 -> 정답지 (사람이 실제로 말한 것)

    print(f"[서기] 시험 '{name}' — 기록: {base}.csv / .md")
    print("엔터=회차 시작   글+엔터=정답지 기록   q+엔터=종료\n")
    if VicaIntent is None:
        print("  (vica_interfaces 미소싱 — 의도 칸은 비게 됩니다)\n")

    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        text = line.strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            with scribe.lock:
                scribe.trial += 1
                n = scribe.trial
            trial_starts[n] = time.time()
            print(f"  [{n}회차 시작] 시험하세요")
        else:
            with scribe.lock:
                n = scribe.trial
            if n == 0:
                print("  (아직 회차 전 — 엔터로 회차를 먼저 시작하세요)")
                continue
            answers[n] = (answers[n] + " / " + text) if n in answers else text
            print(f"  [{n}회차 정답지] {answers[n]!r}")

    with scribe.lock:
        events = list(scribe.events)
        last = scribe.trial
    rclpy.shutdown()

    # CSV 원자료 — 회차 귀속 포함 전체 사건
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "시각", "회차", "출처", "내용"])
        for t, trial, source, detail in events:
            stamp = datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]
            w.writerow([f"{t:.3f}", stamp, trial, source, detail])

    # 마크다운 표 — devlog 에 그대로 붙이는 용도
    lines = [
        f"### 실기 서기 — {name} ({datetime.datetime.now():%Y-%m-%d %H:%M})",
        "",
        "| 회차 | 시각 | 정답지(사람) | 기계가 들은 것 | 창 사건 | 의도 | 특이 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in range(1, last + 1):
        trial_events = [e for e in events if e[1] == n]
        lines.append(_summary_row(
            n, trial_starts.get(n, 0), answers.get(n, ""), trial_events))
    md = "\n".join(lines) + "\n"
    with open(base + ".md", "w") as f:
        f.write(md)

    print("\n" + md)
    print(f"저장: {base}.csv (사건 {len(events)}개) / {base}.md (회차 {last}개)")


if __name__ == "__main__":
    main()
