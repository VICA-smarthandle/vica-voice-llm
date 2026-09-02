#!/usr/bin/env python3
"""귀 실기 시험의 서기 — 사람이 시험하고, 프로그램은 받아 적는다.

로그를 뒤져서 어느 사건이 몇 회차였는지 복원하는 작업이 실기 30회를
지웠다(2026-09-01). vica_mark.py 철학 그대로: 사람이 회차를 찍고,
서기는 그 회차 동안 파이프라인 토픽에서 벌어진 일을 자동으로 붙여 적는다.
시험을 대신 진행하는 기능은 없다 — 발행하는 토픽 0개, 구독만 한다.

    python3 scripts/ear_scribe.py            # 시험 목록과 순서를 본다
    python3 scripts/ear_scribe.py 조기마감    # 대본 모드 (회차마다 대사 표시)
    python3 scripts/ear_scribe.py 아무이름     # 자유 모드 (대본 없이 기록만)

엔터        = 다음 회차 시작 (이후 사건은 이 회차 소속) — 대본 모드면 대사 표시
글 + 엔터   = 방금 회차의 정답지(실제로 말한 내용) 기록
. + 엔터    = 대본대로 말했다는 뜻 (대본 문장이 정답지로 들어간다)
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
import unicodedata

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Empty, String
    from rcl_interfaces.msg import Log
except ImportError:  # ROS 미소싱 터미널 (venv 여부와 무관 — rclpy 는 /opt/ros 에 있다)
    sys.exit(
        "rclpy 를 찾지 못했습니다 — 이 터미널에 ROS 를 소싱하세요:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source ~/VICA-smarthandle/vica_ros2_ws/install/setup.bash\n"
        "(서기는 토픽만 듣습니다. 음성 venv 는 필요 없습니다.)")

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


# ---------------------------------------------------------------- 시험 대본
#
# 실기 중에 "이번엔 뭐라고 말하지?" 를 기억에 맡기면 회차마다 대사가 흔들려
# 표본이 섞인다. 대본을 여기 고정해 두고 회차마다 화면에 띄운다.
# 순서 근거: 귀가 깨져 있으면 주행 시험은 헛수고다 — 제자리 시험을 먼저 건다.
ORDER = [
    ("호출", "제자리", "웨이크워드와 '네?' 가 매번 나오는가"),
    ("조기마감", "제자리", "뜸 들여 말해도 창이 안 닫히는가 (6초 복원 확인)"),
    ("에코", "제자리", "로봇 말이 자기 말에 끊기지 않는가 (barge-in 0)"),
    ("짧은답", "바퀴 띄움", "네·응·어 같은 한마디가 어느 층에서 죽는가"),
    ("주행", "바닥 주행", "3컷 종단 — 사람접근 / 주행-도착 / 도착 후 대기"),
]

_CANCEL_TAIL = "확인 질문이 오면 '아니' 로 취소한다 (주행 방지)"


def _rep(step: dict, times: int) -> list:
    return [dict(step) for _ in range(times)]


def _short_answer_steps() -> list:
    """짧은 답 6단어 × 5회. 단어를 5회씩 묶는다 — 회차마다 갈아타면
    사람이 헷갈려 대사가 섞이고, 그게 곧 표본 오염이다."""
    steps = []
    for word in ("네", "응", "어", "그래", "아니", "대기해줘"):
        drives = word in ("네", "응", "어", "그래")
        for i in range(5):
            steps.append({
                "제목": f"짧은 답 '{word}' ({i + 1}/5)",
                "말": f'"비카야"  →  "화장실로 가자"  →  (질문 뒤) "{word}"',
                "기대": f"질문 창에서 '{word}' 접수 (전사 + 의도)",
                "주의": ("수락되면 주행 시작 — 바퀴 띄움 확인, 시작되면 "
                         "'비카야 → 취소해줘' 로 되돌린다") if drives
                        else "거절되어 제자리 유지",
                "정답지": word,
            })
    return steps


PLANS = {
    "호출": {
        "제목": "호출 응답 — '네?' 가 매번 나오는가",
        "준비": "로봇 제자리. 바퀴는 바닥이어도 된다 (주행 없음).",
        "steps": _rep({
            "제목": "호출",
            "말": '"비카야"  (그리고 아무 말도 하지 않는다)',
            "기대": "wake → '네?' 재생 → 6초 뒤 빈손 종료",
            "주의": "'네?' 가 안 들리면 그 회차가 증발 사례다",
            "정답지": "비카야 (침묵)",
        }, 5),
    },
    "조기마감": {
        "제목": "조기 마감 — 뜸 들여도 창이 버티는가",
        "준비": "로봇 제자리. 확인 질문에는 '아니' 로 답해 주행을 막는다.",
        "steps": _rep({
            "제목": "뜸 들이고 명령",
            "말": '"비카야"  →  (속으로 둘 셋 센 뒤)  →  "화장실로 가자"',
            "기대": "창이 2초 침묵을 견디고 명령을 접수",
            "주의": _CANCEL_TAIL + " · 창 사건이 2.5s 내 마감이면 조기 마감",
            "정답지": "비카야 (2초) 화장실로 가자",
        }, 5),
    },
    "에코": {
        "제목": "에코 barge-in — 로봇이 자기 말에 안 끊기는가",
        "준비": "로봇 제자리. 로봇이 말하는 동안 사람은 완전히 침묵한다.",
        "steps": _rep({
            "제목": "멘트 중 침묵",
            "말": '"비카야"  →  "화장실로 가자"  →  로봇 멘트 동안 침묵',
            "기대": "멘트가 끝까지 재생 (특이 칸에 barge-in 0)",
            "주의": _CANCEL_TAIL,
            "정답지": "비카야 화장실로 가자 (이후 침묵)",
        }, 5),
    },
    "짧은답": {
        "제목": "짧은 발화 — 한마디가 어느 층에서 죽는가",
        "준비": "⚠ 바퀴를 띄운다. 긍정 답이 통과하면 로봇이 실제로 출발한다.",
        "steps": _short_answer_steps(),
    },
    "주행": {
        "제목": "3컷 종단 — 사람접근 / 주행-도착 / 도착 후 대기",
        "준비": "바닥 주행. AMCL 초기 위치를 먼저 찍는다.",
        "steps": [
            {"제목": "① 사람접근 컷", "말": "핸들 앞에 서서 로봇 질문에 답한다",
             "기대": "질문 → 수락 → 온보딩", "주의": "", "정답지": ""},
            {"제목": "② 주행-도착 컷", "말": '"비카야"  →  "안내소로 가줘"',
             "기대": "주행 시작 → 도착", "주의": "", "정답지": "비카야 안내소로 가줘"},
            {"제목": "③ 도착 후 대기 컷", "말": '"비카야"  →  "대기해줘"',
             "기대": "wait 접수", "주의": "최다 오류 지점 — 씹히면 그대로 기록",
             "정답지": "비카야 대기해줘"},
        ],
    },
}


def _pad(text: str, width: int) -> str:
    """한글은 터미널에서 두 칸을 먹는다 — 글자 수로 맞추면 표가 어긋난다."""
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in text)
    return text + " " * max(0, width - used)


def _print_order(current: str = "") -> None:
    print("전체 시험 순서 (귀부터 걸러야 주행 시험이 헛돌지 않는다)")
    for i, (key, place, why) in enumerate(ORDER, 1):
        mark = "▶" if key == current else " "
        print(f" {mark} {i}. {_pad(key, 10)}{_pad('[' + place + ']', 13)}{why}")
    print()


def _show_step(idx: int, total: int, step: dict) -> None:
    bar = "━" * 62
    print(f"\n{bar}")
    print(f" [{idx}/{total}]  {step['제목']}")
    print(f"   말할 것 : {step['말']}")
    print(f"   기대    : {step['기대']}")
    if step.get("주의"):
        print(f"   주의    : {step['주의']}")
    print(f"{bar}")


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


def _summary_row(idx, start, answer, trial_events, title=""):
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
    head = f"| {idx} | {title} " if title else f"| {idx} "
    return (f"{head}| {stamp} | {answer or '—'} | {cell(heard)} "
            f"| {cell(_windows(trial_events))} | {cell(intents)} "
            f"| {cell(extra)} |")


def main() -> None:
    if len(sys.argv) < 2:
        _print_order()
        print("쓰는 법:  python3 scripts/ear_scribe.py <시험이름>")
        print("  대본 있는 이름: " + " / ".join(PLANS))
        print("  그 밖의 이름은 자유 모드 (대본 없이 기록만)")
        return

    name = sys.argv[1]
    plan = PLANS.get(name)
    steps = plan["steps"] if plan else []
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

    if plan:
        _print_order(name)
        print(f"■ {plan['제목']}")
        print(f"  준비: {plan['준비']}")
        print(f"  회차: {len(steps)}회\n")
    print(f"[서기] 시험 '{name}' — 기록: {base}.csv / .md")
    print("엔터=회차 시작   글+엔터=정답지 기록"
          + ("   .+엔터=대본대로 함" if plan else "") + "   q+엔터=종료\n")
    if VicaIntent is None:
        print("  (vica_interfaces 미소싱 — 의도 칸은 비게 됩니다)\n")
    if plan:
        print("엔터를 치면 1회차 대사가 나옵니다.")

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
            if steps and n <= len(steps):
                _show_step(n, len(steps), steps[n - 1])
            elif steps:
                print(f"  [{n}회차 — 대본 끝, 추가 회차] 시험하세요")
            else:
                print(f"  [{n}회차 시작] 시험하세요")
        else:
            with scribe.lock:
                n = scribe.trial
            if n == 0:
                print("  (아직 회차 전 — 엔터로 회차를 먼저 시작하세요)")
                continue
            # '.' 은 "대본대로 말했다" 는 뜻 — 실기 중 타이핑을 줄인다.
            if text == "." and steps and n <= len(steps):
                text = steps[n - 1].get("정답지") or steps[n - 1]["말"]
            answers[n] = (answers[n] + " / " + text) if n in answers else text
            print(f"  [{n}회차 정답지] {answers[n]!r}")
            if steps and n < len(steps):
                nxt = steps[n]["제목"]
                print(f"  (엔터 → 다음 {n + 1}/{len(steps)}: {nxt})")

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
    head_extra = "대본 | " if plan else ""
    dash_extra = "--- | " if plan else ""
    lines = [
        f"### 실기 서기 — {name} ({datetime.datetime.now():%Y-%m-%d %H:%M})",
        "",
        (f"| 회차 | {head_extra}시각 | 정답지(사람) | 기계가 들은 것 "
         f"| 창 사건 | 의도 | 특이 |"),
        f"| --- | {dash_extra}--- | --- | --- | --- | --- | --- |",
    ]
    for n in range(1, last + 1):
        trial_events = [e for e in events if e[1] == n]
        title = steps[n - 1]["제목"] if steps and n <= len(steps) else ""
        lines.append(_summary_row(
            n, trial_starts.get(n, 0), answers.get(n, ""), trial_events, title))
    md = "\n".join(lines) + "\n"
    with open(base + ".md", "w") as f:
        f.write(md)

    print("\n" + md)
    print(f"저장: {base}.csv (사건 {len(events)}개) / {base}.md (회차 {last}개)")


if __name__ == "__main__":
    main()
