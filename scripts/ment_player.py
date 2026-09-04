#!/usr/bin/env python3
"""구워 둔 멘트를 골라서 바로 트는 도구 — 영상 촬영용.

    .venv/bin/python scripts/ment_player.py

한글을 그냥 치면 걸러지고, ↑↓ 로 고르고, Enter 로 재생한다.
재생 중에 아무 키나 누르면 멈춘다.

⚠️ 음성 스택(ros_tts_node·효과음 노드)이 떠 있으면 스피커를 선점해
   재생이 실패한다. 그때는 아래에 빨간 줄로 알려준다 — 스택을 내리고 쓴다.
"""
from __future__ import annotations

import curses
import json
import sys
import time
import unicodedata
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BAKED = ROOT / "assets" / "baked"

# 파일 이름 앞머리 -> 사람이 읽는 갈래. 위에서부터 먼저 맞는 것을 쓴다.
GROUPS = [
    ("handle_grip_", "핸들"),
    ("approach_", "접근"),
    ("dest_", "목적지"),
    ("mission_msg_", "미션"),
    ("reply_", "응답"),
    ("wake_", "호출"),
    ("cue_", "효과음"),
]


def group_of(filename: str) -> str:
    for prefix, label in GROUPS:
        if filename.startswith(prefix):
            return label
    return "기타"


def width(text: str) -> int:
    """터미널에서 차지하는 칸 수 (한글은 두 칸)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def clip(text: str, limit: int) -> str:
    """limit 칸에 맞춰 자른다 (한글 두 칸 계산)."""
    if width(text) <= limit:
        return text
    out, used = "", 0
    for c in text:
        w = 2 if unicodedata.east_asian_width(c) in "WF" else 1
        if used + w > limit - 1:
            return out + "…"
        out += c
        used += w
    return out


def load_ments() -> list[dict]:
    """manifest 와 실제 파일을 합쳐 목록을 만든다."""
    manifest = {}
    mpath = BAKED / "manifest.json"
    if mpath.exists():
        manifest = json.load(open(mpath))

    items = []
    for path in sorted(BAKED.glob("*.wav")):
        text = manifest.get(path.name, "")
        try:
            info = sf.info(str(path))
            seconds = info.frames / info.samplerate
        except Exception:
            seconds = 0.0
        items.append({
            "file": path.name,
            "path": path,
            "text": text or f"(manifest 에 없음) {path.stem}",
            "group": group_of(path.name),
            "sec": seconds,
            "key": f"{text} {path.stem} {group_of(path.name)}".lower(),
        })
    # 갈래끼리 모으되 갈래 안에서는 파일 이름 순
    order = {label: i for i, (_, label) in enumerate(GROUPS)}
    items.sort(key=lambda d: (order.get(d["group"], 99), d["file"]))
    return items


class Player:
    """sounddevice 를 늦게 불러온다 — 목록만 볼 때는 오디오 장치를 안 연다."""

    def __init__(self) -> None:
        self.sd = None
        self.error = ""

    def play(self, path: Path) -> bool:
        try:
            if self.sd is None:
                import sounddevice
                self.sd = sounddevice
            wav, rate = sf.read(str(path))
            self.sd.play(wav, rate)
            return True
        except Exception as exc:
            msg = str(exc).strip().splitlines()[0] if str(exc) else exc.__class__.__name__
            self.error = f"재생 실패: {msg} — 음성 스택이 스피커를 잡고 있는지 보세요"
            return False

    def stop(self) -> None:
        if self.sd is not None:
            try:
                self.sd.stop()
            except Exception:
                pass

    def busy(self) -> bool:
        if self.sd is None:
            return False
        try:
            return self.sd.get_stream().active
        except Exception:
            return False


def run(stdscr, items: list[dict]) -> None:
    curses.use_default_colors()
    for i, fg in enumerate((curses.COLOR_YELLOW, curses.COLOR_CYAN,
                            curses.COLOR_RED, curses.COLOR_GREEN), start=1):
        curses.init_pair(i, fg, -1)
    C_ACCENT, C_DIM, C_ERR, C_OK = (curses.color_pair(i) for i in (1, 2, 3, 4))

    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)

    def put(y: int, x: int, text: str, attr: int = 0) -> None:
        """마지막 칸에 쓰면 curses 가 ERR 를 낸다 — 잘라 쓰고 실패는 삼킨다."""
        h, w = stdscr.getmaxyx()
        if not (0 <= y < h) or x >= w - 1:
            return
        try:
            stdscr.addnstr(y, x, text, max(0, w - 1 - x), attr)
        except curses.error:
            pass

    player = Player()
    query = ""
    cursor = 0
    top = 0
    status = "한글을 치면 걸러집니다.  ↑↓ 고르고  Enter 재생"
    status_attr = C_DIM
    playing = ""

    while True:
        shown = [d for d in items if query.lower() in d["key"]] if query else items
        if not shown:
            cursor = top = 0
        else:
            cursor = max(0, min(cursor, len(shown) - 1))

        h, w = stdscr.getmaxyx()
        body = max(1, h - 5)
        if cursor < top:
            top = cursor
        elif cursor >= top + body:
            top = cursor - body + 1

        stdscr.erase()

        # 머리
        head = f" 멘트 재생기 · 총 {len(items)}개"
        if query:
            head += f" · 걸러짐 {len(shown)}개"
        put(0, 0, head.ljust(max(0, w - 1)), C_ACCENT | curses.A_BOLD)

        # 찾기 줄
        put(1, 0, f" 찾기: {query}▏".ljust(max(0, w - 1)), curses.A_BOLD)

        # 목록
        sec_w = 7
        for row, d in enumerate(shown[top:top + body]):
            y = 2 + row
            picked = (top + row) == cursor
            tag = f" {d['group']:<4}"
            text_room = max(8, w - width(tag) - sec_w - 3)
            line = f"{tag} {clip(d['text'], text_room)}"
            attr = curses.A_REVERSE | curses.A_BOLD if picked else 0
            put(y, 0, line.ljust(max(0, w - 1 - sec_w))[:w], attr)
            put(y, max(0, w - 1 - sec_w), f"{d['sec']:>5.1f}s ",
                attr if picked else C_DIM)

        if not shown:
            put(2, 1, "걸러진 결과가 없습니다 — ESC 로 지우세요", C_DIM)

        # 상태 줄
        if playing and player.busy():
            put(h - 2, 0, f" ▶ 재생 중: {clip(playing, max(4, w - 16))}"
                .ljust(max(0, w - 1)), C_OK | curses.A_BOLD)
        else:
            put(h - 2, 0, f" {status}".ljust(max(0, w - 1)), status_attr)

        put(h - 1, 0, " Enter 재생   Space 다시   ESC 지우기/끝   Ctrl+C 끝"
            .ljust(max(0, w - 1)), C_DIM)
        stdscr.refresh()

        # 재생 중에는 짧게 깨어나 상태 줄을 갱신한다
        stdscr.timeout(200 if player.busy() else -1)
        try:
            ch = stdscr.get_wch()
        except curses.error:
            continue                      # timeout — 화면만 다시 그린다
        except KeyboardInterrupt:
            break

        # 재생 중 아무 키나 -> 정지
        if player.busy():
            player.stop()
            playing = ""
            status, status_attr = "정지했습니다.", C_DIM
            if ch in ("\n", " ", curses.KEY_ENTER):
                continue

        if ch == curses.KEY_UP:
            cursor -= 1
        elif ch == curses.KEY_DOWN:
            cursor += 1
        elif ch == curses.KEY_PPAGE:
            cursor -= body
        elif ch == curses.KEY_NPAGE:
            cursor += body
        elif ch == curses.KEY_HOME:
            cursor = 0
        elif ch == curses.KEY_END:
            cursor = len(shown) - 1
        elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
            query = query[:-1]
        elif ch == "\x15":                                  # Ctrl+U
            query = ""
        elif ch == "\x1b":                                  # ESC
            if query:
                query = ""
            else:
                break
        elif ch in ("\n", "\r", curses.KEY_ENTER, " "):
            if shown:
                d = shown[cursor]
                if player.play(d["path"]):
                    playing = d["text"]
                    status, status_attr = "", C_DIM
                else:
                    playing = ""
                    status, status_attr = player.error, C_ERR
        elif isinstance(ch, str) and ch.isprintable():
            query += ch

        if shown:
            cursor = max(0, min(cursor, len(shown) - 1))

    player.stop()


def main() -> None:
    if not BAKED.exists():
        sys.exit(f"구운 멘트 폴더가 없습니다: {BAKED}")
    items = load_ments()
    if not items:
        sys.exit(f"재생할 wav 가 없습니다: {BAKED}")
    try:
        curses.wrapper(run, items)
    except KeyboardInterrupt:
        pass
    print(f"멘트 재생기 종료 ({len(items)}개 목록)")


if __name__ == "__main__":
    main()
