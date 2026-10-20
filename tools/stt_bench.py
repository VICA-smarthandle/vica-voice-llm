"""STT 비교기 — 휴대폰 버튼을 누르는 동안 reSpeaker 소리를 turbo(⑫ 의 whisper)와 Qwen3-ASR 0.6B 에 넣어 나란히 본다."""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import statistics
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path.home() / "vica_data" / "stt_bench"
PAGE = Path(__file__).with_name("stt_bench_page.html")
DEFAULT_PORT = 8771
TRANSCRIBE_DIR = Path.home() / "tools" / "transcribe.cpp"
QWEN_MODEL = Path.home() / "tools" / "models" / "qwen3asr" / "Qwen3-ASR-0.6B-Q8_0.gguf"
QWEN_LANG = "ko"
MAX_TAKE_SEC = 20.0
RATE = 16000


SHORT, LONG, STOP, ROBOT_QUIET, ROBOT_STOP, FREE = "short", "long", "stop", "robot_quiet", "robot_stop", "free"
CATEGORY_KO = {SHORT: "짧은 대답", LONG: "긴 대답", STOP: "멈춰", ROBOT_QUIET: "로봇 말할 때 가만히",
               ROBOT_STOP: "로봇 말할 때 멈춰", FREE: "자유"}


@dataclass(frozen=True)
class Item:
    id: str
    cat: str
    say: str
    accept: tuple = ()
    hint: str = ""


def _short(word: str, *more: str) -> Item:
    return Item(f"short_{word}", SHORT, word, (word, *more))


def _long(text: str, *more: str) -> Item:
    return Item(f"long_{normalize(text)}", LONG, text, (text, *more))


def normalize(text: str) -> str:
    """띄어쓰기·문장부호·기호를 뺀 글자만(한글·영문·숫자)."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text or "")


SCRIPT: tuple = (
    _short("그래"), _short("아니"), _short("네", "예"), _short("응", "어"), _short("아니요", "아니오"),
    _short("좋아"), _short("싫어"), _short("맞아"), _short("괜찮아"), _short("됐어", "됬어"),
    _long("407호로 가자", "사백칠호로 가자"), _long("화장실 가고 싶어"), _long("엘리베이터 어디 있어"),
    _long("안내 취소해 줘", "안내 취소해줘"), _long("다시 출발하자"), _long("여기서 기다려 줘"),
    _long("식당까지 얼마나 걸려"), _long("테스트 1로 가자", "테스트 일로 가자"),
    Item("stop_normal", STOP, "멈춰", ("멈춰",), "평소 목소리로"),
    Item("stop_quiet", STOP, "멈춰", ("멈춰",), "작게"),
    Item("stop_loud", STOP, "멈춰!", ("멈춰",), "크게 외치듯"),
    Item("stop_jeongji", STOP, "정지", ("정지",), "평소 목소리로"),
    *(Item(f"robot_quiet_{i}", ROBOT_QUIET, "(가만히)", (),
           "[로봇 말하기]를 누르고, 로봇이 말하는 동안 녹음 버튼을 누른 채 아무 말도 하지 마세요")
      for i in (1, 2, 3)),
    *(Item(f"robot_stop_{i}", ROBOT_STOP, "멈춰", ("멈춰",),
           "[로봇 말하기]를 누르고, 로봇이 말하는 도중에 녹음 버튼을 누른 채 \"멈춰\"")
      for i in (1, 2, 3)),
)
FREE_ITEM = Item("free", FREE, "(자유)", (), "아무 말이나 — 채점하지 않습니다")


def build_order(seed: int = 7) -> list:
    """묶음이 섞이게 고정 순서로 섞는다(같은 seed 면 매번 같은 순서 — 사람끼리 비교하기 좋게)."""
    items = list(SCRIPT)
    random.Random(seed).shuffle(items)
    return items


def judge(item: Item, text: str) -> Optional[bool]:
    """정답이면 True, 아니면 False, 채점 안 하는 문항(자유)은 None."""
    got = normalize(text)
    if item.cat == FREE:
        return None
    if item.cat == ROBOT_QUIET:
        return got == ""
    if item.cat in (STOP, ROBOT_STOP):
        return any(normalize(a) in got for a in item.accept)
    return any(got == normalize(a) for a in item.accept)


def summarize(records: Sequence[dict]) -> dict:
    """묶음별 정답 수·문항 수와 모델별 '버튼 뗀 뒤 글자까지' 중앙값. 같은 문항을 다시 했으면 마지막 것만 센다."""
    last: dict = {}
    for r in records:
        if r.get("cat") != FREE:
            last[r["item"]] = r
    out: dict = {"by_cat": {}, "sec": {}}
    for cat in (SHORT, LONG, STOP, ROBOT_QUIET, ROBOT_STOP):
        rows = [r for r in last.values() if r["cat"] == cat]
        out["by_cat"][cat] = {"name": CATEGORY_KO[cat], "n": len(rows),
                              "turbo": sum(1 for r in rows if r["turbo"]["ok"]),
                              "qwen": sum(1 for r in rows if r["qwen"]["ok"])}
    for model in ("turbo", "qwen"):
        secs = [r[model]["sec"] for r in records if r.get(model, {}).get("sec") is not None]
        out["sec"][model] = round(statistics.median(secs), 3) if secs else None
    return out


@dataclass
class Take:
    item: Item
    t_start: float
    frames: list = field(default_factory=list)
    robot_clip: str = ""


def peak_db(audio: np.ndarray, win: int = 320) -> Optional[float]:
    """20 ms 조각 RMS 가운데 가장 큰 것(dBFS). 나중에 '소리 크기 문턱으로 빈 녹음을 거를 수 있나'를 보려고 남긴다."""
    n = len(audio) // win
    if n == 0:
        return None
    fr = audio[: n * win].astype(np.float64).reshape(n, win)
    return round(float(20 * np.log10(np.sqrt((fr ** 2).mean(1)).max() / 32768 + 1e-9)), 1)


class Bench:
    def __init__(self, server, mic, turbo, qwen, clips: list, record_dir: Path, play, stop_play):
        self.server, self.mic, self.turbo, self.qwen = server, mic, turbo, qwen
        self.clips, self.record_dir, self._play, self._stop_play = clips, record_dir, play, stop_play
        self.order = build_order()
        self.pos = 0
        self.free = False
        self.records: list = []
        self.take: Optional[Take] = None
        self.done = False
        self._robot_i = 0
        self._robot_playing = False
        self._robot_last = ""
        self._log = (record_dir / "results.jsonl").open("a", encoding="utf-8")

    @property
    def item(self) -> Item:
        return FREE_ITEM if self.free else self.order[self.pos]

    def publish_item(self) -> None:
        it = self.item
        self.server.publish({"type": "item", "id": it.id, "cat": it.cat, "cat_ko": CATEGORY_KO[it.cat],
                             "say": it.say, "hint": it.hint, "pos": self.pos + 1, "total": len(self.order),
                             "free": self.free, "robot": it.cat in (ROBOT_QUIET, ROBOT_STOP)}, keep="item")

    def publish_summary(self) -> None:
        self.server.publish({"type": "summary", **summarize(self.records)}, keep="summary")

    def handle(self, cmd: dict) -> None:
        c = cmd.get("cmd")
        if c == "start":
            self.start_take()
        elif c == "stop":
            self.stop_take()
        elif c == "next" and self.take is None:
            self.free = False
            self.pos = min(self.pos + 1, len(self.order) - 1)
            self.publish_item()
        elif c == "prev" and self.take is None:
            self.free = False
            self.pos = max(self.pos - 1, 0)
            self.publish_item()
        elif c == "free" and self.take is None:
            self.free = not self.free
            self.publish_item()
        elif c == "robot":
            self.play_robot()
        elif c == "robot_stop":
            self._stop_play()
        elif c == "finish":
            self.done = True

    def play_robot(self) -> None:
        if self._robot_playing or not self.clips:
            return
        clip = self.clips[self._robot_i % len(self.clips)]
        self._robot_i += 1
        self._robot_playing = True
        self._robot_last = clip.name
        self.server.publish({"type": "robot", "playing": True, "text": clip.text})

        def run() -> None:
            try:
                self._play(clip.pcm, clip.rate)
            except Exception as exc:  # noqa: BLE001 — 재생 실패는 화면에 알리고 계속 간다
                self.server.publish({"type": "toast", "text": f"로봇 재생 실패: {exc}"})
            finally:
                self._robot_playing = False
                self.server.publish({"type": "robot", "playing": False})
        threading.Thread(target=run, name="robot-play", daemon=True).start()

    def start_take(self) -> None:
        if self.take is not None:
            return
        while True:
            try:
                self.mic.frames.get_nowait()
            except queue.Empty:
                break
        self.take = Take(self.item, time.monotonic(),
                         robot_clip=self._robot_last if self._robot_playing else "")
        self.server.publish({"type": "rec", "on": True})

    def on_frame(self, frame: np.ndarray) -> None:
        t = self.take
        if t is None:
            return
        t.frames.append(frame)
        if time.monotonic() - t.t_start > MAX_TAKE_SEC:
            self.stop_take()

    def stop_take(self) -> None:
        t = self.take
        if t is None:
            return
        while True:
            try:
                f = self.mic.frames.get_nowait()
            except queue.Empty:
                break
            t.frames.append(f)
        self.server.publish({"type": "rec", "on": False})
        audio = np.concatenate(t.frames) if t.frames else np.zeros(0, dtype=np.int16)
        t0 = time.monotonic()
        qwen_text = self.qwen(audio) if len(audio) else ""
        qwen_sec = time.monotonic() - t0
        t1 = time.monotonic()
        turbo_text = self.turbo(audio) if len(audio) else ""
        turbo_sec = time.monotonic() - t1
        self.take = None
        n = len(self.records) + 1
        wav_name = f"take_{n:03d}_{t.item.id}.wav"
        with wave.open(str(self.record_dir / wav_name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.astype(np.int16).tobytes())
        rec = {"n": n, "item": t.item.id, "cat": t.item.cat, "say": t.item.say, "dur": round(len(audio) / RATE, 2),
               "robot_clip": t.robot_clip, "wav": wav_name, "peak_db": peak_db(audio),
               "turbo": {"text": turbo_text, "ok": judge(t.item, turbo_text), "sec": round(turbo_sec, 3)},
               "qwen": {"text": qwen_text, "ok": judge(t.item, qwen_text), "sec": round(qwen_sec, 3)}}
        self.records.append(rec)
        self._log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._log.flush()
        self.server.publish({"type": "result", **rec})
        self.publish_summary()

    def close(self) -> None:
        (self.record_dir / "summary.json").write_text(
            json.dumps(summarize(self.records), ensure_ascii=False, indent=1), encoding="utf-8")
        self._log.close()


def _load_turbo():
    """⑫ 와 같은 모델 이름·장치·설정(장소 귀띔만 뺌)으로 whisper 를 띄우고 받아쓰기 함수를 돌려준다."""
    from src import stt  # noqa: F401 — CUDA 판 CTranslate2 를 먼저 올린다(⑫ 와 같은 순서)
    from faster_whisper import WhisperModel

    from src.stt_guard import accept_segments
    from src.wakeword_monitor import ear_stt_model_names

    name = ear_stt_model_names()[0]
    device = os.environ.get("VICA_STT_DEVICE", "cpu")
    compute = os.environ.get("VICA_STT_COMPUTE", "float16" if device == "cuda" else "int8")
    print(f"  turbo: {name} · {device} · {compute}")
    model = WhisperModel(name, device=device, compute_type=compute)

    def transcribe(audio: np.ndarray) -> str:
        segs, _ = model.transcribe(audio.astype(np.float32) / 32768.0, language="ko", beam_size=5,
                                   temperature=0.0, condition_on_previous_text=False)
        return accept_segments(segs)

    transcribe(np.zeros(RATE, dtype=np.int16))
    return transcribe


def _load_qwen(model_path: Path, lib_path: Optional[Path]):
    """Qwen3-ASR 를 띄우고 받아쓰기 함수를 돌려준다. 한 번에 받아쓰는 모델이라 말하는 중 글자는 없다."""
    if lib_path is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(lib_path)
    sys.path.insert(0, str(TRANSCRIBE_DIR / "bindings" / "python" / "src"))
    import transcribe_cpp

    model = transcribe_cpp.Model(str(model_path))
    print(f"  Qwen: {model.arch}/{model.variant} · {model.backend}")
    session = model.session()

    def transcribe(audio: np.ndarray) -> str:
        return session.run(audio.astype(np.float32) / 32768.0, language=QWEN_LANG).text.strip()

    transcribe(np.zeros(RATE, dtype=np.int16))
    return transcribe


def _find_lib() -> Optional[Path]:
    hits = sorted((TRANSCRIBE_DIR / "build-shared").rglob("libtranscribe.so*"))
    return hits[0] if hits else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="STT 비교기 (turbo vs Qwen3-ASR 0.6B)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--qwen-model", default=str(QWEN_MODEL))
    ap.add_argument("--lib", default="", help="libtranscribe.so 경로(없으면 build-shared 에서 찾음)")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from src.env_file import EnvEditError, values_from
    from tools.mic_check_web import LiveServer
    from tools.mic_venue_check import (Mic, _chip_apply_for, _find_input, _proc_cmdlines, lan_ip, load_clips,
                                       voice_stack_running, write_chip)

    os.chdir(ROOT)
    running = voice_stack_running(_proc_cmdlines())
    if running:
        print("⑫·⑬ 음성 칸이 켜져 있습니다. 그 칸을 Ctrl+C 로 끈 뒤 다시 실행하세요:")
        for line in running[:5]:
            print("   ", line[:120])
        return 1
    try:
        env_values = values_from(os.environ)
    except EnvEditError as exc:
        print(exc)
        return 1
    lib = Path(args.lib) if args.lib else _find_lib()
    if lib is None or not lib.exists():
        print(f"transcribe.cpp 라이브러리(libtranscribe.so)를 찾지 못했습니다: {lib} — build-shared 빌드를 먼저 하세요")
        return 1
    if not Path(args.qwen_model).exists():
        print(f"Qwen 모델 파일이 없습니다: {args.qwen_model}")
        return 1

    record_dir = DATA / datetime.now().strftime("%Y%m%d-%H%M%S")
    record_dir.mkdir(parents=True, exist_ok=True)
    clips, missing = load_clips(ROOT / "assets" / "baked")
    for name in missing:
        print(f"  (멘트 {name} 없음 — 건너뜀)")

    mic = Mic(_find_input())
    server = LiveServer(PAGE.read_text(encoding="utf-8"), port=args.port)
    server.start()
    print(f"\n  휴대폰으로 여세요:  http://{lan_ip()}:{server.port}\n")
    server.publish({"type": "prep", "text": "젯슨이 모델을 올리는 중입니다(1분 안팎)…"}, keep="prep")

    from src import audio_out

    chip_apply = _chip_apply_for(mic)
    print("  칩에 .env 값을 씁니다(⑫ 가 켤 때와 같은 일)…")
    for p in chip_apply(env_values):
        print(f"  ⚠ {p}")
    print("  모델을 올립니다…")
    turbo = _load_turbo()
    qwen = _load_qwen(Path(args.qwen_model), lib)
    bench = Bench(server, mic, turbo, qwen, clips, record_dir,
                  play=lambda pcm, rate: audio_out.play(pcm, rate, blocking=True), stop_play=audio_out.stop)
    server.publish({"type": "prep", "text": ""}, keep="prep")
    bench.publish_item()
    bench.publish_summary()
    mic.open()
    print("  준비 끝 — 휴대폰 화면을 따라 하세요. 끝낼 때는 화면의 '끝내기' 또는 Ctrl+C.")
    try:
        while not bench.done:
            while True:
                try:
                    bench.handle(server.commands.get_nowait())
                except queue.Empty:
                    break
            try:
                frame = mic.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            if bench.take is None:
                continue
            bench.on_frame(frame)
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        for step, fn in (("재생 멈추기", audio_out.stop), ("마이크 닫기", mic.close), ("기록 마무리", bench.close),
                         ("칩 되돌리기", lambda: write_chip(env_values)), ("서버 끄기", server.stop)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — 한 단계가 터져도 나머지는 돈다
                print(f"  ⚠ {step} 문제: {exc}")
        s = summarize(bench.records)
        print(f"  기록: {record_dir}")
        for c in s["by_cat"].values():
            if c["n"]:
                print(f"   {c['name']:<14} turbo {c['turbo']}/{c['n']}  Qwen {c['qwen']}/{c['n']}")
        print(f"   글자까지 중앙값: turbo {s['sec']['turbo']} s · Qwen {s['sec']['qwen']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
