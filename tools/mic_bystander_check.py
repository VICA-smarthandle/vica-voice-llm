"""마이크 주변 대화 시험기 — reSpeaker 칩 설정(AGC 증폭·말소리 감지)을 바꿔 가며, 손잡이의 사용자 목소리와 1~3 m 옆사람"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
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

DATA = Path.home() / "vica_data" / "mic_bystander"
PAGE = Path(__file__).with_name("mic_bystander_page.html")
DEFAULT_PORT = 8773
RATE = 16000
OWW_FRAME = 1280
SETTLE_SEC = 2.0
VAD_POLL_SEC = 0.1
MAX_TAKE_SEC = 20.0


PARAMS = {
    "AGCONOFF": (19, 0, True),
    "AGCMAXGAIN": (19, 1, False),
    "AGCDESIREDLEVEL": (19, 2, False),
    "AGCGAIN": (19, 3, False),
    "GAMMAVAD_SR": (19, 39, False),
}
TOUCHED = ("AGCONOFF", "AGCMAXGAIN", "GAMMAVAD_SR")


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    note: str
    changes: dict = field(default_factory=dict)


PRESETS: tuple = (
    Preset("base", "지금 설정", "⑫ 와 같음(.env 값 — 증폭 한도 줄이 없으면 ×31.6)"),
    Preset("gain8", "증폭 최대 ×8", "먼 소리를 덜 키운다", {"AGCMAXGAIN": 8.0}),
    Preset("gain4", "증폭 최대 ×4", "×8 과 ×2 사이", {"AGCMAXGAIN": 4.0}),
    Preset("gain2", "증폭 최대 ×2", "먼 소리를 거의 안 키운다", {"AGCMAXGAIN": 2.0}),
    Preset("agcoff", "증폭 끔(고정 ×2)", "거리 차이가 그대로 남는다", {"AGCONOFF": 0, "AGCGAIN": 2.0}),
    Preset("vad3", "말소리 감지 둔감 ×3", "칩이 약한 목소리를 말로 덜 본다", {"GAMMAVAD_SR": ("x", 3.0)}),
    Preset("gain2vad3", "증폭 ×2 + 감지 둔감 ×3", "둘 다", {"AGCMAXGAIN": 2.0, "GAMMAVAD_SR": ("x", 3.0)}),
)
ROLES = (("user", "사용자(손잡이)"), ("userwake", "손잡이: '비카야, 화장실 가자'"), ("far1", "사용자 1 m 앞(접근 대답)"), ("far15", "사용자 1.5 m 앞(접근 대답)"),
         ("near1", "옆사람 1 m"), ("near2", "옆사람 2 m"), ("near3", "옆사람 3 m"),
         ("shout2", "2 m 에서 '멈춰!'"), ("shout3", "3 m 에서 '멈춰!'"))
FAR_ROLES = ("far1", "far15")
NEAR_ROLES = ("near1", "near2", "near3")
SHOUT_ROLES = ("shout2", "shout3")
USER_PHRASES = ("네", "아니", "멈춰", "비카야", "비카야 화장실 가자")
FAR_PHRASES = ("그래", "아니", "네")
WAKE_SENTENCE = "비카야, 화장실 가자"
STOP_TH_LOW = 0.3
PRE_SEC, POST_SEC = 1.5, 0.5
WAKE_WORD, STOP_WORD = "비카야", "멈춰"


def resolve(preset: Preset, baseline: dict) -> dict:
    """묶음이 칩에 쓸 절대값. 배수는 시작 값(baseline)에 곱한다."""
    out = {}
    for name, value in preset.changes.items():
        if isinstance(value, tuple) and value[0] == "x":
            out[name] = float(baseline[name]) * value[1]
        else:
            out[name] = value
    return out


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-9))


def _ratio(hits: int, n: int) -> str:
    return f"{hits}/{n}"


def summarize(records: Sequence[dict], listen_min_rms: float, wake_th: float, stop_th: float) -> dict:
    """설정 묶음마다 사용자·옆사람 소리 크기와 '가를 수 있는가'. 묶음 순서는 PRESETS 순."""
    out = []
    for p in PRESETS:
        rows = [r for r in records if r["preset"] == p.key]
        if not rows:
            continue
        user = [r for r in rows if r["role"] in ("user", "userwake")]
        far = [r for r in rows if r["role"] in FAR_ROLES]
        near = [r for r in rows if r["role"] in NEAR_ROLES]
        shout = [r for r in rows if r["role"] in SHOUT_ROLES]
        labels = dict(ROLES)
        by_dist = {}
        for key, label in ((k, labels[k]) for k in NEAR_ROLES):
            d = [r["rms"] for r in near if r["role"] == key]
            if d:
                by_dist[key] = {"label": label, "n": len(d), "max_db": round(db(max(d)), 1),
                                "med_db": round(db(statistics.median(d)), 1)}
        u = [r["rms"] for r in user]
        keep = u + [r["rms"] for r in far]
        margin = suggest = None
        if keep and near:
            lo, hi = min(keep), max(r["rms"] for r in near)
            margin = db(lo) - db(hi)
            suggest = math.sqrt(lo * hi) if lo > hi else None
        wake = [r for r in user if r["role"] == "user" and WAKE_WORD in r.get("phrase", "")]
        wake_cont = [r for r in user if r["role"] == "userwake"]
        stop = [r for r in user if r.get("phrase", "") == STOP_WORD]
        shout2 = [r for r in shout if r["role"] == "shout2"]
        out.append({
            "key": p.key, "label": p.label,
            "user": {"n": len(u), "min_db": round(db(min(u)), 1) if u else None,
                     "med_db": round(db(statistics.median(u)), 1) if u else None},
            "far": {"n": len(far), "min_db": round(db(min(r["rms"] for r in far)), 1) if far else None,
                    "med_db": round(db(statistics.median(r["rms"] for r in far)), 1) if far else None},
            "by_dist": by_dist,
            "margin_db": None if margin is None else round(margin, 1),
            "suggest_rms": None if suggest is None else round(suggest, 4),
            "bystander_pass_now": _ratio(sum(r["rms"] >= listen_min_rms for r in near), len(near)),
            "user_drop_now": _ratio(sum(r["rms"] < listen_min_rms for r in user), len(user)),
            "far_drop_now": _ratio(sum(r["rms"] < listen_min_rms for r in far), len(far)),
            "wake": _ratio(sum(r.get("wake_a", 0.0) >= wake_th for r in wake), len(wake)),
            "stop": _ratio(sum(r.get("stop_b", 0.0) >= stop_th for r in stop), len(stop)),
            "far_stop": _ratio(sum(r.get("stop_b", 0.0) >= stop_th for r in shout), len(shout)),
            "wake_cont": _ratio(sum(r.get("wake_a", 0.0) >= wake_th for r in wake_cont), len(wake_cont)),
            "shout2": _ratio(sum(r.get("stop_b", 0.0) >= stop_th for r in shout2), len(shout2)),
            "shout2_low": _ratio(sum(r.get("stop_b", 0.0) >= STOP_TH_LOW for r in shout2), len(shout2)),
        })
    return {"presets": out, "listen_min_rms": listen_min_rms}


class Chip:
    """reSpeaker 칩 값 읽기·쓰기. 쓰기는 마이크·재생 스트림이 닫혀 있을 때만(칩이 거부한다)."""

    def __init__(self):
        import usb.core
        self._dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if self._dev is None:
            raise SystemExit("reSpeaker 칩을 찾지 못했습니다 — USB 를 확인하세요")

    def read(self, name: str) -> float:
        import struct

        import usb.util
        pid, off, is_int = PARAMS[name]
        resp = self._dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | off | (0x40 if is_int else 0), pid, 8, 1000)
        lo, hi = struct.unpack("<ii", resp.tobytes())
        return float(lo) if is_int else lo * (2.0 ** hi)

    def write(self, name: str, value: float) -> Optional[float]:
        from src.dsp_state import _write_param
        pid, off, is_int = PARAMS[name]
        try:
            return _write_param(self._dev, (pid, off), value, is_int)
        except Exception:  # noqa: BLE001 — 실패는 화면에 알린다
            return None

    def snapshot(self) -> dict:
        return {name: round(self.read(name), 4) for name in PARAMS}


class VadPoller:
    """녹음하는 동안 칩의 SPEECHDETECTED 를 0.1 초마다 읽는다 — 칩이 '말소리' 로 본 비율."""

    def __init__(self):
        from src.dsp_state import DspState
        self._dsp = DspState()
        self._flags: list = []
        self._on = threading.Event()
        threading.Thread(target=self._run, name="vad-poll", daemon=True).start()

    def _run(self):
        while True:
            if self._on.is_set() and self._dsp.available:
                try:
                    self._flags.append(bool(self._dsp.speech_detected()))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(VAD_POLL_SEC)

    def start(self):
        self._flags = []
        self._on.set()

    def stop(self) -> Optional[float]:
        self._on.clear()
        return round(sum(self._flags) / len(self._flags), 2) if self._flags else None


class WakeScorer:
    """녹음을 귀와 같은 openWakeWord 모델 두 개(호출 A·긴급 B)에 넣어 최고 점수를 본다."""

    def __init__(self):
        from openwakeword.model import Model
        a = os.environ.get("VICA_WAKE_MODEL_A", os.path.join("models", "vica_bikaya_v1.onnx"))
        b = os.environ.get("VICA_WAKE_MODEL_B", os.path.join("models", "vica_modelb_v2.onnx"))
        self._paths = [a, b]
        self._Model = Model
        self._new()

    def _new(self):
        self._m = self._Model(wakeword_models=self._paths, inference_framework="onnx")
        keys = list(self._m.models.keys())
        self._a = next(k for k in keys if "bikaya" in k)
        self._b = next(k for k in keys if k != self._a)

    def score(self, audio: np.ndarray) -> tuple:
        if hasattr(self._m, "reset"):
            self._m.reset()
        else:
            self._new()
        rng = np.random.default_rng(0)
        audio = np.concatenate([rng.normal(0, 8, int(RATE * PRE_SEC)).astype(np.int16), audio,
                                np.zeros(int(RATE * POST_SEC), dtype=np.int16)])
        best_a = best_b = 0.0
        for i in range(0, len(audio) - OWW_FRAME + 1, OWW_FRAME):
            s = self._m.predict(audio[i:i + OWW_FRAME])
            best_a, best_b = max(best_a, float(s[self._a])), max(best_b, float(s[self._b]))
        return round(best_a, 3), round(best_b, 3)


class Check:
    def __init__(self, server, mic, chip: Chip, baseline: dict, thresholds: dict, vad: VadPoller,
                 wake: WakeScorer, record_dir: Path):
        self.server, self.mic, self.chip, self.baseline = server, mic, chip, baseline
        self.th, self.vad, self.wake, self.record_dir = thresholds, vad, wake, record_dir
        self.preset = "base"
        self.role = "user"
        self.phrase_i = 0
        self.take: Optional[dict] = None
        self.records: list = []
        self.done = False
        self._log = (record_dir / "results.jsonl").open("a", encoding="utf-8")

    def publish_state(self) -> None:
        phrase = self._phrase()
        self.server.publish({
            "type": "state", "preset": self.preset, "role": self.role, "phrase": phrase,
            "presets": [{"key": p.key, "label": p.label, "note": p.note} for p in PRESETS],
            "roles": [{"key": k, "label": v} for k, v in ROLES],
            "listen_min_rms": self.th["listen_min_rms"], "chip": self.chip_now,
        }, keep="state")

    def _phrase(self) -> str:
        if self.role == "user":
            return USER_PHRASES[self.phrase_i % len(USER_PHRASES)]
        if self.role in FAR_ROLES:
            return FAR_PHRASES[self.phrase_i % len(FAR_PHRASES)]
        if self.role == "userwake":
            return WAKE_SENTENCE
        return STOP_WORD if self.role in SHOUT_ROLES else ""

    def publish_summary(self) -> None:
        self.server.publish({"type": "summary", **summarize(self.records, self.th["listen_min_rms"],
                                                             self.th["wake"], self.th["stop"])}, keep="summary")

    @property
    def chip_now(self) -> dict:
        return getattr(self, "_chip_now", {})

    def handle(self, cmd: dict) -> None:
        c = cmd.get("cmd")
        if c == "start":
            self.start_take()
        elif c == "stop":
            self.stop_take()
        elif c == "preset" and self.take is None:
            self.apply_preset(str(cmd.get("key")))
        elif c == "role" and self.take is None and cmd.get("key") in dict(ROLES):
            self.role = cmd["key"]
            self.publish_state()
        elif c == "phrase" and self.take is None:
            self.phrase_i += 1
            self.publish_state()
        elif c == "finish":
            self.done = True

    def apply_preset(self, key: str, *, announce: bool = True) -> None:
        p = next((x for x in PRESETS if x.key == key), None)
        if p is None:
            return
        if announce:
            self.server.publish({"type": "busy", "text": f"칩을 '{p.label}' 로 바꾸는 중…"})
        values = {name: self.baseline[name] for name in TOUCHED}
        values.update(resolve(p, self.baseline))
        problems = []
        self.mic.close()
        from src import audio_out
        audio_out.reset_device_cache()
        for name, value in values.items():
            got = self.chip.write(name, value)
            if got is None or abs(got - value) > 1e-3 * max(1.0, abs(value)):
                problems.append(f"{name}={value} 쓰기 실패(읽은 값 {got})")
        time.sleep(SETTLE_SEC)
        self._chip_now = self.chip.snapshot()
        self.mic.open()
        self.preset = key
        if problems:
            self.server.publish({"type": "toast", "text": " · ".join(problems)})
        self.server.publish({"type": "busy", "text": ""})
        self.publish_state()

    def start_take(self) -> None:
        if self.take is not None:
            return
        while True:
            try:
                self.mic.frames.get_nowait()
            except queue.Empty:
                break
        phrase = self._phrase()
        self.take = {"frames": [], "t0": time.monotonic(), "phrase": phrase}
        self.vad.start()
        self.server.publish({"type": "rec", "on": True})

    def on_frame(self, frame: np.ndarray) -> None:
        if self.take is None:
            return
        self.take["frames"].append(frame)
        if time.monotonic() - self.take["t0"] > MAX_TAKE_SEC:
            self.stop_take()

    def stop_take(self) -> None:
        t = self.take
        if t is None:
            return
        while True:
            try:
                t["frames"].append(self.mic.frames.get_nowait())
            except queue.Empty:
                break
        vad = self.vad.stop()
        self.take = None
        self.server.publish({"type": "rec", "on": False})
        audio = np.concatenate(t["frames"]) if t["frames"] else np.zeros(0, dtype=np.int16)
        from src.wakeword_monitor import capture_stats
        stats = capture_stats(audio)
        wake_a, stop_b = self.wake.score(audio) if len(audio) >= OWW_FRAME else (0.0, 0.0)
        n = len(self.records) + 1
        wav_name = f"take_{n:03d}_{self.preset}_{self.role}.wav"
        with wave.open(str(self.record_dir / wav_name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.astype(np.int16).tobytes())
        rec = {"n": n, "preset": self.preset, "role": self.role, "phrase": t["phrase"], "dur": round(len(audio) / RATE, 2),
               "rms": round(stats["rms"], 5), "rms_db": round(db(stats["rms"]), 1), "peak": round(stats["peak"], 3),
               "clip": round(stats["clip_ratio"], 4), "vad": vad, "wake_a": wake_a, "stop_b": stop_b,
               "kept_now": stats["rms"] >= self.th["listen_min_rms"], "chip": self.chip_now, "wav": wav_name}
        self.records.append(rec)
        self._log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._log.flush()
        self.server.publish({"type": "result", **rec})
        if self.role == "user" or self.role in FAR_ROLES:
            self.phrase_i += 1
            self.publish_state()
        self.publish_summary()

    def close(self) -> None:
        (self.record_dir / "summary.json").write_text(
            json.dumps(summarize(self.records, self.th["listen_min_rms"], self.th["wake"], self.th["stop"]),
                       ensure_ascii=False, indent=1), encoding="utf-8")
        self._log.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="마이크 주변 대화 시험기 (reSpeaker 칩 설정 비교)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from src.env_file import EnvEditError, values_from
    from tools.mic_check_web import LiveServer
    from tools.mic_venue_check import (Mic, _chip_apply_for, _find_input, _proc_cmdlines, lan_ip,
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
    thresholds = {"listen_min_rms": env_values.listen_min_rms, "wake": env_values.wake_threshold,
                  "stop": env_values.emergency_single}

    record_dir = DATA / datetime.now().strftime("%Y%m%d-%H%M%S")
    record_dir.mkdir(parents=True, exist_ok=True)
    mic = Mic(_find_input())
    chip = Chip()
    server = LiveServer(PAGE.read_text(encoding="utf-8"), port=args.port)
    server.start()
    print(f"\n  휴대폰으로 여세요:  http://{lan_ip()}:{server.port}\n")

    print("  칩에 .env 값을 씁니다(⑫ 가 켤 때와 같은 일)…")
    for p in _chip_apply_for(mic)(env_values):
        print(f"  ⚠ {p}")
    baseline = chip.snapshot()
    print(f"  시작 칩 값: {baseline}")
    print("  '비카야'·'멈춰' 모델을 올립니다…")
    check = Check(server, mic, chip, baseline, thresholds, VadPoller(), WakeScorer(), record_dir)
    check._chip_now = baseline
    check.publish_state()
    check.publish_summary()
    mic.open()
    print("  준비 끝 — 휴대폰 화면을 따라 하세요. 끝낼 때는 화면의 '끝내기' 또는 Ctrl+C.")
    try:
        while not check.done:
            while True:
                try:
                    check.handle(server.commands.get_nowait())
                except queue.Empty:
                    break
            try:
                frame = mic.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            check.on_frame(frame)
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        try:
            check.apply_preset("base", announce=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 칩 되돌리기 문제: {exc}")
        for step, fn in (("마이크 닫기", mic.close), ("칩 .env 값", lambda: write_chip(env_values)),
                         ("기록 마무리", check.close), ("서버 끄기", server.stop)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — 한 단계가 터져도 나머지는 돈다
                print(f"  ⚠ {step} 문제: {exc}")
        print(f"  칩 끝 값: {chip.snapshot()}")
        s = summarize(check.records, thresholds["listen_min_rms"], thresholds["wake"], thresholds["stop"])
        print(f"  기록: {record_dir}")
        for p in s["presets"]:
            print(f"   {p['label']:<16} 사용자 최소 {p['user']['min_db']} dB · 여유 {p['margin_db']} dB · "
                  f"추천 문턱 {p['suggest_rms']} · 비카야 {p['wake']} · 멈춰 {p['stop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
