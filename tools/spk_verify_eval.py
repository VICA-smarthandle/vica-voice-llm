"""화자 검증 신뢰도 평가기 — 녹음기(tools/spk_verify_record.py)가 모은 사람별 녹음으로
"A 는 통과, 다른 사람은 거부"가 문턱마다 몇 % 인지 잰다. 오프라인·읽기 전용, ROS·칩에 아무것도 보내지 않는다.

    cd ~/VICA-smarthandle/vica-voice-llm
    ~/vica_data/spk_eval/venv/bin/python -m tools.spk_verify_eval                 # 가장 최근 녹음 폴더
    ~/vica_data/spk_eval/venv/bin/python -m tools.spk_verify_eval --rec ~/vica_data/spk_eval/rec/20261110

sherpa-onnx 는 운영 .venv 를 건드리지 않게 별도 venv(~/vica_data/spk_eval/venv)에만 있다. 이 파일은 가져오기(import)
때 sherpa 를 부르지 않는다 — 순수한 부분(자르기·통과율·EER)은 운영 .venv 의 pytest 로 시험한다.

판정 단위(사람 P 를 A 로 등록하고 사람 Q 의 대기 중 명령을 판정):
  등록  P 의 'enroll' 녹음 6개 — 발화마다 벡터를 뽑아 단위 길이로 맞춘 뒤 평균(sherpa 예제와 같은 방식)
  s1    Q 의 명령 하나
  s12   되묻기 — 그 명령 + Q 의 다음 명령을 이어 붙인 것. 남도 똑같이 두 번 기회를 얻는다
  sw    'Q 의 비카야' + 명령을 이어 붙인 것(호출어까지 판정에 쓰는 경우)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speech_trim import (FLOOR, FRAME_SEC, KEEP_REL, MERGE_GAP_SEC, PAD_SEC, REL_DB,  # noqa: E402,F401
                             SAMPLE_RATE, trim_speech)
from src.spk_embed import cosine, load_embedder, mean_embedding  # noqa: E402,F401

BINS = ((0.6, "<0.6"), (1.0, "0.6~1.0"), (1.5, "1.0~1.5"), (2.5, "1.5~2.5"))
LAST_BIN = "≥2.5"
DEFAULT_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(19))   # 0.00 ~ 0.90
DEFAULT_TARGET = 0.95     # 운영점: 본인 되묻기 포함 통과율이 이 이상인 가장 높은 문턱

DATA = Path.home() / "vica_data" / "spk_eval"
MODEL_DIR = DATA / "models"
REC_ROOT = DATA / "rec"
MODELS = {   # 이름 → 파일(csukuangfj/speaker-embedding-models). 한국어로 학습된 것은 목록에 없다
    "campplus_zh_en": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "campplus_vox": "wespeaker_en_voxceleb_CAM++.onnx",
    "eres2net_vox": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "titanet_small": "nemo_en_titanet_small.onnx",
}


# --- 순수한 부분 -----------------------------------------------------------------------------------------



def attempts_for(cmd_ids: Sequence[str], wake_id: Optional[str]) -> list:
    """명령마다 한 번의 판정 시도. 되묻기는 녹음 순서상 다음 명령(마지막은 첫 명령)을 이어 붙인다."""
    n = len(cmd_ids)
    out = []
    for i, c in enumerate(cmd_ids):
        out.append({
            "first": (c,),
            "second": (c, cmd_ids[(i + 1) % n]) if n >= 2 else None,
            "wake": (wake_id, c) if wake_id else None,
        })
    return out


def duration_bin(sec: float) -> str:
    for limit, label in BINS:
        if sec < limit:
            return label
    return LAST_BIN


def eer(genuine: Sequence[float], impostor: Sequence[float]) -> tuple:
    """(EER, 그 문턱). 문턱 후보는 나온 점수 전부, 점수 ≥ 문턱이면 통과."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    best = None
    for th in sorted(set(g.tolist()) | set(i.tolist())):
        frr = float(np.mean(g < th))
        far = float(np.mean(i >= th))
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, (far + frr) / 2, th)
    return best[1], best[2]


def latest_by_file(lines: Sequence[dict]) -> list:
    """같은 파일을 다시 녹음했으면 마지막 줄만 남긴다(처음 나온 순서는 지킨다)."""
    by_file: dict = {}
    for line in lines:
        by_file[line["file"]] = line
    return list(by_file.values())


def _rate(records: Sequence[dict], passed) -> Optional[float]:
    return sum(1 for r in records if passed(r)) / len(records) if records else None


def summarize(records: Sequence[dict], thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
              target: float = DEFAULT_TARGET) -> dict:
    """판정 기록 → 문턱별 통과율·EER·운영점·길이 구간별 EER."""
    gen = [r for r in records if r["genuine"]]
    imp = [r for r in records if not r["genuine"]]

    def single(th):
        return lambda r: r["s1"] >= th

    def retry(th):
        return lambda r: r["s1"] >= th or (r["s12"] is not None and r["s12"] >= th)

    def wake(th):
        return lambda r: r["sw"] >= th

    gen_w = [r for r in gen if r.get("sw") is not None]
    imp_w = [r for r in imp if r.get("sw") is not None]
    rows = []
    for th in thresholds:
        rows.append({
            "threshold": th,
            "genuine_single": _rate(gen, single(th)) or 0.0,
            "genuine_retry": _rate(gen, retry(th)) or 0.0,
            "impostor_single": _rate(imp, single(th)) or 0.0,
            "impostor_retry": _rate(imp, retry(th)) or 0.0,
            "genuine_wake": _rate(gen_w, wake(th)),
            "impostor_wake": _rate(imp_w, wake(th)),
        })
    op = None
    for row in sorted(rows, key=lambda r: r["threshold"], reverse=True):
        if row["genuine_retry"] >= target:
            op = {"threshold": row["threshold"], "genuine": row["genuine_retry"], "impostor": row["impostor_retry"]}
            break
    by_duration = {}
    for label in [b[1] for b in BINS] + [LAST_BIN]:
        g = [r["s1"] for r in gen if duration_bin(r["dur1"]) == label]
        i = [r["s1"] for r in imp if duration_bin(r["dur1"]) == label]
        by_duration[label] = {"genuine_n": len(g), "impostor_n": len(i),
                              "eer": eer(g, i)[0] if g and i else None}
    return {
        "n_genuine": len(gen), "n_impostor": len(imp),
        "eer_single": eer([r["s1"] for r in gen], [r["s1"] for r in imp]) if gen and imp else None,
        "eer_wake": eer([r["sw"] for r in gen_w], [r["sw"] for r in imp_w]) if gen_w and imp_w else None,
        "op_retry": op,
        "by_threshold": rows,
        "by_duration": by_duration,
    }


# --- 파일·모델 (실기로 확인) --------------------------------------------------------------------------------

def read_wav(path: Path) -> np.ndarray:
    """(샘플, 채널) float32, −1~1."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise ValueError(f"{path}: 16 kHz 16-bit 가 아님")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return (pcm.reshape(-1, w.getnchannels()).astype(np.float32) / 32768.0)


class Embedder:
    """load_embedder 를 감싸 판정마다 걸린 시간을 기록한다(평가 보고용)."""

    def __init__(self, model_path: Path, threads: int = 2):
        try:
            self._embed = load_embedder(str(model_path), threads)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        self.seconds: list = []   # (음성 길이 s, 걸린 시간 s)

    def __call__(self, audio: np.ndarray) -> Optional[np.ndarray]:
        t0 = time.perf_counter()
        v = self._embed(audio)
        if v is not None:
            self.seconds.append((len(audio) / SAMPLE_RATE, time.perf_counter() - t0))
        return v


def load_takes(rec_dir: Path) -> dict:
    """{사람: {"enroll": [파일], "wake": 파일|None, "cmd": [파일]}} — 녹음 순서(idx)대로."""
    lines = [json.loads(s) for s in (rec_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    people: dict = {}
    for line in sorted(latest_by_file(lines), key=lambda r: (r["person"], r["idx"])):
        p = people.setdefault(line["person"], {"enroll": [], "wake": None, "cmd": [], "text": {}})
        p["text"][line["file"]] = line["text"]
        if line["kind"] == "wake":
            p["wake"] = line["file"]
        else:
            p[line["kind"]].append(line["file"])
    return people


def evaluate(rec_dir: Path, embed: Embedder, channel: int) -> list:
    people = load_takes(rec_dir)
    speech = {}
    for p in people.values():
        for f in p["enroll"] + p["cmd"] + ([p["wake"]] if p["wake"] else []):
            speech[f] = trim_speech(read_wav(rec_dir / f)[:, channel])
    cache: dict = {}

    def vec(files: tuple) -> Optional[np.ndarray]:
        if files not in cache:
            audio = np.concatenate([speech[f] for f in files])
            cache[files] = embed(audio) if len(audio) else None
        return cache[files]

    enrolled = {}
    for name, p in people.items():
        vs = [v for v in (vec((f,)) for f in p["enroll"]) if v is not None]
        if vs:
            enrolled[name] = mean_embedding(vs)
    records = []
    for claimed, e in enrolled.items():
        for speaker, p in people.items():
            for a in attempts_for(p["cmd"], p["wake"]):
                v1 = vec(a["first"])
                if v1 is None:
                    continue
                v12 = vec(a["second"]) if a["second"] else None
                vw = vec(a["wake"]) if a["wake"] else None
                records.append({
                    "claimed": claimed, "speaker": speaker, "genuine": claimed == speaker,
                    "file": a["first"][0], "text": p["text"][a["first"][0]],
                    "s1": cosine(e, v1),
                    "s12": cosine(e, v12) if v12 is not None else None,
                    "sw": cosine(e, vw) if vw is not None else None,
                    "dur1": len(speech[a["first"][0]]) / SAMPLE_RATE,
                })
    return records


def _pct(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def _print_report(name: str, channel: int, s: dict, speed: tuple) -> None:
    e = s["eer_single"]
    ew = s["eer_wake"]
    op = s["op_retry"]
    print(f"\n■ {name} · ch{channel}   판정 본인 {s['n_genuine']} · 남 {s['n_impostor']}"
          f"   1회 판정 {speed[0]:.0f} ms(말 1 s 당 {speed[1]:.0f} ms)")
    print(f"  EER 명령만 {_pct(e[0] if e else None)} % (문턱 {e[1]:.3f})" if e else "  EER 명령만  -")
    if ew:
        print(f"  EER 비카야+명령 {_pct(ew[0])} % (문턱 {ew[1]:.3f})")
    if op:
        print(f"  운영점(본인 되묻기 포함 ≥ 목표) 문턱 {op['threshold']:.2f}: 본인 {_pct(op['genuine'])} % · 남 {_pct(op['impostor'])} %")
    else:
        print("  운영점 없음 — 어느 문턱에서도 본인 통과율이 목표에 못 미침")
    print("  문턱 | 본인 1회 | 본인 되묻기 | 남 1회 | 남 되묻기 | 본인 비카야+ | 남 비카야+")
    for r in s["by_threshold"]:
        if r["threshold"] < 0.2 or r["threshold"] > 0.8:
            continue
        print(f"  {r['threshold']:.2f} |  {_pct(r['genuine_single'])}  |    {_pct(r['genuine_retry'])}   |"
              f" {_pct(r['impostor_single'])} |   {_pct(r['impostor_retry'])}  |    {_pct(r['genuine_wake'])}    |"
              f"   {_pct(r['impostor_wake'])}")
    print("  명령 길이별 EER(명령만):", "  ".join(
        f"{k} {_pct(v['eer'])}%({v['genuine_n']}/{v['impostor_n']})" for k, v in s["by_duration"].items()
        if v["genuine_n"] or v["impostor_n"]))


def _latest_rec() -> Optional[Path]:
    dirs = sorted(d for d in REC_ROOT.glob("*") if (d / "manifest.jsonl").exists()) if REC_ROOT.exists() else []
    return dirs[-1] if dirs else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="화자 검증 신뢰도 평가기")
    ap.add_argument("--rec", type=Path, default=None, help="녹음 폴더(manifest.jsonl 이 있는 곳). 없으면 가장 최근")
    ap.add_argument("--models", default=",".join(MODELS), help="쉼표로: " + ",".join(MODELS))
    ap.add_argument("--channels", default="0,1", help="0 = 칩 처리음(⑫ 가 듣는 소리), 1 = 마이크 1 원음")
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET, help="운영점의 본인 통과율 목표")
    args = ap.parse_args(argv)
    rec = args.rec.expanduser() if args.rec else _latest_rec()
    if rec is None or not (rec / "manifest.jsonl").exists():
        print("녹음 폴더를 찾지 못했습니다 — 먼저 tools/spk_verify_record.py 로 녹음하세요.")
        return 1
    people = load_takes(rec)
    print(f"녹음 {rec} · 사람 {len(people)}명: " + ", ".join(
        f"{k}(등록 {len(v['enroll'])}·명령 {len(v['cmd'])}·호출 {1 if v['wake'] else 0})" for k, v in people.items()))
    if len(people) < 2:
        print("사람이 2명 이상이어야 '남' 판정을 할 수 있습니다.")
        return 1
    result = {"rec": str(rec), "target": args.target, "runs": []}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = MODEL_DIR / MODELS[name]
        if not path.exists():
            print(f"\n■ {name}: 모델 파일 없음 {path} — 건너뜀")
            continue
        embed = Embedder(path)
        for ch in [int(c) for c in args.channels.split(",")]:
            records = evaluate(rec, embed, ch)
            s = summarize(records, target=args.target)
            took = [t for _, t in embed.seconds]
            per_sec = [t / d for d, t in embed.seconds if d > 0]
            speed = (1000 * statistics.median(took), 1000 * statistics.median(per_sec)) if took else (0.0, 0.0)
            _print_report(name, ch, s, speed)
            result["runs"].append({"model": name, "channel": ch, "summary": s, "records": records,
                                   "ms_per_embedding": speed[0], "ms_per_audio_sec": speed[1]})
            embed.seconds.clear()
    out = rec / "result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n기록: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
