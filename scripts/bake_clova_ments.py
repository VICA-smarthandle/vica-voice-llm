#!/usr/bin/env python3
"""고정 멘트를 CLOVA Voice 로 구워 캐시(wav)로 만든다.

전략(2026-08-29): 실시간 자유 응답은 로컬 TTS 유지, 고정 멘트만 최고 품질로
한 번 구워 재생 0초·네트워크 무관하게 쓴다. CLOVA 는 스트리밍이 없어 실시간
후보에서 탈락했지만 굽기에는 그 약점이 무관하다.

준비 (사용자 1회):
1. https://www.ncloud.com 가입 → 콘솔 → AI·NAVER API → Application 등록
   → CLOVA Voice 선택 → Client ID/Secret 발급
2. .env 에 추가:
       VICA_CLOVA_CLIENT_ID=...
       VICA_CLOVA_CLIENT_SECRET=...

사용법:
    # 1단계 — 시청회: 대표 문장을 보이스 여러 개로 구워 들어보고 고른다
    .venv/bin/python scripts/bake_clova_ments.py sample --voices nara,mijin,jinho

    # 2단계 — 확정 보이스로 등록 멘트 전체(ment_cache.CACHED_MENTS)를 굽는다
    .venv/bin/python scripts/bake_clova_ments.py all --voice nara

출력은 어느 모드든 -3 dBFS 피크 정규화 + 16 kHz mono PCM16 — 기존 assets
녹음과 같은 규격이다. all 모드는 assets/ 를 직접 덮어쓰므로 굽기 전 git
상태를 깨끗이 해 둘 것 (덮어쓴 판이 마음에 안 들면 git 으로 되돌린다).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENDPOINT = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
TARGET_RATE = 16000
TARGET_PEAK_DBFS = -3.0     # 재생 기준(robot-mount-checklist)과 동일

# 시청회용 대표 문장 — 짧은 응답·질문·안전 멘트 세 결을 다 들어본다
SAMPLE_SENTENCES = [
    "네? 무엇을 도와드릴까요?",
    "별빛관 1층 화장실로 안내해드릴까요?",
    "안전을 위해 멈추겠습니다. 관리자를 호출했습니다.",
]


def _keys() -> tuple[str, str]:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    cid = os.environ.get("VICA_CLOVA_CLIENT_ID", "")
    secret = os.environ.get("VICA_CLOVA_CLIENT_SECRET", "")
    if not cid or not secret:
        raise SystemExit(
            "CLOVA 키가 없다 — .env 에 VICA_CLOVA_CLIENT_ID / "
            "VICA_CLOVA_CLIENT_SECRET 를 넣을 것 (모듈 주석의 준비 절차)")
    return cid, secret


def synthesize(text: str, voice: str, cid: str, secret: str,
               speed: int = 0, emotion: int | None = None) -> tuple[np.ndarray, int]:
    """CLOVA Voice 호출 → float32 mono 오디오. 실패는 예외로 그대로 낸다."""
    params = {"speaker": voice, "text": text, "format": "wav", "speed": str(speed)}
    if emotion is not None:
        params["emotion"] = str(emotion)   # 지원 보이스에서만 유효 (0=중립)
    req = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode(params).encode(),
        headers={
            "X-NCP-APIGW-API-KEY-ID": cid,
            "X-NCP-APIGW-API-KEY": secret,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    wav, rate = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, rate


def to_asset(wav: np.ndarray, rate: int) -> np.ndarray:
    """assets 규격으로: 16 kHz 리샘플 + -3 dBFS 피크 정규화."""
    if rate != TARGET_RATE:
        n = int(len(wav) * TARGET_RATE / rate)
        wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                        np.arange(len(wav)), wav).astype(np.float32)
    peak = float(np.abs(wav).max()) or 1.0
    return wav * (10 ** (TARGET_PEAK_DBFS / 20) / peak)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p_sample = sub.add_parser("sample", help="시청회용 — 보이스 비교 굽기")
    p_sample.add_argument("--voices", default="nara,mijin,jinho",
                          help="쉼표 구분 보이스 코드 (프리미엄 목록은 NCP 문서)")
    p_sample.add_argument("--out", default=str(ROOT / "clova_samples"))
    p_all = sub.add_parser("all", help="확정 보이스로 등록 멘트 전체 굽기")
    p_all.add_argument("--voice", required=True)
    args = ap.parse_args()

    cid, secret = _keys()

    if args.mode == "sample":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for voice in args.voices.split(","):
            voice = voice.strip()
            for i, text in enumerate(SAMPLE_SENTENCES, 1):
                wav, rate = synthesize(text, voice, cid, secret)
                path = out / f"{voice}_{i}.wav"
                sf.write(path, to_asset(wav, rate), TARGET_RATE, subtype="PCM_16")
                print(f"구움: {path}  ('{text[:20]}…')")
        print(f"\n시청: 재생 장치로 {out}/*.wav 를 차례로 들어볼 것")
        return

    from src.ment_cache import ASSETS_DIR, CACHED_MENTS
    for filename, text in CACHED_MENTS.items():
        wav, rate = synthesize(text, args.voice, cid, secret)
        path = ASSETS_DIR / filename
        sf.write(path, to_asset(wav, rate), TARGET_RATE, subtype="PCM_16")
        print(f"구움: {path.name}  ({len(wav)/rate:.1f}s)  '{text[:28]}…'")
    print("\n완료 — git diff 로 바뀐 wav 를 확인하고, 실기 청취 후 커밋할 것")


if __name__ == "__main__":
    main()
