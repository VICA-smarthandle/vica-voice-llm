#!/usr/bin/env python3
"""멘트 하나를 CosyVoice(F2 클로닝)로 다시 굽는다 — 문구 수정 때 쓰는 도구.

사용법 (⚠️ 로봇 스택이 내려가 있을 때만 — 모델이 RAM ~3GB 를 쓴다):
    PYTHONPATH=~/CosyVoice:~/CosyVoice/third_party/Matcha-TTS \
      ~/venvs/cosyvoice/bin/python scripts/bake_one_cv.py \
      mission_msg_estop_released "비상멈춤이 해제되었습니다."

첫 인자는 assets/baked/ 의 파일명(확장자 없이), 둘째는 문구. 프롬프트
참조 음성(F2)은 supertonic 으로 그 자리에서 만들어 쓴다(정본 목소리).
manifest.json 도 함께 갱신한다. 레시피 함정(endofprompt 등)은
메모리 voice-batch-2026-08-30 의 "설치 지뢰 7개" 참고.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BAKED = ROOT / "assets" / "baked"
PROMPT_TEXT = "별빛관 1층 화장실로 안내해드릴까요?"

def main() -> None:
    name, text = sys.argv[1], sys.argv[2]

    # ① 프롬프트 참조 음성 — supertonic F2 로 즉석 생성 (정본 목소리)
    from src.tts import VicaTTS
    ref_wav, ref_rate = VicaTTS(voice="F2").synthesize(PROMPT_TEXT)
    ref_path = "/tmp/bake_one_ref_f2.wav"
    sf.write(ref_path, ref_wav, ref_rate, subtype="PCM_16")

    # ② CosyVoice3 로 클로닝 합성 (레시피: endofprompt + text_frontend=False)
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice3
    model_dir = str(Path.home() / ".cache/huggingface/hub"
                    / "models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512"
                    / "snapshots")
    snap = next(Path(model_dir).iterdir())
    cv = CosyVoice3(str(snap), load_trt=False, fp16=True)
    outs = [o["tts_speech"] for o in cv.inference_zero_shot(
        tts_text=text,
        prompt_text=f"You are a helpful assistant.<|endofprompt|>{PROMPT_TEXT}",
        prompt_wav=ref_path, text_frontend=False)]
    wav = torch.cat(outs, dim=1).squeeze(0).numpy()

    # ③ assets 규격(16kHz mono, -3dBFS)으로 저장 + manifest 갱신
    n = int(len(wav) * 16000 / cv.sample_rate)
    wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                    np.arange(len(wav)), wav).astype(np.float32)
    wav *= 10 ** (-3 / 20) / (np.abs(wav).max() or 1.0)
    sf.write(str(BAKED / f"{name}.wav"), wav, 16000, subtype="PCM_16")
    manifest = json.load(open(BAKED / "manifest.json"))
    manifest[f"{name}.wav"] = text
    json.dump(manifest, open(BAKED / "manifest.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"구움: {name}.wav ({n/16000:.1f}s) '{text}' — 청취 확인 후 커밋할 것")

if __name__ == "__main__":
    main()
