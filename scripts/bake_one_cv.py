#!/usr/bin/env python3
"""멘트 하나를 CosyVoice(F2 클로닝)로 다시 굽는다 — 문구 수정 때 쓰는 도구."""
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

    ref_path = "/tmp/bake_one_ref_f2.wav"
    import subprocess
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c",
         "import soundfile as sf\n"
         "from src.tts import VicaTTS\n"
         f"w, r = VicaTTS(voice='F2').synthesize({PROMPT_TEXT!r})\n"
         f"sf.write({ref_path!r}, w, r, subtype='PCM_16')\n"],
        cwd=ROOT, check=True)

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
