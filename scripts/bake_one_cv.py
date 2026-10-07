#!/usr/bin/env python3
"""멘트를 CosyVoice(F2 클로닝)로 다시 굽는다 — 문구 수정 때 쓰는 도구.

사용법 (⚠️ 로봇 스택이 내려가 있을 때만 — 모델이 RAM ~3GB 를 쓴다):
    PYTHONPATH=~/CosyVoice:~/CosyVoice/third_party/Matcha-TTS \
      ~/venvs/cosyvoice/bin/python scripts/bake_one_cv.py \
      mission_msg_estop_released "비상멈춤이 해제되었습니다."

    # 미션 문장 묶음(src/mission_phrases.baked_mission_ments — 대기 장소 M2·M2′·M3·M6·M7·
    # 비상 한 마디, 2026-10-07)을 모델을 한 번만 올려 한꺼번에 굽는다. 이미 같은 글자로
    # 구운 것은 건너뛴다(--force 면 다시).
    PYTHONPATH=... ~/venvs/cosyvoice/bin/python scripts/bake_one_cv.py --mission

첫 인자는 assets/baked/ 의 파일명(확장자 없이), 둘째는 문구. 프롬프트
참조 음성(F2)은 supertonic 으로 그 자리에서 만들어 쓴다(정본 목소리).
manifest.json 도 함께 갱신한다(한 문장 구울 때마다 — 중간에 끊겨도 앞의 것은 남는다).
레시피 함정(endofprompt 등)은 메모리 voice-batch-2026-08-30 의 "설치 지뢰 7개" 참고.
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


def _make_ref() -> str:
    """① 프롬프트 참조 음성 — supertonic F2 로 즉석 생성 (정본 목소리).

    supertonic 은 프로젝트 .venv 에, CosyVoice 는 ~/venvs/cosyvoice 에 있어 한
    인터프리터에 둘 다 없다 — F2 단계는 .venv 하위 프로세스로 돈다.
    """
    ref_path = "/tmp/bake_one_ref_f2.wav"
    import subprocess
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c",
         "import soundfile as sf\n"
         "from src.tts import VicaTTS\n"
         f"w, r = VicaTTS(voice='F2').synthesize({PROMPT_TEXT!r})\n"
         f"sf.write({ref_path!r}, w, r, subtype='PCM_16')\n"],
        cwd=ROOT, check=True)
    return ref_path


def _load_model():
    """② CosyVoice3 (레시피: endofprompt + text_frontend=False)."""
    from cosyvoice.cli.cosyvoice import CosyVoice3
    model_dir = str(Path.home() / ".cache/huggingface/hub"
                    / "models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512"
                    / "snapshots")
    snap = next(Path(model_dir).iterdir())
    return CosyVoice3(str(snap), load_trt=False, fp16=True)


def _bake(cv, ref_path: str, name: str, text: str, out_dir: Path = BAKED,
          manifest: bool = True) -> float:
    """한 문장을 굽고 ③ assets 규격(16kHz mono, -3dBFS)으로 저장 + manifest 갱신. 길이(초).

    out_dir·manifest=False 는 후보 굽기(--takes)용 — 정본(assets/baked)을 건드리지 않는다.
    """
    import torch
    outs = [o["tts_speech"] for o in cv.inference_zero_shot(
        tts_text=text,
        prompt_text=f"You are a helpful assistant.<|endofprompt|>{PROMPT_TEXT}",
        prompt_wav=ref_path, text_frontend=False)]
    wav = torch.cat(outs, dim=1).squeeze(0).numpy()
    n = int(len(wav) * 16000 / cv.sample_rate)
    wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                    np.arange(len(wav)), wav).astype(np.float32)
    wav *= 10 ** (-3 / 20) / (np.abs(wav).max() or 1.0)
    sf.write(str(Path(out_dir) / f"{name}.wav"), wav, 16000, subtype="PCM_16")
    if manifest:
        table = json.load(open(BAKED / "manifest.json"))
        table[f"{name}.wav"] = text
        json.dump(table, open(BAKED / "manifest.json", "w"),
                  ensure_ascii=False, indent=1)
    return n / 16000


def _takes(args: list[str]) -> None:
    """--takes N OUT_DIR 이름... : 미션 문장 중 고른 것을 N 번씩 구워 후보로 둔다.

    CosyVoice 는 구울 때마다 억양이 조금씩 달라, 청취에서 '다시'가 나온 문장은 후보를
    여럿 들려주고 고르게 한다. 정본(assets/baked·manifest)은 건드리지 않는다 — 고른 것을
    사람이 옮긴다. 파일 이름: <이름>__take<k>.wav
    """
    from src.mission_phrases import baked_mission_ments
    count, out_dir, names = int(args[0]), Path(args[1]), args[2:]
    table = baked_mission_ments()
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit(f"미션 굽기 목록에 없는 이름: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _make_ref()
    cv = _load_model()
    for name in names:
        for k in range(1, count + 1):
            sec = _bake(cv, ref_path, f"{name}__take{k}", table[name], out_dir, manifest=False)
            print(f"후보: {name}__take{k}.wav ({sec:.1f}s) '{table[name]}'", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--takes"]:
        _takes(args[1:])
        return
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if args[:1] == ["--mission"]:
        from src.mission_phrases import baked_mission_ments
        manifest = json.load(open(BAKED / "manifest.json"))
        items = [(name, text) for name, text in baked_mission_ments().items()
                 if force or manifest.get(f"{name}.wav") != text
                 or not (BAKED / f"{name}.wav").exists()]
        if not items:
            print("굽을 것 없음 — 미션 문장이 전부 같은 글자로 구워져 있다")
            return
    else:
        items = [(args[0], args[1])]

    ref_path = _make_ref()
    cv = _load_model()
    for i, (name, text) in enumerate(items, 1):
        sec = _bake(cv, ref_path, name, text)
        print(f"[{i}/{len(items)}] 구움: {name}.wav ({sec:.1f}s) '{text}'", flush=True)
    print("청취 확인 후 커밋할 것")


if __name__ == "__main__":
    main()
