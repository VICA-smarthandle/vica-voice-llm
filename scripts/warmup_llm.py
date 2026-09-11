#!/usr/bin/env python3
"""ollama 에 .env 모델을 미리 적재한다 (부팅 후 첫 발화 30초+ 지연 방지)."""
import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
model = os.environ.get("VICA_LLM_MODEL", "")

if not host.startswith("http://localhost") or not model:
    print(f"[warmup] 로컬 호스트 아님({host}) — 예열 생략")
elif __name__ == "__main__":
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps({"model": model, "keep_alive": -1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=180).read()
        print(f"[warmup] 모델 예열 완료: {model}")
    except Exception as exc:
        print(f"[warmup] 예열 실패(무시 가능): {exc}")
