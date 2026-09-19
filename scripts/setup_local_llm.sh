#!/usr/bin/env bash
# 로컬 폴백 LLM 준비: HF Gemma 4 E2B Q4_K_M 을 받아 텍스트 전용 모델 gemma4-e2b-text 로 조립한다.
#
# 사용 (저장소 루트, ollama 서버가 떠 있어야 한다):
#     scripts/setup_local_llm.sh
# 재실행해도 안전하다(같은 이름으로 다시 만든다). 새 젯슨에서는 이 한 줄이면 된다.
set -euo pipefail

SRC_TAG="hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
SRC_MANIFEST="hf.co/unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
NAME="gemma4-e2b-text"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! curl -sf localhost:11434/api/version >/dev/null; then
    echo "ollama 서버가 없습니다. 먼저 'ollama serve' 를 띄우세요 (음성 launch 가 띄워 줍니다)." >&2
    exit 1
fi

echo "[1/3] 가중치 받기: $SRC_TAG (3.1 GB + projector 0.9 GB)"
ollama pull "$SRC_TAG"

echo "[2/3] projector 를 뺀 가중치 blob 찾기"
BLOB=$(python3 - "$OLLAMA_MODELS/manifests/$SRC_MANIFEST" "$OLLAMA_MODELS" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
digest = next(l["digest"] for l in manifest["layers"] if l["mediaType"].endswith(".model"))
print(f"{sys.argv[2]}/blobs/{digest.replace(':', '-')}")
PY
)
test -f "$BLOB" || { echo "blob 이 없습니다: $BLOB" >&2; exit 1; }

echo "[3/3] 모델 조립: $NAME"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
sed "s|^FROM .*|FROM $BLOB|" "$HERE/ollama/Modelfile.gemma4-e2b-text" > "$TMP"
ollama create "$NAME" -f "$TMP"
ollama show "$NAME" | sed -n '1,12p'
echo "완료. .env 에 VICA_LLM_FALLBACK_MODEL=$NAME 을 넣으면 폴백이 켜진다."
