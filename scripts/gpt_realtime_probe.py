"""gpt-realtime-2.1-mini 단독 탐침 — 실험 전용, 파이프라인과 완전히 무관하다.

## 무엇을 재는가 (docs/handoff-gpt-realtime-test.md 의 측정 4종)

1. 연결·세션 수립 시간   — WebSocket open → session.updated 까지
2. 응답 지연             — response.create 전송 → 첫 인자 / response.done 까지
3. report_intent 스키마 준수 — 인자가 _IntentDraft 형태를 지키는가
4. 한국어 이해           — 텍스트·음성 발화에 대한 분류 품질 (사람이 판정)

## 왜 파이프라인 밖에서 하는가

CLAUDE.md 규칙: LLM/오디오 실험은 실험 파일로 격리한다. 이 스크립트는
ROS·whisper·TTS 를 일절 건드리지 않으며, VICA 노드가 떠 있지 않아도 돈다.
도구 이름이 report_intent 인 이유: 로봇을 움직이는 도구가 아니라 해석 결과를
보고하는 도구다 (금지 목록 move_robot 류와 무관, docs/plan_gpt_realtime_llm.md 3.3).

## 준비

    pip install websockets          # venv 안에만. requirements.txt 에 추가 금지
    .env 에 OPENAI_API_KEY=...      # 커밋 금지 (.gitignore 에 이미 있음)

## 실행

    .venv/bin/python scripts/gpt_realtime_probe.py --text "화장실로 가줘"
    .venv/bin/python scripts/gpt_realtime_probe.py --wav sample.wav
    # 녹음: arecord -r 16000 -f S16_LE -c 1 -d 3 sample.wav

## [미검증] 이 스크립트는 2026-08-11 GA 이벤트 형식 기준으로 작성했고 실행해 본
적이 없다. 서버가 error 이벤트를 돌려주면 그 내용이 그대로 출력되니, 아래
_legacy_session() 주석의 구형 키로 바꿔 재시도해라.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

try:  # 파이프라인과 같은 방식으로 .env 를 읽는다 (없어도 동작).
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "gpt-realtime-2.1-mini"
URL = "wss://api.openai.com/v1/realtime?model={model}"
TARGET_RATE = 24000  # Realtime API 기본 PCM 레이트. 16k wav 는 아래서 리샘플한다.

# 탐침용 축소 목적지 목록. 실제 catalog(destinations.yaml)가 아니다 — 스키마 준수와
# 이해 품질만 보면 되므로 대표 4곳이면 충분하다.
PROBE_DESTINATIONS = "화장실, 식당, 안내소, 행정지원실"

# src/langchain_intent_parser._build_system_prompt 의 축약판. 계약이 같아야
# 비교가 성립한다: 모든 발화 → 정확히 하나의 report_intent 호출.
INSTRUCTIONS = f"""너는 시각장애인 안내 로봇 'VICA'의 음성 의도 분석기다.
사용자의 한국어 발화를 분석해 반드시 report_intent 도구를 정확히 한 번 호출해라.
자유 발화로 답하지 마라. 도구 호출이 응답의 전부다.

[목적지 목록] {PROBE_DESTINATIONS}
- destination_candidate 는 위 목록의 정확한 이름 또는 null. 새로 지어내지 마라.
- 간접 표현도 navigate 다 ("배 아파" -> 화장실, "배고파" -> 식당).
- 이동이 아닌 정보 질문은 question, 모호하면 clarify, 이해 불가면 unknown.
- 진행 중인 안내를 그만두려는 말("취소해줘", "안 갈래")은 cancel.
- reply 는 짧고 친절한 한국어."""

# _IntentDraft(langchain_intent_parser.py)와 같은 형태. LLM 이 내지 않는 값
# (pause/resume/cancel_keep — 규칙층 전담)은 enum 에서 뺐다. 파이프라인과 동일.
REPORT_INTENT_TOOL = {
    "type": "function",
    "name": "report_intent",
    "description": "사용자 발화를 분석한 결과를 보고한다. 로봇을 움직이지 않는다.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["navigate", "question", "clarify", "cancel", "unknown"],
            },
            "destination_candidate": {"type": ["string", "null"]},
            "is_confirmation": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reply": {"type": "string"},
        },
        "required": ["intent", "reply"],
    },
}


def _session_update() -> dict:
    """GA(2025-08 이후) 형식. 오류가 나면 아래 _legacy 주석 형식으로 바꿔 본다.

    구형(beta) 키 대응표:
        session.type / output_modalities  →  "modalities": ["text"]
        audio.input.format               →  "input_audio_format": "pcm16"
        audio.input.turn_detection       →  "turn_detection": None (최상위)
    """
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": INSTRUCTIONS,
            "tools": [REPORT_INTENT_TOOL],
            "tool_choice": "required",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": TARGET_RATE},
                    # VAD 를 끈다 — 파이프라인처럼 발화 창 단위로 직접 commit 한다.
                    "turn_detection": None,
                }
            },
        },
    }


def _load_wav_as_pcm24k(path: Path) -> bytes:
    """wav → mono 16bit 24kHz PCM. numpy 는 저장소 의존성에 이미 있다."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        sys.exit(f"16bit wav 만 지원한다 (지금 {width * 8}bit). arecord -f S16_LE 로 녹음해라.")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]  # 첫 채널만
    if rate != TARGET_RATE:
        duration = len(samples) / rate
        target_n = int(duration * TARGET_RATE)
        old_t = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        new_t = np.linspace(0.0, duration, num=target_n, endpoint=False)
        samples = np.interp(new_t, old_t, samples.astype(np.float32)).astype(np.int16)
        print(f"[리샘플] {rate} Hz → {TARGET_RATE} Hz ({duration:.2f}초)")
    return samples.tobytes()


async def _connect(model: str, api_key: str):
    """websockets 버전에 따라 헤더 인자 이름이 다르다 (v14+: additional_headers)."""
    import websockets

    headers = {"Authorization": f"Bearer {api_key}"}
    url = URL.format(model=model)
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:  # websockets < 14
        return await websockets.connect(url, extra_headers=headers, max_size=None)


async def probe(model: str, text: str | None, wav: Path | None) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit(".env 또는 환경변수에 OPENAI_API_KEY 가 필요하다. 커밋은 금지.")

    t_connect = time.monotonic()
    ws = await _connect(model, api_key)
    print(f"[연결] WebSocket open: {time.monotonic() - t_connect:.2f}초")

    await ws.send(json.dumps(_session_update()))

    t_send = None
    t_first = None
    exit_code = 0

    async def send_input() -> None:
        nonlocal t_send
        if text is not None:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }))
        else:
            pcm = _load_wav_as_pcm24k(wav)
            chunk = 256 * 1024
            for i in range(0, len(pcm), chunk):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i + chunk]).decode(),
                }))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        t_send = time.monotonic()
        await ws.send(json.dumps({"type": "response.create"}))

    sent = False
    async for raw in ws:
        event = json.loads(raw)
        etype = event.get("type", "?")

        if etype == "error":
            # 형식이 안 맞으면 여기로 온다. 전문을 그대로 남긴다 — 디버깅의 출발점.
            print(f"[오류] {json.dumps(event, ensure_ascii=False, indent=2)}")
            exit_code = 1
            break

        if etype in ("session.created", "session.updated"):
            print(f"[세션] {etype}: 연결 후 {time.monotonic() - t_connect:.2f}초")
            if etype == "session.updated" and not sent:
                sent = True
                await send_input()
            continue

        if etype == "response.function_call_arguments.delta":
            if t_first is None and t_send is not None:
                t_first = time.monotonic()
                print(f"[지연] 첫 인자 토큰: {t_first - t_send:.2f}초")
            continue

        if etype == "response.function_call_arguments.done":
            print("[결과] report_intent 인자:")
            try:
                args = json.loads(event.get("arguments", "{}"))
                print(json.dumps(args, ensure_ascii=False, indent=2))
                required_ok = all(k in args for k in ("intent", "reply"))
                enum_ok = args.get("intent") in REPORT_INTENT_TOOL["parameters"]["properties"]["intent"]["enum"]
                print(f"[스키마] 필수 필드 {'OK' if required_ok else '누락!'} / "
                      f"intent enum {'OK' if enum_ok else '위반: ' + repr(args.get('intent'))}")
                if not (required_ok and enum_ok):
                    exit_code = 2
            except json.JSONDecodeError as exc:
                print(f"[스키마] 인자가 JSON 이 아니다: {exc}")
                exit_code = 2
            continue

        if etype == "response.done":
            if t_send is not None:
                print(f"[지연] 전체 응답: {time.monotonic() - t_send:.2f}초 "
                      f"(현행 llm_sec ~3.5초와 비교)")
            usage = event.get("response", {}).get("usage")
            if usage:
                print(f"[토큰] {json.dumps(usage, ensure_ascii=False)}")
            # 도구 호출 없이 끝났는지 확인 — tool_choice=required 가 안 먹은 경우.
            outputs = event.get("response", {}).get("output", [])
            if not any(o.get("type") == "function_call" for o in outputs):
                print("[경고] function_call 없이 응답이 끝났다. tool_choice 강제가 "
                      "동작하지 않는다 — 결과에 기록할 것.")
                exit_code = 3
            break

        # 그 밖의 이벤트는 타입만 — 흐름 파악용.
        print(f"  · {etype}")

    await ws.close()
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="텍스트 발화로 탐침")
    group.add_argument("--wav", type=Path, help="16bit mono wav 로 탐침")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.wav is not None and not args.wav.is_file():
        sys.exit(f"wav 파일이 없다: {args.wav}")

    code = asyncio.run(probe(args.model, args.text, args.wav))
    sys.exit(code)


if __name__ == "__main__":
    main()
