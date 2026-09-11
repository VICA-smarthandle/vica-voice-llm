"""[SIM ONLY] 서비스 계측 — 단계별 시간과 시스템 사용량의 순수 로직."""
from __future__ import annotations

from typing import Optional

TTS_GAP_SEC = 1.5


class SpanTracker:
    """토픽 이벤트 흐름을 상호작용(interaction) 단위의 구간으로 묶는다."""

    def __init__(self):
        self.interactions: list[dict] = []
        self.emergencies: list[dict] = []
        self._cur: Optional[dict] = None
        self._pending_emergency: Optional[float] = None

    def feed(self, kind: str, t: float, detail: str = "") -> None:
        if kind == "wake":
            self._close(t, reason="next_wake")
            self._cur = {"wake": t}
        elif kind == "user_text" and self._cur is not None:
            self._cur["user_text"] = t
            self._cur["text"] = detail
        elif kind == "intent" and self._cur is not None and "intent" not in self._cur:
            self._cur["intent"] = t
        elif kind == "tts_request" and self._cur is not None and "tts_request" not in self._cur:
            self._cur["tts_request"] = t
        elif kind == "tts_start" and self._cur is not None:
            prev_end = self._cur.get("tts_end")
            if prev_end is not None and t - prev_end > TTS_GAP_SEC:
                self._close(prev_end, reason="tts_gap")
            elif "tts_start" not in self._cur:
                self._cur["tts_start"] = t
        elif kind == "tts_end" and self._cur is not None:
            self._cur["tts_end"] = t
        elif kind == "emergency":
            self._pending_emergency = t
            self._ekw = detail
        elif kind == "sim_event" and detail == "estopped" and self._pending_emergency is not None:
            self.emergencies.append({
                "t": self._pending_emergency,
                "keyword": getattr(self, "_ekw", ""),
                "react_sec": round(t - self._pending_emergency, 3),
            })
            self._pending_emergency = None

    def finalize(self, t: float) -> None:
        self._close(t, reason="finalize")

    def _close(self, t: float, reason: str) -> None:
        cur, self._cur = self._cur, None
        if cur is None:
            return
        spans = {"t": cur["wake"], "closed_by": reason}
        if "user_text" in cur:
            spans["listen_stt_sec"] = round(cur["user_text"] - cur["wake"], 3)
            spans["text"] = cur.get("text", "")
        else:
            spans["wake_silent"] = True
        if "intent" in cur and "user_text" in cur:
            spans["llm_sec"] = round(cur["intent"] - cur["user_text"], 3)
        if "tts_start" in cur:
            spans["response_sec"] = round(cur["tts_start"] - cur["wake"], 3)
            if "intent" in cur:
                spans["tts_launch_sec"] = round(cur["tts_start"] - cur["intent"], 3)
            if "tts_request" in cur:
                spans["tts_wait_sec"] = round(cur["tts_start"] - cur["tts_request"], 3)
        if "tts_end" in cur and "tts_start" in cur:
            spans["tts_play_sec"] = round(cur["tts_end"] - cur["tts_start"], 3)
        self.interactions.append(spans)


def sample_system() -> dict:
    """CPU/RAM(+가능하면 GPU) 사용량 스냅샷. 실패 항목은 조용히 뺀다."""
    out: dict = {}
    try:
        import psutil

        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        out["mem_pct"] = mem.percent
        out["mem_used_mb"] = round(mem.used / 1e6)
    except Exception:
        pass
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            gpu, gmem = r.stdout.strip().split(", ")
            out["gpu_pct"] = float(gpu)
            out["gpu_mem_mb"] = float(gmem)
    except Exception:
        pass
    return out
