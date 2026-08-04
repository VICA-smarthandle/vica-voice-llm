"""[SIM ONLY] 서비스 계측 — 단계별 시간과 시스템 사용량의 순수 로직.

사용자 체감 관점의 시간을 만든다:

  wake ─ user_text 간격    = 청취 창 + STT
  user_text ─ intent       = LLM 해석
  intent ─ tts_start       = TTS 합성 착수까지
  tts_request ─ tts_start  = 큐 대기 (발화 요청이 실제 소리가 되기까지)
  wake ─ tts_start         = **체감 응답 시간** (부르고 나서 로봇이 말 시작까지)
  emergency ─ estopped     = 긴급 반응 시간 (발행 → 가상 로봇 정지)

이벤트를 (종류, 시각)으로 먹이는 순수 로직이라 ROS 없이 단위 테스트할 수 있다.
"""
from __future__ import annotations

from typing import Optional

# TTS 노드는 한 발화를 문장 단위로 끊어 재생하므로(ros_tts_node) 재생 상태
# 신호가 발화 하나에 여러 번 켜졌다 꺼진다. 이 간격보다 짧게 이어지면 같은
# 재생으로 잇고, 길게 벌어지면 다른 발화로 보고 상호작용을 끊는다.
# ros_tts_node.TAIL_SEC(0.4) + 합성 착수 여유를 감안한 값.
TTS_GAP_SEC = 1.5


class SpanTracker:
    """토픽 이벤트 흐름을 상호작용(interaction) 단위의 구간으로 묶는다."""

    def __init__(self):
        self.interactions: list[dict] = []   # 완결된 상호작용
        self.emergencies: list[dict] = []    # 긴급 반응 기록
        self._cur: Optional[dict] = None     # 진행 중 상호작용
        self._pending_emergency: Optional[float] = None

    # ---------------------------------------------------------------- 입력
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
                # 공백이 크다 = 이 발화는 앞 상호작용과 무관하다 (예: 한참 뒤의
                # 도착 안내). 앞 상호작용을 마지막 문장 끝에서 닫고 흘려보낸다.
                self._close(prev_end, reason="tts_gap")
            elif "tts_start" not in self._cur:
                self._cur["tts_start"] = t
        elif kind == "tts_end" and self._cur is not None:
            # 문장마다 들어오므로 마지막 값이 곧 재생 종료다. 여기서 닫지 않는다.
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

    # ---------------------------------------------------------------- 내부
    def _close(self, t: float, reason: str) -> None:
        cur, self._cur = self._cur, None
        if cur is None:
            return
        spans = {"t": cur["wake"], "closed_by": reason}
        if "user_text" in cur:
            spans["listen_stt_sec"] = round(cur["user_text"] - cur["wake"], 3)
            spans["text"] = cur.get("text", "")
        else:
            spans["wake_silent"] = True     # 부르고 아무 말 없음 = 호출 오탐 후보
        if "intent" in cur and "user_text" in cur:
            spans["llm_sec"] = round(cur["intent"] - cur["user_text"], 3)
        if "tts_start" in cur:
            spans["response_sec"] = round(cur["tts_start"] - cur["wake"], 3)  # 체감
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
    try:  # PC(nvidia-smi). Jetson 은 tegrastats 별도 — sim-guide 참고
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
