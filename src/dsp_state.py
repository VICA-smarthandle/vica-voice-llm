"""XVF-3000(reSpeaker) 실시간 상태 읽기 — VAD·발화감지·방향(DOA). 읽기 전용.

용도: 음성 barge-in 재설계. RMS(소리 크기)는 자기 잔여 에코와 사람을 못
가린다는 것이 실측으로 두 번 확인됐다(2026-08-24 자책골). 칩의 VAD 는 AEC
처리 뒤에 계산되므로 자기 재생음에는 반응하지 않아야 한다 — **이 가정
자체를 tools/vad_probe 로 실측해 통과한 경우에만** 판정에 쓴다.

레지스터 주소는 Seeed 공식 usb_4_mic_array `tuning.py` 의 표와 같다
(vica-wakeword `recorder/dsp_dump.py` 와 같은 출처·같은 읽기 방식).
이 모듈은 어떤 파라미터도 쓰지 않는다 — DSP 설정 동결(D7)과 무관한
상태값 읽기뿐이다. 장치·권한·pyusb 가 없으면 available=False 로 조용히
비활성화된다(파이프라인을 막지 않는다).
"""
from __future__ import annotations

import struct
from typing import Optional

USB_VID, USB_PID = 0x2886, 0x0018
TIMEOUT_MS = 1000

# name: (param_id, offset) — 전부 int 형 읽기 전용 상태값 (Seeed tuning.py 표)
_LIVE_INT = {
    "VOICEACTIVITY": (19, 32),   # AEC 후단 VAD: 사람 음성 존재
    "SPEECHDETECTED": (19, 22),  # 잡음억제 쪽 발화 감지
    "DOAANGLE": (21, 0),         # 도래 방향 0~359도
}


class DspState:
    """reSpeaker 실시간 상태 리더. 생성 시 1회 시험 읽기로 가용성을 확정한다."""

    def __init__(self) -> None:
        self._dev = None
        try:
            import usb.core

            dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
            if dev is not None:
                self._dev = dev
                self._read("VOICEACTIVITY")   # 권한·주소 검증 겸 시험 읽기
        except Exception:
            self._dev = None

    @property
    def available(self) -> bool:
        return self._dev is not None

    def _read(self, name: str) -> int:
        import usb.util

        param_id, offset = _LIVE_INT[name]
        cmd = 0x80 | 0x40 | offset          # 0x80=read, 0x40=int 형
        resp = self._dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, param_id, 8, TIMEOUT_MS)
        lo, _hi = struct.unpack("ii", resp.tobytes())
        return lo

    def _read_optional(self, name: str) -> Optional[int]:
        if self._dev is None:
            return None
        try:
            return self._read(name)
        except Exception:
            return None

    def voice_activity(self) -> Optional[bool]:
        """칩 VAD. None = 읽을 수 없음(호출자는 '증거 없음'으로 처리)."""
        value = self._read_optional("VOICEACTIVITY")
        return None if value is None else bool(value)

    def speech_detected(self) -> Optional[bool]:
        value = self._read_optional("SPEECHDETECTED")
        return None if value is None else bool(value)

    def doa_angle(self) -> Optional[int]:
        return self._read_optional("DOAANGLE")

    def close(self) -> None:
        if self._dev is not None:
            try:
                import usb.util

                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            self._dev = None
