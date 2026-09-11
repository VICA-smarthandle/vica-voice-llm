"""XVF-3000(reSpeaker) 실시간 상태 읽기 — VAD·발화감지·방향(DOA). 읽기 전용."""
from __future__ import annotations

import struct
from typing import Optional

USB_VID, USB_PID = 0x2886, 0x0018
TIMEOUT_MS = 1000

_LIVE_INT = {
    "VOICEACTIVITY": (19, 32),
    "SPEECHDETECTED": (19, 22),
    "DOAANGLE": (21, 0),
}

_AGC_DESIRED = (19, 2)

_GAMMA_ETAIL = (19, 16)
_NLATTEN = (19, 18)


def _write_param(dev, param: tuple, value, is_int: bool) -> Optional[float]:
    """칩에 한 값을 쓰고 읽어서 확인한다. 실패하면 None."""
    import struct
    import usb.util

    param_id, offset = param
    payload = (struct.pack("<iii", offset, int(value), 1) if is_int
               else struct.pack("<ifi", offset, float(value), 0))
    dev.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR
        | usb.util.CTRL_RECIPIENT_DEVICE, 0, 0, param_id, payload, TIMEOUT_MS)
    cmd = 0x80 | offset | (0x40 if is_int else 0)
    resp = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR
        | usb.util.CTRL_RECIPIENT_DEVICE, 0, cmd, param_id, 8, TIMEOUT_MS)
    lo, hi = struct.unpack("<ii", resp.tobytes())
    return float(lo) if is_int else lo * (2.0 ** hi)


def apply_echo_tuning(gamma_etail: Optional[float],
                      nlatten: Optional[int]) -> dict:
    """에코 억제 나사를 칩에 쓴다. 쓴 값(읽어서 확인한 값)을 돌려준다."""
    result: dict = {}
    if gamma_etail is None and nlatten is None:
        return result
    try:
        import usb.core

        dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
        if dev is None:
            return result
        if nlatten is not None:
            result["NLATTENONOFF"] = _write_param(dev, _NLATTEN, nlatten, True)
        if gamma_etail is not None:
            result["GAMMA_ETAIL"] = _write_param(
                dev, _GAMMA_ETAIL, gamma_etail, False)
    except Exception:
        return result
    return result


def echo_tuning_from_env(raw_etail: str, raw_nlatten: str) -> tuple:
    """(gamma_etail, nlatten) 해석. None = 쓰지 않음(공장 기본 유지)."""
    etail: Optional[float] = None
    raw = (raw_etail or "").strip().lower()
    if raw not in ("", "off", "none"):
        try:
            v = float(raw)
            etail = v if 0.0 <= v <= 3.0 else None
        except ValueError:
            etail = None
    nl: Optional[int] = None
    raw = (raw_nlatten or "").strip().lower()
    if raw in ("1", "on", "true"):
        nl = 1
    elif raw in ("0", "off", "false"):
        nl = 0
    return etail, nl


def agc_desired_from_env(raw: str) -> Optional[float]:
    """VICA_MIC_AGC_DESIRED 해석. None = 쓰지 않음(공장 기본 유지)."""
    raw = (raw or "").strip().lower()
    if raw in ("", "0", "off", "none"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 < value <= 1.0 else None


def apply_agc_desired_level(value: float) -> bool:
    """AGC 목표 레벨을 칩에 쓰고 읽어서 확인한다. 성공 여부를 돌려준다."""
    try:
        import usb.core
        import usb.util

        dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
        if dev is None:
            return False
        param_id, offset = _AGC_DESIRED
        payload = struct.pack("<ifi", offset, float(value), 0)
        dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0, param_id, payload, TIMEOUT_MS)
        resp = dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | offset, param_id, 8, TIMEOUT_MS)
        lo, hi = struct.unpack("<ii", resp.tobytes())
        readback = lo * (2.0 ** hi)
        usb.util.dispose_resources(dev)
        return abs(readback - value) < 1e-4
    except Exception:
        return False


class DspState:
    """reSpeaker 실시간 상태 리더. 생성 시 1회 시험 읽기로 가용성을 확정한다."""

    def __init__(self) -> None:
        self._dev = None
        try:
            import usb.core

            dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
            if dev is not None:
                self._dev = dev
                self._read("VOICEACTIVITY")
        except Exception:
            self._dev = None

    @property
    def available(self) -> bool:
        return self._dev is not None

    def _read(self, name: str) -> int:
        import usb.util

        param_id, offset = _LIVE_INT[name]
        cmd = 0x80 | 0x40 | offset
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
