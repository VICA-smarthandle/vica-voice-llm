"""XVF-3000(reSpeaker) 실시간 상태 읽기 — VAD·발화감지·방향(DOA). 읽기 전용.

용도: 음성 barge-in 재설계. RMS(소리 크기)는 자기 잔여 에코와 사람을 못
가린다는 것이 실측으로 두 번 확인됐다(2026-08-24 자책골). 칩의 VAD 는 AEC
처리 뒤에 계산되므로 자기 재생음에는 반응하지 않아야 한다 — **이 가정
자체를 tools/vad_probe 로 실측해 통과한 경우에만** 판정에 쓴다.

레지스터 주소는 Seeed 공식 usb_4_mic_array `tuning.py` 의 표와 같다
(vica-wakeword `recorder/dsp_dump.py` 와 같은 출처·같은 읽기 방식).
DspState 는 어떤 파라미터도 쓰지 않는다 — DSP 설정 동결(D7)과 무관한
상태값 읽기뿐이다. 장치·권한·pyusb 가 없으면 available=False 로 조용히
비활성화된다(파이프라인을 막지 않는다).

D7 동결의 승인된 예외 1건(2026-08-28 사용자 결정): AGCDESIREDLEVEL.
칩 기본 목표(0.005)가 낮아 1m 밖 발화가 rms 0.03 대로 들어와 짧은 답
오전사('그래'→'굿에이')의 바닥 원인이 됐다 — 목표 2배(0.010) 실측으로
개선 확인. 칩은 전원 재투입마다 초기화되므로 노드 기동 시마다
apply_agc_desired_level 로 다시 쓴다. 다른 파라미터 쓰기는 여전히 금지.
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

# AGC 목표 레벨 (id19, offset2, float) — D7 예외로 승인된 유일한 쓰기 대상
_AGC_DESIRED = (19, 2)

# 에코 억제 나사 (2026-09-04 사용자 승인 — D7 동결의 두 번째 예외).
# 촬영 장소가 복도라 울림이 AEC 한계를 넘었다. 칩이 지우는 세기를 올린다.
# 값은 Seeed 공식 tuning.py 표 기준이며 전부 rw 다. 공장 기본은 셋 다
# 1.0 / 0 이고(2026-09-04 실측), 전원을 껐다 켜면 그 값으로 돌아간다 —
# 그래서 노드가 뜰 때마다 다시 쓴다(AGC 와 같은 이유).
#
#   GAMMA_ETAIL   0~3  에코 **꼬리** 과잉 차감 배율 ← 복도 울림이 이것
#   NLATTENONOFF  0/1  비선형 에코 억제 스위치. 스피커가 크면 원신호와
#                      닮지 않은 왜곡 에코가 생기는데 일반 AEC 로는 못 지운다
#
# 올리면 에코가 더 지워지는 만큼 **사람 목소리도 깎인다.** 올린 뒤에는
# 반드시 "비카야"가 여전히 걸리는지 확인할 것.
_GAMMA_ETAIL = (19, 16)
_NLATTEN = (19, 18)


def _write_param(dev, param: tuple, value, is_int: bool) -> Optional[float]:
    """칩에 한 값을 쓰고 읽어서 확인한다. 실패하면 None.

    전송 형식은 Seeed tuning.py 와 같다 — payload 의 마지막 필드가 형 표시
    (int=1, float=0)이고, 읽기는 0x80(read) | 0x40(int) | offset 이다.
    """
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
    """에코 억제 나사를 칩에 쓴다. 쓴 값(읽어서 확인한 값)을 돌려준다.

    AGC 와 같은 제약: **마이크 스트림을 열기 전에** 부를 것. 스트림과 겹치면
    칩이 제어 전송을 거부한다(Pipe error). 실패해도 예외를 내지 않는다 —
    에코가 조금 더 샐 뿐 감시는 계속돼야 한다.
    """
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
    """(gamma_etail, nlatten) 해석. None = 쓰지 않음(공장 기본 유지).

    범위는 Seeed 표 그대로 — etail 0~3, nlatten 0/1. 벗어난 값은 무시한다.
    """
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
    """VICA_MIC_AGC_DESIRED 해석. None = 쓰지 않음(공장 기본 유지).

    범위는 (0, 1] — rms 목표라 1 을 넘을 수 없고, 음수·쓰레기 값이
    칩에 들어가는 것을 막는다.
    """
    raw = (raw or "").strip().lower()
    if raw in ("", "0", "off", "none"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 < value <= 1.0 else None


def apply_agc_desired_level(value: float) -> bool:
    """AGC 목표 레벨을 칩에 쓰고 읽어서 확인한다. 성공 여부를 돌려준다.

    마이크 스트림을 열기 전에 호출할 것 — 스트림과 겹치면 장치가 제어
    전송을 거부할 수 있다(모듈 상단 주석의 실측). 실패해도 예외를 내지
    않는다: 수음이 조금 작을 뿐 감시는 계속돼야 한다.
    """
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
