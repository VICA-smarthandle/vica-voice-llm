"""빔포밍 회전(호출 방향 회전) 스위치 — 이름·`.env` 해석·앱 알림 문구 (2026-11-19 사용자)."""
from __future__ import annotations

WAKE_DOA_TURN_SERVICE = "/vica/wake_doa_turn/enable"
WAKE_DOA_TURN_STATE_TOPIC = "/vica/wake_doa_turn/enabled"


def wake_doa_turn_from_env(value: str | None) -> bool:
    """`.env` 값 → 켜짐 여부. 1·on·true(대소문자·공백 무시)만 켜짐, 나머지·없음은 꺼짐."""
    return (value or "").strip().lower() in ("1", "on", "true")


def wake_doa_turn_reply(on: bool) -> str:
    """서비스 응답 문구 — 앱이 알림으로 그대로 띄운다(캔버스 101번 보드)."""
    return "빔포밍 회전을 켰습니다" if on else "빔포밍 회전을 껐습니다"
