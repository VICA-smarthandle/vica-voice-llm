"""빔포밍 회전(호출 방향 회전) 스위치 — 이름·`.env` 해석·앱 알림 문구 (2026-11-19 사용자).

켜면 "비카야"가 온 방향(DOA)을 `/vica/wake_doa` 로 미션에 보내 그쪽으로 돈다. 앱 대시보드
'빔포밍 회전' 스위치가 아래 서비스를 부르고 상태 토픽을 보고 그린다. 기동 값은 `.env`
`VICA_WAKE_DOA_TURN`, 음성을 다시 띄우면 그 값으로 돌아간다. ROS 를 import 하지 않는다.
"""
from __future__ import annotations

# 앱(VICA_Supervisor supervisor_provider.dart)이 같은 글자로 부른다.
WAKE_DOA_TURN_SERVICE = "/vica/wake_doa_turn/enable"        # std_srvs/SetBool
WAKE_DOA_TURN_STATE_TOPIC = "/vica/wake_doa_turn/enabled"   # std_msgs/Bool, latched


def wake_doa_turn_from_env(value: str | None) -> bool:
    """`.env` 값 → 켜짐 여부. 1·on·true(대소문자·공백 무시)만 켜짐, 나머지·없음은 꺼짐."""
    return (value or "").strip().lower() in ("1", "on", "true")


def wake_doa_turn_reply(on: bool) -> str:
    """서비스 응답 문구 — 앱이 알림으로 그대로 띄운다(캔버스 101번 보드)."""
    return "빔포밍 회전을 켰습니다" if on else "빔포밍 회전을 껐습니다"
