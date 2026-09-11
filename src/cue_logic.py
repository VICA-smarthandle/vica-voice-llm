"""청각 안내의 판정 로직 (소리·ROS 없이 검증되는 순수 로직).

/vica_goal_event 의 JSON 에서 이벤트 이름을 꺼내는 일만 남았다. 회전 신호를
발화로 바꾸던 TurnAnnouncer 는 2026-09-11 에 뺐다 — 회전 안내는 스마트핸들의
서보·LED 만으로 한다(사용자 결정). 복원은 git 이력 참고.

소리를 내는 것은 audio_cue.py, 토픽을 다루는 것은 ros_audio_cue_node.py 다.
"""
from __future__ import annotations

import json
from typing import Optional

# 안내가 끝났다고 보는 /vica_goal_event 이벤트 이름.
# 정본은 vica_ros2_ws 의 mission_manager_node._publish_goal_event 다.
GUIDANCE_END_EVENTS = frozenset({"goal_succeeded", "goal_canceled", "goal_failed"})
GUIDANCE_ARRIVED_EVENT = "goal_succeeded"

def parse_goal_event(payload: str) -> Optional[str]:
    """/vica_goal_event JSON 에서 event 문자열만 꺼낸다.

    이 토픽의 계약은 JSON 이다 (guideline/vica_architecture.md 토픽 표,
    vica_user_guidance.guidance_priority.parse_goal_event 와 같은 판정).
    평문 이벤트 이름은 계약이 아니므로 받지 않는다 — 두 형식을 다 허용하면
    계약이 다시 흐려진다. 파싱 실패나 event 키 부재는 None 이다. 예외를
    던지지 않는다 — 잘못된 payload 하나가 안내 노드를 죽이면 안 된다.
    """
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    event = data.get("event")
    return event if isinstance(event, str) else None
