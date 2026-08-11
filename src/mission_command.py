"""주행 조작 발화(일시정지·재개) 판정 (순수 로직).

ROS·LLM 의존이 없다. `handle_mode.py`·`cue_logic.py` 와 같은 위치의 모듈이다.

## 왜 취소는 여기 없는가

Mission Manager 는 셋을 다르게 다룬다. 위험이 다르기 때문이다.

    cancel   "안내를 취소할까요?" 로 **되묻는다**   (mission_manager_node:267)
    pause    **즉시** 정지
    resume   **즉시** 출발

취소는 잘못 알아들어도 사용자가 "아니"라고 하면 안내가 이어진다. 회복된다.
일시정지·재개는 되묻지 않으므로 오인식이 곧 실행이다. 손잡이를 잡고 뒤따르던
사용자 앞에서 로봇이 갑자기 서면 부딪힌다.

그래서 **취소만 LLM 재량**으로 두고, 일시정지·재개는 여기서 정해진 말로만
판정한다. 긴급어를 LLM 이전에 규칙으로 거르는 것과 같은 원칙이다.

## "멈춰"는 여기 없다

`emergency_filter` 가 먼저 잡아 E-stop 으로 보낸다. 여기 넣으면 도달하지 않는
죽은 항목이 되고, 두 경로가 같은 말을 다투는 것으로 읽혀 위험하다.
`docs/design.md` 가 "`잠깐`, `천천히`, `느리게`는 E-stop 이 아니다"로 이미
갈라 두었고, 이 모듈은 그 `잠깐` 쪽을 맡는다.
"""
from __future__ import annotations

from typing import Optional

from .handle_mode import AFFIRMATIVES, NEGATIVES, normalize_short_reply

PAUSE = "pause"
RESUME = "resume"

# 일시정지로 인정하는 말.
#
# "잠깐" 계열을 정본으로 둔다 — 사용자가 실제로 쓰는 말이고, design.md 가 이미
# E-stop 이 아니라고 갈라 두었다. "잠깐"과 "잠깐만"을 다르게 다루지 않는다.
# STT 가 어느 쪽으로 옮길지 사용자가 통제할 수 없어서, 나누면 같은 말을 했는데
# 어떤 때는 서고 어떤 때는 안 서게 된다.
#
# "멈춰줘"·"정지해줘"·"스톱"은 넣지 않는다. emergency_filter 가 먼저 잡는다.
PAUSE_PHRASES = frozenset({
    "잠깐", "잠깐만", "잠깐만요", "잠깐요",
    "잠시", "잠시만", "잠시만요",
    "세워", "세워줘", "세워주세요",
    "기다려", "기다려줘", "기다려주세요",
    "일시정지",
})

# 재개로 인정하는 말.
RESUME_PHRASES = frozenset({
    "다시출발", "다시출발해", "다시출발해줘", "다시가자", "다시가",
    "출발", "출발해", "출발해줘", "출발하자",
    "계속", "계속가", "계속가자", "계속해",
    "이제가자", "이제가",
})

# 확인 응답도 되고 재개도 되는 말. 문맥으로 가른다.
#
# `handle_mode.AFFIRMATIVES` 에도 들어 있다. 로봇이 방금 확인 질문을 했으면
# 그 대답이고, 일시정지 상태면 재개 요청이다. 사용자가 같은 말을 두 뜻으로
# 쓰는 것이 자연스러우므로 어느 한쪽을 포기하지 않는다(2026-08-10 결정).
AMBIGUOUS_RESUME_PHRASES = frozenset({"가자", "가줘", "가"})


# 취소 되묻기를 기다리는 시간. Mission Manager 의 confirm_timeout_sec 과 같은 값이다
# (mission_logic 이 이 시한을 넘기면 MSG_CANCEL_KEPT 로 안내를 이어간다). 음성 쪽이
# 더 길게 잡으면 로봇은 이미 포기했는데 음성만 확정을 보내는 어긋남이 생긴다.
DEFAULT_CANCEL_CONFIRM_WINDOW_SEC = 30.0


class CancelConfirm:
    """취소 되묻기에 대한 답을 받는다.

    ## 왜 필요한가

    Mission Manager 는 "취소해줘"에 곧바로 취소하지 않고 "안내를 취소할까요?"로
    되묻는다. 그런데 확정 경로가 `intent == "cancel"` 하나뿐이라
    (`mission_manager_node:267`) 사용자가 "네"라고 답하면 아무 일도 일어나지 않았다.

    목적지 확인 단축(`_pending_confirm_destination`)으로도 잡히지 않는다. 그것은
    직전 AI 발화가 **목적지의** `confirm_prompt` 일 때만 걸리는데, 되묻는 문장은
    Mission Manager 가 말하므로 `ros_node` 의 대화 기록에 아예 없기 때문이다.
    `MODE_ASK` 와 같은 구조의 문제다 — 질문한 쪽과 답을 받는 쪽이 다르다.

    그래서 **되물을 것을 아는 쪽**(취소 intent 를 내보낸 음성 노드)이 직접 기억한다.

    ## 게이트에 막혔을 때

    Mission Manager 가 게이트로 취소를 거부하면 되묻지 않는데, 음성 쪽은 그것을
    모른 채 기다린다. 이때 "네"가 오면 취소를 한 번 더 보내고 다시 거부되어
    거부 문구가 나온다. 안전 쪽으로 기우는 실패이며(취소가 일어나지 않는다)
    시한이 지나면 스스로 풀린다.
    """

    def __init__(
        self, window_sec: float = DEFAULT_CANCEL_CONFIRM_WINDOW_SEC
    ) -> None:
        self.window_sec = window_sec
        self._asked_at: Optional[float] = None

    @property
    def waiting(self) -> bool:
        return self._asked_at is not None

    def on_requested(self, now: float) -> None:
        """취소 intent 를 내보냈다. Mission Manager 가 곧 되묻는다."""
        self._asked_at = now

    def take_answer(self, text: str, now: float) -> Optional[bool]:
        """대기 중이면 발화를 답으로 읽는다. True=확정, False=철회, None=답 아님.

        답으로 읽히면 대기를 푼다. 시한을 넘긴 뒤의 답은 무시하고 대기도 푼다 —
        Mission Manager 쪽 시한도 같은 값이라 이미 포기한 상태다.
        """
        if self._asked_at is None:
            return None
        if now - self._asked_at >= self.window_sec:
            self._asked_at = None
            return None

        word = normalize_short_reply(text)
        # "취소"를 다시 말하는 것도 확정이다. Mission Manager 가 원래 기대하던 형태다.
        if word in AFFIRMATIVES or word in _CANCEL_REPEAT_WORDS:
            self._asked_at = None
            return True
        if word in NEGATIVES:
            self._asked_at = None
            return False
        return None

    def reset(self) -> None:
        """안내가 끝났거나 취소가 확정됐다. 대기를 비운다."""
        self._asked_at = None


# 되묻기에 "취소"라고 다시 답하는 경우. NEGATIVES 에도 "취소"가 있어 순서가
# 중요하다 — 되묻는 중의 "취소"는 부정이 아니라 확정이다.
_CANCEL_REPEAT_WORDS = frozenset({"취소", "취소해", "취소해줘", "취소요"})


def classify_mission_command(
    text: str,
    *,
    confirm_pending: bool = False,
    is_paused: bool = False,
) -> Optional[str]:
    """일시정지·재개 발화면 `PAUSE`/`RESUME`, 아니면 None.

    ``confirm_pending``
        직전에 로봇이 목적지 확인 질문을 했다. 이때 "가자"는 그 대답이므로
        재개로 보지 않는다. 확인 응답 처리가 이 판정보다 앞선다.
    ``is_paused``
        로봇이 목적지를 보관한 채 멈춰 있다(`/vica/robot_state` 의 `is_paused`).
        이때만 모호한 말을 재개로 읽는다.

    실행 가능 여부는 판정하지 않는다. 주행 중이 아닐 때의 거부는 Mission
    Manager 의 게이트(`check_pause_gate` 등)가 맡고 거부 문구도 그쪽이 말한다.
    여기서 미리 걸러내면 같은 판단이 두 저장소로 갈라진다.
    """
    word = normalize_short_reply(text)
    if not word:
        return None
    if word in PAUSE_PHRASES:
        return PAUSE
    if word in RESUME_PHRASES:
        return RESUME
    if is_paused and not confirm_pending and word in AMBIGUOUS_RESUME_PHRASES:
        return RESUME
    return None
