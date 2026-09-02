"""스마트핸들 모드 첫 질문과 그 응답 판정 (순수 로직).

ROS·LLM·langchain 의존이 없어 개발용 컴퓨터에서 그대로 검증한다.
`cue_logic.py` 와 같은 위치의 모듈이며, 배선은 노드가 맡는다.

## 왜 규칙으로 판정하는가

첫 호출에서 로봇이 `replies.MODE_ASK` 로 묻고 사용자는 "네" 한 글자로 답한다.
이 한 글자를 LLM 에 넘기면 문맥이 없어 `unknown` 이 나오고, 로봇은 엉뚱하게
"잘 듣지 못했습니다"라고 답한다. 질문을 던진 쪽이 답도 받아야 한다.

`langchain_intent_parser` 의 목적지 확인 단축(`_pending_confirm_destination`)은
직전 AI 발화가 어떤 목적지의 `confirm_prompt` 일 때만 걸리므로 모드 질문에는
쓸 수 없다. 더구나 `MODE_ASK` 는 웨이크워드 노드가 말하고, `ros_node` 의
대화 history 에는 자기가 만든 응답만 들어가므로 애초에 조회되지 않는다.

## 전체 흐름 (터치센서 도입 후)

```text
"비카야" ──▶ WAKE_GREETING "네?"
   │
   ▼
사용자 발화
   ├─ 질문·잡담 ──▶ 평소대로 답한다. 모드를 묻지 않는다
   └─ 목적지     ──▶ confirm_prompt "○○로 안내해드릴까요?"
                        └ "네" ──▶ 목적지 확정 (Mission 이 보관, 아직 출발 안 함)
                                      │
                                      ├─ 상향 통신 두절 ──▶ HANDLE_UNAVAILABLE
                                      │                      ──▶ 비활성 모드로 출발
                                      ▼
                                   MODE_ASK
                                      ├─ "아니요" ──▶ MODE_DECLINED ──▶ 비활성 모드로 출발
                                      └─ "네"    ──▶ MODE_ENTER_REQUEST
                                                        │ 터치 3초  ← mission_manager
                                                        ▼
                                                     MODE_READY ──▶ MSG_START 출발
```

**모드는 안내 의사가 확인된 뒤에 묻는다.** 호출만으로는 안내를 원하는지 알 수
없다 — 층을 묻거나 잘못 부른 사람에게까지 모드를 물으면 대화가 어긋난다.

목적지가 확정되고도 바로 출발하지 않는다는 점에 주의한다. Mission Manager 가
목적지를 든 채 모드 진입을 기다리는 상태가 필요하다
(`devlog/2026-07-30-smart-handle-mode-decisions.md` §1 의 `[활성화 대기]`.
그 그림은 IDLE 직후였으나 여기서는 목적지 확정 뒤로 옮겨진다).

## 지금 구현하지 않는 것

- **모드 질문 자체를 아직 하지 않는다.** 터치센서가 미장착이라
  (`SmartHandleState.msg` 의 `user_contact` 는 항상 false) 3초를 감지할 수단이
  없다. `vica_scenario.md` 2-1.1 은 이 경우 "질문을 하지 않는다"고 정했다 —
  물어보면 사용자가 아무리 잡아도 왜 안 되는지 알 수 없기 때문이다. 센서가
  들어오기 전까지 첫 호출은 종전대로 `replies.WAKE_GREETING` 으로 답한다.
  이 모듈과 문구는 그때를 위해 미리 갖춰 둔 것이다.
- **모드 상태 자체.** 소유자는 `mission_manager_node` 이고 `/vica/handle_mode`
  신설이 선행된다(`vica_scenario.md` 2-1, `devlog/2026-07-30-smart-handle-mode-decisions.md`
  5절 표 4번 — 공용 계약 신설이라 승인 대상이다). 이 모듈은 "사용자가 뭐라고
  답했는지"까지만 판정한다.
- **터치 3초 진입·0.5초 놓침 판정.** 위와 같은 이유로 판정할 입력이 없다.
- **주행 중 모드 전환.** 시나리오 문서에 활성→비활성 경로가 없고,
  비활성→활성은 "터치로 활성화 불허"로 닫혀 있다. 말로 요청하는 경우는
  문서에 없어 팀 확정 전까지 두지 않는다.
"""
from __future__ import annotations

from typing import Optional

# 짧은 긍정·부정 응답의 정본. `langchain_intent_parser` 의 목적지 확인 단축도
# 이 목록을 쓴다. 두 곳에 따로 두면 한쪽만 고쳐져 "확인은 되는데 모드 질문은
# 안 되는" 상태가 생긴다.
AFFIRMATIVES = frozenset(
    {"네", "예", "응", "어", "그래", "그래요", "맞아", "맞아요",
     "좋아", "좋아요", "네네", "네맞아요", "응응", "가자", "가줘"}
)
NEGATIVES = frozenset(
    {"아니", "아니요", "아뇨", "아니야", "아니에요", "싫어", "싫어요", "취소"}
)
# 혼잣소리 계열 긍정 — whisper 가 "응"을 이렇게 적는다(2026-09-02 짧은답
# 30회: '응' 5회 중 '음' 2회·'으음' 1회). 목록 밖이라 매번 LLM 이 판정했고
# 같은 소리가 두 번은 unknown 으로 버려지고 한 번만 affirm 으로 살았다.
#
# **단독일 때만** 긍정으로 본다 — AFFIRMATIVES 와 나눠 둔 이유가 이것이다.
# 확인 질문의 첫 단어 지름길에는 넣지 않는다: "음… 아니야"의 첫 단어를
# 승낙으로 읽으면 망설임이 곧바로 주행이 된다.
SOFT_AFFIRMATIVES = frozenset({"음", "으음", "음음", "어어", "어어어"})

YES = "yes"
NO = "no"

# 질문 뒤 이 시간이 지나면 답으로 보지 않는다. Mission Manager 의
# `confirm_timeout_sec` 과 같은 값이다 — 사용자가 확인 질문에 답할 시간으로
# 이미 쓰이고 있어 체감이 어긋나지 않는다.
DEFAULT_ANSWER_WINDOW_SEC = 30.0


def normalize_short_reply(text: str) -> str:
    """STT 가 붙이는 구두점·공백을 제거해 짧은 답변을 비교 가능하게 만든다."""
    return "".join(ch for ch in text if ch.isalnum())


def classify_short_reply(text: str) -> Optional[str]:
    """짧은 긍정/부정이면 `YES`/`NO`, 아니면 None.

    "네, 화장실 가줘" 처럼 답과 목적지가 붙어 오면 None 이다. 정규화 결과가
    "네화장실가줘" 라 목록에 없기 때문이다. 이때는 모드를 정하지 않고 일반
    처리로 넘겨 목적지를 뽑는 편이 낫다 — 안내가 막히지 않는다.
    """
    word = normalize_short_reply(text)
    if word in AFFIRMATIVES or word in SOFT_AFFIRMATIVES:
        return YES
    if word in NEGATIVES:
        return NO
    return None


class ModeQuestion:
    """모드를 물었는지 기억하고, 바로 다음 발화를 그 답으로 본다.

    안내 한 건의 첫 호출에만 묻는다. 이 판정 자체는 `cue_logic.GreetingState`
    가 이미 하고 있으므로 여기서 되풀이하지 않는다 — 노드가 그 결과를 받아
    `on_asked()` 를 부른다.
    """

    def __init__(self, answer_window_sec: float = DEFAULT_ANSWER_WINDOW_SEC) -> None:
        self.answer_window_sec = answer_window_sec
        self._asked_at: Optional[float] = None

    @property
    def waiting(self) -> bool:
        """답을 기다리는 중인지. 시간 초과는 노드가 아니라 `take_answer` 가 본다."""
        return self._asked_at is not None

    def on_asked(self, now: float) -> None:
        """`MODE_ASK` 를 발화했다."""
        self._asked_at = now

    def take_answer(self, text: str, now: float) -> Optional[str]:
        """대기 중이면 발화를 답으로 해석한다. `YES`/`NO`, 아니면 None.

        답으로 해석되면 대기를 푼다. 목적지 발화처럼 답이 아닌 말이 오면 대기를
        유지한다 — 사용자가 질문을 못 듣고 목적지부터 말한 경우, 되물으면
        흐름이 끊기므로 그냥 안내를 진행시키고 모드는 정해지지 않은 채 둔다.
        시간이 지나면 `reset()` 없이도 스스로 만료된다.
        """
        if self._asked_at is None:
            return None
        if now - self._asked_at >= self.answer_window_sec:
            self._asked_at = None
            return None
        answer = classify_short_reply(text)
        if answer is not None:
            self._asked_at = None
        return answer

    def reset(self) -> None:
        """안내가 끝났거나 취소됐다. 다음 사용자를 위해 대기를 비운다."""
        self._asked_at = None
