"""장면 → 원인 판정 (설계 2026-11-20 5.3). 로그 사실만 쓰고 추측은 `추정`·`확인 필요`로 표시한다.

status: 확정 | 추정 | 정상 | 미분류 | 확인 필요.  `flagged` 는 정상만이면(또는 비었으면) False.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tools.scene.events import Event
from tools.scene.scenes import Scene, state_at

LLM_TIMEOUT_S = 15.0           # src/langchain_intent_parser.py:554 (timeout: float = 15)
TIMEOUT_TOL_S = 0.2
NEAR_AFTER_FIRE_S = 2.0        # 확정 직후 같은 말
OVERLAP_S = 2.0                # 서로 다른 동작의 질문 요청 간격
SAME_ACTION_S = 0.1            # 이보다 가까운 요청은 한 동작의 연속(사용법 안내 두 문장)
RESERVE_NEAR_S = 0.5           # 말 요청 뒤 질문 예약이 이 안에 나오면 질문
SLOW_RESPONSE_S = 5.0          # user_text → 첫 재생 (ctx["slow_response_s"] 로 바꿈)
APPROACH_STATE = "awaiting_user"   # ros_wakeword_node.py:488 — 이때 접근 대답 최소를 쓴다

STATUSES = ("확정", "추정", "정상", "미분류", "확인 필요")


@dataclass
class Cause:
    name: str
    status: str
    evidence: list[str] = field(default_factory=list)
    note: str = ""


def flagged(causes: list[Cause]) -> bool:
    return any(c.status != "정상" for c in causes)


# ── ctx ──────────────────────────────────────────────────────────────

_RE_WAKE = re.compile(r"호출 문턱 ([\d.]+)")
_RE_LISTEN = re.compile(r"수음 최소 ([\d.]+)")
_RE_APPROACH = re.compile(r"접근 대답 최소 ([\d.]+)")


def make_ctx(events: list[Event], t: float) -> dict:
    """시각 t 앞의 가장 가까운 `startup` 줄들로 문턱을 채운다(실행마다 값이 다르다). 없으면 None."""
    ctx = {"wake_threshold": None, "listen_min_rms": None, "approach_min_rms": None, "llm_timeout": LLM_TIMEOUT_S}
    for pat, key in ((_RE_WAKE, "wake_threshold"), (_RE_LISTEN, "listen_min_rms"), (_RE_APPROACH, "approach_min_rms")):
        for e in reversed(events):
            if e.t <= t and e.kind == "startup":
                m = pat.search(e.data.get("text") or e.raw)
                if m:
                    ctx[key] = float(m.group(1))
                    break
    return ctx


# ── 판정 ─────────────────────────────────────────────────────────────

def _fmt(e: Event) -> str:
    return f"{e.kind}@{e.t:.3f} {e.raw}".strip()[:160]


def _timing_key(e: Event):
    d = e.data
    return tuple(d.get(k) for k in ("wait", "speech", "endpoint", "stt"))


def judge(scene: Scene, ctx: dict) -> list[Cause]:
    evs = sorted(scene.events, key=lambda e: e.t)
    ctx = ctx or {}
    out: list[Cause] = []
    all_events = ctx.get("all_events") or evs
    by = lambda *kinds: [e for e in evs if e.kind in kinds]

    # 호출 놓침 / 확정 직후 같은 말
    fires = [e.t for e in by("wake_fire")] + list(ctx.get("prev_fire_ts", []))
    for e in by("wake_near"):
        th = ctx.get("wake_threshold")
        th = e.data.get("threshold") if th is None else th
        peak = e.data.get("peak", 0.0)
        if th is not None and peak < th:
            continue
        if any(0 <= e.t - f <= NEAR_AFTER_FIRE_S for f in fires):
            out.append(Cause("정상: 확정 직후 같은 말", "정상", [_fmt(e)]))
        else:
            out.append(Cause("호출 놓침", "추정", [_fmt(e)],
                             f"점수 {peak:.2f} ≥ 문턱 {th} 인데 확정 안 됨 (넘은 조각 수는 2단계 기록 뒤 확정)"))

    # 대기 중 위치 알림
    for e in evs:
        if e.kind == "wake_ignored" or (e.kind == "mis_wake" and not e.data.get("heard")):
            out.append(Cause("정상: 대기 중 위치 알림", "정상", [_fmt(e)]))

    # ghost: 목소리 작음 / 짧음
    ghosts = by("listen_ghost")
    for e in ghosts:
        approach = state_at(all_events, e.t) == APPROACH_STATE
        th = ctx.get("approach_min_rms") if approach else None
        if th is None:
            th = ctx.get("listen_min_rms")
        rms = e.data.get("rms", 0.0)
        if th is None:
            out.append(Cause("원인 미분류", "확인 필요", [_fmt(e)], "그때 수음 최소를 모름(기동 줄 없음)"))
        elif rms < th:
            out.append(Cause("목소리 작음", "확정", [_fmt(e)],
                             f"rms {rms:.4f} < {'접근 대답' if approach else '수음'} 최소 {th} · 소리 재생 가능"))
        else:
            out.append(Cause("말이 너무 짧아 버림", "확정", [_fmt(e)],
                             f"rms {rms:.4f} ≥ 문턱 {th}, 발화 {e.data.get('speech')} s (짧은 말 조건)"))

    # 말소리 안 잡힘
    for e in by("listen_empty_free"):
        if not ghosts:
            out.append(Cause("말소리 안 잡힘", "확정", [_fmt(e)], "자유 창 빈손 종료, ghost 없음"))
    for e in by("q_close"):
        if e.data.get("reason") == "silence":
            out.append(Cause("말소리 안 잡힘", "확정", [_fmt(e)], "질문 창 닫힘 사유=silence"))

    # LLM 시간 초과 / 재청취 기각
    timeout = float(ctx.get("llm_timeout") or LLM_TIMEOUT_S)
    fails = by("llm_fail")
    users = by("user_text")
    timed_out_users: set[float] = set()
    for u in users:
        j = next((e for e in evs if e.t > u.t and e.kind in ("relisten_drop", "llm_input", "llm_rt")), None)
        if j is None or abs((j.t - u.t) - timeout) > TIMEOUT_TOL_S:
            continue
        timed_out_users.add(u.t)
        has_fail = any(u.t <= f.t <= j.t + 1.0 for f in fails)
        out.append(Cause("LLM 시간 초과", "확정" if has_fail else "추정", [_fmt(u), _fmt(j)],
                         f"받아쓰기 → 판정 {j.t - u.t:.2f} s (제한 {timeout:g} s)"
                         + ("" if has_fail else " · stderr `[LLM] 호출 실패` 없음")))
    for e in by("relisten_drop"):
        out.append(Cause("재청취 기각", "확정", [_fmt(e)], f"intent={e.data.get('intent')}"))

    # 같은 말 두 번 전달
    for f in by("forward_answer"):
        before = [e for e in evs if e.kind == "user_text" and e.t < f.t]
        after = [e for e in evs if e.kind == "user_text" and e.t >= f.t]
        if not before or not after:
            continue
        b, a = before[-1], after[0]
        tb = [e for e in evs if e.kind == "listen_timing" and e.t >= b.t - 0.5 and e.t < f.t]
        ta = [e for e in evs if e.kind == "listen_timing" and e.t >= a.t - 0.5 and e.t > f.t]
        if tb and ta and b.data.get("text") == a.data.get("text") and _timing_key(tb[0]) == _timing_key(ta[0]):
            out.append(Cause("같은 말 두 번 전달", "확정", [_fmt(b), _fmt(f), _fmt(a)],
                             f"'{a.data.get('text')}' 같은 계측 값 {_timing_key(ta[0])}"))

    # 화자 판정
    for e in by("spk"):
        if e.data.get("dropped"):
            out.append(Cause("화자 판정으로 버림", "확정", [_fmt(e)], "맞게 막았는지는 marks.tsv 가 있을 때만 판정"))
        elif "목소리 미달(대기)" in (e.data.get("reason") or "") or "목소리 미달(대기)" in e.raw:
            out.append(Cause("대기 중 본인 확인 미달", "확정", [_fmt(e)], "맞게 막았는지는 marks.tsv 가 있을 때만 판정"))

    # 질문 겹침
    reserves = [e.t for e in by("listen_reserve")] + [e.t for e in by("q_open")]
    qsays = []
    for e in by("mis_say"):
        if e.data.get("priority") != "response":
            continue
        is_q = any(-0.05 <= r - e.t <= RESERVE_NEAR_S for r in reserves) or \
            (e.data.get("text", "").endswith("?") and len(e.data.get("text", "")) > 3)
        if is_q:
            qsays.append(e)
    actions: list[list[Event]] = []
    for e in qsays:
        if actions and e.t - actions[-1][-1].t < SAME_ACTION_S:
            actions[-1].append(e)
        else:
            actions.append([e])
    for a, b in zip(actions, actions[1:]):
        if b[0].t - a[0].t <= OVERLAP_S:
            out.append(Cause("질문 겹침", "확정", [_fmt(a[0]), _fmt(b[0])],
                             f"서로 다른 동작의 질문 요청 간격 {b[0].t - a[0].t:.2f} s"))

    # 느린 응답
    slow = float(ctx.get("slow_response_s") or SLOW_RESPONSE_S)
    forwards = [f.t for f in by("forward_answer")]
    plays = by("tts_play")
    for i, u in enumerate(users):
        if u.t in timed_out_users or any(0 <= u.t - f <= 0.5 for f in forwards):
            continue                                       # 시간 초과가 이미 설명 / 같은 말 두 번째는 일부러 조용함
        nxt_user = users[i + 1].t if i + 1 < len(users) else float("inf")
        p = next((e for e in plays if u.t < e.t < nxt_user), None)
        if p is not None and p.t - u.t > slow:
            out.append(Cause("느린 응답", "확정", [_fmt(u), _fmt(p)], f"받아쓰기 → 첫 재생 {p.t - u.t:.1f} s (기준 {slow:g} s)"))

    # USB / 재생 죽음
    for e in by("usb"):
        out.append(Cause("USB 끊김", "확정", [_fmt(e)]))
    silent = set(ctx.get("silent_play_ts", []))
    for e in plays:
        if e.t in silent:
            out.append(Cause("재생 죽음", "추정", [_fmt(e)], "재생 줄은 있는데 aec_log 기준 신호 없음"))

    # 미분류
    if not out:
        bad = by("listen_reject") or [
            u for u in users if not any(e.t >= u.t and e.kind in ("tts_play", "mis_say", "mis_req") for e in evs)]
        if bad:
            if users and not by("spk"):
                out.append(Cause("원인 미분류", "확인 필요", [_fmt(bad[0])],
                                 "시연자·옆사람 말인지 spk 사건·marks.tsv 가 없어 가를 수 없음"))
            else:
                out.append(Cause("원인 미분류", "미분류", [_fmt(bad[0])], "흐름만 보임"))
    return out
