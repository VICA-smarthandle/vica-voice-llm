"""휴대폰 화면 정적 검사 — 스크립트가 찾는 고정 id 가 모두 있고, id 가 겹치지 않는다."""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "tools" / "mic_check_page.html"
BUILT = ("ctl-", "val-", "chg-", "warn-", "lbl-")      # 값 고치기 칸은 스크립트가 만든다


def test_ids_used_by_script_exist_once():
    html = PAGE.read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script))
    used = {u for u in used if not u.startswith(BUILT)}
    ids = re.findall(r'\sid="([A-Za-z0-9_-]+)"', html.split("<script>", 1)[0])
    assert len(ids) == len(set(ids)), "id 가 겹친다"
    missing = sorted(used - set(ids))
    assert not missing, f"화면에 없는 id: {missing}"


def test_no_external_resources():
    html = PAGE.read_text(encoding="utf-8")
    assert "http://" not in html.replace("ws://", "") and "https://" not in html   # 시연장 인터넷 없이도


def test_every_saved_value_has_a_control():
    """`.env` 에 저장하는 값(env_file.KEYS)마다 화면에 고치는 칸이 있다 — 10-21 최대 증폭·접근 대답 최소·대역 최소."""
    from src.env_file import KEYS

    script = PAGE.read_text(encoding="utf-8").split("<script>", 1)[1]
    sliders = script.split("const CTL = [", 1)[1].split("]", 1)[0]
    for field in {("echo" if k.startswith("echo_") else k) for k in KEYS}:
        assert f'"{field}"' in sliders or f'seg("{field}"' in script, field


def test_step_count_matches_session():
    from src.mic_check_session import TOTAL_STEPS

    assert f"단계 – / {TOTAL_STEPS}" in PAGE.read_text(encoding="utf-8")


def test_save_rule_ratio_comes_from_the_server():
    """저장 검사 비율(SHORT_OVER_LISTEN)은 화면에 따로 적지 않고 상태 메시지로 받는다 — 둘이 어긋나지 않게."""
    script = PAGE.read_text(encoding="utf-8").split("<script>", 1)[1]
    assert "S.short_over_listen" in script and "listen_min_rms * 1.5" not in script
