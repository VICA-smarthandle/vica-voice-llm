"""휴대폰 화면 정적 검사 — 스크립트가 찾는 고정 id 가 모두 있고, id 가 겹치지 않는다."""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "tools" / "mic_check_page.html"
BUILT = ("ctl-", "val-", "chg-", "warn-", "lbl-")


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
    assert "http://" not in html.replace("ws://", "") and "https://" not in html
