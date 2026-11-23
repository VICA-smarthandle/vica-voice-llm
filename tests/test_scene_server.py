"""대화 장면 분석기 — 웹 서버 (설계 2026-11-20 5.5)."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tools.scene import server
from tools.scene.causes import Cause
from tools.scene.events import Event
from tools.scene.expected import Expected
from tools.scene.scenes import Scene


def _fake_analyze(since, until):
    evs = [Event(1000.0, "wake", "wake_open", {}, "🙋 비카야 호출 #1"),
           Event(1003.0, "wake", "listen_empty_free", {}, "빈손 종료"),
           Event(1005.0, "wake", "user_text", {"text": "안녕"}, "-> /vica/user_text: '안녕'")]
    sc = Scene("20261104-000000-wake", 1000.0, 1700.0, "wake", evs, "time", None)
    ok = Scene("20261104-001000-wake", 1700.0, 1710.0, "wake",
               [Event(1700.0, "wake", "wake_open", {}, "x")], "time", None)
    causes = {sc.id: [Cause("말소리 안 잡힘", "확정", ["e"], "자유 창 빈손 종료")],
              ok.id: [Cause("정상: 확정 직후 같은 말", "정상", [])]}
    tally = {"wake_fire": 1, "wake_near_over": 2, "wake_near_over_after_fire": 1, "ghost": 3, "ghost_followup": 2,
             "ghost_after_wake": 1, "relisten_drop": 0, "after_wake": {}, "overlap": 0, "flagged": [sc.id]}
    return evs, [sc, ok], causes, tally


@pytest.fixture
def api(tmp_path):
    listen = tmp_path / "listen"
    wake = tmp_path / "wake"
    listen.mkdir()
    wake.mkdir()
    (listen / "a.wav").write_bytes(b"RIFFxxxx")
    (tmp_path / "secret.wav").write_bytes(b"no")

    def refuse(scene, path, events):
        raise server.ReaskRefused("주행 중이라 로컬 다시 묻기 거부")

    svc = server.Service(analyze_fn=_fake_analyze, reask_fn=refuse,
                         rule_fn=lambda scene, events: Expected("deny", "규칙", "문장", None, ""),
                         clip_dirs=(listen, wake), cache_dir=tmp_path / "cache")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", svc, (listen, wake)
    srv.shutdown()


H = {"X-Scene-Viewer": "1"}


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_clip_path_outside_dirs_rejected(tmp_path):
    listen = tmp_path / "l"
    listen.mkdir()
    (listen / "a.wav").write_bytes(b"x")
    (tmp_path / "x.wav").write_bytes(b"x")
    dirs = (listen,)
    assert server.resolve_clip("a.wav", dirs) == (listen / "a.wav").resolve()
    for bad in ("../x.wav", "/etc/passwd", str(tmp_path / "x.wav"), "a/../../x.wav", "", "nope.wav"):
        assert server.resolve_clip(bad, dirs) is None
    (listen / "link.wav").symlink_to(tmp_path / "x.wav")
    assert server.resolve_clip("link.wav", dirs) is None


def test_clip_endpoint_statuses(api):
    base, _, _ = api
    assert get(f"{base}/api/clip?file=a.wav")[0] == 200
    assert get(f"{base}/api/clip?file=..%2Fsecret.wav")[0] == 403
    assert get(f"{base}/api/clip?file=%2Fetc%2Fpasswd")[0] == 403


def test_api_scenes_tally_and_list(api):
    base, _, _ = api
    code, body = get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    d = json.loads(body)
    assert code == 200 and d["tally"]["wake_fire"] == 1 and d["tally"]["ghost_followup"] == 2
    assert [s["id"] for s in d["scenes"] if s["flagged"]] == ["20261104-000000-wake"]
    assert d["scenes"][0]["user_text"] == "안녕"


def test_api_scene_json_shape(api):
    base, _, _ = api
    get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    code, body = get(f"{base}/api/scene/20261104-000000-wake")
    d = json.loads(body)
    assert code == 200
    assert {"causes", "expected", "events", "stages", "clips", "silent"} <= set(d)
    assert d["silent"] == "말소리 안 잡힘" and len(d["stages"]) == 8
    assert d["expected"]["rule"]["sentence"] == "문장"
    assert d["events"][-1]["t"] == 1005.0
    assert get(f"{base}/api/scene/%EC%97%86%EC%9D%8C")[0] == 404


def test_reask_endpoint_refuses_when_driving(api):
    base, _, _ = api
    get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    req = urllib.request.Request(f"{base}/api/reask/20261104-000000-wake?path=midm", method="POST",
                                 headers={"X-Scene-Viewer": "1"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 409
    assert "주행 중" in json.loads(ei.value.read())["error"]


def test_reask_endpoint_ok_and_bad_path(api):
    base, svc, _ = api
    svc.reask_fn = lambda scene, path, events: {"intent": "affirm", "by": "LLM 추정", "sentence": "s",
                                                "reply": None, "note": path}
    get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    ok = urllib.request.urlopen(urllib.request.Request(f"{base}/api/reask/20261104-000000-wake?path=gpt", method="POST", headers=H))
    assert json.loads(ok.read())["note"] == "gpt"
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(urllib.request.Request(f"{base}/api/reask/20261104-000000-wake?path=zzz", method="POST", headers=H))
    assert ei.value.code == 400


def test_page_served(api):
    base, _, _ = api
    code, body = get(base + "/")
    assert code == 200 and "장면 분석기".encode() in body


def test_reask_child_reports_refusal(tmp_path, monkeypatch, capsys):
    from tools.scene import __main__ as m, expected

    def boom(*a, **k):
        raise RuntimeError("주행 중이라 로컬 다시 묻기 거부")

    monkeypatch.setattr(expected, "reask", boom)
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"scene": {"id": "x", "start": 1, "end": 2, "trigger": "wake", "events": []},
                             "events": []}), encoding="utf-8")
    assert m.main(["reask", "--scene-json", str(f), "--path", "midm"]) == 3
    assert "주행 중" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("headers,why", [
    ({}, "헤더 없음"),
    ({"X-Scene-Viewer": "1", "Origin": "http://evil.example"}, "foreign origin"),
    ({"X-Scene-Viewer": "1", "Host": "evil.example:80"}, "foreign host"),
])
def test_reask_post_cross_site_rejected(api, headers, why):
    base, svc, _ = api
    calls = []
    svc.reask_fn = lambda scene, path, events: calls.append(path) or {}
    get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    req = urllib.request.Request(f"{base}/api/reask/20261104-000000-wake?path=gpt", method="POST", headers=headers)
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 403 and calls == [], why


def test_own_origin_and_localhost_host_ok(api):
    base, svc, _ = api
    svc.reask_fn = lambda scene, path, events: {"note": "ok"}
    get(f"{base}/api/scenes?since=2026-11-04&until=2026-11-05")
    port = base.rsplit(":", 1)[1]
    req = urllib.request.Request(f"{base}/api/reask/20261104-000000-wake?path=gpt", method="POST",
                                 headers={**H, "Origin": f"http://localhost:{port}", "Host": f"localhost:{port}"})
    assert urllib.request.urlopen(req).status == 200
