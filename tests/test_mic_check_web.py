"""휴대폰 중계 — 페이지, 다시 붙을 때 마지막 상태, 명령 줄 세우기(설계 4·9절)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from websockets.sync.client import connect

from tools.mic_check_web import LiveServer


@pytest.fixture
def server():
    s = LiveServer("<html><body>점검</body></html>", host="127.0.0.1", port=0)
    s.start()
    yield s
    s.stop()


def test_page_and_404(server):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as r:
        assert r.status == 200 and "text/html" in r.headers["Content-Type"]
        assert "점검" in r.read().decode("utf-8")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/nope")
    assert err.value.code == 404


def test_kept_message_sent_on_connect_and_commands_queued(server):
    server.publish({"type": "status", "n": 1}, keep="status")
    server.publish({"type": "status", "n": 2}, keep="status")     # 같은 이름은 마지막 것만
    with connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
        assert json.loads(ws.recv(timeout=2)) == {"type": "status", "n": 2}
        ws.send(json.dumps({"cmd": "record"}))
        ws.send("이상한 글자")               # 무시
        ws.send(json.dumps({"no": "cmd"}))   # 무시
        ws.send(json.dumps({"cmd": "rejudge"}))
        assert server.commands.get(timeout=2) == {"cmd": "record"}
        assert server.commands.get(timeout=2) == {"cmd": "rejudge"}


def test_publish_reaches_connected_client(server):
    with connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
        ws.send(json.dumps({"cmd": "ping"}))                     # 붙은 것을 확인하는 왕복
        assert server.commands.get(timeout=2) == {"cmd": "ping"}
        server.publish({"type": "frame", "rms": 0.01})
        assert json.loads(ws.recv(timeout=2)) == {"type": "frame", "rms": 0.01}


def test_stop_twice_is_safe():
    s = LiveServer("<html></html>", host="127.0.0.1", port=0)
    s.start()
    s.stop()
    s.stop()


def test_stop_before_start_is_safe():
    s = LiveServer("<html></html>", host="127.0.0.1", port=0)
    s.stop()


def test_failed_start_reports_port_in_use_and_stop_is_safe():
    first = LiveServer("<html></html>", host="127.0.0.1", port=0)
    first.start()
    try:
        second = LiveServer("<html></html>", host="127.0.0.1", port=first.port)
        with pytest.raises(RuntimeError) as err:
            second.start()
        assert "휴대폰 중계를 열지 못했습니다" in str(err.value)
        second.stop()
    finally:
        first.stop()


def test_publish_after_stop_is_safe():
    s = LiveServer("<html></html>", host="127.0.0.1", port=0)
    s.start()
    s.stop()
    s.publish({"type": "x"}, keep="status")
