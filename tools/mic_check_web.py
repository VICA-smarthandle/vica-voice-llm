"""마이크 점검기 휴대폰 중계 — 페이지 하나와 실시간 통로 하나(websockets 15, 새 의존성 없음)."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Optional

from websockets.asyncio.server import broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

DEFAULT_PORT = 8770


class LiveServer:
    def __init__(self, page_html: str, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self._page = page_html.encode("utf-8")
        self._host, self._want_port = host, port
        self.port: Optional[int] = None
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self._clients: set = set()
        self._kept: dict = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Future] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="mic-check-web", daemon=True)

    def start(self, timeout: float = 5.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("휴대폰 중계를 열지 못했습니다(시간 초과)")
        if self._error is not None:
            raise RuntimeError(f"휴대폰 중계를 열지 못했습니다: {self._error}")

    def stop(self) -> None:
        """여러 번 불러도, 시작 전·시작 실패 뒤에 불러도 안전하다."""
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            try:
                loop.call_soon_threadsafe(lambda: stop.done() or stop.set_result(None))
            except RuntimeError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def publish(self, message: dict, keep: Optional[str] = None) -> None:
        """어느 스레드에서든 부른다. keep 이름을 주면 새로 붙는 휴대폰에도 이 메시지를 먼저 보낸다."""
        text = json.dumps(message, ensure_ascii=False)
        if keep:
            with self._lock:
                self._kept[keep] = text
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._send_all, text)
            except RuntimeError:
                pass

    def _send_all(self, text: str) -> None:
        broadcast(self._clients, text)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 — 시작 실패(포트 사용 중 등)를 start() 로 넘긴다
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = self._loop.create_future()
        async with serve(self._handler, self._host, self._want_port,
                         process_request=self._process_request) as server:
            self.port = list(server.sockets)[0].getsockname()[1]
            self._ready.set()
            await self._stop

    def _process_request(self, connection, request):
        if request.path == "/ws":
            return None
        if request.path in ("/", "/index.html"):
            headers = Headers([("Content-Type", "text/html; charset=utf-8"),
                               ("Content-Length", str(len(self._page))),
                               ("Cache-Control", "no-store")])
            return Response(200, "OK", headers, self._page)
        return connection.respond(404, "없는 주소입니다\n")

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            with self._lock:
                kept = list(self._kept.values())
            for text in kept:
                await ws.send(text)
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(cmd, dict) and isinstance(cmd.get("cmd"), str):
                    self.commands.put(cmd)
        finally:
            self._clients.discard(ws)
