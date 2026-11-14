"""Loopback-only viewer and trace API."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Literal, cast
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from .trace import TraceStore


class ControlCommand(BaseModel):
    """Represent control command data."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "continue", "step_into", "step_over", "step_out"]
    thread_id: int | None = None
    task_id: StrictStr | None = None


class DebuggerHandler(BaseHTTPRequestHandler):
    """Serve trace and control requests for a single debugger session."""

    store: TraceStore
    html: bytes

    def post(self) -> None:
        """Do POST."""
        if self.path != "/api/control":
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if (
            origin
            and origin != f"http://127.0.0.1:{cast(ThreadingHTTPServer, self.server).server_port}"
        ):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 1024:
                raise ValueError("Request too large")
            command = ControlCommand.model_validate(json.loads(self.rfile.read(length)))
            result = self.store.control(command.action, command.thread_id, command.task_id)
            body = json.dumps(result).encode()
            self.send_response(200)
        except ValueError as exc:
            if isinstance(exc, ValidationError):
                error = exc.errors()[0]
                field = ".".join(str(part) for part in error["loc"]) or "input"
                message = f"{field}: {error['msg']}"
            else:
                message = str(exc)
            body = json.dumps({"error": message}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def get(self) -> None:
        """Do GET."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body, mime = self.html, "text/html; charset=utf-8"
        elif parsed.path == "/api/events":
            try:
                after = max(0, int(parse_qs(parsed.query).get("after", ["0"])[0]))
            except ValueError:
                after = 0
            body = json.dumps(self.store.snapshot(after)).encode()
            mime = "application/json"
        elif parsed.path == "/api/source":
            requested = parse_qs(parsed.query).get("file", [""])[0]
            source = (self.store.root / requested).resolve()
            if (
                not source.is_file()
                or not source.is_relative_to(self.store.root)
                or not self.store.included(str(source))
            ):
                self.send_error(404)
                return
            body = source.read_text(encoding="utf-8", errors="replace").encode()
            mime = "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


setattr(DebuggerHandler, "do_POST", DebuggerHandler.post)
setattr(DebuggerHandler, "do_GET", DebuggerHandler.get)


def start_server(store: TraceStore, port: int) -> ThreadingHTTPServer:
    """Start server."""
    html = (Path(__file__).parent / "index.html").read_bytes()

    handler = type("Handler", (DebuggerHandler,), {"store": store, "html": html})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, name="debugger-http", daemon=True).start()
    return server
