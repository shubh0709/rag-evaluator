"""Loopback-only viewer and trace API."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import cast
from urllib.parse import parse_qs, urlparse

from .trace import TraceStore


def start_server(store: TraceStore, port: int) -> ThreadingHTTPServer:
    html = (Path(__file__).parent / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/api/control":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{cast(ThreadingHTTPServer, self.server).server_port}":
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024:
                    raise ValueError("Request too large")
                command = json.loads(self.rfile.read(length))
                thread_id = command.get("thread_id")
                if thread_id is not None:
                    thread_id = int(thread_id)
                task_id = command.get("task_id")
                if task_id is not None and not isinstance(task_id, str):
                    raise ValueError("task_id must be a string")
                result = store.control(command["action"], thread_id, task_id)
                body = json.dumps(result).encode()
                self.send_response(200)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body, mime = html, "text/html; charset=utf-8"
            elif parsed.path == "/api/events":
                try:
                    after = max(0, int(parse_qs(parsed.query).get("after", ["0"])[0]))
                except ValueError:
                    after = 0
                body = json.dumps(store.snapshot(after)).encode()
                mime = "application/json"
            elif parsed.path == "/api/source":
                requested = parse_qs(parsed.query).get("file", [""])[0]
                source = (store.root / requested).resolve()
                if not source.is_file() or not source.is_relative_to(store.root) or not store.included(str(source)):
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

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, name="debugger-http", daemon=True).start()
    return server
