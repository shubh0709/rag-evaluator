"""Start a Python app and its local execution viewer in one process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import threading
import webbrowser

from .server import start_server
from .trace import TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an app with the live debugger")
    parser.add_argument("mode", choices=("streamlit", "python"))
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("--port", type=int, default=8765, help="debugger port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true")
    args, app_args = parser.parse_known_args()
    entrypoint = args.entrypoint.resolve()
    if not entrypoint.is_file():
        parser.error(f"No such entrypoint: {entrypoint}")
    root = Path.cwd().resolve()
    if not entrypoint.is_relative_to(root):
        root = entrypoint.parent
    if str(entrypoint.parent) not in sys.path:
        sys.path.insert(0, str(entrypoint.parent))
    os.chdir(root)

    store = TraceStore(root, Path(__file__).parent)
    server = start_server(store, args.port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\nExecution debugger: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open_new_tab(url)).start()
    store.install()
    try:
        if args.mode == "streamlit":
            from streamlit.web import cli

            sys.argv = ["streamlit", "run", str(entrypoint), *app_args]
            cli.main()
        else:
            sys.argv = [str(entrypoint), *app_args]
            runpy.run_path(str(entrypoint), run_name="__main__")
    finally:
        store.uninstall()
        server.shutdown()


if __name__ == "__main__":
    main()
