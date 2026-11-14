"""Bounded line-by-line snapshots of project-owned Python frames."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sized
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
import re
import sys
import threading
from time import perf_counter_ns
import tracemalloc
from types import FrameType
from typing import Protocol, TypedDict, cast
from uuid import uuid4
from weakref import WeakSet, WeakValueDictionary


SENSITIVE = re.compile(r"pass(word)?|secret|token|api[_-]?key|auth|cookie|credential|private", re.I)


class ChildDescription(TypedDict):
    name: str
    value: str


class ValueDescription(TypedDict):
    value: str
    type: str
    bytes: int
    object_id: str | None
    children: list[ChildDescription]


class FrameLabel(TypedDict):
    file: str
    line: int
    function: str
    bindings: dict[str, str]
    project: bool


class TaskInfo(TypedDict):
    id: str
    name: str
    project: bool
    state: str
    await_chain: list[FrameLabel]


class LoopSnapshot(TypedDict):
    id: str
    thread_id: int
    running: bool
    debug: bool
    task_count: int
    project_task_count: int
    tasks: list[TaskInfo]
    updated: str


class TraceEvent(TypedDict):
    id: int
    time: str
    kind: str
    thread: str
    thread_id: int | None
    frame_id: int
    loop_id: str | None
    task_id: str | None
    task_name: str | None
    frames: list[FrameLabel]
    locals: dict[str, ValueDescription]
    arguments: dict[str, str]
    return_value: str | None
    duration_ms: float | None
    exception: str | None
    memory: dict[str, int]


class PauseState(TypedDict):
    frame: FrameType
    event: int
    task_id: str | None


class StepPlan(TypedDict):
    mode: str
    frame: FrameType
    parent: FrameType | None
    task_id: str | None


class PausedThread(TypedDict):
    thread_id: int
    event: int
    task_id: str | None


class ControlState(TypedDict):
    paused_thread: int | None
    paused_event: int | None
    paused_threads: list[PausedThread]
    pause_requested: bool
    pause_requests: list[int | None]
    step_mode: str | None


class ThreadInfo(TypedDict):
    id: int | None
    name: str
    alive: bool
    where: str
    frames: list[FrameLabel]
    last_event: int | None
    last_time: str | None
    last_task_id: str | None


class Snapshot(TypedDict):
    events: list[TraceEvent]
    latest: int
    root: str
    session: str
    loops: list[LoopSnapshot]
    control: ControlState
    threads: list[ThreadInfo]


class TraceCallback(Protocol):
    def __call__(self, frame: FrameType, event: str, arg: object, /) -> TraceCallback | None: ...


def display(value: object, *, name: str = "") -> str:
    kind = type(value).__name__
    if SENSITIVE.search(name):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)[:160]
    if isinstance(value, str):
        return repr(value[:120] + ("…" if len(value) > 120 else ""))
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return f"<{kind}, {len(cast(Sized, value))} items>"
    return f"<{kind}>"


def describe(value: object, *, name: str = "") -> ValueDescription:
    """A bounded, safe view of a local object without calling arbitrary repr methods."""
    label = display(value, name=name)
    kind = type(value).__name__
    object_id = hex(id(value))
    if SENSITIVE.search(name):
        return {"value": "<redacted>", "type": "hidden", "bytes": 0, "object_id": None, "children": []}
    try:
        size = sys.getsizeof(value)
    except (TypeError, ValueError):
        size = 0
    children: list[ChildDescription] = []
    if isinstance(value, dict):
        for key, child in islice(cast(dict[object, object], value).items(), 6):
            label = display(key)
            children.append({"name": label, "value": display(child, name=str(key) if isinstance(key, str) else "")})
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(cast(list[object] | tuple[object, ...], value)[:6]):
            children.append({"name": str(index), "value": display(child)})
    return {"value": label, "type": kind, "bytes": size,
            "object_id": object_id, "children": children}


def arguments(frame: FrameType) -> dict[str, str]:
    code = frame.f_code
    count = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & 0x04:
        count += 1
    if code.co_flags & 0x08:
        count += 1
    return {name: display(frame.f_locals[name], name=name) for name in code.co_varnames[:count]
            if name in frame.f_locals}


class TraceStore:
    def __init__(self, root: Path, own_folder: Path, capacity: int = 2500) -> None:
        self.root = root.resolve()
        self.own_folder = own_folder.resolve()
        self.events: deque[TraceEvent] = deque(maxlen=capacity)
        self.project_threads: set[int] = set()
        self.sequence = 0
        self.session = uuid4().hex
        self.lock = threading.Lock()
        self.path_cache: dict[str, bool] = {}
        self.previous_trace = None
        self.previous_thread_trace = None
        self.active = True
        self.started_tracemalloc = False
        self.control_lock = threading.Condition()
        self.pause_requests: dict[int | None, str | None] = {}
        self.paused: dict[int, PauseState] = {}
        self.steps: dict[int, StepPlan] = {}
        self.call_times: dict[int, int] = {}
        self.loops: WeakValueDictionary[int, asyncio.AbstractEventLoop] = WeakValueDictionary()
        self.loop_snapshots: dict[int, LoopSnapshot] = {}
        self.loop_refresh_pending: set[int] = set()
        self.project_tasks: WeakSet[asyncio.Task[object]] = WeakSet()
        self.thread_latest: dict[int, TraceEvent] = {}

    def frame_label(self, frame: FrameType) -> FrameLabel:
        path = Path(frame.f_code.co_filename)
        project = self.included(frame.f_code.co_filename)
        try:
            bindings = {name: display(value, name=name)
                        for name, value in islice(((n, v) for n, v in frame.f_locals.items()
                                                 if not n.startswith("__")), 8)} if project else {}
        except RuntimeError:
            bindings = {}
        return {"file": str(path.resolve().relative_to(self.root)) if project else path.name,
                "line": frame.f_lineno, "function": frame.f_code.co_name,
                "bindings": bindings, "project": project}

    def coroutine_chain(self, coroutine: object) -> list[FrameLabel]:
        chain: list[FrameLabel] = []
        seen: set[int] = set()
        current = coroutine
        while current is not None and id(current) not in seen and len(chain) < 16:
            seen.add(id(current))
            frame = next((getattr(current, attr, None) for attr in ("cr_frame", "gi_frame", "ag_frame")
                          if getattr(current, attr, None) is not None), None)
            if isinstance(frame, FrameType):
                chain.append(self.frame_label(frame))
            next_await = next((getattr(current, attr, None) for attr in ("cr_await", "gi_yieldfrom", "ag_await")
                               if getattr(current, attr, None) is not None), None)
            if next_await is None:
                if frame is None:
                    chain.append({"file": "", "line": 0, "function": type(current).__name__,
                                  "bindings": {}, "project": False})
                break
            current = next_await
        return chain

    def task_info(self, task: asyncio.Task[object]) -> TaskInfo:
        chain = self.coroutine_chain(task.get_coro())
        project = any(frame["project"] for frame in chain) or task in self.project_tasks
        return {"id": hex(id(task)), "name": task.get_name(), "project": project,
                "state": "cancelling" if task.cancelling() else "waiting" if len(chain) > 1 else "ready",
                "await_chain": chain}

    def capture_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run on the event-loop thread; Task objects are not thread-safe."""
        loop_id = id(loop)
        try:
            tasks = [self.task_info(task) for task in asyncio.all_tasks(loop)]
            tasks.sort(key=lambda task: (not task["project"], task["name"]))
            info: LoopSnapshot = {"id": hex(loop_id), "thread_id": threading.get_ident(), "running": loop.is_running(),
                    "debug": loop.get_debug(), "task_count": len(tasks),
                    "project_task_count": sum(task["project"] for task in tasks),
                    "tasks": tasks[:80], "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with self.lock:
                self.loop_snapshots[loop_id] = info
        finally:
            with self.lock:
                self.loop_refresh_pending.discard(loop_id)

    def refresh_loops(self) -> None:
        with self.lock:
            loops = list(self.loops.items())
            known_ids = {loop_id for loop_id, _ in loops}
            for stale_id in list(self.loop_snapshots):
                if stale_id not in known_ids:
                    self.loop_snapshots.pop(stale_id, None)
        for loop_id, loop in loops:
            if loop.is_closed():
                with self.lock:
                    self.loop_snapshots.pop(loop_id, None)
                continue
            if not loop.is_running():
                continue
            with self.lock:
                if loop_id in self.loop_refresh_pending:
                    continue
                self.loop_refresh_pending.add(loop_id)
            try:
                loop.call_soon_threadsafe(self.capture_loop, loop)
            except RuntimeError:
                with self.lock:
                    self.loop_refresh_pending.discard(loop_id)

    def included(self, filename: str) -> bool:
        cached = self.path_cache.get(filename)
        if cached is not None:
            return cached
        if filename.startswith("<"):
            self.path_cache[filename] = False
            return False
        path = Path(filename).resolve()
        parts = path.parts
        result = path.is_file() and path.is_relative_to(self.root) and not path.is_relative_to(self.own_folder) and not any(
            part in {".venv", "venv", "site-packages", ".git", "__pycache__"} for part in parts
        )
        self.path_cache[filename] = result
        return result

    def trace(self, frame: FrameType, event: str, arg: object) -> TraceCallback | None:
        if not self.active:
            return None
        if event == "call" and not self.included(frame.f_code.co_filename):
            return None
        if event not in {"call", "line", "return", "exception"}:
            return self.trace
        frame_id = id(frame)
        if event == "call":
            self.call_times[frame_id] = perf_counter_ns()
        duration_ms = None
        if event == "return":
            started = self.call_times.pop(frame_id, None)
            if started is not None:
                duration_ms = round((perf_counter_ns() - started) / 1_000_000, 3)
        frames: list[FrameLabel] = []
        current = frame
        while current is not None and len(frames) < 12:
            if self.included(current.f_code.co_filename):
                frames.append(self.frame_label(current))
            current = current.f_back
        if not frames:
            return self.trace
        try:
            local_values = {name: describe(value, name=name) for name, value in list(frame.f_locals.items())[:40]
                            if not name.startswith("__")}
        except (RuntimeError, ValueError):
            local_values = {}
        exception = None
        if event == "exception" and isinstance(arg, tuple) and len(cast(tuple[object, ...], arg)) >= 2:
            exception_parts = cast(tuple[object, ...], arg)
            exception = f"{getattr(exception_parts[0], '__name__', 'Exception')}: {str(exception_parts[1])[:160]}"
        thread = threading.current_thread()
        try:
            loop = asyncio.get_running_loop()
            task = asyncio.current_task(loop)
        except RuntimeError:
            loop = None
            task = None
        task_id = hex(id(task)) if task is not None else None
        current_bytes, peak_bytes = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        snapshot: TraceEvent = {"id": 0, "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "kind": event, "thread": thread.name, "thread_id": thread.ident, "frame_id": frame_id,
                    "loop_id": hex(id(loop)) if loop is not None else None,
                    "task_id": task_id, "task_name": task.get_name() if task is not None else None,
                    "frames": frames, "locals": local_values, "arguments": arguments(frame) if event == "call" else {},
                    "return_value": display(cast(object, arg)) if event == "return" else None, "duration_ms": duration_ms,
                    "exception": exception, "memory": {"current": current_bytes, "peak": peak_bytes,
                    "locals_shallow": sum(item["bytes"] for item in local_values.values())}}
        with self.lock:
            self.sequence += 1
            snapshot["id"] = self.sequence
            self.events.append(snapshot)
            if thread.ident is not None:
                self.project_threads.add(thread.ident)
                self.thread_latest[thread.ident] = snapshot
            if loop is not None:
                self.loops[id(loop)] = loop
            if task is not None:
                self.project_tasks.add(task)
        self._maybe_pause(frame, event, snapshot["id"], thread.ident, task_id)
        return self.trace

    def _maybe_pause(self, frame: FrameType, event: str, event_id: int, thread_id: int | None,
                     task_id: str | None) -> None:
        if thread_id is None:
            return
        with self.control_lock:
            request_key = thread_id if thread_id in self.pause_requests else None
            has_request = request_key in self.pause_requests
            requested_task = self.pause_requests.get(request_key) if has_request else None
            stop = has_request and (requested_task is None or requested_task == task_id)
            plan = self.steps.get(thread_id)
            if not stop and plan and (plan["task_id"] is None or plan["task_id"] == task_id):
                if plan["mode"] == "into":
                    stop = True
                elif plan["mode"] == "over":
                    stop = frame is plan["frame"] and event != "call"
                elif plan["mode"] == "out":
                    stop = (frame is plan["parent"] and event != "call") or (
                        plan["parent"] is None and frame is plan["frame"] and event == "return")
            if not stop:
                return
            if has_request:
                self.pause_requests.pop(request_key, None)
            self.steps.pop(thread_id, None)
            self.paused[thread_id] = {"frame": frame, "event": event_id, "task_id": task_id}
            while self.active and thread_id in self.paused:
                self.control_lock.wait()

    def control(self, action: str, thread_id: int | None = None,
                task_id: str | None = None) -> ControlState:
        with self.control_lock:
            if action == "pause":
                self.pause_requests[thread_id] = task_id
            elif action == "continue":
                if thread_id is None:
                    self.pause_requests.clear()
                    self.steps.clear()
                    self.paused.clear()
                else:
                    self.pause_requests.pop(thread_id, None)
                    self.steps.pop(thread_id, None)
                    self.paused.pop(thread_id, None)
                self.control_lock.notify_all()
            elif action in {"step_into", "step_over", "step_out"}:
                if thread_id is None:
                    if len(self.paused) != 1:
                        raise ValueError("Select one paused thread to step")
                    thread_id = next(iter(self.paused))
                state = self.paused.get(thread_id)
                if state is None:
                    raise ValueError("Pause execution before stepping")
                frame = state["frame"]
                parent = frame.f_back
                while parent is not None and not self.included(parent.f_code.co_filename):
                    parent = parent.f_back
                self.steps[thread_id] = {"mode": action.removeprefix("step_"), "frame": frame,
                                         "parent": parent, "task_id": state["task_id"]}
                self.paused.pop(thread_id)
                self.control_lock.notify_all()
            else:
                raise ValueError(f"Unknown debugger action: {action}")
            return self.control_state()

    def control_state(self) -> ControlState:
        first_id = next(iter(self.paused), None)
        first_step = next(iter(self.steps.values()), None)
        return {"paused_thread": first_id,
                "paused_event": self.paused[first_id]["event"] if first_id is not None else None,
                "paused_threads": [{"thread_id": tid, "event": state["event"],
                                    "task_id": state["task_id"]} for tid, state in self.paused.items()],
                "pause_requested": bool(self.pause_requests), "pause_requests": list(self.pause_requests),
                "step_mode": first_step["mode"] if first_step else None}

    def snapshot(self, after: int = 0) -> Snapshot:
        self.refresh_loops()
        with self.lock:
            events = [event for event in self.events if event["id"] > after]
            result: Snapshot = {"events": events, "latest": self.sequence, "root": str(self.root),
                                "session": self.session, "loops": [], "control": self.control_state(),
                                "threads": []}
            project_threads = set(self.project_threads)
            latest_by_thread = dict(self.thread_latest)
            result["loops"] = list(self.loop_snapshots.values())
        with self.control_lock:
            result["control"] = self.control_state()
        current_frames = cast(dict[int, FrameType], getattr(sys, "_current_frames")())
        live_threads = threading.enumerate()
        live_ids = {thread.ident for thread in live_threads}
        with self.lock:
            self.project_threads.intersection_update(live_ids)
            for thread_id in list(self.thread_latest):
                if thread_id not in live_ids:
                    self.thread_latest.pop(thread_id, None)
        threads: list[ThreadInfo] = []
        for thread in live_threads:
            if thread.ident not in project_threads:
                continue
            frame = current_frames.get(thread.ident)
            live_frames: list[FrameLabel] = []
            while frame is not None and len(live_frames) < 12:
                if self.included(frame.f_code.co_filename):
                    live_frames.append(self.frame_label(frame))
                frame = frame.f_back
            last = latest_by_thread.get(thread.ident)
            threads.append({"id": thread.ident, "name": thread.name, "alive": thread.is_alive(),
                            "where": "project" if live_frames else "outside project",
                            "frames": live_frames or (last["frames"] if last else []),
                            "last_event": last["id"] if last else None,
                            "last_time": last["time"] if last else None,
                            "last_task_id": last["task_id"] if last else None})
        result["threads"] = threads
        return result

    def install(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start(1)
            self.started_tracemalloc = True
        self.previous_trace = sys.gettrace()
        self.previous_thread_trace = threading.gettrace()
        threading.settrace(self.trace)
        sys.settrace(self.trace)

    def uninstall(self) -> None:
        self.active = False
        with self.control_lock:
            self.paused.clear()
            self.control_lock.notify_all()
        sys.settrace(self.previous_trace)
        threading.settrace(self.previous_thread_trace)
        if self.started_tracemalloc:
            tracemalloc.stop()
