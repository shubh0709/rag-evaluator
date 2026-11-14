"""Bounded line-by-line snapshots of project-owned Python frames."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
import re
import sys
import threading
from time import perf_counter_ns
import tracemalloc
from types import FrameType
from typing import Any, TypedDict, cast
from uuid import uuid4
from weakref import WeakSet, WeakValueDictionary

SENSITIVE = re.compile(r"pass(word)?|secret|token|api[_-]?key|auth|cookie|credential|private", re.I)


class ChildDescription(TypedDict):
    """Represent child description data."""

    name: str
    value: str


class ValueDescription(TypedDict):
    """Represent value description data."""

    value: str
    type: str
    bytes: int
    object_id: str | None
    children: list[ChildDescription]


class FrameLabel(TypedDict):
    """Represent frame label data."""

    file: str
    line: int
    function: str
    bindings: dict[str, str]
    project: bool


class TaskInfo(TypedDict):
    """Represent task info data."""

    id: str
    name: str
    project: bool
    state: str
    await_chain: list[FrameLabel]


class LoopSnapshot(TypedDict):
    """Represent loop snapshot data."""

    id: str
    thread_id: int
    running: bool
    debug: bool
    task_count: int
    project_task_count: int
    tasks: list[TaskInfo]
    updated: str


class TraceEvent(TypedDict):
    """Represent trace event data."""

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
    """Represent pause state data."""

    frame: FrameType
    event: int
    task_id: str | None


class StepPlan(TypedDict):
    """Represent step plan data."""

    mode: str
    frame: FrameType
    parent: FrameType | None
    task_id: str | None


class PausedThread(TypedDict):
    """Represent paused thread data."""

    thread_id: int
    event: int
    task_id: str | None


class ControlState(TypedDict):
    """Represent control state data."""

    paused_thread: int | None
    paused_event: int | None
    paused_threads: list[PausedThread]
    pause_requested: bool
    pause_requests: list[int | None]
    step_mode: str | None


class ThreadInfo(TypedDict):
    """Represent thread info data."""

    id: int | None
    name: str
    alive: bool
    where: str
    frames: list[FrameLabel]
    last_event: int | None
    last_time: str | None
    last_task_id: str | None


class Snapshot(TypedDict):
    """Represent snapshot data."""

    events: list[TraceEvent]
    latest: int
    root: str
    session: str
    loops: list[LoopSnapshot]
    control: ControlState
    threads: list[ThreadInfo]


type TraceCallback = Callable[[FrameType, str, object], TraceCallback | None]


def display(value: object, *, name: str = "") -> str:
    """Render a readable representation of this value."""
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
        return {
            "value": "<redacted>",
            "type": "hidden",
            "bytes": 0,
            "object_id": None,
            "children": [],
        }
    try:
        size = sys.getsizeof(value)
    except (TypeError, ValueError):
        size = 0
    children: list[ChildDescription] = []
    if isinstance(value, dict):
        for key, child in islice(cast(dict[object, object], value).items(), 6):
            label = display(key)
            children.append(
                {
                    "name": label,
                    "value": display(child, name=str(key) if isinstance(key, str) else ""),
                }
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(cast(list[object] | tuple[object, ...], value)[:6]):
            children.append({"name": str(index), "value": display(child)})
    return {
        "value": label,
        "type": kind,
        "bytes": size,
        "object_id": object_id,
        "children": children,
    }


def arguments(frame: FrameType) -> dict[str, str]:
    """Describe the arguments passed to this call."""
    code = frame.f_code
    count = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & 0x04:
        count += 1
    if code.co_flags & 0x08:
        count += 1
    return {
        name: display(frame.f_locals[name], name=name)
        for name in code.co_varnames[:count]
        if name in frame.f_locals
    }


@dataclass
class TraceData:
    """Project identity and synchronized event storage."""

    root: Path
    own_folder: Path
    events: deque[TraceEvent]
    project_threads: set[int]
    sequence: int
    session: str
    lock: threading.Lock


@dataclass
class TraceRuntime:
    """Interpreter tracing state and call timers."""

    path_cache: dict[str, bool]
    previous_trace: Any
    previous_thread_trace: Any
    active: bool
    started_tracemalloc: bool
    call_times: dict[int, int]


@dataclass
class TraceControl:
    """Pause and step state for inspected threads."""

    control_lock: threading.Condition
    pause_requests: dict[int | None, str | None]
    paused: dict[int, PauseState]
    steps: dict[int, StepPlan]
    thread_latest: dict[int, TraceEvent]


@dataclass
class TraceLoops:
    """Weak loop and task references used in snapshots."""

    loops: WeakValueDictionary[int, asyncio.AbstractEventLoop]
    loop_snapshots: dict[int, LoopSnapshot]
    loop_refresh_pending: set[int]
    project_tasks: WeakSet[asyncio.Task[object]]


def _frame_locals(frame: FrameType) -> dict[str, ValueDescription]:
    """Describe a bounded set of local variables, tolerating frame mutation."""
    try:
        return {
            name: describe(value, name=name)
            for name, value in list(frame.f_locals.items())[:40]
            if not name.startswith("__")
        }
    except (RuntimeError, ValueError):
        return {}


def _exception_text(event: str, arg: object) -> str | None:
    """Format an exception trace event without retaining the full value."""
    if event == "exception" and isinstance(arg, tuple):
        parts = cast(tuple[object, ...], arg)
        if len(parts) >= 2:
            return f"{getattr(parts[0], '__name__', 'Exception')}: {str(parts[1])[:160]}"
    return None


def _task_context() -> tuple[asyncio.AbstractEventLoop | None, asyncio.Task[object] | None]:
    """Return the active loop and task when tracing async code."""
    try:
        loop = asyncio.get_running_loop()
        return loop, asyncio.current_task(loop)
    except RuntimeError:
        return None, None


class TraceStore:
    """Represent trace store data."""

    def __init__(self, root: Path, own_folder: Path, capacity: int = 2500) -> None:
        self._data = TraceData(
            root.resolve(),
            own_folder.resolve(),
            deque(maxlen=capacity),
            set(),
            0,
            uuid4().hex,
            threading.Lock(),
        )
        self._runtime = TraceRuntime({}, None, None, True, False, {})
        self._control = TraceControl(threading.Condition(), {}, {}, {}, {})
        self._loop = TraceLoops(WeakValueDictionary(), {}, set(), WeakSet())

    @property
    def root(self) -> Path:
        """Return the project root used by the source endpoint."""
        return self._data.root

    def frame_label(self, frame: FrameType) -> FrameLabel:
        """Frame label."""
        path = Path(frame.f_code.co_filename)
        project = self.included(frame.f_code.co_filename)
        try:
            bindings = (
                {
                    name: display(value, name=name)
                    for name, value in islice(
                        ((n, v) for n, v in frame.f_locals.items() if not n.startswith("__")), 8
                    )
                }
                if project
                else {}
            )
        except RuntimeError:
            bindings = {}
        return {
            "file": str(path.resolve().relative_to(self._data.root)) if project else path.name,
            "line": frame.f_lineno,
            "function": frame.f_code.co_name,
            "bindings": bindings,
            "project": project,
        }

    def coroutine_chain(self, coroutine: object) -> list[FrameLabel]:
        """Coroutine chain."""
        chain: list[FrameLabel] = []
        seen: set[int] = set()
        current = coroutine
        while current is not None and id(current) not in seen and len(chain) < 16:
            seen.add(id(current))
            frame = next(
                (
                    getattr(current, attr, None)
                    for attr in ("cr_frame", "gi_frame", "ag_frame")
                    if getattr(current, attr, None) is not None
                ),
                None,
            )
            if isinstance(frame, FrameType):
                chain.append(self.frame_label(frame))
            next_await = next(
                (
                    getattr(current, attr, None)
                    for attr in ("cr_await", "gi_yieldfrom", "ag_await")
                    if getattr(current, attr, None) is not None
                ),
                None,
            )
            if next_await is None:
                if frame is None:
                    chain.append(
                        {
                            "file": "",
                            "line": 0,
                            "function": type(current).__name__,
                            "bindings": {},
                            "project": False,
                        }
                    )
                break
            current = next_await
        return chain

    def task_info(self, task: asyncio.Task[object]) -> TaskInfo:
        """Task info."""
        chain = self.coroutine_chain(task.get_coro())
        project = any(frame["project"] for frame in chain) or task in self._loop.project_tasks
        return {
            "id": hex(id(task)),
            "name": task.get_name(),
            "project": project,
            "state": (
                "cancelling" if task.cancelling() else "waiting" if len(chain) > 1 else "ready"
            ),
            "await_chain": chain,
        }

    def capture_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run on the event-loop thread; Task objects are not thread-safe."""
        loop_id = id(loop)
        try:
            tasks = [self.task_info(task) for task in asyncio.all_tasks(loop)]
            tasks.sort(key=lambda task: (not task["project"], task["name"]))
            info: LoopSnapshot = {
                "id": hex(loop_id),
                "thread_id": threading.get_ident(),
                "running": loop.is_running(),
                "debug": loop.get_debug(),
                "task_count": len(tasks),
                "project_task_count": sum(task["project"] for task in tasks),
                "tasks": tasks[:80],
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with self._data.lock:
                self._loop.loop_snapshots[loop_id] = info
        finally:
            with self._data.lock:
                self._loop.loop_refresh_pending.discard(loop_id)

    def refresh_loops(self) -> None:
        """Refresh loops."""
        with self._data.lock:
            loops = list(self._loop.loops.items())
            known_ids = {loop_id for loop_id, _ in loops}
            for stale_id in list(self._loop.loop_snapshots):
                if stale_id not in known_ids:
                    self._loop.loop_snapshots.pop(stale_id, None)
        for loop_id, loop in loops:
            if loop.is_closed():
                with self._data.lock:
                    self._loop.loop_snapshots.pop(loop_id, None)
                continue
            if not loop.is_running():
                continue
            with self._data.lock:
                if loop_id in self._loop.loop_refresh_pending:
                    continue
                self._loop.loop_refresh_pending.add(loop_id)
            try:
                loop.call_soon_threadsafe(self.capture_loop, loop)
            except RuntimeError:
                with self._data.lock:
                    self._loop.loop_refresh_pending.discard(loop_id)

    def included(self, filename: str) -> bool:
        """Included."""
        cached = self._runtime.path_cache.get(filename)
        if cached is not None:
            return cached
        if filename.startswith("<"):
            self._runtime.path_cache[filename] = False
            return False
        path = Path(filename).resolve()
        parts = path.parts
        result = (
            path.is_file()
            and path.is_relative_to(self._data.root)
            and not path.is_relative_to(self._data.own_folder)
            and not any(
                part in {".venv", "venv", "site-packages", ".git", "__pycache__"} for part in parts
            )
        )
        self._runtime.path_cache[filename] = result
        return result

    def _trace_duration(self, frame_id: int, event: str) -> float | None:
        """Measure elapsed time for a returning frame."""
        if event == "call":
            self._runtime.call_times[frame_id] = perf_counter_ns()
        if event == "return":
            started = self._runtime.call_times.pop(frame_id, None)
            if started is not None:
                return round((perf_counter_ns() - started) / 1_000_000, 3)
        return None

    def _trace_frames(self, frame: FrameType) -> list[FrameLabel]:
        """Capture at most twelve project frames from the call stack."""
        frames: list[FrameLabel] = []
        current = frame
        while current is not None and len(frames) < 12:
            if self.included(current.f_code.co_filename):
                frames.append(self.frame_label(current))
            current = current.f_back
        return frames

    def trace(self, frame: FrameType, event: str, arg: object) -> TraceCallback | None:
        """Trace."""
        if not self._runtime.active:
            return None
        if event == "call" and not self.included(frame.f_code.co_filename):
            return None
        if event not in {"call", "line", "return", "exception"}:
            return self.trace
        frame_id = id(frame)
        duration_ms = self._trace_duration(frame_id, event)
        frames = self._trace_frames(frame)
        if not frames:
            return self.trace
        local_values = _frame_locals(frame)
        exception = _exception_text(event, arg)
        thread = threading.current_thread()
        loop, task = _task_context()
        current_bytes, peak_bytes = (
            tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        )
        snapshot: TraceEvent = {
            "id": 0,
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind": event,
            "thread": thread.name,
            "thread_id": thread.ident,
            "frame_id": frame_id,
            "loop_id": hex(id(loop)) if loop is not None else None,
            "task_id": hex(id(task)) if task is not None else None,
            "task_name": task.get_name() if task is not None else None,
            "frames": frames,
            "locals": local_values,
            "arguments": arguments(frame) if event == "call" else {},
            "return_value": display(arg) if event == "return" else None,
            "duration_ms": duration_ms,
            "exception": exception,
            "memory": {
                "current": current_bytes,
                "peak": peak_bytes,
                "locals_shallow": sum(item["bytes"] for item in local_values.values()),
            },
        }
        with self._data.lock:
            self._data.sequence += 1
            snapshot["id"] = self._data.sequence
            self._data.events.append(snapshot)
            if thread.ident is not None:
                self._data.project_threads.add(thread.ident)
                self._control.thread_latest[thread.ident] = snapshot
            if loop is not None:
                self._loop.loops[id(loop)] = loop
            if task is not None:
                self._loop.project_tasks.add(task)
        self._maybe_pause(frame, event, snapshot["id"], thread.ident, snapshot["task_id"])
        return self.trace

    def _maybe_pause(
        self,
        frame: FrameType,
        event: str,
        event_id: int,
        thread_id: int | None,
        task_id: str | None,
    ) -> None:
        if thread_id is None:
            return
        with self._control.control_lock:
            request_key = thread_id if thread_id in self._control.pause_requests else None
            has_request = request_key in self._control.pause_requests
            requested_task = self._control.pause_requests.get(request_key) if has_request else None
            stop = has_request and (requested_task is None or requested_task == task_id)
            plan = self._control.steps.get(thread_id)
            if not stop and plan and (plan["task_id"] is None or plan["task_id"] == task_id):
                if plan["mode"] == "into":
                    stop = True
                elif plan["mode"] == "over":
                    stop = frame is plan["frame"] and event != "call"
                elif plan["mode"] == "out":
                    stop = (frame is plan["parent"] and event != "call") or (
                        plan["parent"] is None and frame is plan["frame"] and event == "return"
                    )
            if not stop:
                return
            if has_request:
                self._control.pause_requests.pop(request_key, None)
            self._control.steps.pop(thread_id, None)
            self._control.paused[thread_id] = {
                "frame": frame,
                "event": event_id,
                "task_id": task_id,
            }
            while self._runtime.active and thread_id in self._control.paused:
                self._control.control_lock.wait()

    def control(
        self, action: str, thread_id: int | None = None, task_id: str | None = None
    ) -> ControlState:
        """Control."""
        with self._control.control_lock:
            if action == "pause":
                self._control.pause_requests[thread_id] = task_id
            elif action == "continue":
                if thread_id is None:
                    self._control.pause_requests.clear()
                    self._control.steps.clear()
                    self._control.paused.clear()
                else:
                    self._control.pause_requests.pop(thread_id, None)
                    self._control.steps.pop(thread_id, None)
                    self._control.paused.pop(thread_id, None)
                self._control.control_lock.notify_all()
            elif action in {"step_into", "step_over", "step_out"}:
                if thread_id is None:
                    if len(self._control.paused) != 1:
                        raise ValueError("Select one paused thread to step")
                    thread_id = next(iter(self._control.paused))
                state = self._control.paused.get(thread_id)
                if state is None:
                    raise ValueError("Pause execution before stepping")
                frame = state["frame"]
                parent = frame.f_back
                while parent is not None and not self.included(parent.f_code.co_filename):
                    parent = parent.f_back
                self._control.steps[thread_id] = {
                    "mode": action.removeprefix("step_"),
                    "frame": frame,
                    "parent": parent,
                    "task_id": state["task_id"],
                }
                self._control.paused.pop(thread_id)
                self._control.control_lock.notify_all()
            else:
                raise ValueError(f"Unknown debugger action: {action}")
            return self.control_state()

    def control_state(self) -> ControlState:
        """Control state."""
        first_id = next(iter(self._control.paused), None)
        first_step = next(iter(self._control.steps.values()), None)
        return {
            "paused_thread": first_id,
            "paused_event": (
                self._control.paused[first_id]["event"] if first_id is not None else None
            ),
            "paused_threads": [
                {"thread_id": tid, "event": state["event"], "task_id": state["task_id"]}
                for tid, state in self._control.paused.items()
            ],
            "pause_requested": bool(self._control.pause_requests),
            "pause_requests": list(self._control.pause_requests),
            "step_mode": first_step["mode"] if first_step else None,
        }

    def snapshot(self, after: int = 0) -> Snapshot:
        """Snapshot."""
        self.refresh_loops()
        with self._data.lock:
            events = [event for event in self._data.events if event["id"] > after]
            result: Snapshot = {
                "events": events,
                "latest": self._data.sequence,
                "root": str(self._data.root),
                "session": self._data.session,
                "loops": [],
                "control": self.control_state(),
                "threads": [],
            }
            project_threads = set(self._data.project_threads)
            latest_by_thread = dict(self._control.thread_latest)
            result["loops"] = list(self._loop.loop_snapshots.values())
        with self._control.control_lock:
            result["control"] = self.control_state()
        current_frames = cast(dict[int, FrameType], getattr(sys, "_current_frames")())
        live_threads = threading.enumerate()
        live_ids = {thread.ident for thread in live_threads}
        with self._data.lock:
            self._data.project_threads.intersection_update(live_ids)
            for thread_id in list(self._control.thread_latest):
                if thread_id not in live_ids:
                    self._control.thread_latest.pop(thread_id, None)
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
            threads.append(
                {
                    "id": thread.ident,
                    "name": thread.name,
                    "alive": thread.is_alive(),
                    "where": "project" if live_frames else "outside project",
                    "frames": live_frames or (last["frames"] if last else []),
                    "last_event": last["id"] if last else None,
                    "last_time": last["time"] if last else None,
                    "last_task_id": last["task_id"] if last else None,
                }
            )
        result["threads"] = threads
        return result

    def install(self) -> None:
        """Install."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(1)
            self._runtime.started_tracemalloc = True
        self._runtime.previous_trace = sys.gettrace()
        self._runtime.previous_thread_trace = threading.gettrace()
        threading.settrace(self.trace)
        sys.settrace(self.trace)

    def uninstall(self) -> None:
        """Uninstall."""
        self._runtime.active = False
        with self._control.control_lock:
            self._control.paused.clear()
            self._control.control_lock.notify_all()
        sys.settrace(self._runtime.previous_trace)
        threading.settrace(self._runtime.previous_thread_trace)
        if self._runtime.started_tracemalloc:
            tracemalloc.stop()
