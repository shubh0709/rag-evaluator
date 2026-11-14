"""Provide test live debugger utilities and tests."""

from pathlib import Path
import runpy
import threading
import time

import pytest

from live_debugger.trace import TraceStore, describe
from live_debugger.server import ControlCommand


def test_values_redact_credentials_and_bound_heap_preview():
    """Verify values redact credentials and bound heap preview."""
    assert describe("top-secret", name="api_token")["value"] == "<redacted>"
    info = describe(list(range(20)))
    assert info["type"] == "list"
    assert len(info["children"]) == 6


def test_control_command_validates_browser_payload():
    """Verify control command validates browser payload."""
    command = ControlCommand.model_validate({"action": "pause", "thread_id": "42", "task_id": None})
    assert command.thread_id == 42
    with pytest.raises(ValueError, match="task_id"):
        ControlCommand.model_validate({"action": "pause", "task_id": 42})


def test_live_pause_step_and_function_io(tmp_path: Path):
    """Verify live pause step and function io."""
    script = tmp_path / "target.py"
    script.write_text(
        "def double(value):\n    result = value * 2\n    return result\n\ndouble(4)\n"
    )
    store = TraceStore(tmp_path, Path(__file__).parents[1] / "live_debugger")
    store.install()
    store.control("pause")
    worker = threading.Thread(target=lambda: runpy.run_path(str(script)), name="target-worker")
    try:
        worker.start()
        deadline = time.monotonic() + 3
        while store.snapshot()["control"]["paused_thread"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        first = store.snapshot()["control"]
        assert first["paused_thread"] == worker.ident
        store.control("step_into")
        while store.snapshot()["control"]["paused_thread"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        second = store.snapshot()["control"]
        assert first["paused_event"] is not None
        assert second["paused_event"] is not None
        assert second["paused_event"] > first["paused_event"]
        store.control("continue")
        worker.join(timeout=3)
        assert not worker.is_alive()
        events = store.snapshot()["events"]
        assert any(
            event["kind"] == "call" and event["arguments"].get("value") == "4" for event in events
        )
        assert any(event["kind"] == "return" and event["return_value"] == "8" for event in events)
        assert any(event["memory"]["peak"] > 0 for event in events)
    finally:
        store.control("continue")
        if worker.ident is not None:
            worker.join(timeout=2)
        store.uninstall()


@pytest.mark.parametrize(
    "action,target_function,target_line",
    [
        ("step_over", "<module>", 5),
        ("step_out", "double", 2),
    ],
)
def test_step_depth_controls(tmp_path: Path, action: str, target_function: str, target_line: int):
    """Verify step depth controls."""
    script = tmp_path / "target.py"
    script.write_text(
        "def double(value):\n    result = value * 2\n    return result\n\ndouble(4)\n"
    )
    store = TraceStore(tmp_path, Path(__file__).parents[1] / "live_debugger")
    store.install()
    store.control("pause")
    worker = threading.Thread(target=lambda: runpy.run_path(str(script)))
    try:
        worker.start()
        deadline = time.monotonic() + 3

        def paused_after(previous: int = 0):
            while time.monotonic() < deadline:
                state = store.snapshot()["control"]
                if state["paused_thread"] is not None and (state["paused_event"] or 0) > previous:
                    return store.snapshot()["events"][-1]
                time.sleep(0.005)
            raise AssertionError("Debugger did not pause")

        event = paused_after()
        while not (
            event["frames"][0]["function"] == target_function
            and event["frames"][0]["line"] == target_line
        ):
            previous = event["id"]
            store.control("step_into")
            event = paused_after(previous)
        store.control(action)
        result = paused_after(event["id"])
        assert result["frames"][0]["function"] == "<module>"
        assert result["id"] > event["id"]
    finally:
        store.control("continue")
        if worker.ident is not None:
            worker.join(timeout=2)
        store.uninstall()


def test_two_threads_can_pause_and_step_independently(tmp_path: Path):
    """Verify two threads can pause and step independently."""
    script = tmp_path / "worker.py"
    script.write_text(
        "import time\nfor number in range(50):\n    value = number * 2\n    time.sleep(0.01)\n"
    )
    store = TraceStore(tmp_path, Path(__file__).parents[1] / "live_debugger")
    store.install()
    workers = [
        threading.Thread(target=lambda: runpy.run_path(str(script)), name=f"worker-{i}")
        for i in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            store.control("pause", worker.ident)
        deadline = time.monotonic() + 3
        while (
            len(store.snapshot()["control"]["paused_threads"]) < 2 and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        paused = store.snapshot()["control"]["paused_threads"]
        assert {state["thread_id"] for state in paused} == {worker.ident for worker in workers}
        first_id = workers[0].ident
        old_event = next(state["event"] for state in paused if state["thread_id"] == first_id)
        store.control("step_into", first_id)
        while time.monotonic() < deadline:
            states = store.snapshot()["control"]["paused_threads"]
            first = next((state for state in states if state["thread_id"] == first_id), None)
            if first and first["event"] > old_event:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("First thread did not step")
        assert any(state["thread_id"] == workers[1].ident for state in states)
    finally:
        store.control("continue")
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=2)
        store.uninstall()


def test_asyncio_tasks_and_await_chains_are_visible(tmp_path: Path):
    """Verify asyncio tasks and await chains are visible."""
    script = tmp_path / "async_app.py"
    script.write_text(
        (
            "import asyncio\nasync def child(n):\n    await asyncio.sleep(0.5)\n "
            "   return n\nasync def parent():\n    return await "
            "asyncio.gather(child(1), child(2))\nasyncio.run(parent())\n"
        )
    )
    store = TraceStore(tmp_path, Path(__file__).parents[1] / "live_debugger")
    store.install()
    worker = threading.Thread(target=lambda: runpy.run_path(str(script)), name="async-worker")
    try:
        worker.start()
        deadline = time.monotonic() + 2
        snapshot = store.snapshot()
        while time.monotonic() < deadline:
            snapshot = store.snapshot()
            if snapshot["loops"] and snapshot["loops"][0]["project_task_count"] >= 3:
                break
            time.sleep(0.01)
        assert snapshot["loops"][0]["project_task_count"] >= 3
        tasks = snapshot["loops"][0]["tasks"]
        assert (
            sum(
                any(frame["function"] == "child" for frame in task["await_chain"]) for task in tasks
            )
            == 2
        )
        assert any(event["task_id"] and event["loop_id"] for event in snapshot["events"])
        assert any(thread["name"] == "async-worker" for thread in snapshot["threads"])
    finally:
        if worker.ident is not None:
            worker.join(timeout=2)
        store.uninstall()
