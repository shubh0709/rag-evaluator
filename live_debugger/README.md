# Live execution debugger

Copy this entire `live_debugger/` folder into the root of any Python project. It uses only the Python standard library; the target project's usual dependencies are still needed.

## Start with a Streamlit app

From the project root, replace your usual `streamlit run app.py` command with:

```bash
python3 -m live_debugger streamlit app.py
```

The Streamlit server and debugger run together. The debugger opens in a browser tab at `http://127.0.0.1:8765`; the terminal also prints the link. To see it *inside VS Code*, run **Simple Browser: Show** from the Command Palette and enter that URL. The debugger stays live for the life of the dev server. Stop both with Ctrl-C in the launch terminal.

Pass Streamlit flags after the app path, for example `python3 -m live_debugger streamlit app.py --server.port=8502`. Use `--no-browser` to avoid opening a tab, or `--port 8766` to change the debugger port.

## Start with a regular Python script

```bash
python3 -m live_debugger python main.py
```

The viewer records executed source lines, function calls and returns, argument values, local variables, stack frames, exceptions, thread identity, and sampled Python allocations. **Function flow** is on by default and advances through function entry and exit events. Click **Pause on next line** to stop a running project thread, then use **Step into**, **Step over**, **Step out**, or **Continue**. The thread selector lets you target a particular thread. You can also click the timeline or use the arrow keys to inspect a past event; **Follow latest** jumps to the newest event. Notes are stored in the browser's local storage. The trace is held in memory and capped at 2,500 recent events; it is not written to disk.

The top row keeps **Program**, **Call stack**, and **Variables** in view. Use the **Views** buttons to show or hide panels. Drag a panel's lower-right corner to change its height, or click **⛶** to expand it across the screen. **Esc** restores an expanded panel. View choices and panel heights are saved in this browser; **Reset layout** restores the defaults.

## Threads and asyncio

**Parallel execution** shows observed Python threads side by side, with each thread's current project stack and recent event lane. Use **Inspect** to load that thread's latest trace event into the main source, stack, variable, and heap views. You can pause two threads independently, step one while the other stays paused, or choose **Continue all**.

For `asyncio`, the same panel shows each observed event loop, its pending tasks, and the coroutine chain down to the current `await`. Task cards can be inspected or targeted for pause. Turn on **Runtime tasks** to include tasks outside the project. A task's trace events carry its task and event-loop identity, so its history remains in the timeline after it finishes.

An asyncio event loop runs its tasks on one Python thread. Pausing a task therefore pauses that loop's thread and all other tasks on the same loop until you step or continue it. The panel samples task state while the loop is running; a closed loop disappears from the live loop list, while retained trace events remain available.

The viewer captures Python execution in the launched process. It can pause and step project Python frames, but does not trace separate subprocesses or JavaScript execution. It observes event loops when project code runs inside them; unrelated loops with no traced project activity are not listed. For Streamlit, events appear when a browser session loads or interacts with the app. The heap panel shows objects referenced by the current frame, with shallow sizes and bounded collection previews; it is not a full process heap dump. The memory panel shows allocations tracked by Python's `tracemalloc`, not total operating system memory. Local variables with names suggesting credentials are redacted; other strings are truncated and collections are summarized. The viewer binds only to `127.0.0.1`.
