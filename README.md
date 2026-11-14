# Confluence RAG Evaluation Lab

A local dashboard for running and inspecting Confluence RAG evaluation cases. The first version uses a synthetic corpus and implements Recall@K. It separates evaluator correctness, bot behavior, and evaluation-method validity.

## Start

```bash
python3 -m pip install -r requirements.txt
python3 -m live_debugger streamlit app.py
```

Open the local URL shown by Streamlit. The [live execution debugger](live_debugger/README.md) opens in a second browser tab; it shows executed lines, stack frames, local bindings, and a replayable timeline. Select cases and choose **Run selected cases** in the sidebar. Use **Test cases** to edit fixture inputs and outputs. Use **Connections** to configure a future RAG endpoint and check Confluence access. Saved runs are in `artifacts/runs/`; live runs redact question and response content on disk by default.

The **Test cases** tab can export and import a synthetic case pack so edits can be shared or restored. **Compare runs** shows case-level status and Recall@K changes, with warnings when dataset versions or K differ.

When Streamlit is bound to a loopback address (the included default is `127.0.0.1`), **Check understanding** loads `study_quiz/QUESTIONS.md` and `study_quiz/ANSWER_KEY.md` into a local MCQ test. Submit all answers to see your score, correct answers, and explanations, or retake it. The `study_quiz/` folder is Git-ignored and must exist locally for the quiz to load; the tab is hidden when the server binds to a non-loopback address.

## Test

```bash
python3 -m pytest -q
python3 -m pyright
```

Pyright checks the app, evaluation modules, debugger, and tests in strict mode.

## Current limits

- Offline fixture scores do not measure the real bot. No bot access has been provided yet.
- The live adapter expects a JSON POST endpoint and configurable paths to answer and ranked retrieval fields. Verify the actual bot contract before relying on live scores.
- Confluence connection testing checks a supplied URL only; it does not import pages.
- Recall@K is the sole implemented metric. Permission and generation cases are visible but not yet scored by dedicated checks.
- External LLM judges are not connected; only approved providers may be used when this work begins.

See [HLD](docs/HLD.md), [LLD](docs/LLD.md), and [work log](docs/WORK_LOG.md) for architecture and decision rationale.
