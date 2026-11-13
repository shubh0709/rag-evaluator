# Confluence RAG Evaluation Lab

A local dashboard for running and inspecting Confluence RAG evaluation cases. The first version uses a synthetic corpus and implements Recall@K. It separates evaluator correctness, bot behavior, and evaluation-method validity.

## Start

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit. Select cases and choose **Run selected cases** in the sidebar. Use **Test cases** to edit fixture inputs and outputs. Use **Connections** to configure a future RAG endpoint and check Confluence access. Saved runs are in `artifacts/runs/`; live runs redact question and response content on disk by default.

## Test

```bash
python3 -m pytest -q
```

## Current limits

- Offline fixture scores do not measure the real bot. No bot access has been provided yet.
- The live adapter expects a JSON POST endpoint and configurable paths to answer and ranked retrieval fields. Verify the actual bot contract before relying on live scores.
- Confluence connection testing checks a supplied URL only; it does not import pages.
- Recall@K is the sole implemented metric. Permission and generation cases are visible but not yet scored by dedicated checks.
- External LLM judges are not connected; only approved providers may be used when this work begins.

See [HLD](docs/HLD.md), [LLD](docs/LLD.md), and [work log](docs/WORK_LOG.md) for architecture and decision rationale.
