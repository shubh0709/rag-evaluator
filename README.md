# Confluence RAG Evaluation Lab

A local dashboard for running and inspecting Confluence RAG evaluation cases. The synthetic baseline covers seven retrieval metrics and all nine Phase 4 generation checks. It separates evaluator correctness, bot behavior, and evaluation-method validity.

## Start

```bash
python3 -m pip install -r requirements.txt
python3 -m live_debugger streamlit app.py
```

Streamlit automatically reruns the open dashboard when you save changes to the app or imported Python modules. The setting lives in `.streamlit/config.toml`; restart an already running server once after changing that configuration. You can also launch it with `streamlit run app.py` if you do not need the execution debugger. `uv` is optional for environment management; it is not needed for live reload.

Open the local URL shown by Streamlit. The [live execution debugger](live_debugger/README.md) opens in a second browser tab; it shows executed lines, stack frames, local bindings, and a replayable timeline. Select cases and choose **Run selected cases** in the sidebar. Use **Test cases** to edit fixture inputs and outputs. Use **Connections** to configure a future RAG endpoint and check Confluence access. Saved runs are in `artifacts/runs/`; live runs redact question and response content on disk by default.

The **Test cases** tab can export and import a synthetic case pack so edits can be shared or restored. **Compare runs** shows case-level status and retrieval-score changes, with warnings when dataset versions or K differ. [Contracts and examples](docs/CONTRACTS.md) describe the saved result and event trace.

Input DTOs in [rag_eval/dto.py](rag_eval/dto.py) validate case packs, live bot responses, judge responses, and saved run envelopes when those values enter the program. The case editor uses the same case DTO. Pydantic checks required fields, nested shapes, and strict scalar types; the existing domain objects then check rules such as known document IDs and evidence support. This keeps external data checks at the boundaries while preserving metric functions as independently callable APIs.

The **Metric explanations** tab introduces the retrieval and context metrics and includes a toy playground. Change ranked results, labels, K, and a short answer to see evaluator functions update without changing saved runs. MRR and MAP are cross-case averages of reciprocal rank and average precision. Phase 4 details live in the guide below and in each case's generation trace.

## Phase 4 generation evaluation

[Phase 4 guide](PHASE4.md) gives each metric's formula or rubric, test cases, implementation, validation, dashboard trace, demo, and review gate. Run the nine synthetic cases with **Synthetic reviewed labels** selected to exercise the full trace. Those labels are hand authored examples, not LLM outputs. The **OpenAI (.env)** option reads only `OPENAI_API_KEY` and `OPENAI_MODEL`, and requires explicit approval before sending the selected case content. Select one case and one metric for an initial API smoke test. A custom approved gateway is also available. Judge scores stay provisional until a human accepts or rejects the item verdicts in the dashboard.

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
- The seven retrieval metrics and all nine generation checks are implemented. Context metrics use explicit relevance and reference-point support labels against retrieved items. Strategy comparisons remain unavailable because the bot's retrieval architecture and branch traces are unverified.
- The OpenAI model and key are configured in the local `.env`, which is Git-ignored. Contract tests use mocked HTTP responses. A real API smoke call was blocked by automatic approval review because sending question, evidence, and answer content to OpenAI was not explicitly approved; no external judge verdict has been obtained. No real bot has been supplied. The exact context passed to generation is unavailable, so groundedness evaluates recorded retrieved chunks and labels that scope in the trace.

See [HLD](docs/HLD.md), [LLD](docs/LLD.md), and [work log](docs/WORK_LOG.md) for architecture and decision rationale.
