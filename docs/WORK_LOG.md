# Confluence RAG Evaluation System — Work Log

This is the shared record of what was learned, proposed, and decided. Add a dated entry when a choice changes, including its reason, evidence, owner, and effect on later work. **A proposal is not an approved architectural decision.**

## 2026-11-13 — Requirements discovery

**Scope:** Read the supplied project brief and `rag-evals.md`; inspected the local workspace. The workspace is empty, with no Git repository, bot code, fixtures, or existing documentation. No application code, HLD, or LLD was created.

**Source boundary:** The pasted project brief describes a proposed process and roadmap. The user's current message explicitly asks for a work log, identifies three distinct testing layers, and says the suggested architecture must pass HLD approval. This entry records those direct requests as requirements; other details from the brief and reference remain inputs to design, not independently approved instructions or facts about the bot.

### Confirmed requirements and decisions

| ID | Status | Choice / requirement | Why and consequence | Source |
| --- | --- | --- | --- | --- |
| D-001 | Confirmed | Report three separate kinds of evidence: (1) evaluator implementation correctness, (2) Confluence RAG bot performance, and (3) evaluation methodology validity. | A mathematically correct score can still rest on wrong labels or an unrepresentative dataset. Reports and dashboard views must identify which layer each result supports. | Direct user request |
| D-002 | Confirmed | Record architectural choices and their rationale in this work log for collaborators. | Makes decisions, assumptions, and later changes reviewable. | Direct user request |
| D-003 | Confirmed | Do not treat the suggested stack and workflow as approved architecture before HLD review. | The user explicitly reserved architectural approval. | Direct user request |
| D-004 | Confirmed | Start with synthetic fixtures. | They provide a controlled dataset without exposing company content. Synthetic results must be labeled as such. | User clarification, 2026-11-13 |
| D-005 | Confirmed | No bot access is available yet. | Live integration contracts and telemetry cannot be verified; design an adapter boundary and defer bot-performance claims. | User clarification, 2026-11-13 |
| D-006 | Confirmed | External LLM judges may use only approved providers. | Provider approval and data-sharing scope must be checked before any Confluence content is sent. | User clarification, 2026-11-13 |

### Proposed architecture — pending HLD approval

| ID | Proposal | Rationale | Trade-off / question to resolve |
| --- | --- | --- | --- |
| P-001 | Python + pytest for the evaluation backend. | Simple deterministic metric tests and a broad evaluation ecosystem. | Confirm team stack, deployment constraints, and integration language. |
| P-002 | React + TypeScript dashboard, or a lightweight Python UI for the first trace viewer. | A trace viewer should make inputs, evidence, calculations, and errors inspectable. | Choose based on team skills and expected interactivity; avoid building two UIs. |
| P-003 | Start with offline synthetic fixtures. | Repeatable, low-cost tests while bot and Confluence access are unknown. | Fixture success does not establish real bot performance. |
| P-004 | Add real bot integration after the first deterministic metric work. | Avoid blocking learning on Confluence access. | The exact sequence depends on available APIs and traces. |
| P-005 | Store versioned JSON run artifacts initially; add a database when usage requires one. | Keeps initial infrastructure small and runs reproducible. | Need retention, search, concurrency, and access-control requirements before choosing storage. |
| P-006 | Implement one metric per reviewed task. | Smaller changes make formulas and evidence easier to inspect. | Confirm review cadence and ownership. |
| P-007 | Use independently human-verified document or chunk relevance labels. | Prevents the bot or judge from defining its own ground truth. | Define reviewer process, granularity, adjudication, and label versioning. |
| P-008 | Use one logical commit per completed task. | Supports review, comparison, and rollback. | Git repository has not yet been initialized. |

### Evaluation boundaries to preserve in design

| Layer | Example question | Evidence needed | What it does **not** establish |
| --- | --- | --- | --- |
| Evaluator correctness | Does Recall@K return the expected value for known retrieved and relevant IDs? | Formula, fixed inputs, expected numeric output, edge-case tests. | Whether the bot retrieved good evidence on real questions. |
| Bot performance | Does the Confluence bot retrieve authorized, current evidence and answer well? | Real or controlled bot executions, ranked retrieval trace, answer, citations, permissions, configuration. | Whether gold labels, judge scores, or question sample are trustworthy. |
| Methodology validity | Are labels, judge decisions, and sampled questions reliable? | Independent human review, judge-human agreement, dataset coverage, versioned rubrics, permission oracle. | That a particular metric implementation is bug-free. |

For Recall@K, the design must choose the relevance unit (document or chunk), treatment of duplicate IDs, multiple relevant IDs, queries with no relevant labels, and whether `K` means the first K ranked results before deduplication. The supplied `rag-evals.md` illustrates a single-gold hit-rate style calculation, while the user's example asks about gold **sets** and a score of 0.4. The formula and these edge cases need explicit HLD/LLD agreement before a metric is implemented.

### Reference use and applicability

- `rag-evals.md` is a metric catalog for a system called Kitsune. Its field names, retrieval methods, modes, memory, tool-use, and operational data are **not yet verified** for the Confluence bot.
- The reference's numerical targets, schedules, and broad metric inventory are examples to assess, not project acceptance criteria.
- Enterprise scenarios that matter for this project include unauthorized retrieval/context exposure, outdated or conflicting pages, and citation support. Their design requires independently established expected behavior and permission labels.

### Open questions that affect HLD

1. When access becomes available, where is the bot implemented, and what are its request/response API and ranked retrieval trace contracts? Can a sample execution be shared?
2. Will a Confluence test environment be available later? Which spaces, authentication method, and user permission contexts can be used?
3. Who will verify real document/chunk IDs, expected answers, current-versus-superseded status, and authorization expectations? At what granularity?
4. Which LLM providers are approved, and for which data classes? What redaction, retention, and access controls apply to persisted traces and judge inputs?
5. Are Python + pytest and React + TypeScript acceptable to the team, or is there an existing stack to follow?
6. What evidence or access milestone marks readiness for live bot evaluation?

### Next step

Draft an HLD that labels proposed decisions and alternatives. Obtain HLD approval before choosing the final stack or writing application code. Add a new dated log entry for each approval or change; do not silently overwrite the rationale above.

## 2026-11-13 — Clarifications received and HLD draft

**New evidence:** The user confirmed no bot access yet, synthetic fixtures first, and external judge services limited to approved providers. These answers are recorded as D-004 through D-006. The HLD can therefore describe an offline first architecture while leaving the live adapter and judge provider contract conditional.

**Artifact:** `docs/HLD.md` is a review draft. Its technology selections are proposals, not approvals. No application code or Git commit was created.

## 2026-11-13 — HLD approved; first implementation slice

**Approval:** The user said the HLD looks good and requested implementation, with a highly usable dashboard that can manually trigger cases and configure Confluence and RAG chatbot access.

| ID | Status | Implementation decision | Why / trade-off |
| --- | --- | --- | --- |
| D-007 | Implemented | Use Python + Streamlit for the initial dashboard. | Streamlit is available locally and supports interactive forms, case editing, manual runs, trace inspection, and JSON downloads without a separate frontend build. A later React UI remains an option if richer multi-user workflows justify it. |
| D-008 | Implemented | Keep initial run artifacts as JSON and session-edited cases in memory. | Simple reproduction and review. Session edits are lost on restart; saved runs preserve the exact case and response used. |
| D-009 | Implemented | Add a configurable generic JSON POST bot adapter and Confluence URL reachability check. | Enables a future live run without assuming an unavailable bot contract. Reachability is not validation of retrieval telemetry or authorization behavior. |
| D-010 | Implemented | Persist live run content redacted by default; keep tokens only in the dashboard session. | Reduces accidental storage of sensitive text while still preserving IDs and score calculations. The dashboard is local only until account access controls and a data policy are added. |
| D-011 | Implemented | Recall@K v1.0 uses unique relevant IDs found within the first K ranked positions divided by unique labeled relevant IDs. Duplicate hits occupy rank positions; no-label cases are not applicable. | This resolves the multi-gold ambiguity before scoring. The unit is synthetic document ID in this slice; real labels may require chunk granularity. |

**Implemented:** Synthetic six-page corpus and four cases; editable offline fixture responses; selected-case manual runs; K and threshold controls; full case trace and calculation inspection; run history; session connection settings and checks; first deterministic metric; pytest tests; README and LLD.

**Evidence:** Local browser smoke test triggered all four fixture cases and displayed three Recall@K passes, one not-applicable case, ranked retrieval and the calculation breakdown. Unit tests passed (13 tests). No real bot or Confluence test was possible because access remains unavailable.

**Limitations and next decisions:** Permission and answer-quality scenarios are visible but not yet scored by dedicated metrics. The generic live adapter must be checked against an actual API contract. Provider approval, data policy, team deployment, and multi-user access controls remain open. Git was initialized for this first implementation slice; future tasks should use focused commits.

**Final verification:** 13 pytest tests passed; Python compilation passed. One warning comes from the installed AnyIO pytest plugin's use of a name deprecated for pytest 10. The local Streamlit browser smoke test showed the run summary and complete case trace. The server was restarted bound to `127.0.0.1:8501` so the unprotected local dashboard is not exposed on the network.

## Entry template

```text
Date:
Task / decision ID:
Status: proposed | approved | superseded | rejected
Decision and owner:
Context and evidence:
Alternatives considered:
Reason and trade-offs:
Affected files or interfaces:
Tests / review outcome:
Approval reference:
Next action:
```
