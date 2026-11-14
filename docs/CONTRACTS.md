# Evaluation contracts — synthetic baseline

**Version:** run schema 2. The initial relevance unit is a document ID. A later bot integration must explicitly choose document or chunk IDs before real scores are compared.

## Inputs and reference

`TestCase` in `rag_eval/core.py` is the versioned input and reference record. It contains a stable case ID, name, question, `user_id`, category, `relevant_ids`, expected answer, supporting evidence, and reviewer notes. `relevant_ids` must be distinct nonempty strings. An empty list means retrieval metrics are **not applicable**; it does not mean that any retrieved document is authorized. The synthetic corpus in `rag_eval/fixtures.py` supplies the supporting text for these IDs. Real references will require independently reviewed evidence and permission labels.

The RAG request is `{ "question": string, "user_id": string }`. The generic live adapter sends that object as JSON. The offline adapter uses the same fields from the case. Authentication is passed to the adapter separately and never placed in run configuration.

## Response, citations, and trace

`RagResponse` has an answer string, ordered `retrieved` items, citation ID strings, and safe raw metadata. A retrieved item has `id`, `title`, `text`, `source_url`, and optional `document_id` and `chunk_id`. The `id` is the scoring unit and order is rank; the other IDs preserve source granularity when supplied. Citation IDs are references, not automatically proof that a claim is supported. The live adapter rejects missing answer, ranked list, or citations fields. It does not currently receive the exact context supplied to generation, so that trace field is explicitly `missing`.

Each case result records the exact case, response or error, Recall@K and Precision@K calculations, separate metric statuses, and an execution trace. The trace contains the request, start and finish timestamps when captured by the dashboard, a recording timestamp, stage events, and availability markers for ranked retrieval and generation context. A direct `make_run` call without execution timestamps records them as `null` instead of inventing timing evidence. The run contains `schema_version`, run ID, UTC timestamp, dataset hash, mode, safe configuration, K, threshold, case results, and a legacy summary based on Recall@K status. Old schema-1 run files remain readable; they lack Precision@K and event traces.

New runs also contain `reciprocal_rank_calculation` and `reciprocal_rank_status` on each case, plus run-level `mrr_summary` with `score`, `reciprocal_rank_sum`, `scored_cases`, and `excluded_cases`. These optional fields preserve reading of earlier saved runs. MRR uses all ranked IDs, regardless of K. A labeled case with no relevant hit contributes zero; a case with no relevance labels or a response error is excluded. A null MRR means no cases were scorable.

An adapter or validation failure is `error` with no numeric metric. A case without relevance labels is `not_applicable` with a null score. A numeric score below the selected threshold is `failed`. These states remain distinct.

## Example

```json
{
  "schema_version": 2,
  "id": "run-20261114T120000Z-abc123",
  "created_at": "2026-11-14T12:00:00+00:00",
  "dataset_version": "examplehash",
  "mode": "offline",
  "config": {"source": "Offline fixtures"},
  "k": 2,
  "threshold": 0.5,
  "summary": {"passed": 1, "failed": 0, "not_applicable": 0, "error": 0},
  "results": [{
    "case": {"id": "CASE-001", "name": "Example", "question": "What is the current limit?", "relevant_ids": ["DOC-A", "DOC-B"], "expected_answer": "See current policy", "category": "freshness", "user_id": "employee", "notes": "Synthetic", "supporting_evidence": "DOC-A: Current limit"},
    "metric": "recall_at_k",
    "metric_version": "1.0.0",
    "status": "passed",
    "precision_status": "passed",
    "error": null,
    "response": {"answer": "See current policy", "retrieved": [{"id": "DOC-A", "title": "Current policy", "text": "Current limit", "source_url": "", "document_id": "DOC-A", "chunk_id": ""}, {"id": "DOC-X", "title": "Distractor", "text": "Other policy", "source_url": "", "document_id": "DOC-X", "chunk_id": ""}], "citations": ["DOC-A"], "raw": {}},
    "calculation": {"status": "scored", "score": 0.5, "reason": "", "k": 2, "numerator": 1, "denominator": 2, "hits": ["DOC-A"], "misses": ["DOC-B"], "top_k": ["DOC-A", "DOC-X"]},
    "precision_calculation": {"status": "scored", "score": 0.5, "reason": "", "k": 2, "numerator": 1, "denominator": 2, "relevant_ranks": [1], "duplicate_ranks": [], "top_k": ["DOC-A", "DOC-X"]},
    "trace": {"request": {"question": "What is the current limit?", "user_id": "employee"}, "started_at": null, "finished_at": null, "recorded_at": "2026-11-14T12:00:00+00:00", "events": [{"at": "2026-11-14T12:00:00+00:00", "stage": "request", "status": "available", "detail": "offline request recorded"}, {"at": "2026-11-14T12:00:00+00:00", "stage": "response", "status": "available", "detail": "Response captured"}, {"at": "2026-11-14T12:00:00+00:00", "stage": "metrics", "status": "available", "detail": "Recall@K and Precision@K calculated"}], "ranked_retrieval_status": "available", "generation_context_status": "missing"}
  }]
}
```

## Precision@K v1.0

For ordered IDs `R`, relevant set `G`, and `K >= 1`, count each relevant ID at its first appearance within `R[:K]`. Divide that count by `K`. Duplicate hits occupy rank positions but earn no second hit. Empty positions also earn no hit. If `G` is empty, the score is not applicable because this dataset does not independently label all retrieved content as irrelevant.

Example: `R = [A, A, B]`, `G = {A, B}`, `K = 3` gives two useful unique hits in three positions, so Precision@3 is `2/3`. Recall@3 is `2/2 = 1`. Precision describes how much of the top K is useful; recall describes how much of the known relevant set was found. A perfect recall score can coexist with poor precision when many distractors are returned.
