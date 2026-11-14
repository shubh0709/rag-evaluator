"""Run with: streamlit run app.py"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import streamlit as st

from rag_eval.adapters import check_connection, persisted_copy, run_live_case
from rag_eval.core import RagResponse, Result, RetrievedItem, TestCase, make_run, with_recomputed_mrr
from rag_eval.datasets import export_case_pack, import_case_pack
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES
from rag_eval.quiz import is_local_dev_address, load_quiz
from rag_eval.store import RunStore


st.set_page_config(page_title="Confluence RAG Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; max-width: 1500px;}
  h1,h2,h3 {letter-spacing:-0.035em;}
  h1 {font-size:2.35rem!important;}
  div[data-testid="stMetric"] {border:1px solid #dde6e2; border-radius:14px; padding:14px; background:#fbfdfb;}
  div[data-testid="stAlert"] {border-radius:12px;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
  .eyebrow {color:#477566; letter-spacing:.12em; text-transform:uppercase; font-size:.76rem; font-weight:750;}
  .subtle {color:#68756f;}
  .badge {display:inline-block; padding:5px 10px; border-radius:99px; background:#e7f3ec; color:#155a3d; font-size:.78rem; font-weight:700;}
</style>
""", unsafe_allow_html=True)


def init_state() -> None:
    if "cases" not in st.session_state:
        st.session_state.cases = list(CASES)
    else:
        fixtures = {case.id: case for case in CASES}
        st.session_state.cases = [
            case if all(hasattr(case, field) for field in
                        ("relevance_grades", "context_relevance", "reference_points", "point_support")) else TestCase(
                case.id, case.name, case.question, case.relevant_ids, case.expected_answer,
                case.category, case.user_id, case.notes, case.supporting_evidence,
                getattr(case, "relevance_grades", fixtures.get(case.id).relevance_grades if case.id in fixtures else {}),
                fixtures[case.id].context_relevance if case.id in fixtures else {},
                fixtures[case.id].reference_points if case.id in fixtures else (),
                fixtures[case.id].point_support if case.id in fixtures else {})
            for case in st.session_state.cases
        ]
    if "responses" not in st.session_state:
        st.session_state.responses = dict(RESPONSES)
    if "current_run" not in st.session_state:
        st.session_state.current_run = None


def result_row(result: Result) -> dict[str, object]:
    calculation = result["calculation"]
    precision = result.get("precision_calculation")
    rr = result.get("reciprocal_rank_calculation")
    ndcg = result.get("ndcg_calculation")
    ap = result.get("average_precision_calculation")
    cp = result.get("context_precision_calculation")
    cr = result.get("context_recall_calculation")
    return {"Case": result["case"]["id"], "Name": result["case"]["name"],
            "Category": result["case"]["category"],
            "Recall@K": calculation["score"] if calculation is not None else None,
            "Precision@K": precision["score"] if precision is not None else None,
            "Reciprocal rank": rr["score"] if rr is not None else None,
            "NDCG@K": ndcg["score"] if ndcg is not None else None,
            "Average precision": ap["score"] if ap is not None else None,
            "Context precision": cp["score"] if cp is not None else None,
            "Context recall": cr["score"] if cr is not None else None,
            "Recall status": result["status"],
            "Precision status": result.get("precision_status", "not recorded"),
            "RR status": result.get("reciprocal_rank_status", "not recorded"),
            "NDCG status": result.get("ndcg_status", "not recorded"),
            "AP status": result.get("average_precision_status", "not recorded"),
            "Context precision status": result.get("context_precision_status", "not recorded"),
            "Context recall status": result.get("context_recall_status", "not recorded"),
            "Hits": f"{calculation['numerator']}/{calculation['denominator']}"
            if calculation is not None else ""}


METRIC_HELP = {
    "recall": (
        "**Recall@K:** The share of all labeled relevant documents found in the first K results. "
        "Each document counts once. Example: if A, B, and C are relevant and the top 2 are A and X, "
        "Recall@2 is 1/3 = 0.33."
    ),
    "precision": (
        "**Precision@K:** The number of distinct relevant documents in the first K positions, "
        "divided by K. Duplicates and empty positions do not earn a hit. Example: if A is relevant "
        "and the top 2 are A and X, Precision@2 is 1/2 = 0.50."
    ),
    "reciprocal_rank": (
        "**Reciprocal rank:** 1 divided by the position of the first relevant document in the "
        "full ranked list; 0 if none is found. Example: first relevant document at rank 3 gives 1/3 = 0.33."
    ),
    "mrr": (
        "**Mean reciprocal rank (MRR):** The average reciprocal rank across scored cases. "
        "It uses the full ranked list, independent of K. Example: if two cases first find a relevant "
        "document at ranks 1 and 3, MRR is (1 + 1/3) / 2 = 0.67."
    ),
    "ndcg": (
        "**NDCG@K:** A score from 0 to 1 that rewards highly relevant documents appearing "
        "near the top. It compares discounted graded relevance in the first K results with the "
        "ideal ordering. Example: if A has grade 2 and B grade 1, [A, B] scores 1.00 at K=2; "
        "[B, A] scores about 0.80."
    ),
    "average_precision": (
        "**Average precision (AP):** Average precision at each first relevant hit, divided by "
        "the number of all labeled relevant documents. Missing relevant documents contribute zero. "
        "Example: if A and B are relevant and results are [A, X, B], AP is "
        "(1/1 + 2/3) / 2 = 0.83."
    ),
    "map": (
        "**Mean average precision (MAP):** The average of AP across scored cases. "
        "Example: case AP scores of 1.00 and 0.50 give MAP = 0.75."
    ),
    "context_precision": (
        "**Context precision:** Uses reviewed useful/not useful judgments for retrieved contexts. "
        "It averages precision at each useful context's rank. Example: judgments "
        "[useful, not useful, useful] give (1/1 + 2/3) / 2 = 0.83."
    ),
    "context_recall": (
        "**Context recall:** The share of labeled reference answer points supported by at "
        "least one retrieved context. Example: if retrieved contexts support 2 of 3 required "
        "answer points, context recall is 2/3 = 0.67."
    ),
}

RESULT_METRIC_COLUMNS = {
    "Recall@K": "recall",
    "Precision@K": "precision",
    "Reciprocal rank": "reciprocal_rank",
    "NDCG@K": "ndcg",
    "Average precision": "average_precision",
    "Context precision": "context_precision",
    "Context recall": "context_recall",
}

RESULT_STATUS_COLUMNS = {
    "Recall status": "recall",
    "Precision status": "precision",
    "RR status": "reciprocal_rank",
    "NDCG status": "ndcg",
    "AP status": "average_precision",
    "Context precision status": "context_precision",
    "Context recall status": "context_recall",
}

STATUS_HELP = (
    "**Status:** Passed means the score meets this run's threshold; failed means it is below it. "
    "Not applicable means the metric lacks the labels or relevant evidence needed for a score. "
    "Error means the case could not be scored. "
    "Example: with a threshold of 0.80, a score of 0.90 passes and 0.70 fails."
)


init_state()
store = RunStore()

with st.sidebar:
    st.markdown("### ◈ RAG Evaluation Lab")
    st.caption("Confluence evaluation workspace")
    st.divider()
    st.markdown("**Run controls**")
    mode = st.radio("Execution source", ["Offline fixtures", "Live RAG endpoint"],
                    help="Live runs require an accessible bot API using the fixture response shape.")
    k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)
    threshold = st.slider("Metric pass threshold", 0.0, 1.0, 1.0, 0.05,
                          help="This is a local review threshold, not a universal quality target.")
    selected_ids = st.multiselect("Cases to run", [c.id for c in st.session_state.cases],
                                  default=[c.id for c in st.session_state.cases],
                                  format_func=lambda x: next((f"{x} · {c.name}" for c in st.session_state.cases if c.id == x), x))
    st.caption("Every run records its selected cases, K, threshold, and dataset version.")
    run_clicked = st.button("▶ Run selected cases", type="primary", width="stretch", disabled=not selected_ids)
    st.divider()
    st.caption("Metrics · Recall@K, Precision@K, MRR, NDCG@K, MAP, context precision, context recall")

st.markdown('<div class="eyebrow">Evaluation workspace</div>', unsafe_allow_html=True)
st.title("See what your RAG system actually did")
st.markdown('<p class="subtle">Run cases, inspect ranked evidence, and trace every score back to its inputs.</p>', unsafe_allow_html=True)

quiz_enabled = is_local_dev_address(st.get_option("server.address"))
tab_names = ["Overview", "Test cases", "Connections", "Run history", "Compare runs", "Methodology"]
if quiz_enabled:
    tab_names.append("Check understanding")
tabs = st.tabs(tab_names)

with tabs[2]:
    st.subheader("Connections")
    st.info("No bot or Confluence access is available yet. Configure endpoints here when access is granted. Connection settings stay in this browser session; tokens are never written to run artifacts.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### RAG chatbot")
        bot_url = st.text_input("POST endpoint URL", key="bot_url", placeholder="https://bot.example.com/api/ask")
        bot_token = st.text_input("Access token", type="password", key="bot_token")
        bot_header = st.text_input("Authentication header", value="Authorization", key="bot_header")
        bot_prefix = st.text_input("Token prefix", value="Bearer", key="bot_prefix")
        bot_timeout = st.number_input("Bot timeout (seconds)", min_value=1, max_value=120, value=30, key="bot_timeout")
        if st.button("Check bot endpoint", disabled=not bot_url):
            check = check_connection(bot_url, token=bot_token, header_name=bot_header, prefix=bot_prefix)
            if check["ok"]:
                st.success(f"{check['message']} (HTTP {check['status_code']}). This does not validate the POST response contract.")
            else:
                st.warning(f"Connection check: {check['message']} (HTTP {check['status_code']})")
    with right:
        st.markdown("#### Confluence")
        st.caption("This checks access only. The evaluator does not ingest Confluence pages until the corpus and permissions contract is approved.")
        conf_url = st.text_input("Confluence test URL", key="conf_url", placeholder="https://example.atlassian.net/wiki/rest/api/user/current")
        conf_token = st.text_input("Confluence access token", type="password", key="conf_token")
        conf_header = st.text_input("Confluence auth header", value="Authorization", key="conf_header")
        conf_prefix = st.text_input("Confluence token prefix", value="Bearer", key="conf_prefix")
        if st.button("Check Confluence access", disabled=not conf_url):
            check = check_connection(conf_url, token=conf_token, header_name=conf_header, prefix=conf_prefix)
            if check["ok"]:
                st.success(f"Confluence URL reachable (HTTP {check['status_code']}).")
            else:
                st.warning(f"Connection check: {check['message']} (HTTP {check['status_code']})")
        st.markdown("#### Expected bot response")
        st.code('{"answer": "...", "retrieved": [{"id": "DOC-101", "title": "...", "text": "...", "source_url": "..."}], "citations": ["DOC-101"]}', language="json")

if run_clicked:
    selected = [case for case in st.session_state.cases if case.id in selected_ids]
    responses: dict[str, RagResponse | Exception] = {}
    execution_times: dict[str, tuple[str, str]] = {}
    if mode == "Offline fixtures":
        for case in selected:
            started = datetime.now(timezone.utc).isoformat()
            responses[case.id] = st.session_state.responses.get(case.id, ValueError("No offline response configured"))
            execution_times[case.id] = (started, datetime.now(timezone.utc).isoformat())
    elif not st.session_state.get("bot_url"):
        st.error("Set the RAG chatbot POST endpoint in Connections before running live cases.")
    else:
        with st.spinner("Running selected cases against the configured RAG endpoint…"):
            for case in selected:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    responses[case.id] = run_live_case(
                        case, url=st.session_state.bot_url, token=st.session_state.bot_token,
                        header_name=st.session_state.bot_header, prefix=st.session_state.bot_prefix,
                        timeout=int(st.session_state.bot_timeout))
                except Exception as exc:
                    responses[case.id] = exc
                finally:
                    execution_times[case.id] = (started, datetime.now(timezone.utc).isoformat())
    if responses:
        safe_config: dict[str, object] = {"source": mode, "selected_case_ids": selected_ids,
                       "bot_endpoint": st.session_state.get("bot_url", "") if mode == "Live RAG endpoint" else "",
                       "response_contract": "fixture-v1" if mode == "Live RAG endpoint" else ""}
        run = make_run(selected, responses, mode=mode, k=int(k), threshold=float(threshold),
                       config=safe_config, execution_times=execution_times)
        st.session_state.current_run = run
        store.save(persisted_copy(run, include_content=mode == "Offline fixtures"))
        st.toast(f"Run {run['id']} completed", icon="✅")

with tabs[0]:
    run = st.session_state.current_run
    if run is None:
        history = store.list()
        run = history[0] if history else None
    original_run = run
    if run is not None:
        run = with_recomputed_mrr(run)
    if run is None:
        st.info("Ready to run. Select cases and press **Run selected cases** in the sidebar. The first run uses the included synthetic fixtures.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic documents", len(DOCUMENTS))
        c2.metric("Available cases", len(st.session_state.cases))
        c3.metric("Metrics implemented", 7)
    else:
        st.subheader("Latest run")
        st.caption(f"{run['id']} · {run['created_at']} · {run['mode']} · dataset {run['dataset_version']}")
        cols = st.columns(4)
        for col, key, label in zip(cols, ("passed", "failed", "not_applicable", "error"),
                                   ("Passed", "Failed", "Not applicable", "Errors")):
            col.metric(label, run["summary"][key])
        st.caption("Status counts describe Recall@K. Other metrics have separate case statuses.")
        mrr = run.get("mrr_summary")
        if mrr is not None:
            st.metric("Mean reciprocal rank", f"{mrr['score']:.3f}" if mrr["score"] is not None else "N/A",
                      help=METRIC_HELP["mrr"])
            st.caption(f"{mrr['scored_cases']} scored cases · {mrr['excluded_cases']} excluded cases")
        if run.get("mrr_recomputed"):
            st.info("This run predates MRR. Values shown here were calculated from its saved labels and ranked IDs; the saved run file is unchanged.")
        map_summary = run.get("map_summary")
        if map_summary is not None:
            st.metric("Mean average precision", f"{map_summary['score']:.3f}" if map_summary["score"] is not None else "N/A",
                      help=METRIC_HELP["map"])
            st.caption(f"MAP: {map_summary['scored_cases']} scored cases · {map_summary['excluded_cases']} excluded cases")
        st.markdown("#### Case results")
        st.dataframe(
            [result_row(r) for r in run["results"]],
            column_config={
                **{label: st.column_config.NumberColumn(f"{label} ⓘ", help=METRIC_HELP[key])
                   for label, key in RESULT_METRIC_COLUMNS.items()},
                **{label: st.column_config.TextColumn(
                    f"{label} ⓘ", help=f"{METRIC_HELP[key]}\n\n{STATUS_HELP}")
                   for label, key in RESULT_STATUS_COLUMNS.items()},
                "Hits": st.column_config.TextColumn(
                    "Hits ⓘ", help="Relevant documents found in the first K results / all labeled relevant documents. "
                    "Example: 2/3 means two of three relevant documents were retrieved in the first K positions."),
            },
            width="stretch", hide_index=True,
        )
        st.markdown("#### Trace inspector")
        chosen = st.selectbox("Inspect case", [r["case"]["id"] for r in run["results"]], key="inspect_case")
        result = next(r for r in run["results"] if r["case"]["id"] == chosen)
        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            st.markdown("**Input and reference**")
            st.write(result["case"]["question"])
            st.caption(f"User: {result['case']['user_id']} · Category: {result['case']['category']}")
            st.write("Expected relevant IDs:", ", ".join(result["case"]["relevant_ids"]) or "None labeled")
            st.write("Expected answer:", result["case"]["expected_answer"])
            if result["case"].get("supporting_evidence"):
                st.write("Supporting evidence:", result["case"]["supporting_evidence"])
            st.write("Graded relevance:", result["case"].get("relevance_grades", {}))
            st.write("Context relevance judgments:", result["case"].get("context_relevance", {}))
            st.write("Reference points:", result["case"].get("reference_points", ()))
            if result["case"]["notes"]:
                st.caption(result["case"]["notes"])
            st.markdown("**Actual answer**")
            st.write((result["response"] or {}).get("answer") or "Unavailable")
            st.write("Citations:", ", ".join((result["response"] or {}).get("citations", [])) or "None")
            if result["error"]:
                st.error(result["error"])
        with c2:
            st.markdown("**Recall@K calculation**")
            calc = result["calculation"]
            if calc:
                if calc["score"] is None:
                    st.warning(calc["reason"])
                else:
                    st.metric(f"Recall@{calc['k']}", f"{calc['score']:.2f}", help=METRIC_HELP["recall"])
                    st.write(f"**{calc['numerator']} hits / {calc['denominator']} relevant IDs = {calc['score']:.2f}**")
                    st.write("Hits:", ", ".join(calc["hits"]) or "None")
                    st.write("Misses:", ", ".join(calc["misses"]) or "None")
                    st.caption(f"Pass threshold: {run['threshold']:.2f} · Status: {result['status']}")
            st.markdown("**Precision@K calculation**")
            precision = result.get("precision_calculation")
            if precision is None:
                st.caption("Unavailable in this run.")
            elif precision["score"] is None:
                st.warning(precision["reason"])
            else:
                st.metric(f"Precision@{precision['k']}", f"{precision['score']:.2f}",
                          help=METRIC_HELP["precision"])
                st.write(f"**{precision['numerator']} relevant hits / {precision['denominator']} ranks = {precision['score']:.2f}**")
                st.write("Relevant ranks:", ", ".join(map(str, precision["relevant_ranks"])) or "None")
                st.write("Duplicate ranks:", ", ".join(map(str, precision["duplicate_ranks"])) or "None")
                st.caption(f"Pass threshold: {run['threshold']:.2f} · Precision status: {result.get('precision_status', 'unknown')}")
            st.markdown("**Reciprocal rank calculation**")
            rr = result.get("reciprocal_rank_calculation")
            if rr is None:
                st.caption("Unavailable in this run.")
            elif rr["score"] is None:
                st.warning(rr["reason"])
            else:
                st.metric("Reciprocal rank", f"{rr['score']:.3f}",
                          help=METRIC_HELP["reciprocal_rank"])
                st.write("First relevant ID:", rr["first_relevant_id"] or "None")
                st.write("First relevant rank:", rr["first_relevant_rank"] or "None")
                st.caption(f"Status: {result.get('reciprocal_rank_status', 'unknown')}")
            st.markdown("**NDCG@K calculation**")
            ndcg = result.get("ndcg_calculation")
            if ndcg is None:
                st.caption("Unavailable in this run.")
            elif ndcg["score"] is None:
                st.warning(ndcg["reason"])
            else:
                st.metric(f"NDCG@{ndcg['k']}", f"{ndcg['score']:.3f}", help=METRIC_HELP["ndcg"])
                st.write(f"DCG {ndcg['dcg']:.4f} / ideal DCG {ndcg['ideal_dcg']:.4f}")
                st.write("Ranked grades:", ndcg["ranked_grades"])
                st.write("Ideal grades:", ndcg["ideal_grades"])
                st.write("Discounted gains:", [round(value, 4) for value in ndcg["discounted_gains"]])
                st.caption(f"Duplicate ranks: {ndcg['duplicate_ranks'] or 'None'} · Status: {result.get('ndcg_status', 'unknown')}")
            for title, key, status_key, help_key in (
                ("Average precision", "average_precision_calculation", "average_precision_status", "average_precision"),
                ("Context precision", "context_precision_calculation", "context_precision_status", "context_precision"),
                ("Context recall", "context_recall_calculation", "context_recall_status", "context_recall"),
            ):
                st.markdown(f"**{title} calculation**")
                detail = result.get(key)
                if detail is None:
                    st.caption("Unavailable in this run.")
                elif detail["score"] is None:
                    st.warning(detail["reason"])
                    st.json(detail)
                else:
                    st.metric(title, f"{detail['score']:.3f}", help=METRIC_HELP[help_key])
                    st.caption(f"Status: {result.get(status_key, 'unknown')}")
                    st.json(detail)
            st.markdown("**Ranked retrieved context**")
            retrieved = (result["response"] or {}).get("retrieved", [])
            if retrieved:
                for rank, hit in enumerate(retrieved, 1):
                    with st.expander(f"#{rank} · {hit['id']} · {hit['title']}", expanded=rank <= min(int(k), 2)):
                        st.write(hit["text"] or "No text in trace")
                        if hit.get("source_url"):
                            st.caption(hit["source_url"])
            else:
                st.caption("No ranked retrieval available.")
        with st.expander("Complete case trace and provenance"):
            st.json({"run_id": run["id"], "dataset_version": run["dataset_version"],
                     "run_config": run["config"], "k": run["k"], "threshold": run["threshold"],
                     "case_result": result})
        trace = result.get("trace")
        if trace:
            st.markdown("**Execution events**")
            st.dataframe(trace["events"], hide_index=True, width="stretch")
            st.caption(f"Ranked retrieval: {trace['ranked_retrieval_status']} · Generation context: {trace['generation_context_status']} · Started: {trace['started_at'] or 'unavailable'} · Finished: {trace['finished_at'] or 'unavailable'}")
        st.download_button("Download full run JSON", json.dumps(original_run, indent=2), file_name=f"{run['id']}.json", mime="application/json")
        st.caption("A synthetic run checks evaluator behavior against fixtures. It does not establish real bot quality or label validity.")

with tabs[1]:
    st.subheader("Test cases and fixture responses")
    st.caption("Edit a case and its offline response, then run it from the sidebar. Changes live in this dashboard session and are versioned by dataset hash in each run.")
    with st.expander("Share or restore a synthetic case pack"):
        st.caption("Export cases and predefined responses as JSON. Import validates references against the six-page synthetic corpus. Connection tokens are never included.")
        pack = export_case_pack(st.session_state.cases, st.session_state.responses)
        st.download_button("Download case pack", json.dumps(pack, indent=2),
                           file_name="synthetic-case-pack.json", mime="application/json")
        uploaded = st.file_uploader("Import case pack JSON", type="json")
        if st.button("Import case pack", disabled=uploaded is None) and uploaded is not None:
            try:
                imported_cases, imported_responses = import_case_pack(
                    cast(dict[str, object], json.load(uploaded)), set(DOCUMENTS))
                st.session_state.cases = imported_cases
                st.session_state.responses = imported_responses
                st.success(f"Imported {len(imported_cases)} cases. The sidebar selection has been refreshed.")
                st.rerun()
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))
    case_id = st.selectbox("Choose a case to edit", [c.id for c in st.session_state.cases], key="edit_case")
    case = next(c for c in st.session_state.cases if c.id == case_id)
    response = st.session_state.responses.get(case_id, RagResponse("", ()))
    with st.form(f"edit-{case_id}"):
        a, b = st.columns(2, gap="large")
        with a:
            name = st.text_input("Case name", case.name)
            question = st.text_area("Question", case.question)
            relevant = st.text_input("Relevant IDs, comma separated", ", ".join(case.relevant_ids))
            grade_text = st.text_input("Graded relevance (JSON ID to nonnegative integer)",
                                       json.dumps(case.relevance_grades))
            context_text = st.text_area("Context relevance judgments (JSON ID to true/false)",
                                        json.dumps(case.context_relevance))
            points_text = st.text_area("Reference points (one per line)",
                                       "\n".join(case.reference_points))
            support_text = st.text_area("Point support (JSON point to context ID list)",
                                        json.dumps(case.point_support))
            expected = st.text_area("Expected answer / behavior", case.expected_answer)
            category = st.text_input("Category", case.category)
            user_id = st.text_input("User context ID", case.user_id)
            notes = st.text_area("Case notes", case.notes)
            supporting_evidence = st.text_area("Supporting evidence", case.supporting_evidence)
        with b:
            actual_answer = st.text_area("Offline actual answer", response.answer)
            retrieved_ids = st.text_area("Ranked retrieved IDs, comma separated", ", ".join(r.id for r in response.retrieved),
                                         help="Order matters. Duplicate IDs are allowed to exercise duplicate behavior.")
            citations = st.text_input("Offline citations, comma separated", ", ".join(response.citations))
            st.caption("Known synthetic IDs: " + ", ".join(DOCUMENTS))
        save_case = st.form_submit_button("Save case and fixture", type="primary")
    if save_case:
        ids = tuple(x.strip() for x in relevant.split(",") if x.strip())
        ranked = tuple(x.strip() for x in retrieved_ids.split(",") if x.strip())
        try:
            grades = json.loads(grade_text)
            judgments = json.loads(context_text)
            support_raw = json.loads(support_text)
            if not isinstance(support_raw, dict) or any(not isinstance(items, list) for items in support_raw.values()):
                raise ValueError("Point support must be a JSON object of ID lists")
            support = {point: tuple(items) for point, items in support_raw.items()}
            points = tuple(line.strip() for line in points_text.splitlines() if line.strip())
            updated = TestCase(case.id, name, question, ids, expected, category, user_id, notes,
                               supporting_evidence, grades, judgments, points, support)
            updated.validate()
            if any(i not in DOCUMENTS for i in ids + ranked) or any(i not in DOCUMENTS for i in grades) or any(
                i not in DOCUMENTS for i in judgments
            ) or any(i not in DOCUMENTS for items in support.values() for i in items):
                raise ValueError("IDs must refer to the synthetic corpus")
            updated_response = RagResponse(actual_answer,
                tuple(RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"]) for i in ranked),
                tuple(x.strip() for x in citations.split(",") if x.strip()))
            st.session_state.cases = [updated if c.id == case.id else c for c in st.session_state.cases]
            st.session_state.responses[case.id] = updated_response
            st.success("Saved in this session. Run the case to see the new result.")
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(str(exc))
    with st.expander("Add a new synthetic case"):
        with st.form("add-case"):
            new_id = st.text_input("New case ID", placeholder="CASE-005")
            new_name = st.text_input("Name")
            new_question = st.text_area("Question")
            new_relevant = st.text_input("Relevant IDs, comma separated")
            new_grades = st.text_input("Graded relevance (JSON ID to nonnegative integer)", value="{}")
            new_context = st.text_area("Context relevance judgments (JSON ID to true/false)", value="{}")
            new_points = st.text_area("Reference points (one per line)")
            new_support = st.text_area("Point support (JSON point to context ID list)", value="{}")
            new_expected = st.text_input("Expected answer / behavior")
            new_evidence = st.text_area("Supporting evidence for new case")
            new_category = st.text_input("Category", value="custom")
            new_actual = st.text_area("Offline actual answer")
            new_ranked = st.text_input("Ranked retrieved IDs, comma separated")
            add_case = st.form_submit_button("Add case")
        if add_case:
            ids = tuple(x.strip() for x in new_relevant.split(",") if x.strip())
            ranked = tuple(x.strip() for x in new_ranked.split(",") if x.strip())
            try:
                grades = json.loads(new_grades)
                judgments = json.loads(new_context)
                support_raw = json.loads(new_support)
                if not isinstance(support_raw, dict) or any(not isinstance(items, list) for items in support_raw.values()):
                    raise ValueError("Point support must be a JSON object of ID lists")
                support = {point: tuple(items) for point, items in support_raw.items()}
                points = tuple(line.strip() for line in new_points.splitlines() if line.strip())
                if new_id in [c.id for c in st.session_state.cases]:
                    raise ValueError("Case ID already exists")
                new_case = TestCase(new_id, new_name, new_question, ids, new_expected,
                                    new_category, supporting_evidence=new_evidence,
                                    relevance_grades=grades, context_relevance=judgments,
                                    reference_points=points, point_support=support)
                new_case.validate()
                if any(i not in DOCUMENTS for i in ids + ranked) or any(i not in DOCUMENTS for i in grades) or any(
                    i not in DOCUMENTS for i in judgments
                ) or any(i not in DOCUMENTS for items in support.values() for i in items):
                    raise ValueError("IDs must refer to the synthetic corpus")
                st.session_state.cases.append(new_case)
                st.session_state.responses[new_id] = RagResponse(new_actual, tuple(
                    RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"]) for i in ranked))
                st.success("Case added. Select it in the sidebar to run it.")
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))
    with st.expander("Synthetic document catalog"):
        st.dataframe([{"ID": i, **d} for i, d in DOCUMENTS.items()], hide_index=True, width="stretch")

with tabs[3]:
    st.subheader("Run history")
    history = store.list()
    if not history:
        st.info("No saved runs yet.")
    else:
        st.dataframe([{"Run": r["id"], "When (UTC)": r["created_at"], "Source": r["mode"],
                       "Dataset": r["dataset_version"], "Cases": len(r["results"]),
                       **r["summary"]} for r in history], hide_index=True, width="stretch")
        history_id = st.selectbox("Open saved run", [r["id"] for r in history])
        saved = next(r for r in history if r["id"] == history_id)
        if saved.get("content_redacted"):
            st.info("Sensitive content was redacted from this saved live run. The current session may still hold the full trace.")
        if st.button("Show this run in Overview"):
            st.session_state.current_run = saved
            st.info("Open the Overview tab to inspect the selected run.")
        st.download_button("Download saved run", json.dumps(saved, indent=2), file_name=f"{history_id}.json", mime="application/json")

with tabs[4]:
    st.subheader("Compare saved runs")
    history = store.list()
    if len(history) < 2:
        st.info("Run the cases at least twice to compare a baseline and a candidate.")
    else:
        left, right = st.columns(2)
        with left:
            baseline_id = st.selectbox("Baseline run", [r["id"] for r in history], index=1)
        with right:
            candidate_id = st.selectbox("Candidate run", [r["id"] for r in history], index=0)
        baseline = with_recomputed_mrr(next(r for r in history if r["id"] == baseline_id))
        candidate = with_recomputed_mrr(next(r for r in history if r["id"] == candidate_id))
        if baseline_id == candidate_id:
            st.warning("Choose two different runs to see changes.")
        if baseline["dataset_version"] != candidate["dataset_version"]:
            st.warning("Dataset versions differ. Score changes may reflect edited cases or labels rather than bot behavior.")
        if baseline["k"] != candidate["k"]:
            st.warning("K differs between runs. Retrieval scores are not directly comparable.")
        base_results = {r["case"]["id"]: r for r in baseline["results"]}
        candidate_results = {r["case"]["id"]: r for r in candidate["results"]}
        rows: list[dict[str, str | float | None]] = []
        for case_id in sorted(set(base_results) | set(candidate_results)):
            before = base_results.get(case_id)
            after = candidate_results.get(case_id)
            before_calc = before["calculation"] if before is not None else None
            after_calc = after["calculation"] if after is not None else None
            before_score = before_calc["score"] if before_calc is not None else None
            after_score = after_calc["score"] if after_calc is not None else None
            before_precision = before.get("precision_calculation") if before is not None else None
            after_precision = after.get("precision_calculation") if after is not None else None
            before_precision_score = before_precision["score"] if before_precision is not None else None
            after_precision_score = after_precision["score"] if after_precision is not None else None
            before_rr = before.get("reciprocal_rank_calculation") if before is not None else None
            after_rr = after.get("reciprocal_rank_calculation") if after is not None else None
            before_rr_score = before_rr["score"] if before_rr is not None else None
            after_rr_score = after_rr["score"] if after_rr is not None else None
            before_ndcg = before.get("ndcg_calculation") if before is not None else None
            after_ndcg = after.get("ndcg_calculation") if after is not None else None
            before_ndcg_score = before_ndcg["score"] if before_ndcg is not None else None
            after_ndcg_score = after_ndcg["score"] if after_ndcg is not None else None
            row = {"Case": case_id, "Baseline status": before["status"] if before else "not run",
                         "Candidate status": after["status"] if after else "not run",
                         "Baseline recall": before_score, "Candidate recall": after_score,
                         "Score change": round(after_score - before_score, 4)
                         if before_score is not None and after_score is not None else None,
                         "Baseline precision": before_precision_score,
                         "Candidate precision": after_precision_score,
                         "Precision change": round(after_precision_score - before_precision_score, 4)
                         if before_precision_score is not None and after_precision_score is not None else None,
                         "Baseline RR": before_rr_score, "Candidate RR": after_rr_score,
                         "RR change": round(after_rr_score - before_rr_score, 4)
                         if before_rr_score is not None and after_rr_score is not None else None,
                         "Baseline NDCG": before_ndcg_score, "Candidate NDCG": after_ndcg_score,
                         "NDCG change": round(after_ndcg_score - before_ndcg_score, 4)
                         if before_ndcg_score is not None and after_ndcg_score is not None else None}
            for label, field in (("AP", "average_precision_calculation"),
                                 ("Context precision", "context_precision_calculation"),
                                 ("Context recall", "context_recall_calculation")):
                before_metric = before.get(field) if before else None
                after_metric = after.get(field) if after else None
                before_value = before_metric["score"] if before_metric else None
                after_value = after_metric["score"] if after_metric else None
                row[f"Baseline {label}"] = before_value
                row[f"Candidate {label}"] = after_value
                row[f"{label} change"] = round(after_value - before_value, 4) if (
                    before_value is not None and after_value is not None) else None
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("A status change can also result from a threshold change. Inspect both run configurations and case traces before drawing conclusions.")

with tabs[5]:
    st.subheader("What these results mean")
    st.markdown("""
**1 · Evaluator correctness.** Unit tests check retrieval and context metrics against known labels, including duplicates, empty retrieval, and unlabeled cases.

**2 · Bot performance.** A live run can measure the bot only when its API and ranked trace are available. Offline fixture scores are development examples, not evidence of production quality.

**3 · Methodology validity.** Relevant IDs, permission expectations, and future judge outputs require independent human validation. A precise score can still be misleading when labels are wrong.

**Recall@K definition in this version:** unique labeled relevant IDs present in the first K ranked results, divided by all unique labeled relevant IDs. Duplicates occupy ranks but do not increase the numerator. Cases with no labeled relevant IDs are not applicable. The pass threshold is configurable per run.

**Precision@K definition in this version:** unique labeled relevant IDs in the first K ranked positions, divided by K. Duplicate hits and unfilled positions earn no hit. With no relevance labels the score is not applicable. Precision measures how clean the top K is; recall measures how much of all relevant evidence it contains.

**MRR definition in this version:** for each labeled case, reciprocal rank is 1 divided by the first relevant rank in the full retrieved list, or 0 if no relevant ID appears. MRR averages these values across scored cases. Unlabeled cases and execution errors are excluded; if none can be scored, MRR is unavailable. MRR does not use K.

**NDCG@K definition in this version:** use explicit nonnegative integer relevance grades. At rank r, gain is `(2^grade - 1) / log₂(r + 1)`. Sum these gains for the first K ranks to get DCG. Sort labeled grades descending and apply the same discount to get ideal DCG. NDCG is DCG divided by ideal DCG. Duplicate IDs earn zero after their first appearance. Cases without a positive grade are not applicable. Unlabeled retrieved IDs count as grade zero.

**MAP:** average precision sums prefix precision at each first relevant hit and divides by the number of all labeled relevant IDs. MAP averages those case scores over labeled cases with valid responses. A missing relevant ID contributes zero.

**Context precision:** uses separately reviewed true/false judgments for each retrieved context item. It averages prefix precision at ranks judged useful. The denominator is the number of useful retrieved contexts, unlike classical Precision@K's fixed K. Unjudged retrieved IDs make this metric unavailable.

**Context recall:** counts reference answer points supported by at least one retrieved context item, divided by all labeled reference points. It uses explicit point-to-context support annotations, not string matching or a live LLM judge. Both context metrics measure retrieved context; the exact context passed to generation remains unavailable.

**Retrieval strategies:** section routing, vector/keyword, hybrid, and RRF/multi-query comparisons are unavailable until the bot exposes its architecture, per-strategy ranked traces, and comparable run configuration. The current adapter records one final ranked list only.

**Current limits:** Permission and answer quality scenarios are present as fixtures but are not scored by dedicated metrics yet. Confluence connection testing does not import pages. Live bot responses currently use the fixture contract until a real API contract is available. External LLM judges are not connected.
""")

if quiz_enabled:
    with tabs[6]:
        st.subheader("Check understanding")
        st.caption("Test what has been built so far. Your answers and result stay in this dashboard session.")
        quiz_folder = Path(__file__).resolve().parent / "study_quiz"
        try:
            questions = load_quiz(quiz_folder)
        except (OSError, ValueError) as exc:
            st.info(f"Quiz unavailable: {exc}. Add QUESTIONS.md and ANSWER_KEY.md to study_quiz/.")
        else:
            if st.session_state.get("quiz_result") is None:
                with st.form("understanding_quiz"):
                    section = None
                    for question in questions:
                        if question.section != section:
                            section = question.section
                            st.markdown(f"#### {section}")
                        st.markdown(f"**{question.number}. {question.prompt}**")
                        st.radio(
                            f"Answer for question {question.number}",
                            list(question.options),
                            index=None,
                            format_func=lambda choice, options=question.options: f"{choice}. {options[choice]}",
                            key=f"quiz_answer_{question.number}",
                            label_visibility="collapsed",
                        )
                    submitted = st.form_submit_button("Submit answers", type="primary")
                if submitted:
                    answers = {q.number: st.session_state.get(f"quiz_answer_{q.number}") for q in questions}
                    unanswered = [number for number, answer in answers.items() if answer is None]
                    if unanswered:
                        st.warning(f"Answer all {len(questions)} questions before submitting. Unanswered: {', '.join(map(str, unanswered))}.")
                    else:
                        st.session_state.quiz_result = answers
                        st.rerun()
            else:
                answers = st.session_state.quiz_result
                correct = sum(answers.get(q.number) == q.answer for q in questions)
                st.metric("Your score", f"{correct}/{len(questions)}", f"{correct / len(questions):.0%}")
                st.progress(correct / len(questions))
                if correct == len(questions):
                    st.success("All answers correct.")
                else:
                    st.info(f"{len(questions) - correct} question(s) to review. Your answers and the correct answers are below.")
                for question in questions:
                    chosen = answers.get(question.number)
                    is_correct = chosen == question.answer
                    with st.expander(f"{'✅' if is_correct else '❌'} {question.number}. {question.prompt}", expanded=not is_correct):
                        st.write(f"Your answer: **{chosen}. {question.options.get(chosen, 'No answer')}**")
                        st.write(f"Correct answer: **{question.answer}. {question.options[question.answer]}**")
                        st.write(question.explanation)
                if st.button("Retake quiz"):
                    st.session_state.quiz_result = None
                    for question in questions:
                        st.session_state.pop(f"quiz_answer_{question.number}", None)
                    st.rerun()
