"""Run with: streamlit run app.py"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import requests

from rag_eval.adapters import check_connection, persisted_copy, run_live_case
from rag_eval.core import (
    RagResponse,
    Result,
    TestCase,
    make_run,
    with_recomputed_mrr,
)
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES
from rag_eval.generation_fixtures import (
    MODEL as FIXTURE_JUDGE_MODEL,
    MODEL_VERSION as FIXTURE_JUDGE_VERSION,
    fixture_judge,
)
from rag_eval.generation_judge import (
    RUBRICS,
    call_judge_endpoint,
    evaluate_generation_run,
    review_judgment,
)
from rag_eval.openai_judge import load_openai_config, openai_judge
from rag_eval.quiz import is_local_dev_address
from rag_eval.store import RunStore
from rag_eval.ui_views import (
    render_case_editor,
    render_history,
    render_comparison,
    render_interpretation,
    render_guide,
    render_quiz,
)

st.set_page_config(
    page_title="Confluence RAG Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(
    """
<style>
  .block-container {padding-top: 1.4rem; max-width: 1500px;}
  h1,h2,h3 {letter-spacing:-0.035em;}
  h1 {font-size:2.35rem!important;}
  div[data-testid="stMetric"] {
    border:1px solid #dde6e2; border-radius:14px;
    padding:14px; background:#fbfdfb;
  }
  div[data-testid="stAlert"] {border-radius:12px;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
  .eyebrow {
    color:#477566; letter-spacing:.12em; text-transform:uppercase;
    font-size:.76rem; font-weight:750;
  }
  .subtle {color:#68756f;}
  .badge {
    display:inline-block; padding:5px 10px; border-radius:99px;
    background:#e7f3ec; color:#155a3d; font-size:.78rem; font-weight:700;
  }
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialize the session state required by the app."""
    if "cases" not in st.session_state:
        st.session_state.cases = list(CASES)
    else:
        fixtures = {case.id: case for case in CASES}
        st.session_state.cases = [
            (
                case
                if all(
                    hasattr(case, field)
                    for field in (
                        "relevance_grades",
                        "context_relevance",
                        "reference_points",
                        "point_support",
                        "exact_match_references",
                        "answerable",
                        "trap_prior_answer",
                        "trap_context_answer",
                    )
                )
                else TestCase(
                    case.id,
                    case.name,
                    case.question,
                    case.relevant_ids,
                    case.expected_answer,
                    case.category,
                    case.user_id,
                    case.notes,
                    case.supporting_evidence,
                    getattr(
                        case,
                        "relevance_grades",
                        fixtures[case.id].relevance_grades if case.id in fixtures else {},
                    ),
                    fixtures[case.id].context_relevance if case.id in fixtures else {},
                    fixtures[case.id].reference_points if case.id in fixtures else (),
                    fixtures[case.id].point_support if case.id in fixtures else {},
                    fixtures[case.id].exact_match_references if case.id in fixtures else (),
                    fixtures[case.id].answerable if case.id in fixtures else None,
                    fixtures[case.id].trap_prior_answer if case.id in fixtures else "",
                    fixtures[case.id].trap_context_answer if case.id in fixtures else "",
                )
            )
            for case in st.session_state.cases
        ]
    if "responses" not in st.session_state:
        st.session_state.responses = dict(RESPONSES)
    if "current_run" not in st.session_state:
        st.session_state.current_run = None


def result_row(metric_result: Result) -> dict[str, object]:
    """Convert a metric result into a table row."""
    calculation = metric_result["calculation"]
    metric_precision = metric_result.get("precision_calculation")
    metric_rr = metric_result.get("reciprocal_rank_calculation")
    metric_ndcg = metric_result.get("ndcg_calculation")
    metric_ap = metric_result.get("average_precision_calculation")
    metric_cp = metric_result.get("context_precision_calculation")
    metric_cr = metric_result.get("context_recall_calculation")
    metric_em = metric_result.get("exact_match_calculation")
    metric_f1 = metric_result.get("token_f1_calculation")
    return {
        "Case": metric_result["case"]["id"],
        "Name": metric_result["case"]["name"],
        "Category": metric_result["case"]["category"],
        "Recall@K": calculation["score"] if calculation is not None else None,
        "Precision@K": metric_precision["score"] if metric_precision is not None else None,
        "Reciprocal rank": metric_rr["score"] if metric_rr is not None else None,
        "NDCG@K": metric_ndcg["score"] if metric_ndcg is not None else None,
        "Average precision": metric_ap["score"] if metric_ap is not None else None,
        "Context precision": metric_cp["score"] if metric_cp is not None else None,
        "Context recall": metric_cr["score"] if metric_cr is not None else None,
        "Exact Match": metric_em["score"] if metric_em is not None else None,
        "Token F1": metric_f1["score"] if metric_f1 is not None else None,
        "Recall status": metric_result["status"],
        "Precision status": metric_result.get("precision_status", "not recorded"),
        "RR status": metric_result.get("reciprocal_rank_status", "not recorded"),
        "NDCG status": metric_result.get("ndcg_status", "not recorded"),
        "AP status": metric_result.get("average_precision_status", "not recorded"),
        "Context precision status": metric_result.get("context_precision_status", "not recorded"),
        "Context recall status": metric_result.get("context_recall_status", "not recorded"),
        "Exact Match status": metric_result.get("exact_match_status", "not recorded"),
        "Token F1 status": metric_result.get("token_f1_status", "not recorded"),
        "Hits": (
            f"{calculation['numerator']}/{calculation['denominator']}"
            if calculation is not None
            else ""
        ),
    }


METRIC_HELP = {
    "recall": (
        "**Recall@K:** The share of all labeled relevant documents found in the first K results. "
        "Each document counts once. Example: if A, B, and C are relevant "
        "and the top 2 are A and X, "
        "Recall@2 is 1/3 = 0.33."
    ),
    "precision": (
        "**Precision@K:** The number of distinct relevant documents in the first K positions, "
        "divided by K. Duplicates and empty positions do not earn a hit. Example: if A is relevant "
        "and the top 2 are A and X, Precision@2 is 1/2 = 0.50."
    ),
    "reciprocal_rank": (
        "**Reciprocal rank:** 1 divided by the position of the first relevant document in the "
        "full ranked list; 0 if none is found. Example: first relevant "
        "document at rank 3 gives 1/3 = 0.33."
    ),
    "mrr": (
        "**Mean reciprocal rank (MRR):** The average reciprocal rank across scored cases. "
        "It uses the full ranked list, independent of K. Example: if two "
        "cases first find a relevant "
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
    "exact_match": (
        "**Exact Match:** 1 when the normalized answer equals any "
        "explicit gold answer, otherwise 0. "
        "Normalization casefolds text, removes punctuation and English articles (a, an, the), "
        "and collapses whitespace. It is suitable for short checkable answers; a correct longer "
        "sentence can still score 0."
    ),
    "token_f1": (
        "**Token F1:** Bag-of-tokens overlap after the same normalization as Exact Match. "
        "For each gold answer, precision = overlap / answer tokens and "
        "recall = overlap / gold tokens; "
        "F1 = 2PR/(P+R). The highest reference score wins."
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
    "Exact Match": "exact_match",
    "Token F1": "token_f1",
}

RESULT_STATUS_COLUMNS = {
    "Recall status": "recall",
    "Precision status": "precision",
    "RR status": "reciprocal_rank",
    "NDCG status": "ndcg",
    "AP status": "average_precision",
    "Context precision status": "context_precision",
    "Context recall status": "context_recall",
    "Exact Match status": "exact_match",
    "Token F1 status": "token_f1",
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
    mode = st.radio(
        "Execution source",
        ["Offline fixtures", "Live RAG endpoint"],
        help="Live runs require an accessible bot API using the fixture response shape.",
    )
    k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)
    threshold = st.slider(
        "Metric pass threshold",
        0.0,
        1.0,
        1.0,
        0.05,
        help="This is a local review threshold, not a universal quality target.",
    )
    selected_ids = st.multiselect(
        "Cases to run",
        [c.id for c in st.session_state.cases],
        default=[c.id for c in st.session_state.cases],
        format_func=lambda x: next(
            (f"{x} · {c.name}" for c in st.session_state.cases if c.id == x), x
        ),
    )
    st.caption("Every run records its selected cases, K, threshold, and dataset version.")
    run_clicked = st.button(
        "▶ Run selected cases", type="primary", width="stretch", disabled=not selected_ids
    )
    st.markdown("**Generation judge**")
    judge_mode = st.selectbox(
        "Judge source",
        ["Synthetic reviewed labels", "OpenAI (.env)", "Approved judge gateway", "Off"],
        index=0 if mode == "Offline fixtures" else 3,
        help="Synthetic labels exercise the trace; they are not LLM results.",
    )
    judge_url = judge_token = judge_model = judge_version = ""
    judge_approved = False
    judge_metrics = tuple(RUBRICS)
    if judge_mode == "OpenAI (.env)":
        judge_metrics = tuple(
            st.multiselect(
                "OpenAI judge metrics",
                list(RUBRICS),
                default=["faithfulness"],
                help=(
                    "Start with one metric and a small synthetic case selection to "
                    "control API calls."
                ),
            )
        )
        judge_approved = st.checkbox(
            (
                "Approve sending selected questions, retrieved evidence, answers, "
                "and references to OpenAI"
            ),
            value=False,
            help="Applies to this run only. The selected cases are sent to the OpenAI API.",
        )
    if judge_mode == "Approved judge gateway":
        judge_url = st.text_input("Judge gateway URL", key="judge_url")
        judge_model = st.text_input("Judge model", key="judge_model")
        judge_version = st.text_input("Judge model version", key="judge_version")
        judge_token = st.text_input("Judge token", type="password", key="judge_token")
        judge_approved = st.checkbox("Provider and data class approved for this run", value=False)
    st.divider()
    st.caption("Metrics · 7 retrieval, 2 deterministic generation, 7 judge based generation")

st.markdown('<div class="eyebrow">Evaluation workspace</div>', unsafe_allow_html=True)
st.title("See what your RAG system actually did")
st.markdown(
    (
        '<p class="subtle">Run cases, inspect ranked evidence, and trace '
        "every score back to its inputs.</p>"
    ),
    unsafe_allow_html=True,
)

quiz_enabled = is_local_dev_address(st.get_option("server.address"))
tab_names = [
    "Overview",
    "Test cases",
    "Connections",
    "Run history",
    "Compare runs",
    "Methodology",
    "Metric explanations",
]
if quiz_enabled:
    tab_names.append("Check understanding")
tabs = st.tabs(tab_names)

with tabs[2]:
    st.subheader("Connections")
    st.info(
        "No bot or Confluence access is available yet. Configure "
        "endpoints here when access is granted. Connection settings stay "
        "in this browser session; tokens are never written to run "
        "artifacts."
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### RAG chatbot")
        bot_url = st.text_input(
            "POST endpoint URL", key="bot_url", placeholder="https://bot.example.com/api/ask"
        )
        bot_token = st.text_input("Access token", type="password", key="bot_token")
        bot_header = st.text_input("Authentication header", value="Authorization", key="bot_header")
        bot_prefix = st.text_input("Token prefix", value="Bearer", key="bot_prefix")
        bot_timeout = st.number_input(
            "Bot timeout (seconds)", min_value=1, max_value=120, value=30, key="bot_timeout"
        )
        if st.button("Check bot endpoint", disabled=not bot_url):
            check = check_connection(
                bot_url, token=bot_token, header_name=bot_header, prefix=bot_prefix
            )
            if check["ok"]:
                st.success(
                    (
                        f"{check['message']} (HTTP {check['status_code']}). "
                        "This does not validate the POST response contract."
                    )
                )
            else:
                st.warning(f"Connection check: {check['message']} (HTTP {check['status_code']})")
    with right:
        st.markdown("#### Confluence")
        st.caption(
            "This checks access only. The evaluator does not ingest "
            "Confluence pages until the corpus and permissions contract is "
            "approved."
        )
        conf_url = st.text_input(
            "Confluence test URL",
            key="conf_url",
            placeholder="https://example.atlassian.net/wiki/rest/api/user/current",
        )
        conf_token = st.text_input("Confluence access token", type="password", key="conf_token")
        conf_header = st.text_input(
            "Confluence auth header", value="Authorization", key="conf_header"
        )
        conf_prefix = st.text_input("Confluence token prefix", value="Bearer", key="conf_prefix")
        if st.button("Check Confluence access", disabled=not conf_url):
            check = check_connection(
                conf_url, token=conf_token, header_name=conf_header, prefix=conf_prefix
            )
            if check["ok"]:
                st.success(f"Confluence URL reachable (HTTP {check['status_code']}).")
            else:
                st.warning(f"Connection check: {check['message']} (HTTP {check['status_code']})")
        st.markdown("#### Expected bot response")
        st.code(
            (
                '{"answer": "...", "retrieved": [{"id": "DOC-101", "title": '
                '"...", "text": "...", "source_url": "..."}], "citations": '
                '["DOC-101"]}'
            ),
            language="json",
        )

if run_clicked:
    selected = [case for case in st.session_state.cases if case.id in selected_ids]
    responses: dict[str, RagResponse | Exception] = {}
    execution_times: dict[str, tuple[str, str]] = {}
    if mode == "Offline fixtures":
        for case in selected:
            started = datetime.now(timezone.utc).isoformat()
            responses[case.id] = st.session_state.responses.get(
                case.id, ValueError("No offline response configured")
            )
            execution_times[case.id] = (started, datetime.now(timezone.utc).isoformat())
    elif not st.session_state.get("bot_url"):
        st.error("Set the RAG chatbot POST endpoint in Connections before running live cases.")
    else:
        with st.spinner("Running selected cases against the configured RAG endpoint…"):
            for case in selected:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    responses[case.id] = run_live_case(
                        case,
                        url=st.session_state.bot_url,
                        token=st.session_state.bot_token,
                        header_name=st.session_state.bot_header,
                        prefix=st.session_state.bot_prefix,
                        timeout=int(st.session_state.bot_timeout),
                    )
                except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
                    responses[case.id] = exc
                finally:
                    execution_times[case.id] = (started, datetime.now(timezone.utc).isoformat())
    if responses:
        safe_config: dict[str, object] = {
            "source": mode,
            "selected_case_ids": selected_ids,
            "bot_endpoint": (
                st.session_state.get("bot_url", "") if mode == "Live RAG endpoint" else ""
            ),
            "response_contract": "fixture-v1" if mode == "Live RAG endpoint" else "",
        }
        run = make_run(
            selected,
            responses,
            mode=mode,
            k=int(k),
            threshold=float(threshold),
            config=safe_config,
            execution_times=execution_times,
        )
        valid_responses = {
            case_id: response
            for case_id, response in responses.items()
            if isinstance(response, RagResponse)
        }
        if judge_mode == "Synthetic reviewed labels" and mode == "Offline fixtures":
            run = evaluate_generation_run(
                run,
                selected,
                valid_responses,
                fixture_judge,
                model=FIXTURE_JUDGE_MODEL,
                model_version=FIXTURE_JUDGE_VERSION,
            )
            run["config"]["judge_source"] = "synthetic-reviewed-labels"
        elif judge_mode == "OpenAI (.env)":
            if not judge_approved:
                st.warning(
                    "OpenAI judge skipped: approve sending the selected case content "
                    "to OpenAI first."
                )
            elif not judge_metrics:
                st.warning("OpenAI judge skipped: select at least one metric.")
            else:
                try:
                    openai_key, openai_model = load_openai_config()

                    def judge_call(payload: dict[str, Any]) -> dict[str, Any]:
                        """Call the configured judge for this evaluation."""
                        return openai_judge(payload, api_key=openai_key, model=openai_model)

                    run = evaluate_generation_run(
                        run,
                        selected,
                        valid_responses,
                        judge_call,
                        model=openai_model,
                        model_version=openai_model,
                        metrics=judge_metrics,
                    )
                    run["config"].update(
                        {
                            "judge_source": "openai-responses",
                            "judge_model": openai_model,
                            "judge_metrics": list(judge_metrics),
                        }
                    )
                except ValueError as exc:
                    st.error(f"OpenAI judge configuration: {exc}")
        elif judge_mode == "Approved judge gateway":
            if not (judge_approved and judge_url and judge_model and judge_version):
                st.warning(
                    "Judge gateway skipped: confirm provider approval and supply URL, "
                    "model, and version."
                )
            else:

                def judge_call(payload: dict[str, Any]) -> dict[str, Any]:
                    """Call the configured judge for this evaluation."""
                    return call_judge_endpoint(
                        {**payload, "model": judge_model, "model_version": judge_version},
                        url=judge_url,
                        token=judge_token,
                    )

                run = evaluate_generation_run(
                    run,
                    selected,
                    valid_responses,
                    judge_call,
                    model=judge_model,
                    model_version=judge_version,
                )
                run["config"].update(
                    {
                        "judge_source": "approved-gateway",
                        "judge_model": judge_model,
                        "judge_model_version": judge_version,
                    }
                )
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
        st.info(
            "Ready to run. Select cases and press **Run selected cases** in "
            "the sidebar. The first run uses the included synthetic fixtures."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic documents", len(DOCUMENTS))
        c2.metric("Available cases", len(st.session_state.cases))
        c3.metric("Metrics implemented", 16)
    else:
        st.subheader("Latest run")
        st.caption(
            f"{run['id']} · {run['created_at']} · {run['mode']} · dataset {run['dataset_version']}"
        )
        cols = st.columns(4)
        for col, key, label in zip(
            cols,
            ("passed", "failed", "not_applicable", "error"),
            ("Passed", "Failed", "Not applicable", "Errors"),
        ):
            col.metric(label, run["summary"][key])
        st.caption("Status counts describe Recall@K. Other metrics have separate case statuses.")
        mrr = run.get("mrr_summary")
        if mrr is not None:
            st.metric(
                "Mean reciprocal rank",
                f"{mrr['score']:.3f}" if mrr["score"] is not None else "N/A",
                help=METRIC_HELP["mrr"],
            )
            st.caption(
                f"{mrr['scored_cases']} scored cases · {mrr['excluded_cases']} excluded cases"
            )
        if run.get("mrr_recomputed"):
            st.info(
                "This run predates MRR. Values shown here were calculated from "
                "its saved labels and ranked IDs; the saved run file is unchanged."
            )
        map_summary = run.get("map_summary")
        if map_summary is not None:
            st.metric(
                "Mean average precision",
                f"{map_summary['score']:.3f}" if map_summary["score"] is not None else "N/A",
                help=METRIC_HELP["map"],
            )
            st.caption(
                (
                    f"MAP: {map_summary['scored_cases']} scored cases · "
                    f"{map_summary['excluded_cases']} excluded cases"
                )
            )
        generation_summary = run.get("generation_summary")
        if generation_summary:
            st.markdown("#### Abstention error rates")
            with st.container(horizontal=True):
                for label, rate, count, denominator in (
                    (
                        "False confidence",
                        "false_confidence_rate",
                        "false_confidence_count",
                        "unanswerable_scored",
                    ),
                    (
                        "False abstention",
                        "false_abstention_rate",
                        "false_abstention_count",
                        "answerable_scored",
                    ),
                ):
                    value = generation_summary[rate]
                    st.metric(
                        label,
                        f"{value:.0%}" if value is not None else "N/A",
                        help=(
                            f"{generation_summary[count]}/{generation_summary[denominator]} "
                            "scored planted cases; synthetic labels are not a production estimate."
                        ),
                    )
        st.markdown("#### Case results")
        st.dataframe(
            [result_row(r) for r in run["results"]],
            column_config={
                **{
                    label: st.column_config.NumberColumn(f"{label} ⓘ", help=METRIC_HELP[key])
                    for label, key in RESULT_METRIC_COLUMNS.items()
                },
                **{
                    label: st.column_config.TextColumn(
                        f"{label} ⓘ", help=f"{METRIC_HELP[key]}\n\n{STATUS_HELP}"
                    )
                    for label, key in RESULT_STATUS_COLUMNS.items()
                },
                "Hits": st.column_config.TextColumn(
                    "Hits ⓘ",
                    help="Relevant documents found in the first K results / all labeled "
                    "relevant documents. "
                    "Example: 2/3 means two of three relevant documents were "
                    "retrieved in the first K positions.",
                ),
            },
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Trace inspector")
        chosen = st.selectbox(
            "Inspect case", [r["case"]["id"] for r in run["results"]], key="inspect_case"
        )
        result = next(r for r in run["results"] if r["case"]["id"] == chosen)
        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            st.markdown("**Input and reference**")
            st.write(result["case"]["question"])
            st.caption(
                f"User: {result['case']['user_id']} · Category: {result['case']['category']}"
            )
            st.write(
                "Expected relevant IDs:",
                ", ".join(result["case"]["relevant_ids"]) or "None labeled",
            )
            st.write("Expected answer:", result["case"]["expected_answer"])
            st.write("Exact Match references:", result["case"].get("exact_match_references", ()))
            if result["case"].get("supporting_evidence"):
                st.write("Supporting evidence:", result["case"]["supporting_evidence"])
            st.write("Graded relevance:", result["case"].get("relevance_grades", {}))
            st.write("Context relevance judgments:", result["case"].get("context_relevance", {}))
            st.write("Reference points:", result["case"].get("reference_points", ()))
            st.write("Answerable label:", result["case"].get("answerable"))
            if trap_context := result["case"].get("trap_context_answer"):
                st.write("Trap context answer:", trap_context)
                st.write("Trap prior answer:", result["case"].get("trap_prior_answer", ""))
            if result["case"]["notes"]:
                st.caption(result["case"]["notes"])
            st.markdown("**Actual answer**")
            st.write((result["response"] or {}).get("answer") or "Unavailable")
            st.write(
                "Citations:", ", ".join((result["response"] or {}).get("citations", [])) or "None"
            )
            if result["error"]:
                st.error(result["error"])
        with c2:
            st.markdown("**Recall@K calculation**")
            calc = result["calculation"]
            if calc:
                if calc["score"] is None:
                    st.warning(calc["reason"])
                else:
                    st.metric(
                        f"Recall@{calc['k']}", f"{calc['score']:.2f}", help=METRIC_HELP["recall"]
                    )
                    st.write(
                        (
                            f"**{calc['numerator']} hits / {calc['denominator']} "
                            f"relevant IDs = {calc['score']:.2f}**"
                        )
                    )
                    st.write("Hits:", ", ".join(calc["hits"]) or "None")
                    st.write("Misses:", ", ".join(calc["misses"]) or "None")
                    st.caption(
                        f"Pass threshold: {run['threshold']:.2f} · Status: {result['status']}"
                    )
            st.markdown("**Exact Match calculation**")
            em = result.get("exact_match_calculation")
            if em is None:
                st.caption("Unavailable in this run.")
            elif em["score"] is None:
                st.warning(em["reason"])
            else:
                st.metric("Exact Match", f"{em['score']:.0f}", help=METRIC_HELP["exact_match"])
                st.write("Normalized answer:", em["normalized_answer"])
                st.write("Normalized references:", em["normalized_references"])
                matched_reference_label = (
                    em["matched_reference_index"]
                    if em["matched_reference_index"] is not None
                    else "none"
                )
                st.caption(
                    (
                        f"Matched reference index: {matched_reference_label} "
                        f"· Normalization v{em['normalization_version']} "
                        f"· Status: {result.get('exact_match_status', 'unknown')}"
                    )
                )
                if em["score"] == 0:
                    st.info(
                        "A mismatch is a string comparison, not proof that the answer is "
                        "false or unsupported."
                    )
            st.markdown("**Token F1 calculation**")
            f1 = result.get("token_f1_calculation")
            if f1 is None:
                st.caption("Unavailable in this run.")
            elif f1["score"] is None:
                st.warning(f1["reason"])
            else:
                st.metric("Token F1", f"{f1['score']:.3f}", help=METRIC_HELP["token_f1"])
                st.write("Answer tokens:", f1["answer_tokens"])
                st.write("Reference tokens:", f1["reference_tokens"])
                st.caption(
                    (
                        f"Best reference index: {f1['matched_reference_index']} "
                        f"· Overlap: {f1['overlap']} "
                        f"· Precision: {f1['precision']:.3f} · Recall: {f1['recall']:.3f} "
                        f"· Status: {result.get('token_f1_status', 'unknown')}"
                    )
                )
            st.markdown("**Precision@K calculation**")
            precision = result.get("precision_calculation")
            if precision is None:
                st.caption("Unavailable in this run.")
            elif precision["score"] is None:
                st.warning(precision["reason"])
            else:
                st.metric(
                    f"Precision@{precision['k']}",
                    f"{precision['score']:.2f}",
                    help=METRIC_HELP["precision"],
                )
                st.write(
                    (
                        f"**{precision['numerator']} relevant hits / {precision['denominator']} "
                        f"ranks = {precision['score']:.2f}**"
                    )
                )
                st.write(
                    "Relevant ranks:", ", ".join(map(str, precision["relevant_ranks"])) or "None"
                )
                st.write(
                    "Duplicate ranks:", ", ".join(map(str, precision["duplicate_ranks"])) or "None"
                )
                st.caption(
                    (
                        f"Pass threshold: {run['threshold']:.2f} · Precision status: "
                        f"{result.get('precision_status', 'unknown')}"
                    )
                )
            st.markdown("**Reciprocal rank calculation**")
            rr = result.get("reciprocal_rank_calculation")
            if rr is None:
                st.caption("Unavailable in this run.")
            elif rr["score"] is None:
                st.warning(rr["reason"])
            else:
                st.metric(
                    "Reciprocal rank", f"{rr['score']:.3f}", help=METRIC_HELP["reciprocal_rank"]
                )
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
                st.write(
                    "Discounted gains:", [round(value, 4) for value in ndcg["discounted_gains"]]
                )
                st.caption(
                    (
                        f"Duplicate ranks: {ndcg['duplicate_ranks'] or 'None'} "
                        f"· Status: {result.get('ndcg_status', 'unknown')}"
                    )
                )
            for title, key, status_key, help_key in (
                (
                    "Average precision",
                    "average_precision_calculation",
                    "average_precision_status",
                    "average_precision",
                ),
                (
                    "Context precision",
                    "context_precision_calculation",
                    "context_precision_status",
                    "context_precision",
                ),
                (
                    "Context recall",
                    "context_recall_calculation",
                    "context_recall_status",
                    "context_recall",
                ),
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
                    with st.expander(
                        f"#{rank} · {hit['id']} · {hit['title']}", expanded=rank <= min(int(k), 2)
                    ):
                        st.write(hit["text"] or "No text in trace")
                        if hit.get("source_url"):
                            st.caption(hit["source_url"])
            else:
                st.caption("No ranked retrieval available.")
        with st.expander("Complete case trace and provenance"):
            st.json(
                {
                    "run_id": run["id"],
                    "dataset_version": run["dataset_version"],
                    "run_config": run["config"],
                    "k": run["k"],
                    "threshold": run["threshold"],
                    "case_result": result,
                }
            )
        st.markdown("**Generation judge trace**")
        judgments = result.get("generation_judgments", {})
        if judgments:
            st.caption(
                "Judge scores are provisional. Compare item verdicts and evidence "
                "with human labels before using them for decisions."
            )
            st.dataframe(
                [
                    {
                        "Metric": name.replace("_", " "),
                        "Score": detail.get("score"),
                        "Status": detail.get("status"),
                        "Review": detail.get("review_status", "not reviewed"),
                    }
                    for name, detail in judgments.items()
                ],
                hide_index=True,
                width="stretch",
            )
            for name, detail in judgments.items():
                with st.expander(f"{name.replace('_', ' ').title()} · {detail.get('status')}"):
                    st.markdown(RUBRICS[name])
                    st.json(detail)
            reviewable = [
                name for name, detail in judgments.items() if detail.get("status") == "scored"
            ]
            if reviewable:
                with st.form(f"review-{run['id']}-{chosen}"):
                    st.markdown("**Human review gate**")
                    review_metric = st.selectbox("Metric to review", reviewable)
                    reviewer = st.text_input("Reviewer ID")
                    decision = st.selectbox("Decision", ["accept", "reject"])
                    review_note = st.text_area("Evidence and reason for decision")
                    submitted = st.form_submit_button("Record review")
                if submitted:
                    try:
                        reviewed = review_judgment(
                            judgments[review_metric],
                            reviewer=reviewer,
                            decision=decision,
                            note=review_note,
                        )
                        judgments[review_metric] = reviewed
                        st.session_state.current_run = run
                        store.save(
                            persisted_copy(run, include_content=run["mode"] == "Offline fixtures")
                        )
                        st.success("Human review recorded for this run.")
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.caption(
                "No judge ran for this case. Configure an approved judge gateway "
                "or use synthetic reviewed labels."
            )
        trace = result.get("trace")
        if trace:
            st.markdown("**Execution events**")
            st.dataframe(trace["events"], hide_index=True, width="stretch")
            st.caption(
                (
                    f"Ranked retrieval: {trace['ranked_retrieval_status']} "
                    f"· Generation context: {trace['generation_context_status']} "
                    f"· Started: {trace['started_at'] or 'unavailable'} "
                    f"· Finished: {trace['finished_at'] or 'unavailable'}"
                )
            )
        st.download_button(
            "Download full run JSON",
            json.dumps(original_run, indent=2),
            file_name=f"{run['id']}.json",
            mime="application/json",
        )
        st.caption(
            "A synthetic run checks evaluator behavior against fixtures. It "
            "does not establish real bot quality or label validity."
        )

with tabs[1]:
    render_case_editor()

with tabs[3]:
    render_history(store)

with tabs[4]:
    render_comparison(store)

with tabs[5]:
    render_interpretation()

with tabs[6]:
    render_guide()

render_quiz(quiz_enabled, tabs)
