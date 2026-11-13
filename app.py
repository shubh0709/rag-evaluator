"""Run with: streamlit run app.py"""
from __future__ import annotations

from dataclasses import asdict
import json

import streamlit as st

from rag_eval.adapters import check_connection, persisted_copy, run_live_case
from rag_eval.core import RagResponse, RetrievedItem, TestCase, make_run
from rag_eval.datasets import export_case_pack, import_case_pack
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES
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
    if "responses" not in st.session_state:
        st.session_state.responses = dict(RESPONSES)
    if "current_run" not in st.session_state:
        st.session_state.current_run = None


init_state()
store = RunStore()

with st.sidebar:
    st.markdown("### ◈ RAG Evaluation Lab")
    st.caption("Confluence evaluation workspace")
    st.divider()
    st.markdown("**Run controls**")
    mode = st.radio("Execution source", ["Offline fixtures", "Live RAG endpoint"],
                    help="Live runs require an accessible bot API and compatible response mapping.")
    k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)
    threshold = st.slider("Recall pass threshold", 0.0, 1.0, 1.0, 0.05,
                          help="This is a local review threshold, not a universal quality target.")
    selected_ids = st.multiselect("Cases to run", [c.id for c in st.session_state.cases],
                                  default=[c.id for c in st.session_state.cases],
                                  format_func=lambda x: next((f"{x} · {c.name}" for c in st.session_state.cases if c.id == x), x))
    st.caption("Every run records its selected cases, K, threshold, and dataset version.")
    run_clicked = st.button("▶ Run selected cases", type="primary", width="stretch", disabled=not selected_ids)
    st.divider()
    st.caption("Metric 1 of the planned sequence · Recall@K v1.0")

st.markdown('<div class="eyebrow">Evaluation workspace</div>', unsafe_allow_html=True)
st.title("See what your RAG system actually did")
st.markdown('<p class="subtle">Run cases, inspect ranked evidence, and trace every score back to its inputs.</p>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Test cases", "Connections", "Run history", "Compare runs", "Methodology"])

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
        a, b = st.columns(2)
        with a:
            question_key = st.text_input("Question request field", value="question", key="question_key")
            answer_path = st.text_input("Answer response path", value="answer", key="answer_path")
            id_path = st.text_input("Retrieved item ID path", value="id", key="id_path")
        with b:
            user_key = st.text_input("User request field", value="user_id", key="user_key")
            retrieval_path = st.text_input("Ranked retrieval path", value="retrieved", key="retrieval_path")
            citation_path = st.text_input("Citations path", value="citations", key="citation_path")
        title_path = st.text_input("Retrieved title path", value="title", key="title_path")
        text_path = st.text_input("Retrieved text path", value="text", key="text_path")
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
        st.markdown("#### Response mapping example")
        st.code('{"answer": "...", "retrieved": [{"id": "DOC-101", "title": "...", "text": "..."}], "citations": ["DOC-101"]}', language="json")
        st.caption("Nested values use dotted paths, such as `data.answer` or `trace.hits`.")

if run_clicked:
    selected = [case for case in st.session_state.cases if case.id in selected_ids]
    responses: dict[str, RagResponse | Exception] = {}
    if mode == "Offline fixtures":
        responses = {case.id: st.session_state.responses.get(case.id, ValueError("No offline response configured"))
                     for case in selected}
    elif not st.session_state.get("bot_url"):
        st.error("Set the RAG chatbot POST endpoint in Connections before running live cases.")
    else:
        with st.spinner("Running selected cases against the configured RAG endpoint…"):
            for case in selected:
                try:
                    responses[case.id] = run_live_case(
                        case, url=st.session_state.bot_url, token=st.session_state.bot_token,
                        header_name=st.session_state.bot_header, prefix=st.session_state.bot_prefix,
                        question_key=st.session_state.question_key, user_key=st.session_state.user_key,
                        answer_path=st.session_state.answer_path, retrieval_path=st.session_state.retrieval_path,
                        id_path=st.session_state.id_path, title_path=st.session_state.title_path,
                        text_path=st.session_state.text_path, citation_path=st.session_state.citation_path,
                        timeout=int(st.session_state.bot_timeout))
                except Exception as exc:
                    responses[case.id] = exc
    if responses:
        safe_config = {"source": mode, "selected_case_ids": selected_ids,
                       "bot_endpoint": st.session_state.get("bot_url", "") if mode == "Live RAG endpoint" else "",
                       "response_mapping": {key: st.session_state.get(key) for key in
                                            ("answer_path", "retrieval_path", "id_path", "title_path", "text_path", "citation_path")}
                       if mode == "Live RAG endpoint" else {}}
        run = make_run(selected, responses, mode=mode, k=int(k), threshold=float(threshold), config=safe_config)
        st.session_state.current_run = run
        store.save(persisted_copy(run, include_content=mode == "Offline fixtures"))
        st.toast(f"Run {run['id']} completed", icon="✅")

with tabs[0]:
    run = st.session_state.current_run
    if run is None:
        history = store.list()
        run = history[0] if history else None
    if run is None:
        st.info("Ready to run. Select cases and press **Run selected cases** in the sidebar. The first run uses the included synthetic fixtures.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic documents", len(DOCUMENTS))
        c2.metric("Available cases", len(st.session_state.cases))
        c3.metric("Metrics implemented", 1)
    else:
        st.subheader("Latest run")
        st.caption(f"{run['id']} · {run['created_at']} · {run['mode']} · dataset {run['dataset_version']}")
        cols = st.columns(4)
        for col, key, label in zip(cols, ("passed", "failed", "not_applicable", "error"),
                                   ("Passed", "Failed", "Not applicable", "Errors")):
            col.metric(label, run["summary"][key])
        st.markdown("#### Case results")
        st.dataframe([{"Case": r["case"]["id"], "Name": r["case"]["name"],
                       "Category": r["case"]["category"], "Status": r["status"],
                       "Recall@K": None if not r["calculation"] else r["calculation"]["score"],
                       "Hits": "" if not r["calculation"] else f"{r['calculation']['numerator']}/{r['calculation']['denominator']}"}
                      for r in run["results"]], width="stretch", hide_index=True)
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
                    st.metric(f"Recall@{calc['k']}", f"{calc['score']:.2f}", help="Unique relevant IDs found in the first K ranked results divided by all labeled relevant IDs.")
                    st.write(f"**{calc['numerator']} hits / {calc['denominator']} relevant IDs = {calc['score']:.2f}**")
                    st.write("Hits:", ", ".join(calc["hits"]) or "None")
                    st.write("Misses:", ", ".join(calc["misses"]) or "None")
                    st.caption(f"Pass threshold: {run['threshold']:.2f} · Status: {result['status']}")
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
        st.download_button("Download full run JSON", json.dumps(run, indent=2), file_name=f"{run['id']}.json", mime="application/json")
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
        if st.button("Import case pack", disabled=uploaded is None):
            try:
                imported_cases, imported_responses = import_case_pack(json.load(uploaded), set(DOCUMENTS))
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
            expected = st.text_area("Expected answer / behavior", case.expected_answer)
            category = st.text_input("Category", case.category)
            user_id = st.text_input("User context ID", case.user_id)
            notes = st.text_area("Case notes", case.notes)
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
            updated = TestCase(case.id, name, question, ids, expected, category, user_id, notes)
            updated.validate()
            if any(i not in DOCUMENTS for i in ids + ranked):
                raise ValueError("IDs must refer to the synthetic corpus")
            updated_response = RagResponse(actual_answer,
                tuple(RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"]) for i in ranked),
                tuple(x.strip() for x in citations.split(",") if x.strip()))
            st.session_state.cases = [updated if c.id == case.id else c for c in st.session_state.cases]
            st.session_state.responses[case.id] = updated_response
            st.success("Saved in this session. Run the case to see the new result.")
        except ValueError as exc:
            st.error(str(exc))
    with st.expander("Add a new synthetic case"):
        with st.form("add-case"):
            new_id = st.text_input("New case ID", placeholder="CASE-005")
            new_name = st.text_input("Name")
            new_question = st.text_area("Question")
            new_relevant = st.text_input("Relevant IDs, comma separated")
            new_expected = st.text_input("Expected answer / behavior")
            new_category = st.text_input("Category", value="custom")
            new_actual = st.text_area("Offline actual answer")
            new_ranked = st.text_input("Ranked retrieved IDs, comma separated")
            add_case = st.form_submit_button("Add case")
        if add_case:
            ids = tuple(x.strip() for x in new_relevant.split(",") if x.strip())
            ranked = tuple(x.strip() for x in new_ranked.split(",") if x.strip())
            try:
                if new_id in [c.id for c in st.session_state.cases]:
                    raise ValueError("Case ID already exists")
                new_case = TestCase(new_id, new_name, new_question, ids, new_expected, new_category)
                new_case.validate()
                if any(i not in DOCUMENTS for i in ids + ranked):
                    raise ValueError("IDs must refer to the synthetic corpus")
                st.session_state.cases.append(new_case)
                st.session_state.responses[new_id] = RagResponse(new_actual, tuple(
                    RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"]) for i in ranked))
                st.success("Case added. Select it in the sidebar to run it.")
            except ValueError as exc:
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
        baseline = next(r for r in history if r["id"] == baseline_id)
        candidate = next(r for r in history if r["id"] == candidate_id)
        if baseline_id == candidate_id:
            st.warning("Choose two different runs to see changes.")
        if baseline["dataset_version"] != candidate["dataset_version"]:
            st.warning("Dataset versions differ. Score changes may reflect edited cases or labels rather than bot behavior.")
        if baseline["k"] != candidate["k"]:
            st.warning("K differs between runs. Recall@K scores are not directly comparable.")
        base_results = {r["case"]["id"]: r for r in baseline["results"]}
        candidate_results = {r["case"]["id"]: r for r in candidate["results"]}
        rows = []
        for case_id in sorted(set(base_results) | set(candidate_results)):
            before = base_results.get(case_id, {})
            after = candidate_results.get(case_id, {})
            before_score = (before.get("calculation") or {}).get("score")
            after_score = (after.get("calculation") or {}).get("score")
            rows.append({"Case": case_id, "Baseline status": before.get("status", "not run"),
                         "Candidate status": after.get("status", "not run"),
                         "Baseline recall": before_score, "Candidate recall": after_score,
                         "Score change": round(after_score - before_score, 4)
                         if before_score is not None and after_score is not None else None})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("A status change can also result from a threshold change. Inspect both run configurations and case traces before drawing conclusions.")

with tabs[5]:
    st.subheader("What these results mean")
    st.markdown("""
**1 · Evaluator correctness.** Unit tests prove that Recall@K uses the agreed formula for known IDs, including duplicates, empty retrieval, and unlabeled cases.

**2 · Bot performance.** A live run can measure the bot only when its API and ranked trace are available. Offline fixture scores are development examples, not evidence of production quality.

**3 · Methodology validity.** Relevant IDs, permission expectations, and future judge outputs require independent human validation. A precise score can still be misleading when labels are wrong.

**Recall@K definition in this version:** unique labeled relevant IDs present in the first K ranked results, divided by all unique labeled relevant IDs. Duplicates occupy ranks but do not increase the numerator. Cases with no labeled relevant IDs are not applicable. The pass threshold is configurable per run.

**Current limits:** Only Recall@K is implemented. Permission and answer quality scenarios are present as fixtures but are not scored by dedicated metrics yet. Confluence connection testing does not import pages. Live bot mapping is generic until a real API contract is available. External LLM judges are not connected.
""")
