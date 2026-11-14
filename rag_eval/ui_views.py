"""Streamlit views for case editing, history, comparison, guidance, and quiz."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, TypedDict, cast

import streamlit as st
from pydantic import StrictStr, TypeAdapter, ValidationError

from .core import (
    RagResponse,
    RetrievedItem,
    TestCase,
    average_precision,
    context_precision,
    context_recall,
    exact_match,
    ndcg_at_k,
    normalize_exact_match,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    with_recomputed_mrr,
)
from .datasets import export_case_pack, import_case_pack
from .dto import CaseInput, validation_issue
from .fixtures import DOCUMENTS
from .quiz import QuizQuestion, load_quiz
from .store import RunStore
from .records import Result


def parse_point_support(raw_value: str) -> dict[str, tuple[str, ...]]:
    """Validate the editor's JSON mapping from reference points to context IDs."""
    try:
        return TypeAdapter(dict[StrictStr, tuple[StrictStr, ...]]).validate_python(
            json.loads(raw_value)
        )
    except ValidationError as exc:
        raise ValueError(
            f"Point support must map points to lists of ID strings: {validation_issue(exc)}"
        ) from exc


def render_case_editor() -> None:
    """Render the case editor view."""
    st.subheader("Test cases and fixture responses")
    st.caption(
        "Edit a case and its offline response, then run it from the "
        "sidebar. Changes live in this dashboard session and are "
        "versioned by dataset hash in each run."
    )
    _share_case_pack()
    _edit_existing_case()
    _add_new_case()
    with st.expander("Synthetic document catalog"):
        st.dataframe(
            [{"ID": i, **d} for i, d in DOCUMENTS.items()], hide_index=True, width="stretch"
        )


def _share_case_pack() -> None:
    """Render the share case pack controls."""
    with st.expander("Share or restore a synthetic case pack"):
        st.caption(
            "Export cases and predefined responses as JSON. Import validates "
            "references against the synthetic corpus. Connection tokens are "
            "never included."
        )
        pack = export_case_pack(st.session_state.cases, st.session_state.responses)
        st.download_button(
            "Download case pack",
            json.dumps(pack, indent=2),
            file_name="synthetic-case-pack.json",
            mime="application/json",
        )
        uploaded = st.file_uploader("Import case pack JSON", type="json")
        if st.button("Import case pack", disabled=uploaded is None) and uploaded is not None:
            try:
                imported_cases, imported_responses = import_case_pack(
                    cast(dict[str, object], json.load(uploaded)), set(DOCUMENTS)
                )
                st.session_state.cases = imported_cases
                st.session_state.responses = imported_responses
                st.success(
                    (
                        f"Imported {len(imported_cases)} cases. "
                        "The sidebar selection has been refreshed."
                    )
                )
                st.rerun()
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))


def _edit_existing_case() -> None:
    """Edit the selected case and its offline response."""
    case_id = st.selectbox(
        "Choose a case to edit", [c.id for c in st.session_state.cases], key="edit_case"
    )
    case = next(c for c in st.session_state.cases if c.id == case_id)
    response = st.session_state.responses.get(case_id, RagResponse("", ()))
    fields: dict[str, Any] = {}

    with st.form(f"edit-{case_id}"):
        columns = st.columns(2, gap="large")
        with columns[0]:
            fields["name"] = st.text_input("Case name", case.name)
            fields["question"] = st.text_area("Question", case.question)
            fields["relevant"] = st.text_input(
                "Relevant IDs, comma separated", ", ".join(case.relevant_ids)
            )
            fields["grade_text"] = st.text_input(
                "Graded relevance (JSON ID to nonnegative integer)",
                json.dumps(case.relevance_grades),
            )
            fields["context_text"] = st.text_area(
                "Context relevance judgments (JSON ID to true/false)",
                json.dumps(case.context_relevance),
            )
            fields["points_text"] = st.text_area(
                "Reference points (one per line)", "\n".join(case.reference_points)
            )
            fields["support_text"] = st.text_area(
                "Point support (JSON point to context ID list)", json.dumps(case.point_support)
            )
            fields["expected"] = st.text_area("Expected answer / behavior", case.expected_answer)
            fields["exact_references"] = st.text_area(
                "Exact Match gold answers (one per line)", "\n".join(case.exact_match_references)
            )
            fields["answerability"] = st.selectbox(
                "Answerability label",
                ["Unlabeled", "Answerable", "Unanswerable"],
                index=0 if case.answerable is None else 1 if case.answerable else 2,
            )
            fields["trap_prior"] = st.text_input("Trap prior answer", case.trap_prior_answer)
            fields["trap_context"] = st.text_input("Trap context answer", case.trap_context_answer)
            fields["category"] = st.text_input("Category", case.category)
            fields["user_id"] = st.text_input("User context ID", case.user_id)
            fields["notes"] = st.text_area("Case notes", case.notes)
            fields["supporting_evidence"] = st.text_area(
                "Supporting evidence", case.supporting_evidence
            )
        with columns[1]:
            fields["actual_answer"] = st.text_area("Offline actual answer", response.answer)
            fields["retrieved_ids"] = st.text_area(
                "Ranked retrieved IDs, comma separated",
                ", ".join(r.id for r in response.retrieved),
                help="Order matters. Duplicate IDs are allowed to exercise duplicate behavior.",
            )
            fields["citations"] = st.text_input(
                "Offline citations, comma separated", ", ".join(response.citations)
            )
            st.caption("Known synthetic IDs: " + ", ".join(DOCUMENTS))
        save_case = st.form_submit_button("Save case and fixture", type="primary")

    if save_case:
        ids = tuple(x.strip() for x in fields["relevant"].split(",") if x.strip())
        ranked = tuple(x.strip() for x in fields["retrieved_ids"].split(",") if x.strip())
        try:
            grades = json.loads(fields["grade_text"])
            judgments = json.loads(fields["context_text"])
            support = parse_point_support(fields["support_text"])
            points = tuple(
                line.strip() for line in fields["points_text"].splitlines() if line.strip()
            )
            updated = TestCase(
                case.id,
                (fields["name"] or ""),
                (fields["question"] or ""),
                ids,
                (fields["expected"] or ""),
                (fields["category"] or ""),
                (fields["user_id"] or ""),
                (fields["notes"] or ""),
                (fields["supporting_evidence"] or ""),
                grades,
                judgments,
                points,
                support,
                tuple(
                    line.strip() for line in fields["exact_references"].splitlines() if line.strip()
                ),
                (
                    None
                    if fields["answerability"] == "Unlabeled"
                    else fields["answerability"] == "Answerable"
                ),
                (fields["trap_prior"] or ""),
                (fields["trap_context"] or ""),
            )
            updated = CaseInput.model_validate(asdict(updated)).to_domain()
            updated.validate()
            if (
                any(i not in DOCUMENTS for i in ids + ranked)
                or any(i not in DOCUMENTS for i in grades)
                or any(i not in DOCUMENTS for i in judgments)
                or any(i not in DOCUMENTS for items in support.values() for i in items)
            ):
                raise ValueError("IDs must refer to the synthetic corpus")
            updated_response = RagResponse(
                (fields["actual_answer"] or ""),
                tuple(
                    RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"]) for i in ranked
                ),
                tuple(x.strip() for x in fields["citations"].split(",") if x.strip()),
            )
            st.session_state.cases = [
                updated if c.id == case.id else c for c in st.session_state.cases
            ]
            st.session_state.responses[case.id] = updated_response
            st.success("Saved in this session. Run the case to see the new result.")
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(str(exc))


def _add_new_case() -> None:
    """Create a new synthetic case from the form."""
    with st.expander("Add a new synthetic case"):
        with st.form("add-case"):
            fields: dict[str, Any] = {}
            fields["new_id"] = st.text_input("New case ID", placeholder="CASE-005")
            fields["new_name"] = st.text_input("Name")
            fields["new_question"] = st.text_area("Question")
            fields["new_relevant"] = st.text_input("Relevant IDs, comma separated")
            fields["new_grades"] = st.text_input(
                "Graded relevance (JSON ID to nonnegative integer)", value="{}"
            )
            fields["new_context"] = st.text_area(
                "Context relevance judgments (JSON ID to true/false)", value="{}"
            )
            fields["new_points"] = st.text_area("Reference points (one per line)")
            fields["new_support"] = st.text_area(
                "Point support (JSON point to context ID list)", value="{}"
            )
            fields["new_expected"] = st.text_input("Expected answer / behavior")
            fields["new_exact_references"] = st.text_area("Exact Match gold answers (one per line)")
            fields["new_answerability"] = st.selectbox(
                "Answerability label for new case", ["Unlabeled", "Answerable", "Unanswerable"]
            )
            fields["new_trap_prior"] = st.text_input("Trap prior answer for new case")
            fields["new_trap_context"] = st.text_input("Trap context answer for new case")
            fields["new_evidence"] = st.text_area("Supporting evidence for new case")
            fields["new_category"] = st.text_input("Category", value="custom")
            fields["new_actual"] = st.text_area("Offline actual answer")
            fields["new_ranked"] = st.text_input("Ranked retrieved IDs, comma separated")
            add_case = st.form_submit_button("Add case")
        if add_case:
            ids = tuple(x.strip() for x in fields["new_relevant"].split(",") if x.strip())
            ranked = tuple(x.strip() for x in fields["new_ranked"].split(",") if x.strip())
            try:
                grades = json.loads(fields["new_grades"])
                judgments = json.loads(fields["new_context"])
                support = parse_point_support(fields["new_support"])
                points = tuple(
                    line.strip() for line in fields["new_points"].splitlines() if line.strip()
                )
                if fields["new_id"] in [c.id for c in st.session_state.cases]:
                    raise ValueError("Case ID already exists")
                new_case = TestCase(
                    fields["new_id"],
                    fields["new_name"],
                    fields["new_question"],
                    ids,
                    fields["new_expected"],
                    fields["new_category"],
                    supporting_evidence=fields["new_evidence"],
                    relevance_grades=grades,
                    context_relevance=judgments,
                    reference_points=points,
                    point_support=support,
                    exact_match_references=tuple(
                        line.strip()
                        for line in fields["new_exact_references"].splitlines()
                        if line.strip()
                    ),
                    answerable=(
                        None
                        if fields["new_answerability"] == "Unlabeled"
                        else fields["new_answerability"] == "Answerable"
                    ),
                    trap_prior_answer=fields["new_trap_prior"],
                    trap_context_answer=fields["new_trap_context"],
                )
                new_case = CaseInput.model_validate(asdict(new_case)).to_domain()
                new_case.validate()
                if (
                    any(i not in DOCUMENTS for i in ids + ranked)
                    or any(i not in DOCUMENTS for i in grades)
                    or any(i not in DOCUMENTS for i in judgments)
                    or any(i not in DOCUMENTS for items in support.values() for i in items)
                ):
                    raise ValueError("IDs must refer to the synthetic corpus")
                st.session_state.cases.append(new_case)
                st.session_state.responses[fields["new_id"]] = RagResponse(
                    fields["new_actual"],
                    tuple(
                        RetrievedItem(i, DOCUMENTS[i]["title"], DOCUMENTS[i]["text"])
                        for i in ranked
                    ),
                )
                st.success("Case added. Select it in the sidebar to run it.")
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))


def render_history(store: RunStore) -> None:
    """Render the history view."""
    st.subheader("Run history")
    history = store.list()
    if not history:
        st.info("No saved runs yet.")
    else:
        st.dataframe(
            [
                {
                    "Run": r["id"],
                    "When (UTC)": r["created_at"],
                    "Source": r["mode"],
                    "Dataset": r["dataset_version"],
                    "Cases": len(r["results"]),
                    **r["summary"],
                }
                for r in history
            ],
            hide_index=True,
            width="stretch",
        )
        history_id = st.selectbox("Open saved run", [r["id"] for r in history])
        saved = next(r for r in history if r["id"] == history_id)
        if saved.get("content_redacted"):
            st.info(
                "Sensitive content was redacted from this saved live run. The "
                "current session may still hold the full trace."
            )
        if st.button("Show this run in Overview"):
            st.session_state.current_run = saved
            st.info("Open the Overview tab to inspect the selected run.")
        st.download_button(
            "Download saved run",
            json.dumps(saved, indent=2),
            file_name=f"{history_id}.json",
            mime="application/json",
        )


def _metric_score(result: Result | None, field: str) -> float | None:
    """Read a metric score from one run result when it was calculated."""
    if result is None:
        return None
    calculation = result.get(field)
    return calculation["score"] if calculation is not None else None


def _comparison_row(
    case_id: str, before: Result | None, after: Result | None
) -> dict[str, str | float | None]:
    """Compare a case across two saved runs."""
    row: dict[str, str | float | None] = {
        "Case": case_id,
        "Baseline status": before["status"] if before else "not run",
        "Candidate status": after["status"] if after else "not run",
    }
    for label, field, change_label in (
        ("recall", "calculation", "Score change"),
        ("precision", "precision_calculation", "Precision change"),
        ("RR", "reciprocal_rank_calculation", "RR change"),
        ("NDCG", "ndcg_calculation", "NDCG change"),
        ("AP", "average_precision_calculation", "AP change"),
        ("Context precision", "context_precision_calculation", "Context precision change"),
        ("Context recall", "context_recall_calculation", "Context recall change"),
    ):
        before_score = _metric_score(before, field)
        after_score = _metric_score(after, field)
        row[f"Baseline {label}"] = before_score
        row[f"Candidate {label}"] = after_score
        row[change_label] = (
            round(after_score - before_score, 4)
            if before_score is not None and after_score is not None
            else None
        )
    return row


def render_comparison(store: RunStore) -> None:
    """Render the comparison view."""
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
            st.warning(
                "Dataset versions differ. Score changes may reflect edited cases "
                "or labels rather than bot behavior."
            )
        if baseline["k"] != candidate["k"]:
            st.warning("K differs between runs. Retrieval scores are not directly comparable.")
        base_results = {r["case"]["id"]: r for r in baseline["results"]}
        candidate_results = {r["case"]["id"]: r for r in candidate["results"]}
        rows = [
            _comparison_row(case_id, base_results.get(case_id), candidate_results.get(case_id))
            for case_id in sorted(set(base_results) | set(candidate_results))
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "A status change can also result from a threshold change. Inspect "
            "both run configurations and case traces before drawing "
            "conclusions."
        )


def render_interpretation() -> None:
    """Render the interpretation view."""
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

**Current limits:** The nine generation checks run on synthetic labels or a configured approved judge gateway. No real bot or approved judge provider has been supplied. Judge scores remain provisional pending human review, and retrieved chunks are not confirmed to be the exact generation context. Confluence connection testing does not import pages. Live bot responses use the fixture contract until a real API contract is available.
""")


def render_guide() -> None:
    """Render the guide view."""
    st.subheader("Metric explanations")
    st.write(
        "Start with what the system retrieved, then ask whether that "
        "evidence can answer the question, and finally check the answer "
        "itself. A score from 0 to 1 is meaningful only with the right "
        "labels and a representative test set."
    )

    st.markdown("#### Which metrics belong together?")
    st.dataframe(
        [
            {
                "Layer": "1 · Retrieval coverage",
                "Question": "Did we find enough of the known relevant documents?",
                "Choose from": "Recall@K",
                "Current coverage": "Implemented",
            },
            {
                "Layer": "2 · Retrieval ranking",
                "Question": "Are useful results early and noise low?",
                "Choose from": "Precision@K, reciprocal rank / MRR, NDCG@K, AP / MAP",
                "Current coverage": "Implemented",
            },
            {
                "Layer": "3 · Answerable evidence",
                "Question": "Are retrieved chunks useful and do they support the needed facts?",
                "Choose from": "Context precision, context recall",
                "Current coverage": "Implemented with human labels",
            },
            {
                "Layer": "4 · Generated answer",
                "Question": "Is the answer correct, grounded, and complete?",
                "Choose from": "Exact Match for short factual answers; broader checks needed",
                "Current coverage": "Partial",
            },
            {
                "Layer": "5 · Safety and operation",
                "Question": "Were permissions respected, and is the system reliable and timely?",
                "Choose from": "Permission, citation, latency, failure checks",
                "Current coverage": "Not implemented as metrics",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.info(
        "You do **not** need every metric in every evaluation. Pick "
        "measures for the failure modes that matter, and keep the same "
        "definitions across runs. One metric per layer is a useful "
        "planning prompt, not a certificate that evaluation is good: the "
        "answer and safety layers still have gaps here. A production "
        "decision also needs representative cases, trusted labels, and "
        "review of failures."
    )
    st.markdown(
        "**A practical starter set:** Recall@K + one ranking measure "
        "(usually Precision@K or MRR) + context recall + an answer check "
        "suited to the task. Add graded NDCG when some documents are more "
        "useful than others; use Exact Match only for short, stable "
        "answers. Add permission and latency checks before claiming "
        "production readiness."
    )

    st.markdown("#### Learn them in order")
    st.caption(
        "K is the number of top retrieved positions you inspect. AP and "
        "reciprocal rank use the whole returned list. MRR and MAP are "
        "averages over cases, not additional per-case signals."
    )
    _render_lessons()
    _render_playground()


def render_quiz(quiz_enabled: bool, tabs: Sequence[Any]) -> None:
    """Render the optional understanding quiz."""
    if quiz_enabled:
        with tabs[7]:
            st.subheader("Check understanding")
            st.caption(
                "Test what has been built so far. Your answers and result stay in "
                "this dashboard session."
            )
            quiz_folder = Path(__file__).resolve().parent.parent / "study_quiz"
            try:
                questions = load_quiz(quiz_folder)
            except (OSError, ValueError) as exc:
                st.info(
                    f"Quiz unavailable: {exc}. Add QUESTIONS.md and ANSWER_KEY.md to study_quiz/."
                )
            else:
                if st.session_state.get("quiz_result") is None:
                    _quiz_form(questions)
                else:
                    _quiz_results(questions)


def _quiz_form(questions: list[QuizQuestion]) -> None:
    """Collect and validate submitted quiz answers."""
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
                format_func=lambda choice, options=question.options: (
                    f"{choice}. {options[choice]}"
                ),
                key=f"quiz_answer_{question.number}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        answers = {q.number: st.session_state.get(f"quiz_answer_{q.number}") for q in questions}
        unanswered = [number for number, answer in answers.items() if answer is None]
        if unanswered:
            st.warning(
                (
                    f"Answer all {len(questions)} questions before submitting. "
                    f"Unanswered: {', '.join(map(str, unanswered))}."
                )
            )
        else:
            st.session_state.quiz_result = answers
            st.rerun()


def _quiz_results(questions: list[QuizQuestion]) -> None:
    """Show quiz feedback and allow a retake."""
    answers = st.session_state.quiz_result
    correct = sum(answers.get(q.number) == q.answer for q in questions)
    st.metric(
        "Your score",
        f"{correct}/{len(questions)}",
        f"{correct / len(questions):.0%}",
    )
    st.progress(correct / len(questions))
    if correct == len(questions):
        st.success("All answers correct.")
    else:
        st.info(
            (
                f"{len(questions) - correct} question(s) to review. "
                "Your answers and the correct answers are below."
            )
        )
    for question in questions:
        chosen = answers.get(question.number)
        is_correct = chosen == question.answer
        with st.expander(
            f"{'✅' if is_correct else '❌'} {question.number}. {question.prompt}",
            expanded=not is_correct,
        ):
            st.write(
                (f"Your answer: **{chosen}. " f"{question.options.get(chosen, 'No answer')}**")
            )
            st.write(
                (f"Correct answer: **{question.answer}. " f"{question.options[question.answer]}**")
            )
            st.write(question.explanation)
    if st.button("Retake quiz"):
        st.session_state.quiz_result = None
        for question in questions:
            st.session_state.pop(f"quiz_answer_{question.number}", None)
        st.rerun()


class PlaygroundState(TypedDict):
    """Inputs selected in the metric playground."""

    ranked: list[str]
    relevant: list[str]
    k_play: int
    grades: dict[str, int]
    judgments: dict[str, bool]
    fact_one: list[str]
    fact_two: list[str]
    toy_answer: str
    gold_answer: str


def _render_lessons() -> None:
    """Explain metrics in the recommended learning order."""
    lessons = [
        (
            "1 · Recall@K · coverage",
            "Of all relevant documents we knew about, what fraction appeared in the top K?",
            (
                "With relevant A and B, [A, X] at K=2 gives 1/2. It catches "
                "missing evidence. It cannot tell whether the top results are "
                "cluttered."
            ),
        ),
        (
            "2 · Precision@K · top-result cleanliness",
            "What fraction of the K positions contained distinct relevant documents?",
            (
                "For [A, X] at K=2, it is 1/2. Compared with recall, it reveals "
                "wasted positions. It cannot tell whether the useful item was "
                "first or last."
            ),
        ),
        (
            "3 · Reciprocal rank / MRR · first useful hit",
            (
                "How soon did the first relevant result appear? Score = 1 / its "
                "rank; MRR averages this over cases."
            ),
            (
                "A first hit at rank 3 gives 1/3. Compared with Precision@K, it "
                "highlights the user's wait for the first useful result. It "
                "ignores later relevant results."
            ),
        ),
        (
            "4 · NDCG@K · graded ordering",
            (
                "Do highly useful results appear before merely relevant ones? "
                "Compare discounted gains with an ideal order."
            ),
            (
                "A grade-2 result near the top earns more than a grade-1 result. "
                "Compared with MRR, it values several results and their degree of "
                "usefulness. It needs graded labels."
            ),
        ),
        (
            "5 · Average precision / MAP · whole-list ranking",
            (
                "Average precision at each distinct relevant hit, divided by all "
                "labeled relevant documents; MAP averages this over cases."
            ),
            (
                "With relevant A and B and results [A, X, B], AP = (1 + 2/3) / 2 "
                "= 0.83. Compared with NDCG@K, it covers the full list using "
                "binary relevance; it cannot express relevance grades."
            ),
        ),
        (
            "6 · Context precision · useful chunks",
            (
                "Are the actual retrieved context chunks useful for answering "
                "this question? Reward useful chunks at earlier ranks."
            ),
            (
                "Compared with document-ID precision, it uses separate human "
                "judgments about chunk usefulness. It needs a judgment for every "
                "retrieved chunk; it still does not test the generated answer."
            ),
        ),
        (
            "7 · Context recall · answer-point support",
            "What fraction of required answer points have support in at least one retrieved chunk?",
            (
                "Compared with context precision, it catches a missing fact even "
                "if the returned chunks look useful. It needs reference points "
                "and explicit support labels. It checks retrieved context, not "
                "necessarily the context sent to the model."
            ),
        ),
        (
            "8 · Exact Match · short-answer equality",
            (
                "After case, punctuation, and English-article normalization, does "
                "the answer equal one of the labeled gold strings?"
            ),
            (
                "This adds a generated-answer check. It is useful for IDs, names, "
                "and short facts; a correct paraphrase or longer answer can still "
                "score zero. It does not prove grounding or completeness."
            ),
        ),
    ]
    for title, question, significance in lessons:
        with st.expander(title):
            st.write(question)
            st.write(significance)


def _playground_inputs(toy_ids: list[str]) -> PlaygroundState:
    """Collect retrieval and answer labels from the playground controls."""
    relevant = st.multiselect(
        "Known relevant document IDs", toy_ids, default=["A", "B"], key="play_relevant"
    )
    k_play = st.slider("Top K", 1, 5, 3, key="play_k")
    result_count = st.slider("Number of returned results", 0, 5, 3, key="play_result_count")
    ranked = [
        st.selectbox(
            f"Rank {rank}",
            toy_ids,
            index=toy_ids.index(["A", "X", "B", "C", "X"][rank - 1]),
            key=f"play_rank_{rank}",
        )
        for rank in range(1, result_count + 1)
    ]
    st.caption(
        "Duplicate IDs keep their rank but earn no second hit. If fewer "
        "than K results return, empty positions lower Precision@K."
    )
    with st.expander("Advanced labels and answer", expanded=False):
        grades = {
            item: st.slider(
                f"Relevance grade for {item}",
                0,
                3,
                2 if item == "A" else 1 if item == "B" else 0,
                key=f"play_grade_{item}",
            )
            for item in toy_ids
        }
        judgments = {
            item: st.checkbox(
                f"Chunk {item} is useful", value=item in ("A", "B"), key=f"play_useful_{item}"
            )
            for item in toy_ids
        }
        fact_one = st.multiselect(
            "Chunks supporting fact 1", toy_ids, default=["A"], key="play_fact_1"
        )
        fact_two = st.multiselect(
            "Chunks supporting fact 2", toy_ids, default=["B"], key="play_fact_2"
        )
        toy_answer = st.text_input("Generated short answer", value="Alpha", key="play_answer")
        gold_answer = st.text_input("Gold short answer", value="Alpha", key="play_gold")
    return {
        "ranked": ranked,
        "relevant": relevant,
        "k_play": k_play,
        "grades": grades,
        "judgments": judgments,
        "fact_one": fact_one,
        "fact_two": fact_two,
        "toy_answer": toy_answer,
        "gold_answer": gold_answer,
    }


def _playground_results(state: PlaygroundState) -> None:
    """Calculate and display scores for the selected playground inputs."""
    recall = recall_at_k(state["ranked"], state["relevant"], state["k_play"])
    precision = precision_at_k(state["ranked"], state["relevant"], state["k_play"])
    rr = reciprocal_rank(state["ranked"], state["relevant"])
    ndcg = ndcg_at_k(state["ranked"], state["grades"], state["k_play"])
    ap = average_precision(state["ranked"], state["relevant"])
    cp = context_precision(state["ranked"], state["judgments"])
    cr = context_recall(
        state["ranked"],
        ["Fact 1", "Fact 2"],
        {"Fact 1": tuple(state["fact_one"]), "Fact 2": tuple(state["fact_two"])},
    )
    em = (
        exact_match(state["toy_answer"], (state["gold_answer"],))
        if normalize_exact_match(state["gold_answer"])
        else None
    )
    calculations = [
        ("Recall@K", recall),
        ("Precision@K", precision),
        ("Reciprocal rank", rr),
        ("NDCG@K", ndcg),
        ("Average precision", ap),
        ("Context precision", cp),
        ("Context recall", cr),
        ("Exact Match", em),
    ]
    rows = [
        {
            "Metric": name,
            "Score": round(calc["score"], 3) if calc and calc["score"] is not None else None,
            "Status": "Scored" if calc and calc["score"] is not None else "Needs labels",
        }
        for name, calc in calculations
    ]
    chart_rows = [row for row in rows if row["Score"] is not None]
    if chart_rows:
        st.bar_chart(chart_rows, x="Metric", y="Score", horizontal=True)
    else:
        st.info("Add labels to see metric bars.")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Metrics without enough labels appear in the table but are omitted from the chart.")
    st.write(
        f"**Top {state['k_play']}:** {', '.join(state['ranked'][:state['k_play']]) or 'none'} "
        f"· **Relevant:** {', '.join(state['relevant']) or 'none'}"
    )
    if recall["score"] is not None:
        st.write(
            f"Recall: {recall['numerator']} of {recall['denominator']} relevant IDs found. "
            f"Precision: {precision['numerator']} of {state['k_play']} slots earned a distinct hit."
        )
    if rr["score"] is not None:
        st.write(
            f"First relevant rank: {rr['first_relevant_rank'] or 'none'}. "
            "AP rewards all relevant hits across the full list."
        )
    st.write(f"Supported answer points: {len(cr['covered_points'])} of {cr['reference_total']}.")


def _render_playground() -> None:
    """Render the interactive metric playground."""
    st.markdown("#### Visual playground")
    st.caption(
        "A toy case only; changing these controls does not edit test "
        "cases or saved runs. Every number below calls the same scoring "
        "functions used by the evaluator."
    )
    left, right = st.columns([1, 1.2], gap="large")
    toy_ids = ["A", "B", "C", "X"]
    with left:
        state = _playground_inputs(toy_ids)
    with right:
        _playground_results(state)
