"""Streamlit report for OpenLearnLM benchmark results."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_1T_BASELINE_API
from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_1T_INNOSPARK_API
from report_gen.data_loader import report_data
from report_gen.data_loader import report_data_from_praxis_reading1

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


RESPONSES_SUBDIRECTORY_LOCAL = "responses"
RESPONSES_SUBDIRECTORY_PROD = "response_prod"
RESPONSES_SUBDIRECTORY_PRAXIS = "praxis_reading_1"

# Display labels for the "Response set" picker. The underlying value stays the
# subdirectory key; only the rendered text changes (praxis shows as US proprietary).
RESPONSES_SUBDIRECTORY_DISPLAY = {
    RESPONSES_SUBDIRECTORY_PRAXIS: "US proprietary",
}


def display_response_set(subdirectory: str) -> str:
    return RESPONSES_SUBDIRECTORY_DISPLAY.get(subdirectory, subdirectory)

praxis_reading_directory = f'{os.environ["HOME"]}/klee_code/git_repos/parse_evaluation/praxis_reading_1'

# from report_gen import report_data as report_data
# from report_gen import report_data_from_praxis_reading1 as report_data_from_praxis_reading1
# except ModuleNotFoundError as exc:
#     if exc.name != "report_gen":
#         raise
#     import report_data as data

# Active data source module; swapped by configure_data_source based on the
# selected response set. Defaults to the benchmark source.
# data = report_data

BASELINE_MODEL = report_data.BASELINE_MODEL
DEFAULT_OVERALL_REPORT_FILE = report_data.DEFAULT_OVERALL_REPORT_FILE
DEFAULT_LOCAL_BASELINE_MODEL = report_data.BASELINE_MODEL
DEFAULT_LOCAL_EXPERIMENT_MODEL = report_data.EXPERIMENT_MODEL
EXPERIMENT_MODEL = report_data.EXPERIMENT_MODEL
OVERALL_SCORE_OPTION = report_data.OVERALL_SCORE_OPTION
PROJECT_ROOT = report_data.PROJECT_ROOT
RESPONSES_ROOT = report_data.RESPONSES_ROOT
SUBSET_CHOICES = report_data.SUBSET_CHOICES
category_report_files = report_data.category_report_files
category_response_files = report_data.category_response_files
category_summary_frame = report_data.category_summary_frame
comparison_summary_frame = report_data.comparison_summary_frame
display_category = report_data.display_category
display_model = report_data.display_model
list_response_files = report_data.list_response_files
model_summary_frame = report_data.model_summary_frame
overall_score_summary_frame = report_data.overall_score_summary_frame
pair_response_frames = report_data.pair_response_frames
response_frame = report_data.response_frame
safe_float = report_data.safe_float

ALL_SECTIONS_OPTION = "All"
SECTION_OVERALL_PERFORMANCE = "Overall performance"
SECTION_CATEGORY_SUBSETS = "Category subsets"
SECTION_OVERALL_COMPARISON = "Overall side-by-side comparison"
SECTION_QUESTION_COMPARISON = "Question-level side-by-side comparison"
SECTION_SINGLE_FILE_DETAIL = "Single-file experiment detail"
VISIBLE_SECTION_OPTIONS = [
    ALL_SECTIONS_OPTION,
    SECTION_OVERALL_PERFORMANCE,
    # SECTION_CATEGORY_SUBSETS,
    SECTION_OVERALL_COMPARISON,
    SECTION_QUESTION_COMPARISON,
    SECTION_SINGLE_FILE_DETAIL,
]


def configure_data_source(response_subdirectory: str) -> None:
    global data
    global BASELINE_MODEL
    global DEFAULT_OVERALL_REPORT_FILE
    global EXPERIMENT_MODEL
    global OVERALL_SCORE_OPTION
    global PROJECT_ROOT
    global RESPONSES_ROOT
    global SUBSET_CHOICES
    global category_report_files
    global category_response_files
    global category_summary_frame
    global comparison_summary_frame
    global display_category
    global display_model
    global list_response_files
    global model_summary_frame
    global overall_score_summary_frame
    global pair_response_frames
    global response_frame
    global safe_float

    if response_subdirectory == RESPONSES_SUBDIRECTORY_PRAXIS:
        data = report_data_from_praxis_reading1
        report_data.configure_data_source(
            response_subdirectory,
            baseline_model=report_data.BASELINE_MODEL,
            experiment_model=report_data.EXPERIMENT_MODEL,
        )
    else:
        data = report_data
        baseline_model = DEFAULT_LOCAL_BASELINE_MODEL
        experiment_model = DEFAULT_LOCAL_EXPERIMENT_MODEL
        if response_subdirectory == RESPONSES_SUBDIRECTORY_PROD:
            baseline_model = DEFAULT_1T_BASELINE_API.model
            experiment_model = DEFAULT_1T_INNOSPARK_API.model
        report_data.configure_data_source(
            response_subdirectory,
            baseline_model=baseline_model,
            experiment_model=experiment_model,
        )

    # Re-pull every name from the now-active data module so the rest of the
    # report (functions and constants alike) reflects the selected source.
    BASELINE_MODEL = report_data.BASELINE_MODEL
    DEFAULT_OVERALL_REPORT_FILE = report_data.DEFAULT_OVERALL_REPORT_FILE
    EXPERIMENT_MODEL = report_data.EXPERIMENT_MODEL
    OVERALL_SCORE_OPTION = report_data.OVERALL_SCORE_OPTION
    PROJECT_ROOT = report_data.PROJECT_ROOT
    RESPONSES_ROOT = report_data.RESPONSES_ROOT
    SUBSET_CHOICES = report_data.SUBSET_CHOICES
    category_report_files = report_data.category_report_files
    category_response_files = report_data.category_response_files
    category_summary_frame = report_data.category_summary_frame
    comparison_summary_frame = report_data.comparison_summary_frame
    display_category = report_data.display_category
    display_model = report_data.display_model
    list_response_files = report_data.list_response_files
    model_summary_frame = report_data.model_summary_frame
    overall_score_summary_frame = report_data.overall_score_summary_frame
    pair_response_frames = report_data.pair_response_frames
    response_frame = report_data.response_frame
    safe_float = report_data.safe_float


def section_visible(selected_section: str, section: str) -> bool:
    return selected_section in (ALL_SECTIONS_OPTION, section)


@st.cache_data(show_spinner=False)
def load_report(path: str) -> dict[str, Any]:
    return report_data.load_report(path)


@st.cache_data(show_spinner=False)
def load_responses(path: str) -> list[dict[str, Any]]:
    return report_data.load_responses(path)


def render_metric_row(summary: pd.DataFrame, responses: pd.DataFrame) -> None:
    total_items = int(responses.shape[0])
    avg_score = (
        safe_float(responses["score"].dropna().mean()) if "score" in responses else 0.0
    )
    accuracy = (
        safe_float(responses["is_correct"].fillna(False).mean())
        if "is_correct" in responses and total_items
        else 0.0
    )
    avg_latency = (
        safe_float(responses["latency_ms"].dropna().mean())
        if "latency_ms" in responses
        else 0.0
    )

    cols = st.columns(4)
    cols[0].metric("Evaluated items", f"{total_items:,}")
    cols[1].metric("Average score", f"{avg_score:.2f}")
    cols[2].metric("Accuracy", f"{accuracy * 100:.1f}%")
    cols[3].metric("Avg latency", f"{avg_latency:,.0f} ms")

    if not summary.empty:
        best_row = summary.sort_values(["score", "accuracy"], ascending=False).iloc[0]
        st.caption(
            f"Top model in selected report: {best_row['model']} "
            f"(score {safe_float(best_row['score']):.2f}, "
            f"accuracy {safe_float(best_row['accuracy']) * 100:.1f}%)."
        )


def render_performance(summary: pd.DataFrame) -> None:
    st.subheader("Overall performance")
    if summary.empty:
        st.warning("No model summary data found in the selected report.")
        return

    chart_data = summary[["model", "score", "accuracy"]].set_index("model")
    st.bar_chart(chart_data, height=280)
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "model_id": None,
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.2f"),
            "score": st.column_config.NumberColumn("score", format="%.2f"),
            "avg_latency_ms": st.column_config.NumberColumn(
                "avg_latency_ms", format="%.0f"
            ),
        },
    )


# def render_category_summary(summary: pd.DataFrame) -> None:
#     st.subheader("Category subsets")
#     if summary.empty:
#         st.warning("No category response files are available.")
#         return
#
#     chart_data = summary.set_index("subset")[
#         ["baseline_score", "experiment_score", "score_delta"]
#     ]
#     st.bar_chart(chart_data, height=260)
#     st.dataframe(
#         summary,
#         width="stretch",
#         hide_index=True,
#         column_config={
#             "category": None,
#             "baseline_score": st.column_config.NumberColumn(
#                 "baseline_score", format="%.2f"
#             ),
#             "experiment_score": st.column_config.NumberColumn(
#                 "experiment_score", format="%.2f"
#             ),
#             "score_delta": st.column_config.NumberColumn(
#                 "score_delta", format="%+.2f"
#             ),
#             "baseline_accuracy": st.column_config.NumberColumn(
#                 "baseline_accuracy", format="%.3f"
#             ),
#             "experiment_accuracy": st.column_config.NumberColumn(
#                 "experiment_accuracy", format="%.3f"
#             ),
#             "accuracy_delta": st.column_config.NumberColumn(
#                 "accuracy_delta", format="%+.3f"
#             ),
#             "baseline_latency_ms": st.column_config.NumberColumn(
#                 "baseline_latency_ms", format="%.0f"
#             ),
#             "experiment_latency_ms": st.column_config.NumberColumn(
#                 "experiment_latency_ms", format="%.0f"
#             ),
#             "latency_delta_ms": st.column_config.NumberColumn(
#                 "latency_delta_ms", format="%+.0f"
#             ),
#         },
#     )


def render_overall_comparison(
    baseline: pd.DataFrame | None = None,
    experiment: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    st.subheader("Overall side-by-side comparison")
    if comparison is None:
        comparison = comparison_summary_frame(baseline, experiment)
    if comparison.empty:
        st.warning("No baseline or experiment rows are available for comparison.")
        return comparison

    baseline_row = comparison[comparison["group"] == "baseline"].iloc[0]
    experiment_row = comparison[comparison["group"] == "experiment"].iloc[0]

    cols = st.columns(4)
    cols[0].metric(
        "Score",
        f"{safe_float(experiment_row['score']):.2f}",
        f"{safe_float(experiment_row['score_delta_vs_baseline']):+.2f}",
    )
    cols[1].metric(
        "Accuracy",
        f"{safe_float(experiment_row['accuracy']) * 100:.1f}%",
        f"{safe_float(experiment_row['accuracy_delta_vs_baseline']) * 100:+.1f}%",
    )
    cols[2].metric(
        "Avg latency",
        f"{safe_float(experiment_row['avg_latency_ms']):,.0f} ms",
        f"{safe_float(experiment_row['latency_delta_ms_vs_baseline']):+,.0f} ms",
        delta_color="inverse",
    )
    cols[3].metric(
        "Items",
        f"{int(experiment_row['total']):,}",
        f"{int(experiment_row['total']) - int(baseline_row['total']):+,}",
    )

    chart_data = comparison[["group", "score", "accuracy", "avg_latency_ms"]].set_index(
        "group"
    )
    st.bar_chart(chart_data[["score", "accuracy"]], height=260)
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.2f"),
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.3f"),
            "avg_latency_ms": st.column_config.NumberColumn(
                "avg_latency_ms", format="%.0f"
            ),
            "score_delta_vs_baseline": st.column_config.NumberColumn(
                "score_delta_vs_baseline", format="%+.2f"
            ),
            "accuracy_delta_vs_baseline": st.column_config.NumberColumn(
                "accuracy_delta_vs_baseline", format="%+.3f"
            ),
            "latency_delta_ms_vs_baseline": st.column_config.NumberColumn(
                "latency_delta_ms_vs_baseline", format="%+.0f"
            ),
        },
    )
    return comparison


def render_question_comparison(
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    baseline: pd.DataFrame,
    experiment: pd.DataFrame,
) -> None:
    st.subheader("Question-level side-by-side comparison")
    paired = pair_response_frames(baseline, experiment)
    baseline_ids = set(baseline["item_id"].dropna()) if "item_id" in baseline else set()
    experiment_ids = (
        set(experiment["item_id"].dropna()) if "item_id" in experiment else set()
    )
    common_ids = baseline_ids & experiment_ids
    st.caption(
        f"Common answered items: {len(common_ids):,}. "
        f"Baseline-only items hidden: {len(baseline_ids - experiment_ids):,}. "
        f"Experiment-only items hidden: {len(experiment_ids - baseline_ids):,}."
    )
    if paired.empty:
        st.warning("No paired question rows are available.")
        return

    with st.sidebar:
        st.header("Comparison filters")
        only_changed = st.checkbox("Only changed correctness", value=False)
        search = st.text_input("Search paired question / answer / reasoning")
        score_min = safe_float(
            pd.to_numeric(
                pd.concat([paired["baseline_score"], paired["experiment_score"]]),
                errors="coerce",
            )
            .dropna()
            .min(),
            0.0,
        )
        score_max = safe_float(
            pd.to_numeric(
                pd.concat([paired["baseline_score"], paired["experiment_score"]]),
                errors="coerce",
            )
            .dropna()
            .max(),
            10.0,
        )
        score_range = st.slider(
            "Paired score range",
            min_value=0.0,
            max_value=10.0,
            value=(score_min, score_max),
            step=0.5,
        )

    filtered = paired.copy()
    if only_changed:
        filtered = filtered[filtered["correct_changed"]]

    score_values = pd.to_numeric(filtered["experiment_score"], errors="coerce")
    filtered = filtered[
        score_values.isna()
        | ((score_values >= score_range[0]) & (score_values <= score_range[1]))
    ]

    if search:
        text = search.casefold()
        baseline_answers = baseline.set_index("item_id")["model_answer"]
        experiment_answers = experiment.set_index("item_id")["model_answer"]
        haystack = (
            filtered["question"].fillna("")
            + "\n"
            + filtered["baseline_reasoning"].fillna("")
            + "\n"
            + filtered["experiment_reasoning"].fillna("")
            + "\n"
            + filtered["item_id"].map(baseline_answers).fillna("")
            + "\n"
            + filtered["item_id"].map(experiment_answers).fillna("")
        ).str.casefold()
        filtered = filtered[haystack.str.contains(text, regex=False)]

    display_cols = [
        "item_id",
        "baseline_score",
        "experiment_score",
        "score_delta",
        "baseline_correct",
        "experiment_correct",
        "latency_delta_ms",
        "difficulty",
        "domain",
        "question",
    ]
    st.dataframe(
        filtered[display_cols],
        width="stretch",
        hide_index=True,
        height=320,
        column_config={
            "question": st.column_config.TextColumn("question", width="large"),
            "baseline_score": st.column_config.NumberColumn(
                "baseline_score", format="%.1f"
            ),
            "experiment_score": st.column_config.NumberColumn(
                "experiment_score", format="%.1f"
            ),
            "score_delta": st.column_config.NumberColumn(
                "score_delta", format="%+.1f"
            ),
            "latency_delta_ms": st.column_config.NumberColumn(
                "latency_delta_ms", format="%+.0f"
            ),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(paired):,} paired questions.")

    labels = [
        f"item {row.item_id} | baseline {row.baseline_score} | experiment {row.experiment_score}"
        for row in filtered.itertuples()
    ]
    if not labels:
        st.info("No paired item matches the current filters.")
        return

    selected_label = st.selectbox("Select a paired item", labels)
    selected_item_id = selected_label.split(" | ", 1)[0].replace("item ", "")
    baseline_by_item = {str(row.get("item_id")): row for row in baseline_rows}
    experiment_by_item = {str(row.get("item_id")): row for row in experiment_rows}
    baseline_item = baseline_by_item.get(selected_item_id, {})
    experiment_item = experiment_by_item.get(selected_item_id, {})
    selected_question = filtered[filtered["item_id"].astype(str) == selected_item_id][
        "question"
    ].iloc[0]

    render_record_messages(
        experiment_item.get("messages")
        or baseline_item.get("messages"),
        fallback_question=(
            experiment_item.get("question")
            or baseline_item.get("question")
            or selected_question
        ),
    )

    baseline_col, experiment_col = st.columns(2)
    render_compared_item(baseline_col, "Baseline", baseline_item)
    render_compared_item(experiment_col, "Experiment", experiment_item)


def render_record_messages(
    messages: list[dict[str, Any]] | None,
    *,
    fallback_question: str,
) -> None:
    st.markdown("**Prompt messages**")
    prompt_messages = messages or [{"role": "user", "content": fallback_question}]

    for message in prompt_messages:
        role = str(message.get("role", "message")).upper()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        st.markdown(f"**{role}**")
        if role == "USER":
            st.info(content)
        else:
            st.write(content)

# 在生成report的过程中 哪里是根据和response_prod 和response 决定用哪两个pair的
def render_compared_item(container: Any, title: str, item: dict[str, Any]) -> None:
    check_result = item.get("check_result") or {}
    model_name = display_model(item.get("model", "missing")) or "missing"
    with container:
        st.markdown(f"**{title}: {model_name}**")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Score", check_result.get("score", "n/a"))
        metric_cols[1].metric("Correct", str(item.get("is_correct", "n/a")))
        metric_cols[2].metric("Latency", f"{item.get('latency_ms') or 0:,} ms")
        tabs = st.tabs(["Response", "Scoring", "Raw"])
        with tabs[0]:
            st.markdown(item.get("model_answer") or item.get("raw_content") or "")
        with tabs[1]:
            reasoning = check_result.get("reasoning")
            if reasoning:
                st.markdown("**Reasoning**")
                st.write(reasoning)

            scoring_fields = {
                key: value
                for key, value in check_result.items()
                if key != "reasoning" and value not in (None, "")
            }
            if scoring_fields:
                st.markdown("**Check result**")
                st.json(scoring_fields)
            else:
                st.info("No structured check result is available.")

            with st.expander("Full check_result", expanded=False):
                st.json(check_result)
        with tabs[2]:
            st.write(item.get("raw_content") or "")


def render_response_table(responses: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Per-instance results")
    if responses.empty:
        st.warning("No per-instance responses found in the selected JSONL file.")
        return responses

    with st.sidebar:
        st.header("Filters")
        correctness = st.multiselect(
            "Correctness",
            ["correct", "incorrect", "unknown"],
            default=["correct", "incorrect", "unknown"],
        )
        check_types = sorted(v for v in responses["check_type"].dropna().unique() if v)
        selected_check_types = st.multiselect(
            "Check type", check_types, default=check_types
        )
        search = st.text_input("Search question / answer / reasoning")
        score_min, score_max = 0.0, 10.0
        if responses["score"].notna().any():
            score_min = safe_float(responses["score"].min())
            score_max = safe_float(responses["score"].max())
        score_range = st.slider(
            "Score range",
            min_value=0.0,
            max_value=10.0,
            value=(score_min, score_max),
            step=0.5,
        )

    filtered = responses.copy()
    status_map = {
        True: "correct",
        False: "incorrect",
    }
    filtered["_correctness"] = filtered["is_correct"].map(status_map).fillna("unknown")
    filtered = filtered[filtered["_correctness"].isin(correctness)]

    if selected_check_types:
        filtered = filtered[filtered["check_type"].isin(selected_check_types)]

    score_values = pd.to_numeric(filtered["score"], errors="coerce")
    filtered = filtered[
        score_values.isna()
        | ((score_values >= score_range[0]) & (score_values <= score_range[1]))
    ]

    if search:
        text = search.casefold()
        haystack = (
            filtered["question"].fillna("")
            + "\n"
            + filtered["model_answer"].fillna("")
            + "\n"
            + filtered["reasoning"].fillna("")
        ).str.casefold()
        filtered = filtered[haystack.str.contains(text, regex=False)]

    display_cols = [
        "line_no",
        "item_id",
        "success",
        "is_correct",
        "score",
        "check_type",
        "rubric_source",
        "latency_ms",
        "difficulty",
        "domain",
        "question",
        "reasoning",
    ]
    st.dataframe(
        filtered[display_cols],
        width="stretch",
        hide_index=True,
        height=360,
        column_config={
            "question": st.column_config.TextColumn("question", width="large"),
            "reasoning": st.column_config.TextColumn("reasoning", width="large"),
            "score": st.column_config.NumberColumn("score", format="%.1f"),
            "latency_ms": st.column_config.NumberColumn("latency_ms", format="%.0f"),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(responses):,} response rows.")
    return filtered


def render_detail(rows: list[dict[str, Any]], filtered: pd.DataFrame) -> None:
    st.subheader("Response detail")
    if filtered.empty:
        st.info("No row matches the current filters.")
        return

    labels = [
        f"line {int(row.line_no)} | item {row.item_id} | score {row.score}"
        for row in filtered.itertuples()
    ]
    selected_label = st.selectbox("Select an item", labels)
    selected_line = int(selected_label.split(" | ", 1)[0].replace("line ", ""))
    selected = next(row for row in rows if row.get("line_no") == selected_line)
    check_result = selected.get("check_result") or {}
    metadata = selected.get("metadata") or {}

    meta_cols = st.columns(5)
    meta_cols[0].metric("Item", selected.get("item_id", ""))
    meta_cols[1].metric("Score", check_result.get("score", "n/a"))
    meta_cols[2].metric("Correct", str(selected.get("is_correct", "n/a")))
    meta_cols[3].metric("Latency", f"{selected.get('latency_ms') or 0:,} ms")
    meta_cols[4].metric("Check", check_result.get("check_type", "n/a"))

    st.markdown("**Question**")
    st.write(selected.get("question", ""))

    answer_tabs = st.tabs(["Model response", "Expected answer", "Scoring", "Metadata"])
    with answer_tabs[0]:
        st.markdown(selected.get("model_answer") or selected.get("raw_content") or "")
        with st.expander("Raw content", expanded=False):
            st.write(selected.get("raw_content") or "")
        thinking = selected.get("thinking_content")
        if thinking:
            with st.expander("Thinking content", expanded=False):
                st.write(thinking)
    with answer_tabs[1]:
        st.write(selected.get("expected_answer", ""))
    with answer_tabs[2]:
        st.json(check_result)
    with answer_tabs[3]:
        st.json(metadata)


def main() -> None:
    st.set_page_config(
        page_title="OpenLearnLM Benchmark Report",
        page_icon="",
        layout="wide",
    )
    st.title("OpenLearnLM Benchmark Report")

    with st.sidebar:
        st.header("Data sources")
        response_subdirectory = st.selectbox(
            "Response set",
            [
                RESPONSES_SUBDIRECTORY_LOCAL,
                RESPONSES_SUBDIRECTORY_PROD,
                RESPONSES_SUBDIRECTORY_PRAXIS,
            ],
            format_func=display_response_set,
        )
    configure_data_source(response_subdirectory)

    response_files = list_response_files()
    category_files = category_response_files()
    category_reports = category_report_files()

    with st.sidebar:
        if not response_files:
            st.error(f"No response JSONL files found under {RESPONSES_ROOT}")
            st.stop()
        selected_category = st.selectbox(
            "Category subset",
            SUBSET_CHOICES,
            format_func=display_category,
        )
        selected_report_file = (
            DEFAULT_OVERALL_REPORT_FILE
            if selected_category == OVERALL_SCORE_OPTION
            else category_reports[selected_category]
        )
        if not selected_report_file.exists():
            st.error(f"Report not found: {selected_report_file}")
            st.stop()
        # st.caption(f"Overall report: {selected_report_file.name}")

        st.caption(f"Baseline: {display_model(BASELINE_MODEL)}")
        st.caption(f"Experiment: {display_model(EXPERIMENT_MODEL)}")
        st.header("Display")
        visible_section = st.selectbox(
            "Visible section",
            VISIBLE_SECTION_OPTIONS,
        )

        missing_reports = [
            path for path in category_reports.values() if not path.exists()
        ]
        if missing_reports:
            with st.expander("Missing category report files", expanded=False):
                report_by_path = {
                    path: category for category, path in category_reports.items()
                }
                for path in missing_reports:
                    category = report_by_path.get(path)
                    prefix = f"{display_category(category)}: " if category else ""
                    st.write(f"{prefix}{path.relative_to(PROJECT_ROOT)}")

        missing_files = [
            path
            for files in category_files.values()
            for path in files.values()
            if not path.exists()
        ]
        if missing_files:
            with st.expander("Missing fixed response files", expanded=False):
                category_by_file = {
                    path: category
                    for category, files in category_files.items()
                    for path in files.values()
                }
                for path in missing_files:
                    category = category_by_file.get(path)
                    prefix = f"{display_category(category)}: " if category else ""
                    st.write(f"{prefix}{path.relative_to(PROJECT_ROOT)}")

    report = load_report(str(selected_report_file))
    overall_summary = model_summary_frame(report)

    category_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    category_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for category, files in category_files.items():
        baseline_rows = (
            load_responses(str(files["baseline"])) if files["baseline"].exists() else []
        )
        experiment_rows = (
            load_responses(str(files["experiment"]))
            if files["experiment"].exists()
            else []
        )
        baseline_responses = response_frame(baseline_rows)
        experiment_responses = response_frame(experiment_rows)
        category_rows[category] = (baseline_rows, experiment_rows)
        category_frames[category] = (baseline_responses, experiment_responses)

    category_summary = category_summary_frame(category_frames)
    is_overall_score = selected_category == OVERALL_SCORE_OPTION
    if not is_overall_score:
        baseline_rows, experiment_rows = category_rows[selected_category]
        baseline_responses, experiment_responses = category_frames[selected_category]

    st.caption(
        f"Generated at {report.get('generated_at', 'unknown')} | "
        f"Subset: {display_category(selected_category)} | "
        f"Baseline: {display_model(BASELINE_MODEL)} | "
        f"Experiment: {display_model(EXPERIMENT_MODEL)}"
    )
    if section_visible(visible_section, SECTION_OVERALL_PERFORMANCE):
        render_performance(overall_summary)

    # if section_visible(visible_section, SECTION_CATEGORY_SUBSETS):
    #     render_category_summary(category_summary)

    selected_subset_section_requested = (
        section_visible(visible_section, SECTION_OVERALL_COMPARISON)
        or section_visible(visible_section, SECTION_QUESTION_COMPARISON)
        or section_visible(visible_section, SECTION_SINGLE_FILE_DETAIL)
    )
    if selected_subset_section_requested:
        st.divider()

    if selected_subset_section_requested and selected_category != OVERALL_SCORE_OPTION:
        st.subheader(f"Selected subset: {display_category(selected_category)}")

    if is_overall_score:
        if section_visible(visible_section, SECTION_OVERALL_COMPARISON):
            render_overall_comparison(
                comparison=overall_score_summary_frame(category_summary)
            )
        if visible_section in (
            SECTION_QUESTION_COMPARISON,
            SECTION_SINGLE_FILE_DETAIL,
        ):
            st.info(
                "The score option is category-level only; pick a category subset for item-level views."
            )
        return

    if section_visible(visible_section, SECTION_OVERALL_COMPARISON):
        render_overall_comparison(baseline_responses, experiment_responses)

    if section_visible(visible_section, SECTION_QUESTION_COMPARISON):
        render_question_comparison(
            baseline_rows,
            experiment_rows,
            baseline_responses,
            experiment_responses,
        )

    if section_visible(visible_section, SECTION_SINGLE_FILE_DETAIL):
        st.subheader("Single-file experiment detail")
        render_metric_row(overall_summary, experiment_responses)
        filtered = render_response_table(experiment_responses)
        render_detail(experiment_rows, filtered)


if __name__ == "__main__":
    main()
