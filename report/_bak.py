"""Non-aggregating helpers moved out of get_aggregated_result.py.

Contains:
- Data-source configuration & display helpers
- Path helpers (scored_csv_for_model, category_response_files, …)
- Per-row response loader (load_responses)
- Example usage (main)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from report_gen.data_loader.report_data_openlearnlm import (  # noqa: E402, F401
    category_summary_frame,
    comparison_summary_frame,
    display_model,
    model_summary_frame,
    overall_score_summary_frame,
    pair_response_frames,
    response_frame,
    safe_float,
)
from llm_common.llm_infer.instances import load_system_input           # noqa: E402
from llm_common.llm_infer.instances import response_record_to_prompt_messages  # noqa: E402
from llm_common.report.get_aggregated_result import (
    MODEL_DIR_PREFIX,
    PRAXIS_READING_DIR,
    RESPONSES_ROOT,
    SCORED_CSV_NAME,
    available_models,
    generate_overall_latex_table,
    load_report,
    model_from_path,
)

# --- display / category constants --------------------------------------------

READING_CATEGORY = "reading"
CATEGORY_CHOICES = [READING_CATEGORY]
CATEGORY_DISPLAY_NAMES = {READING_CATEGORY: "US proprietary Reading"}
OVERALL_SCORE_OPTION = "score"
SUBSET_CHOICES = [OVERALL_SCORE_OPTION, *CATEGORY_CHOICES]

PROJECT_ROOT = PRAXIS_READING_DIR
BASELINE_MODEL = "Qwen3-32B"
EXPERIMENT_MODEL = "Qwen3-32B-ceval"
DEFAULT_OVERALL_REPORT_FILE = RESPONSES_ROOT


# --- path helpers ------------------------------------------------------------

def scored_csv_for_model(model: str) -> Path:
    return RESPONSES_ROOT / f"{MODEL_DIR_PREFIX}{model}" / SCORED_CSV_NAME


def configure_data_source(
        *,
        baseline_model: str | None = None,
        experiment_model: str | None = None,
) -> None:
    global BASELINE_MODEL, EXPERIMENT_MODEL, DEFAULT_OVERALL_REPORT_FILE
    if baseline_model is not None:
        BASELINE_MODEL = baseline_model
    if experiment_model is not None:
        EXPERIMENT_MODEL = experiment_model
    DEFAULT_OVERALL_REPORT_FILE = RESPONSES_ROOT


def display_category(category: str | None) -> str:
    if not category:
        return ""
    return CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " "))


def list_response_files() -> list[Path]:
    return [scored_csv_for_model(m) for m in available_models()]


def category_response_files() -> dict[str, dict[str, Path]]:
    return {
        READING_CATEGORY: {
            "baseline"  : scored_csv_for_model(BASELINE_MODEL),
            "experiment": scored_csv_for_model(EXPERIMENT_MODEL),
        }
    }


def category_report_files() -> dict[str, Path]:
    return {READING_CATEGORY: DEFAULT_OVERALL_REPORT_FILE}


# --- per-row response loader -------------------------------------------------

def _is_answered(pred: Any) -> bool:
    if pred is None:
        return False
    try:
        return not pd.isna(pred)
    except (TypeError, ValueError):
        return True


def load_responses(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    model = model_from_path(path)
    system_input = load_system_input(path.parent)
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for line_no, record in enumerate(df.to_dict("records"), start=1):
        pred = record.get("pred")
        gold = record.get("gold")
        answered = _is_answered(pred)
        correct = bool(record.get("correct"))
        llm_response = str(record.get("llm_response", "") or "")
        thinking = ""
        if "</think>" in llm_response:
            thinking = llm_response.split("</think>", 1)[0].replace("<think>", "").strip()
        rows.append(
            {
                "line_no"         : line_no,
                "item_id"         : record.get("id"),
                "model"           : model,
                "success"         : answered,
                "is_correct"      : correct,
                "check_result"    : {
                    "score"        : (1.0 if correct else 0.0) if answered else None,
                    "check_type"   : "letter_match",
                    "rubric_source": "gold_letter",
                    "reasoning"    : f"predicted={pred!r} gold={gold!r}",
                },
                "latency_ms"      : (
                    record.get("latency_ms")
                    if pd.notna(record.get("latency_ms")) else None
                ),
                "usage"           : {
                    k: record.get(k)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if pd.notna(record.get(k))
                },
                "metadata"        : {
                    "question_number": record.get("question_number"),
                    "domain"         : READING_CATEGORY,
                },
                "messages"        : response_record_to_prompt_messages(record, system_input),
                "question"        : record.get("question") or record.get("prompt") or "",
                "model_answer"    : llm_response,
                "raw_content"     : llm_response,
                "thinking_content": thinking,
                "expected_answer" : record.get("answer", ""),
            }
        )
    return rows




def available_models() -> list[str]:
    """Return model names for all scored CSVs found under RESPONSES_ROOT."""
    if not RESPONSES_ROOT.exists():
        return []
    return [
        model_from_path(p)
        for p in sorted(RESPONSES_ROOT.glob(f"{MODEL_DIR_PREFIX}*/{SCORED_CSV_NAME}"))
    ]

def _model_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute accuracy stats from a scored DataFrame (requires pred/correct cols)."""
    total      = int(len(df))
    successful = int(df["pred"].notna().sum())
    correct    = int(df["correct"].fillna(False).sum())
    return {
        "total"                  : total,
        "successful"             : successful,
        "failed"                 : total - successful,
        "correct"                : correct,
        "accuracy"               : correct / successful if successful else 0.0,
        "score"                  : correct / successful if successful else 0.0,
        "avg_latency_ms"         : 0.0,
        "total_prompt_tokens"    : 0,
        "total_completion_tokens": 0,
    }


def load_report() -> dict[str, Any]:
    """Build an aggregate report dict for all available models."""
    models: dict[str, Any] = {}
    for model in available_models():
        scored_path = RESPONSES_ROOT / f"{MODEL_DIR_PREFIX}{model}" / SCORED_CSV_NAME
        df = pd.read_csv(scored_path)
        models[model] = _model_stats(df)
    return {"generated_at": "praxis_reading_1 (computed)", "models": models}

# --- example usage -----------------------------------------------------------

def main() -> None:
    scored_path = Path(_get_scored_file_path(SAMPLE_LLMOUTPUT))
    print(f"Scored file : {scored_path}")

    if not scored_path.exists():
        raw_path = Path(SAMPLE_LLMOUTPUT)
        if not raw_path.exists():
            print("Raw LLM output not found — run prompts_sample_8.py first.")
            return
        print("Scoring raw output …")
        get_scored_file(SAMPLE_LLMOUTPUT)

    # --- load_responses (per-row) ---
    print("\n=== load_responses() ===")

    rows = load_responses(scored_path)
    n_success = sum(1 for r in rows if r["success"])
    n_correct = sum(1 for r in rows if r["is_correct"])
    print(f"  Loaded {len(rows)} rows  |  answered={n_success}  correct={n_correct}"
          f"  acc={n_correct/n_success:.2%}" if n_success else "  no answered rows")

    print("\n=== first 3 rows ===")
    for row in rows[:3]:
        cr = row["check_result"]
        print(f"  {row['item_id']}  model={row['model']}  "
              f"correct={row['is_correct']}  score={cr['score']}  "
              f"latency_ms={row['latency_ms']:.0f}ms")
        print(f"    {cr['reasoning']}")

    # --- response_frame ---
    print("\n=== response_frame() ===")
    rf = response_frame(rows)
    cols = ["item_id", "model", "success", "is_correct", "score",
            "latency_ms", "prompt_tokens", "completion_tokens"]
    print(rf[[c for c in cols if c in rf.columns]].to_string(index=False))

    # --- load_report (aggregate) ---
    print("\n=== load_report() ===")
    report = load_report()
    for m, s in report["models"].items():
        print(f"  {m}: total={s['total']}  correct={s['correct']}"
              f"  acc={s['accuracy']:.2%}")

    # --- generate_latex_table (aggregate) ---
    print("\n=== generate_latex_table() ===")
    latex = generate_overall_latex_table(rows)
    print(latex)

    # --- category helpers ---
    print("\n=== category_response_files() ===")
    for cat, paths in category_response_files().items():
        print(f"  {cat}:")
        for role, p in paths.items():
            print(f"    {role}: {p}  exists={p.exists()}")

    # --- messages (row 0) ---
    print("\n=== messages (row 0) ===")
    if rows:
        for msg in rows[0]["messages"]:
            print(f"  [{msg['role']:6s}] {str(msg['content'])[:100].replace(chr(10),' ')}…")

    print("\nDone.")


if __name__ == "__main__":
    main()
