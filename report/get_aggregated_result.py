"""Aggregate Praxis-Reading-1 results from scored CSVs.

Responsibilities:
- Discover available model output directories
- Compute per-model accuracy stats (_model_stats, load_report)
- Build a LaTeX summary table (generate_latex_table)

Everything else (path helpers, per-row loading, display config, demo) lives in
_bak.py to keep this module focused on aggregation.
"""


from llm_common.report.pipeline_after_llmcall1 import SAMPLE_LLMOUTPUT
from llm_common.report.pipeline_after_llmcall1 import _get_scored_file_path
from llm_common.report.pipeline_after_llmcall1 import get_scored_file

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data-source roots
# ---------------------------------------------------------------------------

PRAXIS_READING_DIR = Path(
    f"{os.environ['HOME']}/klee_code/git_repos/parse_evaluation/praxis_reading_1"
)
RESPONSES_ROOT  = PRAXIS_READING_DIR
SCORED_CSV_NAME = "prompts_scored.csv"
MODEL_DIR_PREFIX = "outputs_batch_infer_"

BASELINE_MODEL   = "Qwen3-32B"
EXPERIMENT_MODEL = "Qwen3-32B-ceval"


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------

def model_from_path(path: str | Path) -> str:
    """Extract model name from a scored-CSV path.

    Checks two conventions:
    1. Parent directory:  …/outputs_batch_infer_<model>/prompts_scored.csv
    2. Filename stem:     …/prompts_sample_8_batch_infer_<model>_scored.csv
    """
    p = Path(path)
    parent = p.parent.name
    if parent.startswith(MODEL_DIR_PREFIX):
        return parent[len(MODEL_DIR_PREFIX):]
    stem = p.stem.removesuffix("_scored")
    if "_batch_infer_" in stem:
        return stem.split("_batch_infer_", 1)[-1]
    return parent


def available_models() -> list[str]:
    """Return model names for all scored CSVs found under RESPONSES_ROOT."""
    if not RESPONSES_ROOT.exists():
        return []
    return [
        model_from_path(p)
        for p in sorted(RESPONSES_ROOT.glob(f"{MODEL_DIR_PREFIX}*/{SCORED_CSV_NAME}"))
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

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


def generate_latex_table(
        rows: list[dict[str, Any]],
        caption: str = "",
        label: str = "tab:praxis_reading",
) -> str:
    """Aggregate ``load_responses`` rows per model → compact LaTeX accuracy table.

    Writes ``latex/tables/<label>.tex``, ``latex/main.tex``, and
    ``latex/compile.sh`` as side-effects (mirroring TeaCH-main/nature/ layout).
    Returns the table fragment as a string.
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "answered": 0, "correct": 0}
    )
    for row in rows:
        model = row.get("model", "unknown")
        stats[model]["total"] += 1
        if row.get("success"):
            stats[model]["answered"] += 1
        if row.get("is_correct"):
            stats[model]["correct"] += 1

    if not caption:
        caption = (
            r"Praxis Reading MCQ accuracy on the sample set. "
            r"Acc\textsubscript{all} = correct / total; "
            r"Acc\textsubscript{ans} = correct / answered."
        )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{tabular}{l r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Total} & \textbf{Answered} & \textbf{Correct}"
        r" & \textbf{Acc\textsubscript{all}} & \textbf{Acc\textsubscript{ans}} \\",
        r"    \midrule",
    ]
    for model, s in sorted(stats.items()):
        n, ans, cor = s["total"], s["answered"], s["correct"]
        acc_all = cor / n   * 100 if n   else 0.0
        acc_ans = cor / ans * 100 if ans else 0.0
        lines.append(
            f"    {model} & {n} & {ans} & {cor}"
            f" & {acc_all:.1f}\\% & {acc_ans:.1f}\\% \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    table_tex = "\n".join(lines)

    # --- write latex artefacts ---
    latex_dir  = Path(__file__).resolve().parent / "latex"
    tables_dir = latex_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{label}.tex").write_text(table_tex, encoding="utf-8")

    main_tex = rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}

\title{{Praxis Reading-1 Evaluation Results}}
\author{{llm\_evals / praxis\_reading\_1}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Results}}

Table~\ref{{{label}}} shows MCQ accuracy on the Praxis Reading sample set
(\texttt{{prompts\_sample\_8}}).
Acc\textsubscript{{all}} = correct / total questions;
Acc\textsubscript{{ans}} = correct / answered questions (unanswered rows excluded).

\input{{tables/{label}}}

\end{{document}}
"""
    (latex_dir / "main.tex").write_text(main_tex, encoding="utf-8")

    compile_sh = """\
#!/usr/bin/env bash
# Compile main.tex → main.pdf  (pdflatex, two passes for cross-references)
set -euo pipefail
cd "$(dirname "$0")"
echo "==> Pass 1: pdflatex"
pdflatex -interaction=nonstopmode main.tex
echo "==> Pass 2: pdflatex"
pdflatex -interaction=nonstopmode main.tex
echo "==> Done: main.pdf"
"""
    compile_path = latex_dir / "compile.sh"
    compile_path.write_text(compile_sh, encoding="utf-8")
    compile_path.chmod(0o755)

    return table_tex

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
    latex = generate_latex_table(rows)
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
