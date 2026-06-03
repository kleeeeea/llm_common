"""Aggregate Praxis-Reading-1 results from scored CSVs.

Responsibilities:
- Discover available model output directories
- Compute per-model accuracy stats (_model_stats, load_report)
- Build a LaTeX summary table (generate_latex_table)

Everything else (path helpers, per-row loading, display config, demo) lives in
_bak.py to keep this module focused on aggregation.
"""
from typing import List

from llm_common.report.get_per_row_result import SAMPLE_LLMOUTPUT

import os
import re
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def generate_overall_latex_table(
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

_LATEX_UNICODE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "--", "—": "---", "…": "...", " ": " ",
}
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _latex_cell(
        text: Any,
        limit: int | None = None,
        preserve_newlines: bool = False,
        strip_line_numbers: bool = False,
) -> str:
    """Make arbitrary text safe for a LaTeX cell: normalise unicode, escape
    special characters, and collapse whitespace. Truncates only when *limit*
    is given (``None`` = keep the full text).

    With ``preserve_newlines=True`` the source line breaks are kept as LaTeX
    ``\\newline`` breaks (intra-line whitespace is still collapsed) — used so the
    a./b./c. answer choices stay on separate lines in a ``p{}`` column.

    With ``strip_line_numbers=True`` a leading integer + space is removed from
    each line (drops the ``5``, ``10``, ``15`` … margin line-numbers embedded in
    the source passages). Only applies in ``preserve_newlines`` mode.
    """
    s = str(text) if text is not None else ""
    for uni, ascii_ in _LATEX_UNICODE.items():
        s = s.replace(uni, ascii_)
    s = s.encode("ascii", "ignore").decode("ascii")  # drop any remaining non-ascii

    def _esc(t: str) -> str:
        return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in t)

    if preserve_newlines:
        cleaned = []
        for ln in s.splitlines():
            ln = " ".join(ln.split())  # collapse intra-line whitespace
            if strip_line_numbers:
                ln = re.sub(r"^\d+\s+", "", ln)  # drop a leading margin line-number
            if ln:
                cleaned.append(_esc(ln))
        out = r" \newline ".join(cleaned)
    else:
        out = _esc(" ".join(s.split()))

    if limit is not None and len(out) > limit:
        out = out[:limit].rstrip() + "..."
    return out


def generate_error_table(
        reports: list,
        caption: str = "",
        label: str = "tab:praxis_errors",
) -> str:
    """Collect the *wrong* answered rows into a LaTeX ``longtable``.

    Each error is rendered as a block of key/value rows (one field per row) in a
    two-column layout, so long question / reasoning text wraps across the full
    page width instead of being squeezed into a narrow column. Writes
    ``latex/tables/<label>.tex`` and returns the fragment string.
    """
    errors = [r for r in reports if r.pred is not None and not r.correct]

    if not caption:
        caption = (
            r"Incorrectly answered Praxis Reading questions: gold answer vs.\ "
            r"the model's prediction and reasoning."
        )

    # Two columns: field name (left) + wide value column that wraps.
    lines = [
        r"\begin{longtable}{@{}l p{13cm}@{}}",
        f"  \\caption{{{caption}}}\\label{{{label}}} \\\\",
        r"  \toprule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  \endhead",
    ]
    if not errors:
        lines.append(r"  \multicolumn{2}{c}{\textit{No incorrect answers.}} \\")
    for r in errors:
        # No truncation — show the full text; keep the a./b./c. choices and
        # passage paragraphs on separate lines.
        passage   = _latex_cell((r.extra or {}).get("passage", ""), preserve_newlines=True, strip_line_numbers=True) or r"\textit{--}"
        question  = _latex_cell((r.extra or {}).get("question", ""), preserve_newlines=True)
        gold      = _latex_cell(r.gold)
        model_ans = _latex_cell(r.pred)
        raw       = _latex_cell(r.llm_response)
        if raw and raw.upper() != model_ans.upper():
            model_ans = f"{model_ans} ({raw})"
        reasoning = _latex_cell(r.reasoning or "", preserve_newlines=True) or r"\textit{--}"

        # One field per row; the ID heads the block spanning both columns.
        lines.append(rf"  \multicolumn{{2}}{{@{{}}l}}{{\textbf{{{_latex_cell(r.id)}}}}} \\")
        lines.append(rf"  \textbf{{Passage}}   & {passage}    \\")
        lines.append(rf"  \textbf{{Question}}  & {question}   \\")
        lines.append(rf"  \textbf{{Gold}}      & {gold}       \\")
        lines.append(rf"  \textbf{{Model}}     & {model_ans}  \\")
        lines.append(rf"  \textbf{{Reasoning}} & {reasoning}  \\")
        lines.append(r"  \midrule")
    lines += [
        r"  \bottomrule",
        r"\end{longtable}",
    ]
    table_tex = "\n".join(lines)

    tables_dir = Path(__file__).resolve().parent / "latex" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{label}.tex").write_text(table_tex, encoding="utf-8")
    return table_tex


def _model_settings_dict(reports: list) -> dict:
    """Return the run's model_settings dict from the first report that has one.

    The settings are shared across rows; in a loaded CSV they live in
    ``extra['model_settings']`` as a stringified dict.
    """
    import ast
    for r in reports:
        ms = (getattr(r, "extra", None) or {}).get("model_settings")
        if isinstance(ms, str):
            try:
                ms = ast.literal_eval(ms)
            except (ValueError, SyntaxError):
                ms = None
        if isinstance(ms, dict):
            return ms
    return {}


def generate_model_settings_table(
        reports: list,
        caption: str = "",
        label: str = "tab:model_settings",
) -> str:
    """Render the run's ``model_settings`` as a key/value LaTeX table.

    Nested dicts (e.g. ``api``) are flattened to ``key.subkey`` rows; any
    secret-looking field (``api_key`` / ``*_key``) is redacted. Writes
    ``latex/tables/<label>.tex`` and returns the fragment string.
    """
    settings = _model_settings_dict(reports)

    def _is_secret(key: str) -> bool:
        k = key.lower()
        return "api_key" in k or k == "key" or k.endswith("_key")

    # ModelSettings.__post_init__ mirrors the ApiConfig (``api.base_url`` /
    # ``api.api_key`` / ``api.model``) up to top-level fields, so those values
    # appear twice. Dedupe by leaf name + value: the first occurrence wins.
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(key: str, value: Any) -> None:
        val = "***redacted***" if _is_secret(key) else str(value)
        leaf = key.rsplit(".", 1)[-1]
        if (leaf, val) in seen:
            return
        seen.add((leaf, val))
        rows.append((key, val))

    for k, v in settings.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                _add(f"{k}.{sk}", sv)
        else:
            _add(k, v)

    if not caption:
        caption = r"Model settings used to produce the responses (secrets redacted)."

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{tabular}{@{}l p{10cm}@{}}",
        r"    \toprule",
        r"    \textbf{Setting} & \textbf{Value} \\",
        r"    \midrule",
    ]
    if not rows:
        lines.append(r"    \multicolumn{2}{c}{\textit{No model settings recorded.}} \\")
    for k, v in rows:
        lines.append(f"    {_latex_cell(k)} & {_latex_cell(v)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    table_tex = "\n".join(lines)

    tables_dir = Path(__file__).resolve().parent / "latex" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{label}.tex").write_text(table_tex, encoding="utf-8")
    return table_tex


def _figure_block(relpath: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    """A centred ``figure`` environment embedding *relpath* (a PDF under latex/)."""
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"  \centering",
        rf"  \includegraphics[width={width}]{{{relpath}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ])


def _write_combined_document(
        table_labels: list[str],
        figures: list[tuple[str, str, str]] | None = None,
) -> None:
    """Write ``latex/main.tex`` inputting every table fragment and figure.

    *figures* is a list of ``(relpath, caption, label)`` tuples; each is embedded
    with ``\\includegraphics`` (paths are relative to the ``latex/`` directory).
    """
    latex_dir = Path(__file__).resolve().parent / "latex"
    blocks = [rf"\input{{tables/{lbl}}}" for lbl in table_labels]
    for relpath, caption, label in (figures or []):
        blocks.append(_figure_block(relpath, caption, label))
    body = "\n\n".join(blocks)
    main_tex = rf"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}
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

{body}

\end{{document}}
"""
    (latex_dir / "main.tex").write_text(main_tex, encoding="utf-8")


def main():
    """Demo: score the sample output, then build a LaTeX accuracy table."""
    from llm_common.report.get_per_row_result import LLMInferPerRowReport
    from llm_common.report.get_per_row_result import get_scored_file

    # Score the raw LLM output if the scored CSV doesn't exist yet.
    scored_path = LLMInferPerRowReport.get_output_path_hint(SAMPLE_LLMOUTPUT)
    if not Path(scored_path).exists():
        get_scored_file(SAMPLE_LLMOUTPUT)

    # Load the scored rows as typed reports (gold/pred/correct restored).
    reports: List[LLMInferPerRowReport] = LLMInferPerRowReport.from_csv(scored_path)

    # Reshape into the {model, success, is_correct} rows generate_latex_table
    # wants. The model name comes from each report's embedded model_settings
    # (LLMInferInput.model), not the filename.
    rows = [
        {
            "model"     : r.model,
            "success"   : r.pred is not None,
            "is_correct": r.correct,
        }
        for r in reports
    ]

    overall_tex = generate_overall_latex_table(rows)
    print(overall_tex)

    # Render the run's model_settings as a key/value table (secrets redacted).
    settings_tex = generate_model_settings_table(reports)
    print("\n" + settings_tex)

    # Nature-style plots of the numeric fields (prompt/completion/total tokens,
    # latency): univariate distributions + multivariate pairwise relationships.
    from llm_common.report.plot_numeric import generate_numeric_plots
    latex_dir = Path(__file__).resolve().parent / "latex"
    uni_pdf, multi_pdf = generate_numeric_plots(reports, latex_dir / "figures")
    figures = [
        (f"figures/{uni_pdf.name}",
         r"Univariate distributions of the per-row numeric fields, conditioned "
         r"on whether the answer was correct (histogram + KDE + rug).",
         "fig:numeric_univariate"),
        (f"figures/{multi_pdf.name}",
         r"Pairwise relationships between the numeric fields, coloured by "
         r"correctness: scatter (lower triangle), per-group distribution "
         r"(diagonal), Pearson $r$ (upper triangle).",
         "fig:numeric_multivariate"),
    ]

    # Collect the wrong questions into a LaTeX table: question, correct answer,
    # the model's answer, and its reasoning/process.
    error_tex = generate_error_table(reports)
    print("\n" + error_tex)

    # Write a combined document with all tables + figures (overrides the
    # single-table main.tex written by generate_overall_latex_table).
    _write_combined_document(
        ["tab:praxis_reading", "tab:model_settings", "tab:praxis_errors"],
        figures=figures,
    )


if __name__ == "__main__":
    main()


