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


# Subjective judges (llm_judge / attitude_judge) score on a 1–10 scale.
JUDGE_MAX_SCORE = 10.0


def _normalized_score(r) -> float | None:
    """Per-row score in ``[0, 1]`` so objective + subjective rows can be mixed.

    - subjective (judge): ``score / JUDGE_MAX_SCORE``
    - objective (mcq/rule, answered): ``1.0`` if correct else ``0.0``
    - unanswered: ``None``
    """
    if getattr(r, "score", None) is not None:
        return float(r.score) / JUDGE_MAX_SCORE
    if r.pred is not None:
        return 1.0 if r.correct else 0.0
    return None


def _cat_display(category: str) -> str:
    """Short English label for a category dir (``01_기능_skills`` → ``Skills``)."""
    if not category:
        return "Overall"
    return category.split("_")[-1].title()


def _group_by_model(reports: list) -> dict[str, dict[str, dict[str, float]]]:
    """Group reports as ``model -> category -> {total, score_sum}``.

    Mixing objective (0/1) and subjective (score/10) rows by their normalized
    score is the only sound way to aggregate the two — a plain correct/total
    accuracy would throw away the subjective rows' graded scores. Splitting by
    category lets each one be reported separately, plus an overall.
    """
    def _blank():
        return {"total": 0, "answered": 0, "correct": 0, "score_sum": 0.0}

    stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(_blank)
    )
    for r in reports:
        model = r.model or "unknown"
        category = (r.extra or {}).get("category", "") or ""
        ns = _normalized_score(r)
        for cat in (category, "__overall__"):
            cell = stats[model][cat]
            cell["total"] += 1
            if ns is not None:
                cell["answered"] += 1
                cell["score_sum"] += ns
            if r.correct:
                cell["correct"] += 1
    return {m: dict(c) for m, c in stats.items()}


def generate_overall_latex_table(
        stats: dict[str, dict[str, dict[str, float]]],
        caption: str = "",
        label: str = "tab:praxis_reading",
) -> str:
    """Per-model × per-category continuous-score matrix.

    Each cell is the mean normalized score for that (model, category): objective
    rows count as 0/1, subjective (judge) rows as ``score/10``. The last column
    is the overall mean across all categories. With exactly two models an extra
    ``Δ`` row shows the per-column difference (second − first).
    """
    def _score(cell: dict[str, float] | None) -> float | None:
        if not cell or not cell["total"]:
            return None
        return cell["score_sum"] / cell["total"] * 100

    # All categories present (excluding the synthetic overall bucket), sorted.
    categories = sorted(
        {c for m in stats for c in stats[m] if c != "__overall__"}
    )
    col_keys = categories + ["__overall__"]
    headers = [_cat_display(c) if c != "__overall__" else "Overall" for c in col_keys]

    if not caption:
        caption = (
            r"Continuous score per model and category (objective: correct = 1; "
            r"subjective: judge score / 10; cell = mean normalized score)."
        )

    col_spec = "l" + " r" * len(col_keys)
    head_row = r"    \textbf{Model} & " + " & ".join(
        rf"\textbf{{{h}}}" for h in headers
    ) + r" \\"
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        head_row,
        r"    \midrule",
    ]

    def _row(label_: str, cells: list[float | None], pct: bool) -> str:
        vals = []
        for v in cells:
            if v is None:
                vals.append("---")
            else:
                vals.append(f"{v:+.1f}\\%" if pct else f"{v:.1f}\\%")
        return f"    {label_} & " + " & ".join(vals) + r" \\"

    ordered = sorted(stats.items())
    for model, per_cat in ordered:
        cells = [_score(per_cat.get(c)) for c in col_keys]
        lines.append(_row(model, cells, pct=False))

    if len(ordered) == 2:
        (m1, c1), (m2, c2) = ordered
        deltas = []
        for c in col_keys:
            v1, v2 = _score(c1.get(c)), _score(c2.get(c))
            deltas.append(v2 - v1 if (v1 is not None and v2 is not None) else None)
        lines.append(r"    \midrule")
        lines.append(_row(f"$\\Delta$ ({m2} $-$ {m1})", deltas, pct=True))
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


def generate_category_table(
        stats: dict[str, dict[str, dict[str, float]]],
        category: str,
        label: str,
        caption: str = "",
) -> str:
    """Per-model detail table for a single *category* (same format as the old
    overall table: Total / Answered / Correct / Score, plus a Δ row for two
    models). Writes ``latex/tables/<label>.tex`` and returns the fragment."""
    def _score(cell: dict[str, float] | None) -> float:
        return cell["score_sum"] / cell["total"] * 100 if cell and cell["total"] else 0.0

    disp = _cat_display(category)
    if not caption:
        caption = (
            rf"{disp}: continuous score per model "
            r"(objective: correct = 1; subjective: judge score / 10)."
        )

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \begin{tabular}{l r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Total} & \textbf{Answered} & \textbf{Correct}"
        r" & \textbf{Score} \\",
        r"    \midrule",
    ]
    ordered = sorted(stats.items())
    for model, per_cat in ordered:
        c = per_cat.get(category) or {"total": 0, "answered": 0, "correct": 0, "score_sum": 0.0}
        n, ans, cor = int(c["total"]), int(c["answered"]), int(c["correct"])
        lines.append(f"    {model} & {n} & {ans} & {cor} & {_score(c):.1f}\\% \\\\")
    if len(ordered) == 2:
        (m1, c1), (m2, c2) = ordered
        d = _score(c2.get(category)) - _score(c1.get(category))
        lines.append(r"    \midrule")
        lines.append(f"    $\\Delta$ ({m2} $-$ {m1}) & & & & {d:+.1f}\\% \\\\")
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


_LATEX_UNICODE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "--", "—": "---", "…": "...", " ": " ",
}
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    # In LaTeX text mode bare < / > render as inverted ! / ? — escape them so
    # e.g. <think>…</think> tags display literally.
    "<": r"\textless{}", ">": r"\textgreater{}",
}


def _latex_cell(
        text: Any,
        limit: int | None = None,
        preserve_newlines: bool = False,
        strip_line_numbers: bool = False,
        max_lines: int | None = None,
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

    With ``max_lines`` (preserve_newlines mode) only the first N lines are kept
    and a ``[… truncated]`` marker is appended — caps how much vertical space a
    single cell (e.g. a long reasoning) can take.
    """
    s = str(text) if text is not None else ""
    for uni, ascii_ in _LATEX_UNICODE.items():
        s = s.replace(uni, ascii_)
    s = s.encode("ascii", "ignore").decode("ascii")  # drop any remaining non-ascii
    # Dropping non-ascii (e.g. Chinese terms) can leave empty brackets/quotes
    # like ``()`` / ``""`` — strip those husks so they don't litter the text.
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\[\s*\]", "", s)
    s = re.sub(r'"\s*"', "", s)
    s = re.sub(r"'\s*'", "", s)

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
        if max_lines is not None and len(cleaned) > max_lines:
            cleaned = cleaned[:max_lines]
            cleaned.append(r"\textit{[\ldots\ truncated]}")
        out = r" \newline ".join(cleaned)
    else:
        out = _esc(" ".join(s.split()))

    if limit is not None and len(out) > limit:
        out = out[:limit].rstrip() + "..."
    return out


# Max lines of reasoning shown per error block (keeps one block from filling a page).
REASONING_MAX_LINES = 12


def generate_error_table(
        reports: list,
        caption: str = "",
        label: str = "tab:praxis_errors",
        per_model_limit: int | None = None,
) -> str:
    """Collect the *wrong* answered rows into a LaTeX ``longtable``.

    Each error is rendered as a block of key/value rows (one field per row) in a
    two-column layout, so long question / reasoning text wraps across the full
    page width instead of being squeezed into a narrow column. Writes
    ``latex/tables/<label>.tex`` and returns the fragment string.

    *per_model_limit*: keep only the first N errors per model (reasoning stays
    full) — caps the report size when there are many long-reasoning errors.
    """
    errors = [r for r in reports if r.pred is not None and not r.correct]

    if per_model_limit is not None:
        grouped: dict[str, list] = defaultdict(list)
        for r in errors:
            grouped[r.model].append(r)
        errors = [r for rs in grouped.values() for r in rs[:per_model_limit]]

    if not caption:
        suffix = (f" (first {per_model_limit} per model)"
                  if per_model_limit is not None else "")
        caption = (
            r"Incorrectly answered questions" + suffix + r": gold answer vs.\ "
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
        # Use the full prompt (question stem + A./B./C./D. options) so the
        # answer choices are shown; fall back to the bare question.
        e = r.extra or {}
        question  = _latex_cell(e.get("prompt") or e.get("question", ""), preserve_newlines=True)
        gold      = _latex_cell(r.gold)
        # Model row: just the extracted answer (letter / judge score).
        model_ans = _latex_cell(r.pred) or _latex_cell(r.score) or r"\textit{--}"
        # Reasoning row: the model's full response / thinking — prefer an
        # explicit reasoning/judge field, else the raw llm_response (which for
        # praxis carries the <think>…</think> block).
        reasoning_src = r.reasoning or r.judge_reasoning or r.llm_response or ""
        # Cap reasoning length so one long block can't fill a page.
        reasoning = _latex_cell(
            reasoning_src, preserve_newlines=True, max_lines=REASONING_MAX_LINES,
        ) or r"\textit{--}"

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
        r"\begin{table}[H]",
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
        r"\begin{figure}[H]",
        r"  \centering",
        rf"  \includegraphics[width={width}]{{{relpath}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ])


def _write_combined_document(
        table_labels: list[str],
        figures: list[tuple[str, str, str]] | None = None,
        trailing_table_labels: list[str] | None = None,
) -> None:
    """Write ``latex/main.tex`` inputting every table fragment and figure.

    Order: *table_labels* → *figures* → *trailing_table_labels*. The trailing
    slot is for long tables (e.g. the error longtable) that should come last so
    they don't bury the figures / settings.

    *figures* is a list of ``(relpath, caption, label)`` tuples; each is embedded
    with ``\\includegraphics`` (paths are relative to the ``latex/`` directory).
    """
    latex_dir = Path(__file__).resolve().parent / "latex"
    blocks = [rf"\input{{tables/{lbl}}}" for lbl in table_labels]
    for relpath, caption, label in (figures or []):
        blocks.append(_figure_block(relpath, caption, label))
    blocks += [rf"\input{{tables/{lbl}}}" for lbl in (trailing_table_labels or [])]
    body = "\n\n".join(blocks)
    main_tex = rf"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{float}}
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

def get_aggregated_result_main(scored_paths: "list | None" = None):
    """Build the LaTeX report from one or more scored CSVs.

    *scored_paths*: the scored CSVs to aggregate (their reports are merged so a
    multi-model run produces one combined report). When omitted, falls back to
    scoring + using the bundled SAMPLE_LLMOUTPUT.
    """
    from llm_common.report.get_per_row_result import LLMInferPerRowReport
    from llm_common.report.get_per_row_result import get_scored_file

    if scored_paths:
        # Merge the per-row reports from every scored CSV.
        reports: List[LLMInferPerRowReport] = []
        for p in scored_paths:
            if p is not None and Path(p).exists():
                reports.extend(LLMInferPerRowReport.from_csv(p))
    else:
        # Fallback demo: score the bundled sample if needed, then load it.
        scored_path = LLMInferPerRowReport.get_output_path_hint(SAMPLE_LLMOUTPUT)
        if not Path(scored_path).exists():
            get_scored_file(SAMPLE_LLMOUTPUT)
        reports = LLMInferPerRowReport.from_csv(scored_path)

    # Group reports by model (accumulating the continuous normalized score so
    # objective + subjective items aggregate soundly), then render. With exactly
    # two models the table also shows their score delta.
    stats = _group_by_model(reports)
    overall_tex = generate_overall_latex_table(stats)
    print(overall_tex)
    if len(stats) == 2:
        (m1, s1), (m2, s2) = sorted(stats.items())
        ov = lambda s: (s["__overall__"]["score_sum"] / s["__overall__"]["total"] * 100
                        if s.get("__overall__", {}).get("total") else 0.0)
        print(f"\nΔ overall score ({m2} − {m1}): {ov(s2) - ov(s1):+.1f}%")

    # One detail table per category (same format), in addition to the matrix.
    categories = sorted({c for m in stats for c in stats[m] if c != "__overall__"})
    category_labels: list[str] = []
    for cat in categories:
        lbl = f"tab:cat_{_cat_display(cat).lower()}"
        generate_category_table(stats, cat, label=lbl)
        category_labels.append(lbl)

    # Render the run's model_settings as a key/value table (secrets redacted).
    settings_tex = generate_model_settings_table(reports)
    print("\n" + settings_tex)

    # Nature-style plots of the numeric fields (prompt/completion/total tokens,
    # latency): univariate distributions + multivariate pairwise relationships.
    from llm_common.report.plot_numeric import generate_numeric_plots
    latex_dir = Path(__file__).resolve().parent / "latex"
    plot_paths = generate_numeric_plots(reports, latex_dir / "figures")
    figures = []
    if plot_paths:
        uni_pdf, multi_pdf = plot_paths
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
    error_tex = generate_error_table(reports, per_model_limit=20)
    print("\n" + error_tex)

    # Order: score tables → model settings → figures → error table (last, since
    # it's the longest and shouldn't bury the settings/figures).
    _write_combined_document(
        ["tab:praxis_reading", *category_labels, "tab:model_settings"],
        figures=figures,
        trailing_table_labels=["tab:praxis_errors"],
    )


if __name__ == "__main__":
    get_aggregated_result_main()


