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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import pandas as pd

from llm_common.llm_infer.api_info.dataclass_ import model_alias_for


def _model_display(model: str) -> str:
    """Human-facing model name for tables: the registered alias, else the raw name.

    Grouping/keys keep using the raw ``r.model``; only what the reader sees is
    aliased (e.g. ``kimi`` -> ``Innospark-1T``).
    """
    return model_alias_for(model or "unknown")


# ---------------------------------------------------------------------------
# Document schema — a typed description of one report section, so the assembly
# code can't silently misspell a key (the old loose ``dict`` approach).
# ---------------------------------------------------------------------------

# A table entry is either a bare table label, or a ``(subsection_title, label)``
# pair that prefixes a ``\subsection`` heading before the ``\input``.
TableEntry = Union[str, tuple[str, str]]


@dataclass(frozen=True)
class FigureEntry:
    """One ``\\includegraphics`` block: a figure path, caption, and label."""
    relpath: str
    caption: str
    label: str


@dataclass
class ReportSection:
    """One ``\\section`` of the report: a title plus its tables and figures.

    Rendered in order by ``_write_combined_document``; the caller controls
    layout simply by the order it appends sections.
    """
    title: str
    tables: list[TableEntry] = field(default_factory=list)
    figures: list[FigureEntry] = field(default_factory=list)


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


def _write_table_tex(label: str, table_tex: str) -> None:
    """Write one LaTeX table fragment under ``latex/tables``."""
    tables_dir = Path(__file__).resolve().parent / "latex" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{label}.tex").write_text(table_tex, encoding="utf-8")


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
        lines.append(_row(_model_display(model), cells, pct=False))

    if len(ordered) == 2:
        (m1, c1), (m2, c2) = ordered
        deltas = []
        for c in col_keys:
            v1, v2 = _score(c1.get(c)), _score(c2.get(c))
            deltas.append(v2 - v1 if (v1 is not None and v2 is not None) else None)
        lines.append(r"    \midrule")
        lines.append(_row(
            f"$\\Delta$ ({_model_display(m2)} $-$ {_model_display(m1)})",
            deltas, pct=True))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    table_tex = "\n".join(lines)

    _write_table_tex(label, table_tex)
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
        lines.append(f"    {_model_display(model)} & {n} & {ans} & {cor} & {_score(c):.1f}\\% \\\\")
    if len(ordered) == 2:
        (m1, c1), (m2, c2) = ordered
        d = _score(c2.get(category)) - _score(c1.get(category))
        lines.append(r"    \midrule")
        lines.append(f"    $\\Delta$ ({_model_display(m2)} $-$ {_model_display(m1)}) & & & & {d:+.1f}\\% \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    table_tex = "\n".join(lines)

    _write_table_tex(label, table_tex)
    return table_tex


def per_instance_deltas(reports: list) -> list[dict]:
    """Per-item score difference between the two models.

    Pairs reports by item id, computes each model's normalized score, and the
    delta (second model − first, alphabetical). Items not answered by exactly
    two models (or missing a score) are skipped. Sorted by ``|delta|`` desc.
    """
    by_item: dict = defaultdict(dict)
    for r in reports:
        ns = _normalized_score(r)
        by_item[r.id][r.model] = (ns, (r.extra or {}).get("category", ""))

    rows: list[dict] = []
    for item_id, md in by_item.items():
        models = sorted(md)
        if len(models) != 2:
            continue
        m1, m2 = models
        (s1, cat), (s2, _) = md[m1], md[m2]
        if s1 is None or s2 is None:
            continue
        rows.append({
            "item_id": item_id, "category": cat,
            "model1": m1, "score1": s1,
            "model2": m2, "score2": s2,
            "delta": s2 - s1,
        })
    rows.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return rows


# Heavy rule separating per-item blocks (more visible than \midrule).
_BLOCK_SEP = r"  \specialrule{1.2pt}{8pt}{8pt}"

# |Δ| magnitude buckets (lo ≤ |Δ| < hi), largest first.
_DELTA_BUCKETS = [
    (0.75, 1.01, r"$|\Delta| \geq 75\%$"),
    (0.50, 0.75, r"$50\% \leq |\Delta| < 75\%$"),
    (0.25, 0.50, r"$25\% \leq |\Delta| < 50\%$"),
    (1e-9, 0.25, r"$0 < |\Delta| < 25\%$"),
]

# Signed Δ buckets for detailed answer-comparison sections. These do not use
# absolute value: negative means the second model scored lower than the first.
_SIGNED_DELTA_BUCKETS = [
    (-1.01, -0.75, r"$\Delta \leq -75\%$", "neg75"),
    (-0.75, -0.50, r"$-75\% < \Delta \leq -50\%$", "neg50"),
    (-0.50, -0.25, r"$-50\% < \Delta \leq -25\%$", "neg25"),
    (-0.25, -1e-9, r"$-25\% < \Delta < 0$", "neg0"),
    (1e-9, 0.25, r"$0 < \Delta < 25\%$", "pos0"),
    (0.25, 0.50, r"$25\% \leq \Delta < 50\%$", "pos25"),
    (0.50, 0.75, r"$50\% \leq \Delta < 75\%$", "pos50"),
    (0.75, 1.01, r"$\Delta \geq 75\%$", "pos75"),
]


def generate_disagreement_table(
        reports: list,
        per_bucket_limit: int = 40,
        label: str = "tab:disagreement",
        caption: str = "",
) -> str:
    """Per-instance score differences grouped into |Δ| magnitude buckets, with
    the concrete instances listed under each bucket (largest first).

    Writes ``latex/tables/<label>.tex`` and returns the fragment (empty string
    when there aren't exactly two comparable models)."""
    deltas = per_instance_deltas(reports)
    if not deltas:
        return ""
    m1, m2 = deltas[0]["model1"], deltas[0]["model2"]
    n_diff = sum(1 for d in deltas if abs(d["delta"]) > 1e-9)

    if not caption:
        caption = (
            rf"Per-instance score differences ($\Delta = $ {_model_display(m2)} $-$ "
            rf"{_model_display(m1)}), grouped "
            rf"by magnitude; {n_diff} of {len(deltas)} shared items differ. "
            r"Normalized scores (objective 0/1, subjective score/10)."
        )

    header = (
        rf"  \textbf{{ID}} & \textbf{{Category}} & \textbf{{{_latex_cell(_model_display(m1))}}}"
        rf" & \textbf{{{_latex_cell(_model_display(m2))}}} & \textbf{{$\Delta$}} \\"
    )
    lines = [
        r"\begin{longtable}{@{}l l r r r@{}}",
        rf"  \caption{{{caption}}}\label{{{label}}} \\",
        r"  \toprule", header, r"  \midrule", r"  \endfirsthead",
        r"  \toprule", header, r"  \midrule", r"  \endhead",
    ]
    for lo, hi, title in _DELTA_BUCKETS:
        items = sorted(
            (d for d in deltas if lo <= abs(d["delta"]) < hi),
            key=lambda d: d["delta"],
        )
        if not items:
            continue
        lines.append(
            rf"  \multicolumn{{5}}{{@{{}}l}}{{\textbf{{{title}}} \quad "
            rf"({len(items)} items)}} \\"
        )
        lines.append(r"  \midrule")
        for d in items[:per_bucket_limit]:
            lines.append(
                f"  {_latex_cell(d['item_id'])} & {_cat_display(d['category'])} &"
                f" {d['score1'] * 100:.0f}\\% & {d['score2'] * 100:.0f}\\%"
                f" & {d['delta'] * 100:+.0f}\\% \\\\"
            )
        if len(items) > per_bucket_limit:
            lines.append(
                rf"  \multicolumn{{5}}{{@{{}}l}}{{\textit{{\ldots\ "
                rf"{len(items) - per_bucket_limit} more}}}} \\"
            )
        lines.append(r"  \midrule")
    lines += [r"  \bottomrule", r"\end{longtable}"]
    table_tex = "\n".join(lines)

    _write_table_tex(label, table_tex)
    return table_tex


def _paired_reports(reports: list) -> list[tuple]:
    """Pair reports by item id → ``(item_id, m1, r1, m2, r2, delta)``.

    Only items answered by exactly two models (with scores) are kept; sorted by
    ``|delta|`` desc."""
    by_item: dict = defaultdict(dict)
    for r in reports:
        by_item[r.id][r.model] = r
    pairs: list[tuple] = []
    for item_id, md in by_item.items():
        if len(md) != 2:
            continue
        m1, m2 = sorted(md)
        r1, r2 = md[m1], md[m2]
        s1, s2 = _normalized_score(r1), _normalized_score(r2)
        if s1 is None or s2 is None:
            continue
        pairs.append((item_id, m1, r1, m2, r2, s2 - s1))
    pairs.sort(key=lambda p: abs(p[5]), reverse=True)
    return pairs


def generate_comparison_table(
        reports: list,
        top_n: int = 15,
        label: str = "tab:comparison",
        caption: str = "",
        category: str | None = None,
        delta_range: tuple[float, float] | None = None,
        delta_title: str = "",
) -> str:
    """Side-by-side answer comparison for the top-N most divergent items.

    One block per item: the question (+ options), the reference/gold answer, and
    each model's answer + reasoning — so the two models' differing responses to
    the *same* question sit together. Writes ``latex/tables/<label>.tex``.

    When *category* is given, only items in that benchmark category are compared.
    When *delta_range* is given, only signed deltas in ``[lo, hi)`` are included,
    so positive and negative movements are reported separately."""
    if category is not None:
        reports = [r for r in reports if (r.extra or {}).get("category", "") == category]
    pairs = [p for p in _paired_reports(reports) if abs(p[5]) > 1e-9]
    if delta_range is not None:
        lo, hi = delta_range
        pairs = [p for p in pairs if lo <= p[5] < hi]
        pairs.sort(key=lambda p: p[5], reverse=lo > 0)
    pairs = pairs[:top_n]
    if not pairs:
        return ""

    if not caption:
        scope = f"{_cat_display(category)} " if category is not None else ""
        delta_scope = f" ({delta_title})" if delta_title else ""
        caption = (
            rf"{scope}per-instance answer comparison{delta_scope} for the {len(pairs)} most "
            r"divergent items: same question, each model's answer and reasoning."
        )

    def _ans(r) -> str:
        if r.score is not None:
            return rf"score {r.score:g}/10"
        return "answer " + (_latex_cell(r.pred) or r"\textit{--}")

    def _reason(r) -> str:
        src = r.reasoning or r.judge_reasoning or r.llm_response or ""
        return _latex_cell(src, preserve_newlines=True, max_lines=8) or r"\textit{--}"

    lines = [
        r"\begin{longtable}{@{}l p{12.5cm}@{}}",
        rf"  \caption{{{caption}}}\label{{{label}}} \\",
        r"  \toprule", r"  \endfirsthead", r"  \toprule", r"  \endhead",
    ]
    for item_id, ma, ra, mb, rb, delta in pairs:
        e = ra.extra or {}
        question = _latex_cell(e.get("prompt") or e.get("question", ""), preserve_newlines=True)
        ref = _latex_cell(e.get("answer", ""), preserve_newlines=True, max_lines=8)
        lines.append(
            rf"  \multicolumn{{2}}{{@{{}}l}}{{\textbf{{{_latex_cell(item_id)}}} "
            rf"\quad $\Delta = {delta * 100:+.0f}\%$}} \\"
        )
        lines.append(rf"  \textbf{{Question}} & {question} \\")
        if ra.score is not None and ref:
            lines.append(rf"  \textbf{{Reference}} & {ref} \\")
        elif ra.gold is not None:
            lines.append(rf"  \textbf{{Gold}} & {_latex_cell(ra.gold)} \\")
        lines.append(r"  \midrule")
        # each model's answer + reasoning
        lines.append(rf"  \textbf{{{_latex_cell(_model_display(ma))}}} & {_ans(ra)} \\")
        lines.append(rf"  & {_reason(ra)} \\")
        lines.append(rf"  \textbf{{{_latex_cell(_model_display(mb))}}} & {_ans(rb)} \\")
        lines.append(rf"  & {_reason(rb)} \\")
        # heavy rule between items so different questions are clearly separated.
        lines.append(_BLOCK_SEP)
    lines += [r"  \bottomrule", r"\end{longtable}"]
    table_tex = "\n".join(lines)

    _write_table_tex(label, table_tex)
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
            # Keep the head (setup) AND the tail (conclusion), cut the middle —
            # so a final "the answer is …" line isn't lost to truncation.
            tail = min(3, max(1, max_lines // 3))
            head = max(1, max_lines - tail)
            cleaned = (
                cleaned[:head]
                + [r"\textit{[\ldots\ truncated \ldots]}"]
                + cleaned[-tail:]
            )
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
        category: str | None = None,
) -> str:
    """Collect the *wrong* answered rows into a LaTeX ``longtable``.

    Each error is rendered as a block of key/value rows (one field per row) in a
    two-column layout, so long question / reasoning text wraps across the full
    page width instead of being squeezed into a narrow column. Writes
    ``latex/tables/<label>.tex`` and returns the fragment string.

    *per_model_limit*: keep only the first N errors per model (reasoning stays
    full) — caps the report size when there are many long-reasoning errors.
    *category*: restrict to errors from one benchmark category.
    """
    # What counts as an "error":
    # - objective (has pred): wrong answer.
    # - subjective (has score): score below 50% of max (regardless of the
    #   binary is_correct flag, which may use a different threshold).
    def _is_error(r) -> bool:
        if r.score is not None:
            return float(r.score) < JUDGE_MAX_SCORE * 0.5
        if r.pred is not None:
            return not r.correct
        return False

    errors = [r for r in reports if _is_error(r)]
    if category is not None:
        errors = [r for r in errors if (r.extra or {}).get("category", "") == category]

    if per_model_limit is not None:
        grouped: dict[str, list] = defaultdict(list)
        for r in errors:
            grouped[r.model].append(r)
        errors = [r for rs in grouped.values() for r in rs[:per_model_limit]]

    if not caption:
        scope = f"{_cat_display(category)} " if category is not None else ""
        suffix = (f" (first {per_model_limit} per model)"
                  if per_model_limit is not None else "")
        caption = (
            scope + r"failed questions" + suffix +
            r" (objective: wrong; subjective: score $<$ 50\%): "
            r"gold/score vs.\ the model's prediction and reasoning."
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
        e = r.extra or {}
        # No truncation for question/passage; reasoning + reference answer are
        # capped so one block can't fill a page.
        passage   = _latex_cell(e.get("passage", ""), preserve_newlines=True, strip_line_numbers=True)
        # Full prompt (question stem + A./B./C./D. options); fall back to question.
        question  = _latex_cell(e.get("prompt") or e.get("question", ""), preserve_newlines=True)
        # Reference / expected answer (the grading reference for subjective items).
        answer    = _latex_cell(e.get("answer", ""), preserve_newlines=True, max_lines=REASONING_MAX_LINES)
        reasoning_src = r.reasoning or r.judge_reasoning or r.llm_response or ""
        reasoning = _latex_cell(
            reasoning_src, preserve_newlines=True, max_lines=REASONING_MAX_LINES,
        ) or r"\textit{--}"

        # One field per row; the ID heads the block spanning both columns.
        lines.append(rf"  \multicolumn{{2}}{{@{{}}l}}{{\textbf{{{_latex_cell(r.id)}}}}} \\")
        if passage:  # only objective reading items have a passage
            lines.append(rf"  \textbf{{Passage}}   & {passage}    \\")
        lines.append(rf"  \textbf{{Question}}  & {question}   \\")
        if r.score is not None:
            # subjective: reference answer + judge score (no letter gold/pred).
            if answer:
                lines.append(rf"  \textbf{{Reference}} & {answer}      \\")
            lines.append(rf"  \textbf{{Score}}     & {r.score:g}/10  \\")
        else:
            # objective: gold answer vs model's predicted letter.
            lines.append(rf"  \textbf{{Gold}}      & {_latex_cell(r.gold)}       \\")
            lines.append(rf"  \textbf{{Model}}     & {_latex_cell(r.pred) or r'\textit{--}'}  \\")
        lines.append(rf"  \textbf{{Reasoning}} & {reasoning}  \\")
        # heavy rule between items so different questions are clearly separated.
        lines.append(_BLOCK_SEP)
    lines += [
        r"  \bottomrule",
        r"\end{longtable}",
    ]
    table_tex = "\n".join(lines)

    _write_table_tex(label, table_tex)
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

    # ChatCompletionRequest.__post_init__ mirrors the ApiConfig (``api.base_url`` /
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

    _write_table_tex(label, table_tex)
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


def _write_combined_document(sections: "list[ReportSection]") -> None:
    """Write ``latex/main.tex`` from an ordered list of :class:`ReportSection`.

    Each section becomes a ``\\section`` followed by its table ``\\input``s and
    figure blocks. Sections are emitted in order, so the caller controls layout
    (and which tables come last).
    """
    parts: list[str] = []
    for sec in sections:
        if sec.title:
            parts.append(rf"\section{{{sec.title}}}")
        for item in sec.tables:
            # item is a label, or (subsection_title, label) to prefix a heading.
            if isinstance(item, tuple):
                subtitle, lbl = item
                parts.append(rf"\subsection{{{subtitle}}}")
            else:
                lbl = item
            parts.append(rf"\input{{tables/{lbl}}}")
        for fig in sec.figures:
            parts.append(_figure_block(fig.relpath, fig.caption, fig.label))
    body = "\n\n".join(parts)
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

{body}

\end{{document}}
"""
    latex_dir = Path(__file__).resolve().parent / "latex"
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
    if len(stats) == 2:
        (m1, s1), (m2, s2) = sorted(stats.items())
        ov = lambda s: (s["__overall__"]["score_sum"] / s["__overall__"]["total"] * 100
                        if s.get("__overall__", {}).get("total") else 0.0)
        print(f"\nΔ overall score ({_model_display(m2)} − {_model_display(m1)}): {ov(s2) - ov(s1):+.1f}%")

    # One detail table per category (same format), in addition to the matrix.
    categories = sorted({c for m in stats for c in stats[m] if c != "__overall__"})
    category_labels: list[str] = []
    for cat in categories:
        lbl = f"tab:cat_{_cat_display(cat).lower()}"
        generate_category_table(stats, cat, label=lbl)
        category_labels.append(lbl)

    # Per-instance comparison of the two models (skipped unless exactly two).
    deltas = per_instance_deltas(reports)
    disagreement_label = None
    if deltas:
        m1, m2 = deltas[0]["model1"], deltas[0]["model2"]
        n = len(deltas)
        mean_abs = sum(abs(d["delta"]) for d in deltas) / n
        n_better2 = sum(1 for d in deltas if d["delta"] > 0)
        n_tie     = sum(1 for d in deltas if d["delta"] == 0)
        n_better1 = sum(1 for d in deltas if d["delta"] < 0)
        print(f"\nPer-instance ({n} shared items): mean |Δ| = {mean_abs * 100:.1f}% | "
              f"{_model_display(m2)} better: {n_better2} | tie: {n_tie} | "
              f"{_model_display(m1)} better: {n_better1}")
        disagreement_label = "tab:disagreement"
        generate_disagreement_table(reports, per_bucket_limit=40, label=disagreement_label)
        # Split the answer comparison into signed Δ sections within each category.
        # Negative and positive deltas are intentionally separate: Δ is not
        # bucketed by absolute value here.
        comparison_items: list[tuple[str, str]] = []
        for cat in categories:
            disp = _cat_display(cat)
            for lo, hi, title, suffix in _SIGNED_DELTA_BUCKETS:
                lbl = f"tab:cmp_{disp.lower()}_{suffix}"
                tex = generate_comparison_table(
                    reports,
                    top_n=15,
                    label=lbl,
                    category=cat,
                    delta_range=(lo, hi),
                    delta_title=title,
                )
                if tex:  # skip empty category × band combinations
                    comparison_items.append((f"{disp}: {title}", lbl))

    # Render the run's model_settings as a key/value table (secrets redacted).
    settings_tex = generate_model_settings_table(reports)
    print("\n" + settings_tex)

    # Nature-style plots of the numeric fields (prompt/completion/total tokens,
    # latency): univariate distributions + multivariate pairwise relationships.
    from llm_common.report.plot_numeric import generate_numeric_plots
    latex_dir = Path(__file__).resolve().parent / "latex"
    plot_paths = generate_numeric_plots(reports, latex_dir / "figures")
    figures: list[FigureEntry] = []
    if plot_paths:
        uni_pdf, multi_pdf = plot_paths
        figures = [
            FigureEntry(
                f"figures/{uni_pdf.name}",
                r"Univariate distributions of the per-row numeric fields, conditioned "
                r"on whether the answer was correct (histogram + KDE + rug).",
                "fig:numeric_univariate"),
            FigureEntry(
                f"figures/{multi_pdf.name}",
                r"Pairwise relationships between the numeric fields, coloured by "
                r"correctness: scatter (lower triangle), per-group distribution "
                r"(diagonal), Pearson $r$ (upper triangle).",
                "fig:numeric_multivariate"),
        ]

    # One error table per category (question, correct answer / score, the
    # model's answer, and its reasoning). Each is capped per model.
    # (category display name, table label) so each gets a \subsection heading.
    error_items: list[tuple[str, str]] = []
    for cat in categories:
        disp = _cat_display(cat)
        lbl = f"tab:err_{disp.lower()}"
        generate_error_table(reports, label=lbl, per_model_limit=20, category=cat)
        error_items.append((disp, lbl))

    # Assemble the document as titled sections (errors last — longest content).
    # ReportSection is a dataclass, so the schema (title/tables/figures) is
    # enforced — a typo'd field raises at construction instead of silently
    # vanishing the way a dict key would.
    sections = [
        ReportSection("Overall scores", tables=["tab:praxis_reading"]),
        ReportSection("Per-category scores", tables=category_labels),
    ]
    if disagreement_label:
        sections.append(
            ReportSection("Per-instance model differences", tables=[disagreement_label])
        )
        if comparison_items:
            sections.append(
                ReportSection("Per-instance answer comparison", tables=comparison_items)
            )
    sections += [
        ReportSection("Model settings", tables=["tab:model_settings"]),
        ReportSection("Numeric field distributions", figures=figures),
        ReportSection("Errors by category", tables=error_items),
    ]
    _write_combined_document(sections)


if __name__ == "__main__":
    get_aggregated_result_main()
