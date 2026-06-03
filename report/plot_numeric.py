"""Nature-style univariate + multivariate plots of the numeric output fields.

Plots the four per-row numeric measurements produced by inference —
``prompt_tokens``, ``completion_tokens``, ``total_tokens``, ``latency_ms`` —
as publication-quality figures:

- univariate: a 2x2 panel of distributions (histogram + KDE), one per field.
- multivariate: a pairwise scatter matrix (correlations + KDE diagonal).

Figures are written as vector PDFs under ``latex/figures/`` so they embed
crisply in the LaTeX report.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import pandas as pd

NUMERIC_FIELDS = ["prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"]
FIELD_LABELS = {
    "prompt_tokens"    : "Prompt tokens",
    "completion_tokens": "Completion tokens",
    "total_tokens"     : "Total tokens",
    "latency_ms"       : "Latency (ms)",
}
# Colour-blind-safe palette (Nature-ish), one colour per field.
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Conditioning variable: split distributions by whether the answer was correct.
GROUP_FIELD = "correct"
# (value, legend label, colour) — drawn in this order.
GROUPS = [
    (False, "Incorrect", "#C44E52"),  # red
    (True,  "Correct",   "#55A868"),  # green
]

# Single-/double-column widths in inches (Nature: ~89mm / ~183mm).
SINGLE_COL = 3.5
DOUBLE_COL = 7.0


def _apply_nature_style() -> None:
    """Set Matplotlib rcParams to a clean, Nature-like publication style."""
    matplotlib.rcParams.update({
        "figure.dpi"        : 300,
        "savefig.dpi"       : 300,
        "savefig.bbox"      : "tight",
        "savefig.pad_inches": 0.02,
        "font.family"       : "sans-serif",
        "font.sans-serif"   : ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size"         : 8,
        "axes.labelsize"    : 8,
        "axes.titlesize"    : 8,
        "axes.titleweight"  : "bold",
        "xtick.labelsize"   : 7,
        "ytick.labelsize"   : 7,
        "legend.fontsize"   : 7,
        "legend.frameon"    : False,
        "axes.linewidth"    : 0.6,
        "xtick.major.width" : 0.6,
        "ytick.major.width" : 0.6,
        "xtick.direction"   : "out",
        "ytick.direction"   : "out",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "lines.linewidth"   : 1.0,
    })


def numeric_frame(reports: list) -> pd.DataFrame:
    """Build a DataFrame of the four numeric fields plus the ``correct`` flag."""
    data = {f: [] for f in NUMERIC_FIELDS}
    data[GROUP_FIELD] = []
    for r in reports:
        for f in NUMERIC_FIELDS:
            data[f].append(getattr(r, f, None))
        data[GROUP_FIELD].append(bool(getattr(r, GROUP_FIELD, False)))
    df = pd.DataFrame(data)
    df[NUMERIC_FIELDS] = df[NUMERIC_FIELDS].apply(pd.to_numeric, errors="coerce")
    return df


def plot_univariate(df: pd.DataFrame, out_dir: Path) -> Path:
    """2x2 panel of per-field distributions, conditioned on ``correct``.

    Each panel overlays the distribution for the correct vs incorrect rows
    (histogram + KDE + rug). Returns the PDF path.
    """
    _apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    panel = "abcd"
    for i, field in enumerate(NUMERIC_FIELDS):
        ax = axes.flat[i]
        # Pad the x-range a little so KDE tails and rugs aren't clipped.
        col = df[field].dropna()
        lo, hi = (float(col.min()), float(col.max())) if not col.empty else (0.0, 1.0)
        pad = (hi - lo) * 0.08 or 1.0
        ymax = 0.0
        for k, (val, glabel, color) in enumerate(GROUPS):
            series = df.loc[df[GROUP_FIELD] == val, field].dropna()
            if series.empty:
                continue
            # Filled KDE (avoids misleading big histogram blocks at small n).
            if series.nunique() > 1:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(series)
                    xs = _linspace(lo - pad, hi + pad, 200)
                    ys = kde(xs)
                    ymax = max(ymax, float(max(ys)))
                    ax.fill_between(xs, ys, color=color, alpha=0.22, linewidth=0)
                    ax.plot(xs, ys, color=color, linewidth=1.3,
                            label=glabel if i == 0 else None)
                except Exception:
                    pass
            if i == 0 and series.nunique() <= 1:
                # ensure the legend still gets a handle for tiny groups
                ax.plot([], [], color=color, linewidth=1.3, label=glabel)
        # Two-colour rug below the axis: incorrect on the lower row, correct above.
        rug_y = {False: -0.04, True: -0.09}
        span = ymax or 1.0
        for val, glabel, color in GROUPS:
            series = df.loc[df[GROUP_FIELD] == val, field].dropna()
            if series.empty:
                continue
            ax.plot(series, [rug_y[val] * span] * len(series), "|",
                    color=color, markersize=7, markeredgewidth=1.0, clip_on=False)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(FIELD_LABELS[field])
        ax.set_ylabel("Density")
        ax.set_yticks([])
        ax.text(-0.08, 1.05, panel[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="right")
    # one shared legend (correct vs incorrect) for the whole figure.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.02), title="Answer")
    fig.tight_layout(w_pad=2.0, h_pad=2.0, rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "numeric_univariate.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multivariate(df: pd.DataFrame, out_dir: Path) -> Path:
    """Pairwise scatter matrix (KDE diagonal, Pearson r upper triangle)."""
    import numpy as np

    _apply_nature_style()
    cols = NUMERIC_FIELDS
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(DOUBLE_COL, DOUBLE_COL))

    for i, yf in enumerate(cols):
        for j, xf in enumerate(cols):
            ax = axes[i, j]
            x = df[xf]
            y = df[yf]
            if i == j:
                # diagonal: distribution per group (correct vs incorrect).
                for val, glabel, color in GROUPS:
                    s = df.loc[df[GROUP_FIELD] == val, xf].dropna()
                    if s.empty:
                        continue
                    ax.hist(s, bins=min(8, max(3, len(s))), density=True,
                            color=color, alpha=0.40, edgecolor="white",
                            linewidth=0.4, label=glabel if (i == 0) else None)
                ax.set_yticks([])
            elif i > j:
                # lower triangle: scatter, coloured by group.
                for val, glabel, color in GROUPS:
                    m = df[GROUP_FIELD] == val
                    ax.scatter(x[m], y[m], s=16, color=color, alpha=0.85,
                               edgecolor="white", linewidth=0.3)
            else:
                # upper triangle: Pearson r (overall), shaded by sign/strength.
                mask = x.notna() & y.notna()
                r = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() > 1 else float("nan")
                ax.set_facecolor(plt.cm.RdBu((r + 1) / 2) if r == r else "white")
                ax.text(0.5, 0.5, "" if r != r else f"r = {r:.2f}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, fontweight="bold")
                ax.set_xticks([]); ax.set_yticks([])
            # axis labels only on the outer edges.
            if i == n - 1:
                ax.set_xlabel(FIELD_LABELS[xf], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(FIELD_LABELS[yf], fontsize=7)
            else:
                ax.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.02), title="Answer")
    fig.tight_layout(w_pad=0.5, h_pad=0.5, rect=(0, 0, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "numeric_multivariate.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * k for k in range(n)]


def generate_numeric_plots(reports: list, out_dir: Path | str) -> List[Path]:
    """Build both the univariate and multivariate figures. Returns their paths."""
    out_dir = Path(out_dir)
    df = numeric_frame(reports)
    return [plot_univariate(df, out_dir), plot_multivariate(df, out_dir)]
