import json
from pathlib import Path

from llm_common.report.get_per_row_result import LLMInferPerRowReport

RESPONSES_ROOT = Path(
    "/Users/l/klee_code/git_repos/llm_evals/openlearnlm-benchmark-17D4/outputs/responses"
)
# Models to backfill (one jsonl per model per category dir). Only items answered
# by *every* model in this list are kept, so the scored CSVs align across models.
MODELS = [
    "kimi",
    "Kimi-K2.6",
]

# Apply to all categories: every sub-directory under RESPONSES_ROOT.
CATEGORIES = sorted(
    p.name for p in RESPONSES_ROOT.iterdir() if p.is_dir()
) if RESPONSES_ROOT.is_dir() else []


def _item_ids(jsonl: Path) -> set:
    """The set of item_ids present in a response jsonl."""
    ids: set = set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line).get("item_id"))
        except json.JSONDecodeError:
            pass
    ids.discard(None)
    return ids


def _common_item_ids(category: str) -> set:
    """item_ids answered by *every* model in MODELS for *category*.

    If any model is missing its file, there is no common set (empty), since we
    only want instances where all models responded.
    """
    category_dir = RESPONSES_ROOT / category
    per_model: list[set] = []
    for model in MODELS:
        f = category_dir / f"{model}.jsonl"
        if not f.exists():
            return set()
        per_model.append(_item_ids(f))
    return set.intersection(*per_model) if per_model else set()


def _user_prompt(messages: list) -> str:
    """Join the non-system message contents into a single user prompt."""
    return "\n\n".join(
        str(m.get("content", "")) for m in (messages or [])
        if m.get("role") != "system"
    ).strip()


def _to_report(r: dict) -> LLMInferPerRowReport:
    """Map one openlearnlm response record to an LLMInferPerRowReport.

    Objective rows (``check_type`` without ``judge``) keep the letter
    ``gold``/``pred``/``correct``; subjective rows (``*_judge``) carry the
    numeric ``score`` and the judge's ``judge_reasoning`` instead.
    """
    usage = r.get("usage") or {}
    thinking = r.get("thinking_content")
    item_id = r.get("item_id")
    prompt = _user_prompt(r.get("messages")) or str(r.get("question", ""))
    check = r.get("check_result") or {}
    judge_type = check.get("check_type") or "mcq"
    # Any *_judge check is subjective (score-based): llm_judge, attitude_judge, …
    is_judge = "judge" in judge_type

    # `extra` is what to_dict() inlines into the row; include `prompt` so it
    # round-trips (to_dict doesn't serialise the prompt field separately).
    extra = {
        "id"      : item_id,
        "prompt"  : prompt,
        "question": r.get("question", ""),
        "answer"  : r.get("expected_answer", ""),
    }
    # Judge-specific extra dimensions (e.g. attitude_judge adds 'dimension' /
    # 'attitude_category' / 'rubric_source') go into a single nested ``extra``
    # field rather than flat top-level columns — so every category's scored CSV
    # keeps the same column structure.
    judge_extra = {
        k: check[k]
        for k in ("dimension", "attitude_category", "rubric_source")
        if k in check
    }
    if judge_extra:
        extra["extra"] = judge_extra

    return LLMInferPerRowReport(
        id                = str(item_id) if item_id is not None else "",
        prompt            = prompt,
        extra             = extra,
        llm_response      = r.get("raw_content") or "",
        reasoning         = thinking if thinking not in (None, "None", "") else None,
        prompt_tokens     = usage.get("prompt_tokens"),
        completion_tokens = usage.get("completion_tokens"),
        total_tokens      = usage.get("total_tokens"),
        latency_ms        = r.get("latency_ms"),
        # objective letter match (only meaningful for non-judge rows).
        gold              = None if is_judge else r.get("expected_answer"),
        pred              = None if is_judge else r.get("model_answer"),
        correct           = bool(r.get("is_correct")),
        # subjective judge scoring.
        score             = check.get("score") if is_judge else None,
        judge_reasoning   = check.get("reasoning") if is_judge else None,
        judge_type        = judge_type,
    )


def backfill_one(src: Path, allowed_ids: set | None = None) -> Path | None:
    """Convert one openlearnlm response jsonl to a scored per-row report CSV.

    When *allowed_ids* is given, only rows whose ``item_id`` is in that set are
    kept (used to align across models — keep only items all models answered).
    """
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if allowed_ids is not None:
        rows = [r for r in rows if r.get("item_id") in allowed_ids]
    if not rows:
        print(f"no records in {src}")
        return None

    # Convert the openlearnlm responses straight into per-row reports.
    reports = [_to_report(r) for r in rows]

    # Dump to the inferred LLMInferPerRowReport path (…_scored.csv next to source).
    report_path = LLMInferPerRowReport.get_output_path_hint(src.with_suffix(".csv"))
    LLMInferPerRowReport.to_csv(reports, [rep.extra for rep in reports], report_path)

    # Validate by reloading via from_csv.
    loaded = LLMInferPerRowReport.from_csv(report_path)
    n = len(loaded)
    n_judge = sum(1 for r in loaded if "judge" in (r.judge_type or ""))
    if n_judge:
        jt = loaded[0].judge_type
        scores = [r.score for r in loaded if r.score is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{src.parent.name}/{src.name}: {n} rows | {jt} {n_judge}/{n} | "
              f"avg score {avg:.2f} ({len(scores)} scored) -> {report_path.name}")
    else:
        n_correct = sum(1 for r in loaded if r.correct)
        acc = f"{n_correct / n:.2%}" if n else "n/a"
        print(f"{src.parent.name}/{src.name}: {n} rows | mcq | "
              f"correct {n_correct}/{n} = {acc} -> {report_path.name}")
    return report_path


def main() -> list[Path]:
    report_paths: list[Path] = []
    for category in CATEGORIES:
        category_dir = RESPONSES_ROOT / category
        if not category_dir.is_dir():
            continue
        # Only keep items answered by every model, so the scored CSVs align.
        common = _common_item_ids(category)
        print(f"[{category}] {len(common)} item(s) answered by all {len(MODELS)} models")
        for model in MODELS:
            src = category_dir / f"{model}.jsonl"
            if not src.exists():
                continue
            report_path = backfill_one(src, allowed_ids=common)
            if report_path is not None:
                report_paths.append(report_path)
    print(f"\nbackfilled {len(report_paths)} file(s).")
    return report_paths


if __name__ == '__main__':
    main()
