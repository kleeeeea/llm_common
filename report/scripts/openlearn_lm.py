# adapt to open learn lm
# the backfill script for open learn lm is accidentally deleted, restored it
from llm_common.llm_infer.scripts.backfill_openlearnlm import main as backfill_openlearnlm_main
from llm_common.report.get_aggregated_result import get_aggregated_result_main


def main() -> None:
    """Backfill all openlearnlm scored CSVs, then build the aggregate report.

    Unlike the praxis pipeline, backfill_openlearnlm already writes fully-scored
    per-row CSVs (gold/pred/correct for mcq, score/judge_reasoning for judge),
    so we feed those scored paths straight into the aggregator.
    """
    scored_paths = backfill_openlearnlm_main()
    get_aggregated_result_main(scored_paths)


if __name__ == "__main__":
    main()
