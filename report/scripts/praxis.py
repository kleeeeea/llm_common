from llm_common.llm_infer.scripts.backfill_praxis import backfill_praxis_main
from llm_common.report.get_aggregated_result import get_aggregated_result_main
from llm_common.report.get_per_row_result import get_scored_file

def get_scored_file_praxis_main() -> None:
    """Backfill the praxis runs, then score each resulting output CSV."""
    get_aggregated_result_main(
            [get_scored_file(output_csv) for output_csv in backfill_praxis_main()]
    )



if __name__ == "__main__":
    get_scored_file_praxis_main()
