import json
from pathlib import Path

from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_32B_OFFICIAL_API
from llm_common.llm_infer.instances import LLMInferResultRecord
from llm_common.llm_infer.run import get_llm_output_from_file

praxis_reading1_file = '/Users/l/klee_code/git_repos/llm_evals/parse_evaluation/praxis_reading_1/outputs/prompts.csv'


def _ids_in_jsonl(path: Path) -> set:
    ids: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def backfill(output_jsonl: Path, source_jsonl: Path) -> int:
    """Append backfill records for any id not already in *output_jsonl*.

    get_llm_output_from_file reads its sibling ``.jsonl`` as a resume log and
    skips ids already present — so seeding it from a previous run's jsonl makes
    those rows resume instead of triggering fresh LLM calls.
    """
    if not source_jsonl.exists():
        print(f"backfill source not found: {source_jsonl}")
        return 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    have = _ids_in_jsonl(output_jsonl)
    added = 0
    with output_jsonl.open("a", encoding="utf-8") as out:
        for line in source_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rid = str(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
            if rid not in have:
                out.write(line + "\n")
                have.add(rid)
                added += 1
    print(f"backfilled {added} record(s) into {output_jsonl}")
    return added


def backfill_praxis_main() -> list[Path]:
    # Same path get_llm_output_from_file would pick (passed explicitly so the
    # seed and the run agree on the resume jsonl).
    from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_32B_INNOSPARK_API
    # backfill sources live next to the prompts file as outputs_batch_infer_<model>/
    praxis_dir = Path(praxis_reading1_file).parent.parent

    output_paths: list[Path] = []
    for modelapi in [
            DEFAULT_32B_INNOSPARK_API,
            DEFAULT_32B_OFFICIAL_API,
    ]:
        output_path = LLMInferResultRecord.get_output_path_hint(praxis_reading1_file, modelapi.model)
        output_jsonl = Path(output_path).with_suffix(".jsonl")

        # Derive the backfill source dir from the model name.
        data_backfill_source = praxis_dir / f"outputs_batch_infer_{modelapi.model}"
        backfill(output_jsonl, data_backfill_source / "prompts.jsonl")

        output_paths.append(get_llm_output_from_file(
            csv_path=praxis_reading1_file,
            apiconfig=modelapi,
            output_path=output_path,
        ))
    return output_paths


if __name__ == "__main__":
    backfill_praxis_main()
