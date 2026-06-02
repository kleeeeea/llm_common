# implement calling from a jsonl file with prompts stored in 'prompt' field
# default to save with a '_called' suffix, and 'reasoning' and 'llm_response' field
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import dataclasses
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_32B_OFFICIAL_API
from llm_common.llm_infer.api_info.dataclass_ import GEMINI_API
from llm_common.llm_infer.api_info.dataclass_ import MODEL_TO_APICONFIG
from llm_common.llm_infer.api_info.dataclass_ import apiconfig_for_model
from llm_common.llm_infer.call import ModelSettings
from llm_common.llm_infer.call import call_openai
from llm_common.llm_infer.instances import LLMInferInput
from llm_common.llm_infer.instances import LLMInferOutput
from llm_common.llm_infer.instances import load_system_input  # noqa: F401  re-exported
from llm_common.llm_infer.instances import response_record_to_prompt_messages  # noqa: F401  re-exported
from llm_common.llm_infer.test.data.index import LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH


# ---------------------------------------------------------------------------
# Output schema + adapters
# ---------------------------------------------------------------------------
#
# batch_call writes two parallel artifacts next to each other, plus a settings
# sidecar, under `<input_dir>_batch_infer_<model>/`:
#
#   prompts.jsonl          one JSON object per processed row (append-only; the
#                          resume log — re-running skips ids already present)
#   prompts.csv            the same data as a table (input columns + llm_response)
#   model_settings.json    asdict(ModelSettings) — notably `system_input`, which
#                          is the system prompt shared by every row (NOT stored
#                          per-record, so reconstruct prompts via this sidecar)
#
# A record is the original input row carried through verbatim plus `llm_response`.


# Columns the input CSV must provide; everything else is optional metadata that
# is carried through to the output untouched.



# default to write the output to the with same filebasename under sibiling parent directory under the suffix '_batch_infer'

# Output-only fields (LLMInferOutput minus LLMInferInput), computed once at
# import time so the resume loop is never hardcoded.
_INPUT_FIELD_NAMES: frozenset[str] = frozenset(
    f.name for f in dc_fields(LLMInferInput)
)
_OUTPUT_ONLY_FIELDS = [
    f for f in dc_fields(LLMInferOutput) if f.name not in _INPUT_FIELD_NAMES
]


def cached_batch_call_file(
        csv_path: Path = LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH,
        apiconfig = DEFAULT_32B_OFFICIAL_API,
        max_workers: int = 1,
        output_path: Optional[Path] = None,
) -> Path:
    csv_path = Path(csv_path)
    rows: list[Dict[str, Any]] = pd.read_csv(csv_path).to_dict("records")

    model_settings = ModelSettings(
            api=apiconfig,
            max_tokens=12000,
            system_input='Reasoning effort should be low. Maxmium tokens for reasoning or thinking can not be more than 100 tokens.',
            disable_maxtoken_hint=1,
    )

    # Convert every row to LLMInferInput up-front: schema validation (required
    # columns, non-empty prompt) fires immediately for all rows before any LLM
    # calls are made.
    inputs: list[LLMInferInput] = [
        LLMInferInput.from_dict(row, model_settings=model_settings) for row in rows
    ]
    ids = [str(r["id"]) for r in rows]

    if output_path is None:
        output_dir = csv_path.parent.parent / (
            csv_path.parent.name + "_batch_infer_" + model_settings.model
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / csv_path.name

    # One LLMInferOutput per row; None until the row is processed or resumed.
    outputs: list[Optional[LLMInferOutput]] = [None] * len(inputs)

    output_jsonl_path = output_path.with_suffix(".jsonl")

    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    done_ids: set = set()
    if output_jsonl_path.exists():
        for line in output_jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                row_id = str(record["id"])
                done_ids.add(row_id)
                if row_id in id_to_index:
                    idx = id_to_index[row_id]
                    field_vals: Dict[str, Any] = {}
                    for f in _OUTPUT_ONLY_FIELDS:
                        val = record.get(f.name)
                        if val is None and f.default is not dataclasses.MISSING:
                            val = f.default
                        field_vals[f.name] = val
                    outputs[idx] = LLMInferOutput(
                        prompt=inputs[idx].prompt,
                        model_settings=model_settings,
                        extra=inputs[idx].extra,
                        **field_vals,
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor, output_jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        futures = {
            executor.submit(call_openai, input_=inputs[row_index]): (row_index, row_id)
            for row_index, row_id in enumerate(ids)
            if row_id not in done_ids
        }
        for future in as_completed(futures):
            row_index, row_id = futures[future]
            outputs[row_index] = future.result()
            record = outputs[row_index].to_dict()
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()
    # preserve original CSV row order (outputs[row_index] is indexed by input position)
    output_rows = [
        outputs[i].to_dict() if outputs[i] is not None else rows[i]
        for i in range(len(rows))
    ]
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    print(f"saved to {output_path}")
    return output_path

def cached_batch_call_directory(
        directory: Path = LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH.parent,
        apiconfig=DEFAULT_32B_OFFICIAL_API,
        max_workers: int = 1,
) -> list[Path]:
    """Process every ``*.csv`` in *directory*, writing results to a sibling
    output directory named ``<directory>_batch_infer_<model>``.

    Returns a list of output CSV paths (one per input file).
    """
    directory = Path(directory)
    output_dir = directory.parent / (directory.name + "_batch_infer_" + apiconfig.model)
    results: list[Path] = []
    for csv_path in sorted(directory.glob("*.csv")):
        results.append(
            cached_batch_call_file(
                csv_path=csv_path,
                apiconfig=apiconfig,
                max_workers=max_workers,
                output_path=output_dir / csv_path.name,
            )
        )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch call an LLM over prompts in a CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH,
        help="CSV with a 'prompt' and 'id' column.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_TO_APICONFIG),
        default=GEMINI_API.model,
        help="Model name to select the api config from MODEL_TO_APICONFIG.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    apiconfig = apiconfig_for_model(args.model)
    csv_path = Path(args.csv_path)
    output_dir = csv_path.parent.parent / (
        csv_path.parent.name + "_batch_infer_" + apiconfig.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_batch_call_file(
        csv_path=csv_path,
        apiconfig=apiconfig,
        max_workers=args.max_workers,
        output_path=output_dir / csv_path.name,
    )


if __name__ == "__main__":
    main()
