# implement calling from a jsonl file with prompts stored in 'prompt' field
# default to save with a '_called' suffix, and 'reasoning' and 'llm_response' field
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import pandas as pd

from llm_common.llm_infer.api_info.dataclass_ import DEFAULT_32B_OFFICIAL_API
from llm_common.llm_infer.api_info.dataclass_ import MODEL_TO_APICONFIG
from llm_common.llm_infer.api_info.dataclass_ import apiconfig_for_model
from llm_common.llm_infer.call import ModelSettings
from llm_common.llm_infer.call import call_openai
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

MODEL_SETTINGS_FILENAME = "model_settings.json"

# Columns the input CSV must provide; everything else is optional metadata that
# is carried through to the output untouched.
REQUIRED_INPUT_COLUMNS = ("id", "prompt")


@dataclass
class BatchResponseRecord:
    """Schema of a single batch_call output record (one jsonl line / csv row).

    Required (must come from the input CSV):
        id (str):           stable per-row identifier; also the jsonl resume key.
        prompt (str):       the user prompt sent to the model.
    Added by batch_call:
        llm_response (str): raw model output, possibly wrapping a
                            ``<think>...</think>`` reasoning block.
    extra (dict):
        Any other input columns carried through verbatim. For the
        parse_evaluation reading set these include ``question_number``,
        ``passage``, ``question``, ``answer`` and the screenshot path columns.

    The system prompt is shared across rows and lives in the sibling
    ``model_settings.json`` (``system_input``), not on the record. Use
    :meth:`prompt_messages` to reconstruct the chat-style prompt for display.
    """

    id: str
    prompt: str
    llm_response: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = ("id", "prompt", "llm_response")

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "BatchResponseRecord":
        return cls(
            id=str(record.get("id", "")),
            prompt=str(record.get("prompt", "") or ""),
            llm_response=str(record.get("llm_response", "") or ""),
            extra={k: v for k, v in record.items() if k not in cls._KNOWN_FIELDS},
        )

    def prompt_messages(self, system_input: str | None = None) -> list[dict[str, str]]:
        """Adapter: render the prompt as a chat ``messages`` list.

        Returns ``[{"role": "system", ...}, {"role": "user", ...}]`` (the system
        turn is dropped when no system prompt is available). This is exactly the
        shape the Streamlit report's ``render_record_messages`` consumes, so
        downstream visualizations can show the full prompt instead of falling
        back to a bare question string.
        """
        messages: list[dict[str, str]] = []
        if system_input:
            messages.append({"role": "system", "content": str(system_input)})
        messages.append({"role": "user", "content": self.prompt})
        return messages


def load_system_input(output_dir: str | Path) -> str:
    """Read ``system_input`` from the ``model_settings.json`` sidecar in a run dir.

    Returns an empty string when the sidecar is missing or has no system prompt.
    """
    settings_path = Path(output_dir) / MODEL_SETTINGS_FILENAME
    if not settings_path.exists():
        return ""
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return str(settings.get("system_input") or "")


def response_record_to_prompt_messages(
    record: dict[str, Any], system_input: str | None = None
) -> list[dict[str, str]]:
    """Convenience adapter: dict record -> chat ``messages`` list (see
    :meth:`BatchResponseRecord.prompt_messages`)."""
    return BatchResponseRecord.from_dict(record).prompt_messages(system_input)


# default to write the output to the with same filebasename under sibiling parent directory under the suffix '_batch_infer'

def batch_call(
        csv_path: Path = LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH,
        apiconfig = DEFAULT_32B_OFFICIAL_API,
        max_workers: int = 1) -> Path:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"input CSV {csv_path} is missing required column(s): {missing}; "
            f"got {list(df.columns)}"
        )

    model_settings = ModelSettings(
            api=apiconfig,
            max_tokens=12000,
            system_input='Reasoning effort should be low. Maxmium tokens for reasoning or thinking can not be more than 100 tokens.',
            disable_maxtoken_hint=1,
    )

    output_dir = csv_path.parent.parent / (csv_path.parent.name + "_batch_infer_" + model_settings.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / csv_path.name

    prompts = [str(p) for p in df["prompt"]]
    ids = list(df["id"])
    llm_responses = [None] * len(prompts)
    prompt_tokens = [None] * len(prompts)
    completion_tokens = [None] * len(prompts)
    total_tokens = [None] * len(prompts)
    latency_ms = [None] * len(prompts)

    def timed_call(prompt: str):
        start = time.perf_counter()
        response = call_openai(prompt=prompt, model_settings=model_settings)
        return response, (time.perf_counter() - start) * 1000.0

    (output_dir / MODEL_SETTINGS_FILENAME).write_text(
        json.dumps(asdict(model_settings), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    jsonl_path = output_path.with_suffix(".jsonl")

    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    done_ids: set = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                row_id = record["id"]
                done_ids.add(row_id)
                if row_id in id_to_index:
                    idx = id_to_index[row_id]
                    llm_responses[idx] = record["llm_response"]
                    prompt_tokens[idx] = record.get("prompt_tokens")
                    completion_tokens[idx] = record.get("completion_tokens")
                    total_tokens[idx] = record.get("total_tokens")
                    latency_ms[idx] = record.get("latency_ms")
            except (json.JSONDecodeError, KeyError):
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor, jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        futures = {
            executor.submit(timed_call, p): (i, row_id)
            for i, (p, row_id) in enumerate(zip(prompts, ids))
            if row_id not in done_ids
        }
        for future in as_completed(futures):
            i, row_id = futures[future]
            response, latency_ms[i] = future.result()
            llm_responses[i] = response
            # call_openai returns a str subclass carrying token counts in `.usage`
            # (None when the server didn't report them).
            usage = getattr(response, "usage", None) or {}
            prompt_tokens[i] = usage.get("prompt_tokens")
            completion_tokens[i] = usage.get("completion_tokens")
            total_tokens[i] = usage.get("total_tokens")
            record = {
                **df.iloc[i].to_dict(),
                "llm_response": str(response),
                "prompt_tokens": prompt_tokens[i],
                "completion_tokens": completion_tokens[i],
                "total_tokens": total_tokens[i],
                "latency_ms": latency_ms[i],
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    df["llm_response"] = llm_responses
    df["prompt_tokens"] = prompt_tokens
    df["completion_tokens"] = completion_tokens
    df["total_tokens"] = total_tokens
    df["latency_ms"] = latency_ms
    df.to_csv(output_path, index=False)
    print(f"saved to {output_path}")
    return output_path


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
        default=DEFAULT_32B_OFFICIAL_API.model,
        help="Model name to select the api config from MODEL_TO_APICONFIG.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    batch_call(
        csv_path=args.csv_path,
        apiconfig=apiconfig_for_model(args.model),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
