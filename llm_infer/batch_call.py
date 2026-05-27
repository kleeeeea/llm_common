# implement calling from a jsonl file with prompts stored in 'prompt' field
# default to save with a '_called' suffix, and 'reasoning' and 'llm_response' field
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from llm_common.llm_infer.api_info.dataclass_ import GEMINI_API
from llm_common.llm_infer.call import ModelSettings, call_openai
from llm_common.llm_infer.test.data.index import LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH

# default to write the output to the with same filebasename under sibiling parent directory under the suffix '_batch_infer'


def batch_call(csv_path: Path = LLM_INFER_TEST_SAMPLE_BATCH_PROMPT_CSV_PATH, max_workers: int = 4) -> Path:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    model_settings = ModelSettings(api_key=GEMINI_API.api_key, base_url=GEMINI_API.base_url, model=GEMINI_API.model)

    output_dir = csv_path.parent.parent / (csv_path.parent.name + "_batch_infer_" + model_settings.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / csv_path.name

    prompts = [str(p) for p in df["prompt"]]
    ids = list(df["id"])
    llm_responses = [None] * len(prompts)

    (output_dir / "model_settings.json").write_text(
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
                    llm_responses[id_to_index[row_id]] = record["llm_response"]
            except (json.JSONDecodeError, KeyError):
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor, jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        futures = {
            executor.submit(call_openai, prompt=p, model_settings=model_settings): (i, row_id)
            for i, (p, row_id) in enumerate(zip(prompts, ids))
            if row_id not in done_ids
        }
        for future in as_completed(futures):
            i, row_id = futures[future]
            response = future.result()
            llm_responses[i] = response
            jsonl_file.write(json.dumps({**df.iloc[i].to_dict(), "llm_response": response}, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    df["llm_response"] = llm_responses
    df.to_csv(output_path, index=False)
    print(f"saved to {output_path}")
    return output_path


if __name__ == "__main__":
    batch_call()
