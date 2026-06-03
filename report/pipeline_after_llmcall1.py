import os
import re

import pandas as pd

from dataclasses import dataclass

from llm_common.llm_infer.instances import LLMInferOutput


@dataclass(frozen=True)
class EvaluationInstance(LLMInferOutput):
    """LLMInferOutput extended with MCQ scoring columns.

    Built from a scored CSV row via ``from_dict``; the three scoring fields
    (``gold``, ``pred``, ``correct``) are typed so they are schema-enforced
    rather than left as untyped keys in ``extra``.
    """

    gold: str | None = None
    pred: str | None = None
    correct: bool = False

    def __post_init__(self) -> None:
        # Minimal post-init: normalise image_data_urls but skip the
        # model_settings / system_input validation that LLMInferInput enforces
        # — EvaluationInstance is built from a scored CSV, not a live config.
        image_data_urls = [u.strip() for u in (self.image_data_urls or ()) if u.strip()]
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))

    @classmethod
    def from_dict(cls, row: dict) -> "EvaluationInstance":
        """Construct from a scored CSV row, enforcing the output schema.

        NaN values (pandas missing cells) are normalised to ``None`` so the
        resulting ``to_dict()`` serialises cleanly to valid JSON / CSV.
        """
        import math
        sanitised = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
        }
        return cls(
                prompt=str(sanitised.get("prompt", "") or ""),
                llm_response=str(sanitised.get("llm_response", "") or ""),
                reasoning=sanitised.get("reasoning") or None,
                prompt_tokens=sanitised.get("prompt_tokens"),
                completion_tokens=sanitised.get("completion_tokens"),
                total_tokens=sanitised.get("total_tokens"),
                latency_ms=sanitised.get("latency_ms"),
                gold=sanitised.get("gold"),
                pred=sanitised.get("pred"),
                correct=bool(sanitised.get("correct", False)),
                extra=sanitised,
        )

    def to_dict(self) -> dict:
        """Extend LLMInferOutput.to_dict() with the scoring columns."""
        d = super().to_dict()
        d["gold"] = self.gold
        d["pred"] = self.pred
        d["correct"] = self.correct
        return d


def gold_letter(answer: str) -> str | None:
    # gold answers look like "1. d. <explanation>" — the choice letter follows the number.
    m = re.match(r"\s*\d+\.\s*([a-eA-E])\b", str(answer))
    return m.group(1).upper() if m else None


def pred_letter(llm_response: str) -> str | None:
    """Extract the predicted choice letter from the model response.

    Handles two formats:
    - Thinking models: reasoning wrapped in ``<think>...</think>``; the choice
      letter is the first token after the closing tag.
    - Direct-answer models (e.g. gemini-2.5-flash): the response is just the
      letter, optionally followed by punctuation or whitespace.
    """
    text = str(llm_response).strip()
    if "</think>" in text:
        post = text.split("</think>")[-1].strip()
    else:
        post = text
    m = re.match(r"([a-eA-E])\b", post)
    return m.group(1).upper() if m else None


SAMPLE_LLMOUTPUT = '/Users/l/klee_code/git_repos/llm_evals/parse_evaluation/praxis_reading_1/outputs/prompts_sample_8_batch_infer_gemini-2.5-flash.csv'


def get_scored_file(llm_response_file: str) -> None:
    df = pd.read_csv(llm_response_file)
    df["gold"] = df["answer"].map(gold_letter)
    df["pred"] = df["llm_response"].map(pred_letter)
    df["correct"] = df["gold"] == df["pred"]

    # Enforce output schema: convert every row through EvaluationInstance so
    # typed fields (gold / pred / correct) and NaN normalisation are applied,
    # then rebuild the DataFrame from the serialised dicts.
    scored_rows = [EvaluationInstance.from_dict(row).to_dict()
                   for row in df.to_dict("records")]
    scored_path = _get_scored_file_path(llm_response_file)
    pd.DataFrame(scored_rows).to_csv(scored_path, index=False)

    n = len(df)
    n_answered = int(df["pred"].notna().sum())
    n_correct = int(df["correct"].sum())
    print(f"saved to {scored_path}")
    print(f"  accuracy        : {n_correct}/{n} = {n_correct / n:.2%}")
    if n_answered:
        print(f"  answered        : {n_answered}/{n}")
        print(f"  acc (answered)  : {n_correct}/{n_answered} = {n_correct / n_answered:.2%}")


def _get_scored_file_path(llm_response_file: str) -> str:
    return llm_response_file.replace(".csv", "_scored.csv")


if __name__ == "__main__":
    get_scored_file(SAMPLE_LLMOUTPUT)
