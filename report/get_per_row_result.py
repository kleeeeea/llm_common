import re
from dataclasses import dataclass
from pathlib import Path

from llm_common.llm_infer.instances import LLMInferOutput

@dataclass(frozen=True)
class LLMInferPerRowReport(LLMInferOutput):
    """LLMInferOutput augmented with per-row MCQ scoring columns.

    Constructed via ``from_output`` so callers don't need to replicate the
    field-copying boilerplate; serialised via the inherited ``to_dict()``
    which already inlines ``extra`` + all output fields.
    """

    gold   : str | None = None
    pred   : str | None = None
    correct: bool       = False

    def __post_init__(self) -> None:
        # Skip model_settings / system_input validation — reports are built
        # from already-completed inference outputs, not live inference configs.
        image_data_urls = [u.strip() for u in (self.image_data_urls or ()) if u.strip()]
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))

    @classmethod
    def from_output(
            cls,
            out: LLMInferOutput,
            gold: str | None,
            pred: str | None,
    ) -> "LLMInferPerRowReport":
        """Create a scored report from an existing ``LLMInferOutput``."""
        correct = gold is not None and pred is not None and gold == pred
        return cls(
            id                = out.id,
            prompt            = out.prompt,
            image_data_urls   = out.image_data_urls,
            extra             = out.extra,
            llm_response      = out.llm_response,
            reasoning         = out.reasoning,
            prompt_tokens     = out.prompt_tokens,
            completion_tokens = out.completion_tokens,
            total_tokens      = out.total_tokens,
            latency_ms        = out.latency_ms,
            gold              = gold,
            pred              = pred,
            correct           = correct,
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["gold"]    = self.gold
        d["pred"]    = self.pred
        d["correct"] = self.correct
        return d


    @classmethod
    def get_output_path_hint(cls, input_path: "str | Path") -> Path:
        """Default scored-output path for a raw LLM-output CSV.

        ``…/file.csv`` → ``…/file_scored.csv`` (the scoring columns gold/pred/
        correct are appended to the same rows, so it stays a sibling file).
        """
        p = Path(input_path)
        return p.with_stem(p.stem + "_scored")


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


def get_scored_file(
        llm_response_file: str
) -> None:
    outputs = LLMInferOutput.from_csv(llm_response_file)

    reports: list[LLMInferPerRowReport] = [
        LLMInferPerRowReport.from_output(
            out,
            gold=gold_letter((out.extra or {}).get("answer", "")),
            pred=pred_letter(out.llm_response),
        )
        for out in outputs
    ]

    n          = len(reports)
    n_answered = sum(1 for r in reports if r.pred is not None)
    n_correct  = sum(1 for r in reports if r.correct)
    scored_path = LLMInferPerRowReport.get_output_path_hint(llm_response_file)
    # Inherited from LLMInferOutput: serialises each report via to_dict()
    # (which appends gold/pred/correct). No row is None, so fallback is unused.
    LLMInferPerRowReport.to_csv(reports, [r.extra for r in reports], scored_path)
    print(f"saved to {scored_path}")
    print(f"  accuracy        : {n_correct}/{n} = {n_correct / n:.2%}")
    if n_answered:
        print(f"  answered        : {n_answered}/{n}")
        print(f"  acc (answered)  : {n_correct}/{n_answered} = {n_correct / n_answered:.2%}")


if __name__ == "__main__":
    get_scored_file(SAMPLE_LLMOUTPUT)
