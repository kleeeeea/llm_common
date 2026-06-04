import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from llm_common.llm_infer.instances import LLMInferOutput


class JudgeType(StrEnum):
    """How a row was scored.

    A ``StrEnum`` so members *are* their string value — they compare equal to
    the raw strings and serialise to plain text in CSV/JSON without special
    handling.
    """

    # only these 2 , in fact
    MCQ            = "mcq"             # objective letter match
    LLM_JUDGE      = "llm_judge"      # subjective, generic LLM judge

    # RULE           = "rule"           # objective rule-based check
    # ATTITUDE_JUDGE = "attitude_judge"  # subjective, attitude-specific judge

    @classmethod
    def coerce(cls, value) -> "JudgeType":
        """Best-effort conversion of an arbitrary value to a JudgeType.

        Unknown values are bucketed by name: anything containing ``judge`` is
        treated as a subjective LLM judge, otherwise objective ``mcq`` — so the
        field is always a valid enum member (back-compat: ``None`` -> ``mcq``).
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.MCQ
        s = str(value).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return cls.LLM_JUDGE if "judge" in s else cls.MCQ


@dataclass(frozen=True)
class LLMInferPerRowReport(LLMInferOutput):
    """LLMInferOutput augmented with per-row MCQ scoring columns.

    Constructed via ``from_output`` so callers don't need to replicate the
    field-copying boilerplate; serialised via the inherited ``to_dict()``
    which already inlines ``extra`` + all output fields.
    """

    # objective (mcq / rule) scoring
    gold   : str | None = None
    pred   : str | None = None
    correct: bool       = False

    # subjective (llm_judge) scoring
    score          : float | None = None  # numeric judge score
    judge_reasoning: str | None   = None  # the judge's explanation

    # Constrained to the JudgeType enum; defaults to MCQ for backward
    # compatibility with older scored files that predate this field.
    judge_type: JudgeType = JudgeType.MCQ

    def __post_init__(self) -> None:
        # Skip model_settings / system_input validation — reports are built
        # from already-completed inference outputs, not live inference configs.
        image_data_urls = [u.strip() for u in (self.image_data_urls or ()) if u.strip()]
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))
        # Normalise judge_type (callers pass plain strings from CSV/JSON).
        object.__setattr__(self, "judge_type", JudgeType.coerce(self.judge_type))

    @classmethod
    def from_output(
            cls,
            out: LLMInferOutput,
            gold: str | None = None,
            pred: str | None = None,
            *,
            score: float | None = None,
            judge_reasoning: str | None = None,
            judge_type: "JudgeType | str" = JudgeType.MCQ,
    ) -> "LLMInferPerRowReport":
        """Create a scored report from an existing ``LLMInferOutput``.

        Objective (mcq/rule): pass ``gold`` / ``pred`` — ``correct`` is derived.
        Subjective (llm_judge): pass ``score`` / ``judge_reasoning`` and
        ``judge_type="llm_judge"``.
        """
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
            score             = score,
            judge_reasoning   = judge_reasoning,
            judge_type        = judge_type,
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["gold"]            = self.gold
        d["pred"]            = self.pred
        d["correct"]        = self.correct
        d["score"]           = self.score
        d["judge_reasoning"] = self.judge_reasoning
        d["judge_type"]      = str(self.judge_type)  # plain str for CSV/JSON
        return d

    @classmethod
    def from_dict(cls, row, model_settings=None, image_paths=None, image_data_urls=None):
        """Round-trip a scored CSV/JSONL record, restoring gold/pred/correct.

        Layers the scoring columns on top of ``LLMInferOutput.from_dict`` so the
        inherited ``from_csv`` / ``from_jsonl`` produce fully-typed reports.
        """
        base = LLMInferOutput.from_dict(
            row, model_settings=model_settings,
            image_paths=image_paths, image_data_urls=image_data_urls,
        )
        e = base.extra  # already NaN-sanitised by LLMInferOutput.from_dict
        return cls(
            id                = base.id,
            prompt            = base.prompt,
            image_data_urls   = base.image_data_urls,
            extra             = e,
            llm_response      = base.llm_response,
            reasoning         = base.reasoning,
            prompt_tokens     = base.prompt_tokens,
            completion_tokens = base.completion_tokens,
            total_tokens      = base.total_tokens,
            latency_ms        = base.latency_ms,
            gold              = e.get("gold"),
            pred              = e.get("pred"),
            correct           = bool(e.get("correct")),
            score             = e.get("score"),
            judge_reasoning   = e.get("judge_reasoning"),
            # back-compat: rows without judge_type are objective MCQ.
            judge_type        = e.get("judge_type") or "mcq",
        )

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
