import base64
import json
import os
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import read_env_file

api_innospark_cn_v_ = "https://api.innospark.cn/v1"

MODEL_SETTINGS_FILENAME = "model_settings.json"
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
    """Read ``system_input`` for a batch-call run directory.

    Checks two sources in order:

    1. ``model_settings.json`` sidecar (legacy / explicit write).
    2. The ``model_settings.system_input`` field embedded in the first valid
       record of any ``*.jsonl`` file in the directory (new default — written
       by ``LLMInferOutput.to_dict()`` instead of a separate sidecar).

    Returns an empty string when neither source is available.
    """
    output_dir = Path(output_dir)

    # 1. legacy sidecar
    settings_path = output_dir / MODEL_SETTINGS_FILENAME
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            value = str(settings.get("system_input") or "")
            if value:
                return value
        except (json.JSONDecodeError, OSError):
            pass

    # 2. embedded in first JSONL record
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        try:
            for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                ms = record.get("model_settings")
                if isinstance(ms, dict):
                    value = str(ms.get("system_input") or "")
                    if value:
                        return value
        except (json.JSONDecodeError, OSError):
            continue

    return ""


def response_record_to_prompt_messages(
    record: dict[str, Any], system_input: str | None = None
) -> list[dict[str, str]]:
    """Convenience adapter: dict record -> chat ``messages`` list (see
    :meth:`BatchResponseRecord.prompt_messages`)."""
    return BatchResponseRecord.from_dict(record).prompt_messages(system_input)

def require_positive_number(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")

@dataclass(frozen=True)
class ModelSettings:
    thinking: Dict[str, Any] = field(default_factory=lambda: {"type": "disabled"})
    temperature: float = 0.1
    stream: bool = True
    system_input: Optional[str] = None
    max_tokens: Optional[int] = None
    disable_maxtoken_hint: bool = False
    # Pass an ApiConfig in `api` to populate api_key / base_url / model in one
    # shot. Explicit values for those fields still win — they're used as
    # per-field overrides on top of the ApiConfig.
    api: Optional[ApiConfig] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = None

    def __post_init__(self):
        if self.api is not None:
            if self.api_key is None:
                object.__setattr__(self, 'api_key', self.api.api_key)
            if self.base_url is None:
                object.__setattr__(self, 'base_url', self.api.base_url)
            if self.model is None:
                object.__setattr__(self, 'model', self.api.model)

        local_env = read_env_file(ENV_FILE)
        local_env.update(os.environ)
        api_key = (self.api_key or local_env.get("LLM_API_KEY", "")).strip()
        base_url = (self.base_url or local_env.get("LLM_BASE_URL", api_innospark_cn_v_)).strip().rstrip("/")
        model = (self.model or local_env.get("LLM_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash"
        timeout = self.timeout if self.timeout is not None else float(local_env.get("TIMEOUT_SECONDS", "60"))
        require_positive_number("timeout", timeout)
        is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not api_key and not is_local:
            print(f"ERROR: LLM_API_KEY is empty. Set it in {ENV_FILE} or export LLM_API_KEY.", file=sys.stderr)
            sys.exit(2)
        max_tokens = self.max_tokens if self.max_tokens is not None else int(local_env.get("MAX_TOKENS", "12000"))
        require_positive_number("max_tokens", max_tokens)
        system_input = (self.system_input or local_env.get("SYSTEM_INPUT", "你是测试助手。回答必须按照字数要求。")).strip()
        # Idempotent: `replace()` (used in call_openai) re-runs __post_init__, so
        # only append the hint if it isn't already present.
        hint_marker = "Total token budget including reasoning is:"
        if not self.disable_maxtoken_hint and hint_marker not in system_input:
            system_input += f"\n{hint_marker} {max_tokens}. Reasoning budget can not be more than {int(max_tokens * 0.8)} tokens"

        # Frozen dataclass — write resolved values back in place (no `replace`,
        # which would recurse through __post_init__).
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "system_input", system_input)


def image_path_to_data_url(image_path: Any) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_type = {
            ".jpg" : "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png" : "image/png",
            ".gif" : "image/gif",
            ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    image_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"



@dataclass(frozen=True)
class LLMInferInput(object):
    prompt: str
    image_paths: Optional[Sequence[Any]] = None
    image_data_urls: Optional[Sequence[str]] = None
    model_settings: Optional[ModelSettings] = None
    # Full original input row (id + all extra CSV columns) carried through so
    # LLMInferOutput.to_dict() can produce a self-contained record without any
    # external dict merge.
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        ms = (self.model_settings if self.model_settings is not None else ModelSettings())

        image_data_urls = [u.strip() for u in (self.image_data_urls or ()) if u.strip()]
        image_data_urls.extend(image_path_to_data_url(p) for p in (self.image_paths or ()))
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))
        object.__setattr__(self, "model_settings", ms)
        if not self.prompt:
            raise ValueError("prompt is empty")
        if not ms.system_input:
            raise ValueError("system_input is empty")

    @classmethod
    def from_dict(
        cls,
        row: Dict[str, Any],
        model_settings: Optional["ModelSettings"] = None,
        image_paths: Optional[Sequence[Any]] = None,
        image_data_urls: Optional[Sequence[str]] = None,
    ) -> "LLMInferInput":
        """Construct from a CSV row dict, validating required fields.

        Raises ``ValueError`` if any column in ``REQUIRED_INPUT_COLUMNS``
        (``id``, ``prompt``) is missing from *row* — this is the canonical
        place for schema validation so callers don't need to repeat it.
        """
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in row]
        if missing:
            raise ValueError(
                f"row is missing required field(s): {missing}; "
                f"got {list(row.keys())}"
            )
        return cls(
            prompt=str(row["prompt"]),
            model_settings=model_settings,
            image_paths=image_paths,
            image_data_urls=image_data_urls,
            extra=dict(row),
        )


@dataclass(frozen=True)
class LLMInferOutput(LLMInferInput):
    """Extends LLMInferInput with the model's response and runtime stats.

    Serialises to a self-contained dict (via ``to_dict``) that embeds
    ``model_settings`` inline — no separate ``model_settings.json`` sidecar
    is needed.  ``load_system_input`` falls back to reading this embedded
    field when the sidecar is absent.
    """

    llm_response: str = ""
    reasoning: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        ms = self.model_settings
        # Start with the original input row (id + all extra CSV columns) so
        # the output record is self-contained without any external dict merge.
        d: Dict[str, Any] = dict(self.extra)
        d.update({
            "llm_response"     : self.llm_response,
            "reasoning"        : self.reasoning,
            "prompt_tokens"    : self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens"     : self.total_tokens,
            "latency_ms"       : self.latency_ms,
        })
        if ms is not None:
            from dataclasses import asdict as _asdict
            d["model_settings"] = _asdict(ms)
        return d


