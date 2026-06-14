import base64
import json
import os
import sys
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

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
class ChatCompletionContentPartText:
    """Source: OpenAI `ChatCompletionContentPartText`.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ChatCompletionImageURL:
    """Source: OpenAI `ChatCompletionContentPartImage.image_url`.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


@dataclass(frozen=True)
class ChatCompletionContentPartImage:
    """Source: OpenAI `ChatCompletionContentPartImage`.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    image_url: ChatCompletionImageURL
    type: Literal["image_url"] = "image_url"


ChatCompletionUserContentPart = Union[
    ChatCompletionContentPartText,
    ChatCompletionContentPartImage,
]
ChatCompletionUserContent = Union[str, tuple[ChatCompletionUserContentPart, ...]]


def build_user_content(
        prompt: Union[str, list],
        image_data_urls: Sequence[str],
) -> ChatCompletionUserContent:
    if not image_data_urls:
        return prompt
    return (
        ChatCompletionContentPartText(text=str(prompt)),
        *(
            ChatCompletionContentPartImage(
                image_url=ChatCompletionImageURL(url=image_data_url),
            )
            for image_data_url in image_data_urls
        ),
    )


@dataclass(frozen=True)
class ChatCompletionMessage:
    """Source: OpenAI Chat Completions `messages` request parameter.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    role: str
    content: ChatCompletionUserContent


@dataclass(frozen=True)
class ChatCompletionStreamOptions:
    """Source: OpenAI Chat Completions `stream_options` request parameter.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    include_usage: bool = True


@dataclass(frozen=True)
class ChatCompletionThinking:
    """Anthropic-compatible extension; not an OpenAI standard field.

    https://docs.anthropic.com/en/api/messages
    """

    type: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    """OpenAI Chat Completions request plus local transport metadata.

    Source:
    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

    ``api``, ``timeout``, ``system_input``, and ``disable_maxtoken_hint`` are
    local runtime metadata and are excluded from :meth:`to_dict`.
    """

    messages: tuple[ChatCompletionMessage, ...] = ()
    temperature: float = 0.1
    stream: bool = True
    max_tokens: Optional[int] = None
    stream_options: Optional[ChatCompletionStreamOptions] = None
    thinking: Optional[ChatCompletionThinking] = None
    chat_template_kwargs: Optional[dict[str, Any]] = None
    enable_thinking: Optional[bool] = None
    api: Optional[ApiConfig] = None
    timeout: Optional[float] = None
    system_input: Optional[str] = None
    disable_maxtoken_hint: bool = False

    def __post_init__(self):
        api = self.api
        local_env = read_env_file(ENV_FILE)
        local_env.update(os.environ)
        api_key = ((api.api_key if api else None) or local_env.get("LLM_API_KEY", "")).strip()
        base_url = ((api.base_url if api else None) or local_env.get("LLM_BASE_URL", api_innospark_cn_v_)).strip().rstrip("/")
        model = ((api.model if api else None) or local_env.get("LLM_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash"
        timeout = self.timeout if self.timeout is not None else float(local_env.get("TIMEOUT_SECONDS", "60"))
        require_positive_number("timeout", timeout)
        is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not api_key and not is_local:
            print(f"ERROR: LLM_API_KEY is empty. Set it in {ENV_FILE} or export LLM_API_KEY.", file=sys.stderr)
            sys.exit(2)
        # max_tokens = self.max_tokens if self.max_tokens is not None else int(local_env.get("MAX_TOKENS", "12000"))
        # require_positive_number("max_tokens", max_tokens)
        # system_input = (self.system_input or local_env.get("SYSTEM_INPUT", "你是测试助手。回答必须按照字数要求。")).strip()
        # Idempotent: `replace()` (used in call_openai) re-runs __post_init__, so
        # only append the hint if it isn't already present.
        hint_marker = "Total token budget including reasoning is:"
        if self.max_tokens is not None and self.system_input is not None and (
                not self.disable_maxtoken_hint and hint_marker not in self.system_input):
            system_input=self.system_input
            system_input += f"\n{hint_marker} {self.max_tokens}. Reasoning budget can not be more than {int(self.max_tokens * 0.8)} tokens"
            object.__setattr__(self, "system_input", system_input)

        # Frozen dataclass — write resolved values back in place (no `replace`,
        # which would recurse through __post_init__).
        object.__setattr__(self, "api", ApiConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            model_alias=api.model_alias if api else None,
        ))
        object.__setattr__(self, "timeout", timeout)

    @property
    def api_key(self) -> str:
        return self.api.api_key

    @property
    def base_url(self) -> str:
        return self.api.base_url

    @property
    def model(self) -> str:
        return self.api.model

    def with_user_input(
            self,
            *,
            prompt: Union[str, list],
            image_data_urls: Optional[Sequence[str]] = None,
            disable_thinking: bool = False,
    ) -> "ChatCompletionRequest":
        messages: list[ChatCompletionMessage] = []
        if self.system_input:
            messages.append(ChatCompletionMessage(
                role="system",
                content=self.system_input,
            ))
        messages.append(ChatCompletionMessage(
            role="user",
            content=build_user_content(prompt, image_data_urls or ()),
        ))
        return replace(
            self,
            messages=tuple(messages),
            stream_options=ChatCompletionStreamOptions() if self.stream else None,
            thinking=(
                ChatCompletionThinking(type="disabled")
                if disable_thinking
                else None
            ),
            chat_template_kwargs=(
                {"enable_thinking": False}
                if disable_thinking
                else None
            ),
            enable_thinking=False if disable_thinking else None,
        )

    def to_dict(self) -> dict[str, Any]:
        def to_json_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: to_json_value(item)
                    for key, item in value.items()
                    if item is not None
                }
            if isinstance(value, (list, tuple)):
                return [to_json_value(item) for item in value]
            return value

        payload = asdict(self)
        for local_field in (
            "api",
            "timeout",
            "system_input",
            "disable_maxtoken_hint",
        ):
            payload.pop(local_field, None)
        payload["model"] = self.model
        return to_json_value(payload)


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
class LLMInferInputRecord(object):
    # allow list
    prompt: Union[str, list]
    image_paths: Optional[Sequence[Any]] = None
    image_data_urls: Optional[Sequence[str]] = None
    model_settings: Optional[ChatCompletionRequest] = None
    # Full original input row (id + all extra CSV columns) carried through so
    # LLMInferOutput.to_dict() can produce a self-contained record without any
    # external dict merge.
    extra: Dict[str, Any] = field(default_factory=dict)
    # Stable row identifier — a normal dataclass field (not a property) so it
    # can be read, compared, and used in dataclasses.replace() without indirection.
    # Populated from extra['id'] in __post_init__ when not passed explicitly.
    id: str = ""

    def __post_init__(self) -> None:
        ms = (
            self.model_settings
            if self.model_settings is not None
            else ChatCompletionRequest()
        )

        image_data_urls = [u.strip() for u in (self.image_data_urls or ()) if u.strip()]
        image_data_urls.extend(image_path_to_data_url(p) for p in (self.image_paths or ()))
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))
        object.__setattr__(self, "model_settings", ms)
        if not self.prompt:
            raise ValueError("prompt is empty")
        # if not ms.system_input:
        #     raise ValueError("system_input is empty")
        # Auto-populate id from extra when not provided explicitly.
        if not self.id and self.extra:
            object.__setattr__(self, "id", str(self.extra.get("id", "")))

    @property
    def model(self) -> str:
        """Model name for this instance.

        For a *loaded* record (CSV/JSONL) the authoritative settings are the
        ones embedded in ``extra['model_settings']`` — a dict, or a stringified
        dict when round-tripped through a CSV cell — so check that first. When
        absent (a freshly-built instance), fall back to the live
        ``model_settings.model``. The order matters: loading passes
        ``model_settings=None``, so ``__post_init__`` fills in a *default*
        ChatCompletionRequest whose ``.model`` would otherwise mask the real value.
        """
        # A direct ``model`` column in extra wins (e.g. openlearnlm records carry
        # the model name as a field rather than inside model_settings).
        direct = (self.extra or {}).get("model")
        if direct:
            return str(direct)
        ms = (self.extra or {}).get("model_settings")
        if isinstance(ms, str):
            import ast
            try:
                ms = ast.literal_eval(ms)
            except (ValueError, SyntaxError):
                ms = None
        if isinstance(ms, dict):
            # model lives at the top level (legacy) or nested under ``api``
            # (current ChatCompletionRequest serialises it inside the ApiConfig).
            m = ms.get("model") or (ms.get("api") or {}).get("model")
            if m:
                return str(m)
        if self.model_settings is not None and self.model_settings.model:
            return self.model_settings.model
        return ""

    @classmethod
    def from_dict(
        cls,
        row: Dict[str, Any],
        model_settings: Optional["ChatCompletionRequest"] = None,
        image_paths: Optional[Sequence[Any]] = None,
        image_data_urls: Optional[Sequence[str]] = None,
    ) -> "LLMInferInputRecord":
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
        import math
        # pandas uses float('nan') for missing CSV cells; json.dumps would
        # emit the bare token NaN which is not valid JSON — normalise to None.
        sanitised = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
        }
        return cls(
            id=str(sanitised.get("id", "")),
            prompt=str(row["prompt"]),
            model_settings=model_settings,
            image_paths=image_paths,
            image_data_urls=image_data_urls,
            extra=sanitised,
        )

    @classmethod
    def from_csv(
            cls,
            path: "str | Path",
            model_settings: "Optional[ChatCompletionRequest]" = None,
    ) -> "list[LLMInferInputRecord]":
        """Load a CSV file and convert every row to an instance via ``from_dict``.

        NaN values (pandas missing cells) are normalised to ``None`` inside
        ``from_dict``, so the resulting ``extra`` dicts are JSON-safe.
        Replaces the inline ``pd.read_csv(...).to_dict("records")`` +
        list-comprehension pattern in ``run.py``.
        """
        import pandas as pd
        rows = pd.read_csv(path).to_dict("records")
        return [cls.from_dict(row, model_settings=model_settings) for row in rows]

    @classmethod
    def from_jsonl(
            cls,
            path: "str | Path",
            model_settings: "Optional[ChatCompletionRequest]" = None,
    ) -> "list[LLMInferInputRecord]":
        """Deserialise a JSONL file into a list of instances.

        Each non-empty line is parsed as JSON and forwarded to ``cls.from_dict``.
        When called as ``LLMInferOutput.from_jsonl(...)`` the subclass classmethod
        is inherited automatically, so output fields are populated correctly via
        ``LLMInferOutput.from_dict``.

        Replaces the inline resume-loading loop in
        ``llm_common/llm_infer/run.py:82``.
        """
        instances = []
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            instances.append(cls.from_dict(record, model_settings=model_settings))
        return instances


@dataclass(frozen=True)
class LLMInferResultRecord(LLMInferInputRecord):
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

    @classmethod
    def from_dict(
            cls,
            row: Dict[str, Any],
            input_: Optional[LLMInferInputRecord] = None,
            *,
            model_settings: Optional["ChatCompletionRequest"] = None,
            image_paths: Optional[Sequence[Any]] = None,
            image_data_urls: Optional[Sequence[str]] = None,
    ) -> "LLMInferResultRecord":
        """Construct from a JSONL/CSV record dict, populating all output fields.

        ``input_`` is authoritative for input fields and request configuration;
        ``row`` supplies the response and runtime fields. When ``input_`` is
        omitted, one is built from ``row`` for inherited CSV/JSONL loaders.
        """
        import math
        sanitised: Dict[str, Any] = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
        }
        if input_ is None:
            input_ = LLMInferInputRecord.from_dict(
                sanitised,
                model_settings=model_settings,
                image_paths=image_paths,
                image_data_urls=image_data_urls,
            )
        extra = dict(input_.extra)
        extra.update(sanitised)
        return cls(
            id                = input_.id,
            prompt            = input_.prompt,
            model_settings    = input_.model_settings,
            # image_data_urls on LLMInferInput are already normalised and
            # include converted image_paths; do not convert paths a second time.
            image_data_urls   = input_.image_data_urls,
            extra             = extra,
            llm_response      = str(sanitised.get("llm_response", "") or ""),
            reasoning         = sanitised.get("reasoning") or None,
            prompt_tokens     = sanitised.get("prompt_tokens"),
            completion_tokens = sanitised.get("completion_tokens"),
            total_tokens      = sanitised.get("total_tokens"),
            latency_ms        = sanitised.get("latency_ms"),
        )

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

    @classmethod
    def to_csv(
            cls,
            outputs: "list[Optional[LLMInferResultRecord]]",
            fallback_rows: "list[Dict[str, Any]]",
            path: "str | Path",
    ) -> Path:
        """Write outputs to a CSV file, preserving original row order.

        For each position ``i``:
        - If ``outputs[i]`` is not None, serialise via ``to_dict()``.
        - Otherwise fall back to ``fallback_rows[i]`` (the original input row).

        Moves the logic from ``llm_common/llm_infer/run.py:120``.
        """
        import pandas as pd
        output_rows = [
            out.to_dict() if out is not None else fallback_rows[i]
            for i, out in enumerate(outputs)
        ]
        path = Path(path)
        pd.DataFrame(output_rows).to_csv(path, index=False)
        return path


    @classmethod
    def get_output_path_hint(
            cls,
            input_path: "str | Path",
            model: str,
    ) -> Path:
        """Compute the default output path for inference results.

        Unifies the two conventions in ``run.py``:

        * **CSV file** (``run.py:84``): same directory, stem gets the
          ``_batch_infer_<model>`` suffix →
          ``data/prompts/file.csv`` → ``data/prompts/file_batch_infer_model.csv``

        * **Directory** (``run.py:132``): a sibling directory whose name gets
          the suffix →
          ``data/prompts/`` → ``data/prompts_batch_infer_model/``
        """
        p = Path(input_path)
        if p.is_dir():
            return p.parent / (p.name + "_batch_infer_" + model)
        return p.with_stem(p.stem + "_batch_infer_" + model)
