# llm_common

Shared utilities for LLM inference and local model serving.

## Modules

```
llm_common/
├── llm_infer/
│   ├── call.py              # Core inference: ModelSettings, call_openai()
│   ├── batch_call.py        # Batch CSV inference with multi-threading and resume
│   ├── load_env.py          # .env file loading
│   ├── api_info/
│   │   └── dataclass_.py    # ApiConfig + pre-configured API instances
│   └── test/
│       └── data/            # Sample data and path index
└── llm_serve/
    └── vllm_serve.py        # Local vLLM server launch and management
```

## Quick Start

### Single inference

```python
from llm_common.llm_infer.call import call_openai, ModelSettings
from llm_common.llm_infer.api_info.dataclass_ import GEMINI_API

text = call_openai(
    prompt="tell me a joke",
    api_config=GEMINI_API,
    system_input="You are a helpful assistant",
)
```

Or via `ModelSettings` for full control:

```python
ms = ModelSettings(
    api_key="sk-...",
    base_url="https://api.innospark.cn/v1",
    model="gemini-2.5-flash",
    max_tokens=4096,
    system_input="You are a helpful assistant",
)
text = call_openai(model_settings=ms, prompt="tell me a joke")
```

### Batch inference from CSV

Input CSV must have an `id` column and a `prompt` column:

```csv
id,prompt
0,tell me a joke
1,tell me a joke about politics
```

```python
from llm_common.llm_infer.batch_call import cached_batch_call_file

cached_batch_call_file("path/to/prompts.csv", max_workers=4)
```

Output is written to `<parent>_batch_infer_<model>/`:
- `prompts.csv` — input columns + `llm_response`
- `prompts.jsonl` — one record per line, written as each thread completes
- `model_settings.json` — the config used for this run

Resumable: rows already present in the output JSONL are skipped on re-run.

### Local vLLM server

```python
from llm_common.llm_serve.vllm_serve import create_ApiInfo_from_localModelInfo
from llm_common.llm_infer.call import call_openai

api_config = create_ApiInfo_from_localModelInfo(
    PORT=8000,
    model_name="Qwen3-4B-Instruct-2507",
    model_path="/path/to/model",
)
text = call_openai(api_config=api_config, prompt="tell me a joke")
```

## Configuration

Copy `.env.example` to `llm_infer/.env` and fill in:

```
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.innospark.cn/v1
LLM_MODEL=gemini-2.5-flash
MAX_TOKENS=16000
TIMEOUT_SECONDS=60
SYSTEM_INPUT=你是测试助手。
```

All values can also be set via environment variables, which take precedence over `.env`.

## Pre-configured APIs

Defined in `llm_infer/api_info/dataclass_.py`:

| Name | Model |
|------|-------|
| `GEMINI_API` | gemini-2.5-flash |
| `DEFAULT_1T_BASELINE_API` | Kimi-K2.6 |
| `DEFAULT_4B_BASELINE_API` | Qwen3-4B-Instruct-2507-Official |
| `Qwen27b_API` | Qwen3.6-27B |
| `GLM51FP8_API` | GLM-5.1-FP8 |

## ModelSettings fields

| Field | Default | Description |
|-------|---------|-------------|
| `api_key` | env `LLM_API_KEY` | API key |
| `base_url` | env `LLM_BASE_URL` | OpenAI-compatible endpoint |
| `model` | env `LLM_MODEL` | Model name |
| `timeout` | `60` | Request timeout in seconds |
| `max_tokens` | `16000` | Max output tokens |
| `system_input` | env `SYSTEM_INPUT` | System prompt |
| `temperature` | `0.1` | Sampling temperature |
| `thinking` | `{"type": "disabled"}` | Reasoning mode |
| `disable_maxtoken_hint` | `False` | Suppress token budget hint in system prompt |
