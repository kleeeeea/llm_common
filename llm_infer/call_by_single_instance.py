import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.instances import LLMInferInput
from llm_common.llm_infer.instances import LLMInferOutput
from llm_common.llm_infer.instances import ModelSettings
from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file
from llm_common.llm_infer.load_env import require_env


def stream_sse_lines(response):
    buffer = ""
    while True:
        chunk = response.read(1024)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        parts = buffer.replace("\r\n", "\n").split("\n\n")
        buffer = parts.pop() or ""
        for part in parts:
            for line in part.splitlines():
                if line.startswith("data: "):
                    yield line[6:].strip()
    for line in buffer.splitlines():
        if line.startswith("data: "):
            yield line[6:].strip()




def build_chat_completions_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("base_url is empty")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)
    return ""


def extract_stream_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") or {}
        if isinstance(delta, dict):
            text = normalize_text_content(delta.get("content"))
            if text:
                return text
        message = first.get("message") or {}
        if isinstance(message, dict):
            text = normalize_text_content(message.get("content"))
            if text:
                return text
    return normalize_text_content(payload.get("content"))


def extract_stream_reasoning_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") or {}
        if isinstance(delta, dict):
            text = normalize_text_content(delta.get("reasoning_content"))
            if text:
                return text
        message = first.get("message") or {}
        if isinstance(message, dict):
            text = normalize_text_content(message.get("reasoning_content"))
            if text:
                return text
    return normalize_text_content(payload.get("reasoning_content"))


def build_user_content(prompt: str, image_data_urls: Sequence[str]) -> Any:
    if not image_data_urls:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_data_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return content




def main() -> int:
    load_env_file(ENV_FILE)

    from llm_common.llm_infer.instances import api_innospark_cn_v_
    base_url = require_env("LLM_BASE_URL", api_innospark_cn_v_).rstrip("/")
    api_key = require_env("LLM_API_KEY", '')
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    prompt = os.environ.get("PROMPT", "output a paragraph with 100 words").strip()
    timeout = float(os.environ.get("TIMEOUT_SECONDS", "60"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "16000"))

    text = call_openai(api_key=api_key, base_url=base_url, max_tokens=max_tokens, model=model, prompt=prompt, timeout=timeout,
                       system_input="You're a helpful assistant")

    print("\nOK: stream returned content.")
    print(text)
    return 0

def call_openai(
        input_: Optional[LLMInferInput]=None,
        api_config: Optional[ApiConfig]=None,
        api_key: str=None, base_url: str=None, max_tokens: int=None,
        model: str=None, prompt: str=None,
        system_input: str=None, image_paths: Optional[Sequence[Any]]=None,
        image_data_urls: Optional[Sequence[str]]=None, timeout: float=None,
        do_print_one_response_per_line=None, disable_maxtoken_hint=None,
        model_settings: Optional[ModelSettings]=None,
) -> LLMInferOutput:
    if input_ is None:
        if model_settings is None:
            model_settings = ModelSettings()
        # api_key / base_url / model are no longer standalone ModelSettings
        # fields — fold any explicit overrides (direct kwargs or api_config)
        # into the ModelSettings' ApiConfig instead.
        cur = model_settings.api
        ovr_key   = api_key  or (api_config.api_key  if api_config else None)
        ovr_url   = base_url or (api_config.base_url if api_config else None)
        ovr_model = model    or (api_config.model    if api_config else None)
        if cur is not None or ovr_key or ovr_url or ovr_model:
            merged_api = ApiConfig(
                base_url=ovr_url   or (cur.base_url if cur else "") or "",
                api_key =ovr_key   or (cur.api_key  if cur else "") or "",
                model   =ovr_model or (cur.model    if cur else "") or "",
            )
        else:
            merged_api = None
        model_settings = replace(model_settings,
                                 api=merged_api,
                                 timeout=model_settings.timeout or timeout,
                                 system_input=model_settings.system_input or system_input,
                                 max_tokens=model_settings.max_tokens or max_tokens,
                                 disable_maxtoken_hint=model_settings.disable_maxtoken_hint or bool(disable_maxtoken_hint))
        input_ = LLMInferInput(
                prompt=prompt,
                image_paths=image_paths,
                image_data_urls=image_data_urls,
                model_settings=model_settings,
        )
    ms = input_.model_settings
    body = {
            "model"      : ms.model,
            "thinking"   : ms.thinking,
            "temperature": ms.temperature,
            "stream"     : ms.stream,
            # ask OpenAI-compatible servers to emit a final usage-only chunk
            # (choices=[], usage={...}) at the end of the stream.
            "stream_options": {"include_usage": True},
            "max_tokens" : ms.max_tokens,
            "messages"   : [
                    {"role": "system", "content": ms.system_input},
                    {"role": "user", "content": build_user_content(input_.prompt, input_.image_data_urls)},
            ],
    }

    print('*' * 50 + f'''\n{ms.system_input}\n^^^(ms.system_input)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:242\n''' + '*' * 50)
    print('*' * 50 + f'''\n{input_.prompt}\n^^^(input_.prompt)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:243\n''' + '*' * 50)

    url = build_chat_completions_url(ms.base_url)
    request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                    "Authorization": f"Bearer {ms.api_key}",
                    "Content-Type" : "application/json",
            },
            method="POST",
    )

    print(f"POST {url}")
    print(f"model={ms.model} stream=true")

    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    all_payloads: list[str] = []
    usage: Optional[Dict[str, Any]] = None
    _t0 = time.perf_counter()
    #     Out[1]:
    # ['{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":"The","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" user","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" wants","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" a","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" paragraph","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" with","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" exactly","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":" ","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    #  '{"id":"8fe1630f301c48a3bf7c4600cd5b17bc","object":"chat.completion.chunk","created":1779609667,"model":"Kimi-K2.6","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":"100","tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":null}',
    # 需要收集推理模型的输入
    try:
        with urllib.request.urlopen(request, timeout=ms.timeout) as response:
            content_type = response.headers.get("content-type", "")
            print(f"HTTP {response.status} content-type={content_type}")
            if "text/event-stream" not in content_type:
                raw = response.read().decode("utf-8", errors="replace")
                print("ERROR: response is not text/event-stream", file=sys.stderr)
                print(raw[:4000], file=sys.stderr)
                raise Exception('llm error')

            for payload in stream_sse_lines(response):
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # the usage-only trailer chunk (and some servers on every chunk)
                # carries token counts; keep the last non-null one.
                if isinstance(data, dict) and data.get("usage"):
                    usage = data["usage"]
                text = extract_stream_text(data)
                reasoning_text = extract_stream_reasoning_text(data)
                all_payloads.append(payload)
                if reasoning_text:
                    reasoning_chunks.append(reasoning_text)
                    if do_print_one_response_per_line:
                        print(f"reasoning_chunk {reasoning_text}", flush=True)
                        print('-' * 30)
                    else:
                        print(f"\033[31m{reasoning_text}\033[0m", end='', flush=True)
                if text:
                    chunks.append(text)
                    if do_print_one_response_per_line:
                        print(f"chunk {text}", flush=True)
                        print('=' * 30)
                    else:
                        print(text, end='', flush=True)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}", file=sys.stderr)
        print(error.read().decode("utf-8", errors="replace")[:4000], file=sys.stderr)
        raise Exception('llm error')
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise Exception('llm error')

    text = "".join(chunks).strip()
    if not text:
        print("ERROR: stream completed but produced no text", file=sys.stderr)
        raise Exception('llm error')
    reasoning = "".join(reasoning_chunks).strip() or None
    return LLMInferOutput(
        prompt=input_.prompt,
        image_data_urls=input_.image_data_urls,
        model_settings=input_.model_settings,
        extra=input_.extra,
        llm_response=text,
        reasoning=reasoning,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=(usage or {}).get("completion_tokens"),
        total_tokens=(usage or {}).get("total_tokens"),
        latency_ms=(time.perf_counter() - _t0) * 1000.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
