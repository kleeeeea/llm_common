#!/usr/bin/env python3
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.load_env import ENV_FILE, load_env_file, read_env_file, require_env

api_innospark_cn_v_ = "https://api.innospark.cn/v1"


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




def require_positive_number(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


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


def main() -> int:
    load_env_file(ENV_FILE)

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

@dataclass(frozen=True)
class CallOpenaiInput(object):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    system_input: Optional[str] = None
    image_paths: Optional[Sequence[Any]] = None
    image_data_urls: Optional[Sequence[str]] = None
    timeout: Optional[float] = None
    disable_maxtoken_hint: Optional[bool] = None

    def __post_init__(self) -> None:
        local_env = read_env_file(ENV_FILE)
        local_env.update(os.environ)

        api_key = (self.api_key or local_env.get("LLM_API_KEY", "")).strip()
        if not api_key:
            print(f"ERROR: LLM_API_KEY is empty. Set it in {ENV_FILE} or export LLM_API_KEY.", file=sys.stderr)
            sys.exit(2)

        #
        max_tokens = self.max_tokens if self.max_tokens is not None else int(local_env.get("MAX_TOKENS", "16000"))
        system_input = (self.system_input or local_env.get("SYSTEM_INPUT", "你是测试助手。回答必须按照字数要求。")).strip()
        if max_tokens is not None and not self.disable_maxtoken_hint:
            system_input += f"\nTotal token budget including reasoning is: {max_tokens}. Reasoning budget can not be more than {int(max_tokens * 0.8)} tokens"
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", (self.base_url or local_env.get("LLM_BASE_URL", api_innospark_cn_v_)).strip().rstrip("/"))
        object.__setattr__(self, "model", (self.model or local_env.get("LLM_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash")
        object.__setattr__(self, "prompt", (self.prompt or local_env.get("PROMPT", "用中文输出一百个字的笑话")).strip())
        object.__setattr__(self, "system_input", system_input)
        image_data_urls = [image_data_url.strip() for image_data_url in (self.image_data_urls or ()) if image_data_url.strip()]
        image_data_urls.extend(image_path_to_data_url(image_path) for image_path in (self.image_paths or ()))
        object.__setattr__(self, "image_data_urls", tuple(image_data_urls))
        object.__setattr__(self, "timeout", self.timeout if self.timeout is not None else float(local_env.get("TIMEOUT_SECONDS", "60")))
        object.__setattr__(self, "max_tokens", max_tokens)
        require_positive_number("timeout", self.timeout)
        require_positive_number("max_tokens", self.max_tokens)
        if not self.prompt:
            raise ValueError("prompt is empty")
        if not self.system_input:
            raise ValueError("system_input is empty")

def call_openai(
        input_: Optional[CallOpenaiInput]=None,
        api_config: Optional[ApiConfig]=None,
        api_key: str=None, base_url: str=None, max_tokens: int=None, model: str=None, prompt: str=None,
        system_input: str=None, image_paths: Optional[Sequence[Any]]=None,
        image_data_urls: Optional[Sequence[str]]=None, timeout: float=None,
        do_print_one_response_per_line=None, disable_maxtoken_hint=None) -> str:
    if api_config is not None:
        api_key = api_key or api_config.api_key
        base_url = base_url or api_config.base_url
        model = model or api_config.model
    if input_ is None:
        input_ = CallOpenaiInput(
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                model=model,
                prompt=prompt,
                system_input=system_input,
                image_paths=image_paths,
                image_data_urls=image_data_urls,
                timeout=timeout,
                disable_maxtoken_hint=disable_maxtoken_hint,
        )
    body = {
            "model"      : input_.model,
    "thinking": {"type": "disabled"},   # 关键：强制关闭思考
            "stream"     : True,
            "temperature": 0.1,
            "max_tokens" : input_.max_tokens,
            "messages"   : [
                    {"role": "system", "content": input_.system_input},
                    {"role": "user", "content": build_user_content(input_.prompt, input_.image_data_urls)},
            ],
    }

    print('*' * 50 + f'''\n{input_.system_input}\n^^^(input_.system_input)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:242\n''' + '*' * 50)
    print('*' * 50 + f'''\n{input_.prompt}\n^^^(input_.prompt)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:243\n''' + '*' * 50)

    url = build_chat_completions_url(input_.base_url)
    request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                    "Authorization": f"Bearer {input_.api_key}",
                    "Content-Type" : "application/json",
            },
            method="POST",
    )

    print(f"POST {url}")
    print(f"model={input_.model} stream=true")

    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    all_payloads: list[str] = []
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
        with urllib.request.urlopen(request, timeout=input_.timeout) as response:
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
    return text


if __name__ == "__main__":
    raise SystemExit(main())
