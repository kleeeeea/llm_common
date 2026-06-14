import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.instances import ChatCompletionContentPartImage
from llm_common.llm_infer.instances import ChatCompletionContentPartText
from llm_common.llm_infer.instances import ChatCompletionImageURL
from llm_common.llm_infer.instances import ChatCompletionRequest
from llm_common.llm_infer.instances import LLMInferInputRecord
from llm_common.llm_infer.instances import LLMInferResultRecord
from llm_common.llm_infer.instances import build_user_content
from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file
from llm_common.llm_infer.load_env import require_env


def _stream_sse_lines(response):
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




def _build_chat_completions_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("base_url is empty")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _check_server_health(base_url: str, api_key: str, timeout: float = 10.0) -> None:
    """正式发流式请求前，先用 curl 探测远端服务器是否可达，失败则快速报错。

    打到 OpenAI 兼容的 /models 端点：
      - 2xx：服务正常；
      - 4xx：服务在线但鉴权/路径有差异（仍视为可达，放行）；
      - 其它（连不上、超时、5xx、curl 本身报错）：抛异常，避免请求发到一半才挂。
    """
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    models_url = f"{base}/models"
    cmd = [
        "curl", "-sS",
        "-o", os.devnull,            # 丢弃响应体，只看状态码
        "-w", "%{http_code}",
        "--max-time", str(timeout),
        "-H", f"Authorization: Bearer {api_key}",
        models_url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 5
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise Exception(f"remote server health check failed (curl 无法执行: {error})")
    code = (result.stdout or "").strip()
    print(f"health check: GET {models_url} -> {code or '(no http_code)'}")
    if not code.startswith(("2", "4")):
        stderr = (result.stderr or "").strip()[:200]
        raise Exception(
            f"remote server health check failed (http_code={code!r}, stderr={stderr!r})"
        )


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


@dataclass(frozen=True)
class ChatCompletionChunkDelta:
    """Source: OpenAI Chat Completions streaming choice `delta`.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

    `reasoning_content` is an OpenAI-compatible server extension.
    """

    content: Any = None
    role: Optional[str] = None
    reasoning_content: Any = None

    @classmethod
    def from_dict(cls, value: Any) -> "ChatCompletionChunkDelta":
        value = value if isinstance(value, dict) else {}
        return cls(
            content=value.get("content"),
            role=value.get("role") if isinstance(value.get("role"), str) else None,
            reasoning_content=value.get("reasoning_content"),
        )


@dataclass(frozen=True)
class ChatCompletionChunkChoice:
    """Source: OpenAI Chat Completions streaming `choices` item.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

    `message` is retained as a compatibility fallback for servers that return
    a non-streaming message inside a streamed response.
    """

    index: int = 0
    delta: ChatCompletionChunkDelta = ChatCompletionChunkDelta()
    finish_reason: Optional[str] = None
    message: ChatCompletionChunkDelta = ChatCompletionChunkDelta()

    @classmethod
    def from_dict(cls, value: Any) -> "ChatCompletionChunkChoice":
        value = value if isinstance(value, dict) else {}
        index = value.get("index")
        return cls(
            index=index if isinstance(index, int) else 0,
            delta=ChatCompletionChunkDelta.from_dict(value.get("delta")),
            finish_reason=(
                value.get("finish_reason")
                if isinstance(value.get("finish_reason"), str)
                else None
            ),
            message=ChatCompletionChunkDelta.from_dict(value.get("message")),
        )


@dataclass(frozen=True)
class ChatCompletionChunk:
    """Source: OpenAI Chat Completions streamed response chunk.

    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

    Top-level `content` and `reasoning_content` are compatibility extensions.
    """

    choices: tuple[ChatCompletionChunkChoice, ...] = ()
    content: Any = None
    reasoning_content: Any = None

    @classmethod
    def from_dict(cls, value: Any) -> "ChatCompletionChunk":
        value = value if isinstance(value, dict) else {}
        choices = value.get("choices")
        return cls(
            choices=tuple(
                ChatCompletionChunkChoice.from_dict(choice)
                for choice in choices
                if isinstance(choice, dict)
            ) if isinstance(choices, list) else (),
            content=value.get("content"),
            reasoning_content=value.get("reasoning_content"),
        )


def _extract_stream_text(payload: dict[str, Any]) -> str:
    chunk = ChatCompletionChunk.from_dict(payload)
    if chunk.choices:
        first = chunk.choices[0]
        text = normalize_text_content(first.delta.content)
        if text:
            return text
        text = normalize_text_content(first.message.content)
        if text:
            return text
    return normalize_text_content(chunk.content)


def extract_stream_reasoning_text(payload: dict[str, Any]) -> str:
    chunk = ChatCompletionChunk.from_dict(payload)
    if chunk.choices:
        first = chunk.choices[0]
        text = normalize_text_content(first.delta.reasoning_content)
        if text:
            return text
        text = normalize_text_content(first.message.reasoning_content)
        if text:
            return text
    return normalize_text_content(chunk.reasoning_content)


def _build_chat_completion_request(
        input_: LLMInferInputRecord,
        disable_thinking: bool = False,
) -> ChatCompletionRequest:
    return input_.model_settings.with_user_input(
        prompt=input_.prompt,
        image_data_urls=input_.image_data_urls,
        disable_thinking=disable_thinking,
    )




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
        input_: Optional[LLMInferInputRecord]=None,
        api_config: Optional[ApiConfig]=None,
        api_key: str=None, base_url: str=None, max_tokens: int=None,
        model: str=None, prompt: Union[str,list]=None,
        system_input: str=None, image_paths: Optional[Sequence[Any]]=None,
        image_data_urls: Optional[Sequence[str]]=None, timeout: float=None,
        do_print_one_response_per_line=None, disable_maxtoken_hint=None,
        model_settings: Optional[ChatCompletionRequest]=None,
        disable_thinking=None,) -> LLMInferResultRecord:
    if input_ is None:
        if model_settings is None:
            model_settings = ChatCompletionRequest()
        # api_key / base_url / model are no longer standalone request settings
        # fields — fold any explicit overrides (direct kwargs or api_config)
        # into the ChatCompletionRequest's ApiConfig instead.
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
        input_ = LLMInferInputRecord(
                prompt=prompt,
                image_paths=image_paths,
                image_data_urls=image_data_urls,
                model_settings=model_settings,
        )
    ms = input_.model_settings
    body = _build_chat_completion_request(
        input_,
        disable_thinking=bool(disable_thinking),
    ).to_dict()


    print('*' * 50 + f'''\n{ms.system_input}\n^^^(ms.system_input)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:242\n''' + '*' * 50)
    print('*' * 50 + f'''\n{input_.prompt}\n^^^(input_.prompt)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:243\n''' + '*' * 50)

    # first test remote server healthiness using curl
    _check_server_health(ms.api.base_url, ms.api.api_key, timeout=min(ms.timeout or 10.0, 10.0))

    url = _build_chat_completions_url(ms.api. base_url)
    request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                    "Authorization": f"Bearer {ms.api. api_key}",
                    "Content-Type" : "application/json",
            },
            method="POST",
    )

    print(f"POST {url}")

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

            for payload in _stream_sse_lines(response):
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
                text = _extract_stream_text(data)
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
    print()
    return LLMInferResultRecord.from_dict({
        "llm_response": text,
        "reasoning": reasoning,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "total_tokens": (usage or {}).get("total_tokens"),
        "latency_ms": (time.perf_counter() - _t0) * 1000.0,
    }, input_=input_)


if __name__ == "__main__":
    raise SystemExit(main())
