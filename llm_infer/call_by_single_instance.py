import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

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


def _dump_curl_command(
        chat_completions_url: str,
        api_key: str,
        body: dict[str, Any],
        timeout: Optional[float],
        output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "curl_latest.sh",
        )
    json_body = json.dumps(body, ensure_ascii=False, indent=2)

    command_lines = [
        "#!/usr/bin/env bash",
        f"LLM_API_KEY=${{LLM_API_KEY:-{shlex.quote(api_key)}}}",
        "curl -sS -N -X POST \\",
    ]
    if timeout is not None:
        command_lines.append(f"  --max-time {shlex.quote(str(timeout))} \\")
    command_lines.extend([
        f"  {shlex.quote(chat_completions_url)} \\",
        '  -H "Authorization: Bearer ${LLM_API_KEY}" \\',
        "  -H 'Content-Type: application/json' \\",
        f"  -d {shlex.quote(json_body)}",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(command_lines))
    os.chmod(output_path, 0o700)
    return output_path


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
        check_command = (
            "curl -sS -o /dev/null -w '%{http_code}\\n' "
            f"--max-time {shlex.quote(str(timeout))} "
            '-H "Authorization: Bearer $LLM_API_KEY" '
            f"{shlex.quote(models_url)}"
        )
        print(
            "health check failed; reproduce with:\n"
            f"export LLM_API_KEY='<your-api-key>'\n{check_command}",
            file=sys.stderr,
        )
        raise Exception(
            "remote server health check failed "
            f"(http_code={code!r}, stderr={stderr!r}); "
            f"check_command={check_command!r}"
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


def _build_mock_response(prompt: Any) -> str:
    if isinstance(prompt, list):
        return next(
            (
                "mock response with prompt: " + part.get("text", "")
                for part in prompt
                if isinstance(part, dict) and part.get("type") == "text"
            ),
            "",
        )
    return "mock response with prompt: " + str(prompt)


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
) -> ChatCompletionRequest:
    return input_.chat_completion_request


def _build_chat_completion_payload(
        input_: LLMInferInputRecord,
        disable_thinking: bool = False,
) -> dict[str, Any]:
    payload = _build_chat_completion_request(input_).to_dict()
    if disable_thinking:
        # OpenAI-compatible server extensions; deliberately outside
        # ChatCompletionRequest because they are not OpenAI schema fields.
        payload["thinking"] = {"type": "disabled"}
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["enable_thinking"] = False
        payload.update({
            "reasoning_effort": "none",
        })
        model_name = str(payload.get("model", "")).lower()
        if "intern" in model_name:
            messages = payload.get("messages", [])

            system_content = (
        "You must answer directly and concisely. "
        "Do not write any reasoning process. "
        "Do not output <think>, </think>, hidden reasoning, analysis, or explanation. "
        "Only output the final answer."
            )

            if not messages or messages[0].get("role") != "system":
                payload["messages"] = [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    *messages,
                ]
    return payload




def main() -> int:
    load_env_file(ENV_FILE)

    from llm_common.llm_infer.instances import api_innospark_cn_v_
    base_url = require_env("LLM_BASE_URL", api_innospark_cn_v_).rstrip("/")
    api_key = require_env("LLM_API_KEY", '')
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    prompt = os.environ.get("PROMPT", "output a paragraph with 100 words").strip()
    timeout = float(os.environ.get("TIMEOUT_SECONDS", "60"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "16000"))

    text = call_openai(LLMInferInputRecord(
        prompt=prompt,
        system_input="You're a helpful assistant",
        api=ApiConfig(base_url=base_url, api_key=api_key, model=model),
        chat_completion_request=ChatCompletionRequest(
            model=model,
            max_tokens=max_tokens,
        ),
        timeout=timeout,
    ))

    print("\nOK: stream returned content.")
    print(text)
    return 0

def call_openai(
        input_: LLMInferInputRecord,
) -> LLMInferResultRecord:
    if input_.model == "mock":
        return LLMInferResultRecord(
            id=input_.id,
            prompt=input_.prompt,
            image_data_urls=input_.image_data_urls,
            chat_completion_request=input_.chat_completion_request,
            api=input_.api,
            timeout=input_.timeout,
            disable_thinking=input_.disable_thinking,
            disable_maxtoken_hint=input_.disable_maxtoken_hint,
            do_print_one_response_per_line=input_.do_print_one_response_per_line,
            extra=input_.extra,
            system_input=input_.system_input,
            llm_response=_build_mock_response(input_.prompt),
            reasoning=None,
            latency_ms=0.0,
        )

    body = _build_chat_completion_payload(
        input_,
        disable_thinking=input_.disable_thinking,
    )

    print('*' * 50 + f'''\n{input_.system_input}\n^^^(input_.system_input)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:242\n''' + '*' * 50)
    print('*' * 50 + f'''\n{input_.prompt}\n^^^(input_.prompt)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:243\n''' + '*' * 50)

    # first test remote server healthiness using curl
    _check_server_health(input_.api.base_url, input_.api.api_key, timeout=min(input_.timeout or 10.0, 10.0))

    # support simple non streaming option, just merge in scripts/inference/models.py:289
    request = urllib.request.Request(
            _build_chat_completions_url(input_.api.base_url),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                    "Authorization": f"Bearer {input_.api.api_key}",
                    "Content-Type" : "application/json",
            },
            method="POST",
    )

    # dump to the curl command to curl_latest.sh at the same directory
    curl_path = _dump_curl_command(
        _build_chat_completions_url(input_.api.base_url),
        input_.api.api_key,
        body,
        input_.timeout,
    )
    print(f"dumped latest curl command to {curl_path}")

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
        with urllib.request.urlopen(request, timeout=input_.timeout) as response:
            content_type = response.headers.get("content-type", "")
            print(f"HTTP {response.status} content-type={content_type}")

            def collect_response_data(data: dict[str, Any], raw: str) -> None:
                nonlocal usage
                if data.get("usage"):
                    usage = data["usage"]
                text = _extract_stream_text(data)
                reasoning_text = extract_stream_reasoning_text(data)
                all_payloads.append(raw)
                if reasoning_text:
                    reasoning_chunks.append(reasoning_text)
                    if input_.do_print_one_response_per_line:
                        print(f"reasoning_chunk {reasoning_text}", flush=True)
                        print('-' * 30)
                    else:
                        print(f"\033[31m{reasoning_text}\033[0m", end='', flush=True)
                if text:
                    chunks.append(text)
                    if input_.do_print_one_response_per_line:
                        print(f"chunk {text}", flush=True)
                        print('=' * 30)
                    else:
                        print(f"\033[32m{text}\033[0m", end='', flush=True)

            if "text/event-stream" in content_type:
                for payload in _stream_sse_lines(response):
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        collect_response_data(data, payload)
            else:
                raw = response.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as error:
                    print("ERROR: response is not valid JSON", file=sys.stderr)
                    print(raw[:4000], file=sys.stderr)
                    raise Exception("llm error") from error
                if not isinstance(data, dict):
                    raise Exception("llm error")
                collect_response_data(data, raw)
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
