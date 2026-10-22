import codecs
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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
from llm_common.llm_infer.llm_response_parse import _extract_stream_text
from llm_common.llm_infer.llm_response_parse import extract_stream_reasoning_text
from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file
from llm_common.llm_infer.load_env import require_env


def _stream_sse_lines(response):
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = response.read(1024)
        if not chunk:
            break
        buffer += decoder.decode(chunk)
        parts = buffer.replace("\r\n", "\n").split("\n\n")
        buffer = parts.pop() or ""
        for part in parts:
            for line in part.splitlines():
                if line.startswith("data: "):
                    yield line[6:].strip()
    buffer += decoder.decode(b"", final=True)
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


def _is_anthropic_messages_url(base_url: str) -> bool:
    return base_url.strip().rstrip("/").endswith("/messages")


def _build_anthropic_payload(input_: LLMInferInputRecord) -> dict[str, Any]:
    request = _build_chat_completion_request(input_).to_dict()
    messages = request.get("messages", [])
    system_parts = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "system"
    ]
    payload = {
        "model": request["model"],
        "messages": [
            message for message in messages if message.get("role") != "system"
        ],
        "max_tokens": request.get("max_tokens") or 4096,
        "temperature": request.get("temperature", 0.1),
        "stream": request.get("stream", True),
    }
    if system_parts:
        payload["system"] = "\n".join(str(part) for part in system_parts)
    return payload


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
        f'LLM_API_KEY={api_key}',
        "curl -sS -N -X POST \\",
    ]
    if timeout is not None:
        command_lines.append(f"  --max-time {shlex.quote(str(timeout))} \\")
    command_lines.extend([
        f"  {shlex.quote(chat_completions_url)} \\",
        '  -H "Authorization: Bearer ${LLM_API_KEY}" \\',
        "  -H 'Content-Type: application/json' \\",
    ])
    if _is_anthropic_messages_url(chat_completions_url):
        command_lines.append("  -H 'anthropic-version: 2023-06-01' \\")
    command_lines.extend([f"  -d {shlex.quote(json_body)}", ""])

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
    reasoning: Any = None

    @classmethod
    def from_dict(cls, value: Any) -> "ChatCompletionChunk":
        value = value if isinstance(value, dict) else {}
        choices = value.get("choices")
        first_delta = (
            choices[0].get("delta", {})
            if (
                isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                and isinstance(choices[0].get("delta"), dict)
            )
            else {}
        )
        return cls(
            choices=tuple(
                ChatCompletionChunkChoice.from_dict(choice)
                for choice in choices
                if isinstance(choice, dict)
            ) if isinstance(choices, list) else (),
            content=value.get("content"),
            reasoning_content=value.get("reasoning_content"),
            reasoning=value.get("reasoning") or first_delta.get("reasoning"),
        )

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

    is_anthropic = _is_anthropic_messages_url(input_.api.base_url)
    body = (
        _build_anthropic_payload(input_)
        if is_anthropic
        else _build_chat_completion_payload(
            input_,
            disable_thinking=input_.disable_thinking,
        )
    )

    if input_.chat_completion_request.stream:
        print('*' * 50 + f'''\n{input_.system_input}\n^^^(input_.system_input)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:242\n''' + '*' * 50)
        print('*' * 50 + f'''\n{input_.prompt}\n^^^(input_.prompt)^^^\n''' + '''\nat:\nllm_common/llm_infer/call.py:243\n''' + '*' * 50)

    # first test remote server healthiness using curl
    # Anthropic-compatible gateways do not expose OpenAI's /models endpoint;
    # the request itself is the authoritative health check for those URLs.
    if not is_anthropic:
        _check_server_health(input_.api.base_url, input_.api.api_key, timeout=min(input_.timeout or 10.0, 10.0))

    # support simple non streaming option, just merge in scripts/inference/models.py:289
    request_url = (
        input_.api.base_url.strip().rstrip("/")
        if is_anthropic
        else _build_chat_completions_url(input_.api.base_url)
    )
    request_headers = {
        "Authorization": f"Bearer {input_.api.api_key}",
        "Content-Type" : "application/json",
    }
    if is_anthropic:
        request_headers["anthropic-version"] = "2023-06-01"
    request = urllib.request.Request(
            request_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=request_headers,
            method="POST",
    )

    # dump to the curl command to curl_latest.sh at the same directory
    curl_path = _dump_curl_command(
        request_url,
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


    def collect_response_data(data: dict[str, Any], raw: str) -> None:
                    nonlocal usage
                    event_usage = data.get("usage")
                    if is_anthropic and not event_usage:
                        message = data.get("message", {})
                        if isinstance(message, dict):
                            event_usage = message.get("usage")
                    if event_usage:
                        usage = {**(usage or {}), **event_usage}
                    if is_anthropic:
                        delta = data.get("delta", {})
                        text = delta.get("text", "") if isinstance(delta, dict) else ""
                        reasoning_text = delta.get("thinking", "") if isinstance(delta, dict) else ""
                        content = data.get("content", [])
                        if isinstance(content, list):
                            text = text or "".join(
                                str(part.get("text", ""))
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            )
                            reasoning_text = reasoning_text or "".join(
                                str(part.get("thinking", ""))
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "thinking"
                            )
                    else:
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
        print('---'* 3)
        print(input_.model)
        print('---'* 3)
        if (
            isinstance(input_.prompt, list)
            and input_.prompt
            and isinstance(input_.prompt[0], dict)
        ):
            print(input_.prompt[0].get('text', input_.prompt[0]))
        else:
            print(input_.prompt)
        print('---'* 3)
        print('paste the response there into reponse_latest.txt')

        txt_ = Path(__file__).resolve().parent / 'reponse_latest.txt'
        open(txt_,'w').write('')
        input()
        text = open(txt_).read().strip()
        if text:
            chunks.append(text.strip())
        else:
            raise Exception('llm error')
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise Exception('llm error')

    text = "".join(chunks).strip()
    reasoning = "".join(reasoning_chunks).strip() or None
    if not text and not reasoning:
        print("ERROR: stream completed but produced no text", file=sys.stderr)
        import importlib
        hostname = importlib.import_module("socket").gethostname()
        if (hostname in {"wsy-pretrain2-cpu-kq--169ddeec5a64-onoqafwz75", "cpu3"} or "mac" in hostname.lower()) and importlib.import_module("os").environ.get("LOCAL_RANK", "0") == "0":
            importlib.import_module("sys").path.append(str("/home/likeqian/klee_code" if importlib.import_module("os").path.exists("/home/likeqian/klee_code") else importlib.import_module("os").path.expanduser("~/klee_code")));from python_common.borrowed_klee_python_code.pdb import set_trace;set_trace()

        if 'https://api.agicto.cn/v1' in input_.api.base_url:
            # commercial api shouldn't lead to infra error
            text = 'Score：0'

        else:

            raise Exception('llm error')

    print()
    return LLMInferResultRecord.from_dict({
        "llm_response": text,
        "reasoning": reasoning,
        "prompt_tokens": (usage or {}).get("prompt_tokens", (usage or {}).get("input_tokens")),
        "completion_tokens": (usage or {}).get("completion_tokens", (usage or {}).get("output_tokens")),
        "total_tokens": (
            (usage or {}).get("total_tokens")
            if (usage or {}).get("total_tokens") is not None
            else sum(
                value or 0
                for value in (
                    (usage or {}).get("input_tokens"),
                    (usage or {}).get("output_tokens"),
                )
            ) or None
        ),
        "latency_ms": (time.perf_counter() - _t0) * 1000.0,
    }, input_=input_)


if __name__ == "__main__":
    raise SystemExit(main())
