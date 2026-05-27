import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig

_current_vllm_server: "Optional[LocalVllmServer]" = None


def tee_process_output(proc, log_file):
    """Mirror process stdout/stderr to terminal and a log file."""
    with log_file.open("a", encoding="utf-8") as file:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            file.write(line)
            file.flush()


def make_vllm_log_file():
    now = datetime.now()
    log_dir = Path("/tmp/vllm_log_local/by_date") / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"vllm_{now:%H%M%S}.log"


def wait_for_vllm_start(port=8000, proc=None, timeout=300, interval=2):
    """Wait until the vLLM OpenAI-compatible server is ready."""
    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{port}/health"
    models_url = f"http://127.0.0.1:{port}/v1/models"
    last_error = None

    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM process exited early with code {proc.returncode}")

        for url in (health_url, models_url):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server is ready: {url}")
                    return
                last_error = f"{url} returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)

        print(f"Waiting for vLLM server on port {port}...")
        time.sleep(interval)

    raise TimeoutError(f"Timed out waiting for vLLM server on port {port}: {last_error}")

# curl -s http://127.0.0.1:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen3-4B-Instruct-2507",
#     "stream": false,
#     "temperature": 0.1,
#     "max_tokens": 512,
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "tell me a joke"}
#     ]
#   }' | python3 -m json.tool
# {
#     "id": "chatcmpl-888ca2dbf043020d",
#     "object": "chat.completion",
#     "created": 1779696566,
#     "prompt_routed_experts": null,
#     "model": "Qwen3-4B-Instruct-2507",
#     "choices": [
#         {
#             "index": 0,
#             "message": {
#                 "role": "assistant",
#                 "content": "Sure! Here's a light-hearted one for you:\n\nWhy did the coffee file a police report?\n\nBecause it got mugged! \u2615\ud83d\ude04",
#                 "refusal": null,
#                 "annotations": null,
#                 "audio": null,
#                 "function_call": null,
#                 "tool_calls": [],
#                 "reasoning": null
#             },
#             "logprobs": null,
#             "finish_reason": "stop",
#             "stop_reason": null,
#             "token_ids": null,
#             "routed_experts": null
#         }
#     ],
#     "service_tier": null,
#     "system_fingerprint": "vllm-0.21.0-ef5ea05f",
#     "usage": {
#         "prompt_tokens": 23,
#         "total_tokens": 53,
#         "completion_tokens": 30,
#         "prompt_tokens_details": null
#     },
#     "prompt_logprobs": null,
#     "prompt_token_ids": null,
#     "prompt_text": null,
#     "kv_transfer_params": null
# }


@dataclass(frozen=True)
class LocalModelInfo(object):
    model_path: str = "/inspire/qb-ilm/project/ai4education/public/models/Qwen3-4B-Instruct-2507"
    model_name: Optional[str] = None

    def __post_init__(self):
        # default to the basename
        if self.model_name is None:
            object.__setattr__(self, "model_name", Path(self.model_path).name)


def main():

    PORT = 8000
    model_info = LocalModelInfo()
    model_path = model_info.model_path
    model_name = model_info.model_name

    r: ApiConfig = create_ApiInfo_from_localModelInfo(PORT, model_name, model_path)
    from llm_common.llm_infer.call import call_openai
    call_openai(
        api_config=r,
        prompt="tell me a joke",
        system_input="You're a helpful assistant",
    )
    # 接口调用
    # test_by_port(PORT, model_name)


def test_by_port(PORT: int, model_name: str | None):
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
            "model"      : model_name,
            "messages"   : [{"role": "user", "content": "tell me a joke"}],
            "temperature": 0.7
    }
    res = requests.post(url, json=payload)
    print(res.json())

@dataclass(frozen=True)
class LocalVllmServer:
    api_config: ApiConfig
    _proc: subprocess.Popen

    def __del__(self):
        proc = self._proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def create_ApiInfo_from_localModelInfo(PORT: int, model_name: str | None, model_path: str) -> ApiConfig:
    cmd_str = f"""
CUDA_VISIBLE_DEVICES=0 \
/usr/local/bin/vllm serve {model_path} \
    --host 0.0.0.0 \
    --port {PORT} \
    --served-model-name {model_name} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
	    --reasoning-parser qwen3 \
	    --default-chat-template-kwargs  """ + """'{"enable_thinking": false}'
	    """
    log_file = make_vllm_log_file()
    print(f"vLLM log: {log_file}")
    proc = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
    )
    threading.Thread(
            target=tee_process_output,
            args=(proc, log_file),
            daemon=True,
    ).start()

    wait_for_vllm_start(port=PORT, proc=proc)
    global _current_vllm_server
    _current_vllm_server = LocalVllmServer(
        api_config=ApiConfig(
            base_url=f"http://127.0.0.1:{PORT}/v1",
            api_key="",
            model=model_name or "",
        ),
        _proc=proc,
    )
    return _current_vllm_server.api_config


if __name__ == '__main__':
    main()
