import os

from llm_common.llm_infer.call_by_single_instance import call_openai
from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.instances import ChatCompletionRequest
from llm_common.llm_infer.instances import LLMInferInputRecord
from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file
from llm_common.llm_infer.load_env import require_env
from llm_common.llm_infer.test.data.index import LLM_INFER_TEST_IMAGE_JPEG_PATH


def main() -> int:
    load_env_file(ENV_FILE)

    from llm_common.llm_infer.instances import api_innospark_cn_v_
    base_url = require_env("LLM_BASE_URL", api_innospark_cn_v_).rstrip("/")
    api_key = require_env("LLM_API_KEY", '')
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    prompt = os.environ.get("PROMPT", "describe the image using 5 words").strip()
    timeout = float(os.environ.get("TIMEOUT_SECONDS", "60"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "16000"))

    text = call_openai(LLMInferInputRecord(
        prompt=prompt,
        system_input="You're a helpful assistant",
        image_paths=(LLM_INFER_TEST_IMAGE_JPEG_PATH,),
        api=ApiConfig(base_url=base_url, api_key=api_key, model=model),
        chat_completion_request=ChatCompletionRequest(
            max_tokens=max_tokens,
        ),
        timeout=timeout,
    ))

    print("\nOK: returned content.")
    print(type(text))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
