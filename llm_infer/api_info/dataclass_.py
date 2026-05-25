import os
from dataclasses import dataclass

from llm_common.llm_infer.load_env import ENV_FILE, load_env_file

load_env_file(ENV_FILE)
SII_API_KEY = os.environ.get("SII_API_KEY", "")


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(
        cls,
        *,
        defaults: "ApiConfig",
        url_env: str,
        key_env: str,
        model_env: str,
    ) -> "ApiConfig":
        def env_or_default(name: str, default: str) -> str:
            value = os.environ.get(name, "").strip()
            return value if value else default

        return cls(
            base_url=env_or_default(url_env, defaults.base_url),
            api_key=env_or_default(key_env, defaults.api_key),
            model=env_or_default(model_env, defaults.model),
        )

# @dataclass
# class APIInfo(object):
#     model_name: str
#     api_baseurl: str
#     apikey: str

DEFAULT_1T_BASELINE_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-53234a68-283c-482d-ae93-2f11c0701f49?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
        base_url="https://qobqj89deqk5chm9heeqm59eo8dpekkg.openapi-qb-ai.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Kimi-K2.6",
    ),
    url_env="DEFAULT_1T_BASELINE_BASE_URL",
    key_env="DEFAULT_1T_BASELINE_API_KEY",
    model_env="DEFAULT_1T_BASELINE_MODEL",
)


DEFAULT_4B_INNOSPARK_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://e8p5ocom8hcgcecckoc8jbhhohqhahhg.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Qwen3-4B-Instruct-2507",
    ),
    url_env="DEFAULT_4B_INNOSPARK_BASE_URL",
    key_env="DEFAULT_4B_INNOSPARK_API_KEY",
    model_env="DEFAULT_4B_INNOSPARK_MODEL",
)


DEFAULT_4B_BASELINE_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-667f46c5-314e-46ac-87e8-9ee38d91fab3?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        base_url="https://eeg5ceodb9cqcekohgqhjqqbhpj95kmb.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Qwen3-4B-Instruct-2507-Official",
    ),
    url_env="DEFAULT_4B_BASELINE_BASE_URL",
    key_env="DEFAULT_4B_BASELINE_API_KEY",
    model_env="DEFAULT_4B_BASELINE_MODEL",
)


DEFAULT_1T_INNOSPARK_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-9833105f-d358-4cac-bc0e-eb9c95af3469?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        base_url="https://jhbb98d5pbdhcokomo8qqmjojdk5bcej.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="kimi",
    ),
    url_env="DEFAULT_1T_INNOSPARK_BASE_URL",
    key_env="DEFAULT_1T_INNOSPARK_API_KEY",
    model_env="DEFAULT_1T_INNOSPARK_MODEL",
)


DEFAULT_INNOSPARK_API = ApiConfig.from_env(
    defaults=DEFAULT_1T_INNOSPARK_API,
    url_env="DEFAULT_INNOSPARK_BASE_URL",
    key_env="DEFAULT_INNOSPARK_API_KEY",
    model_env="DEFAULT_INNOSPARK_MODEL",
)

DEFAULT_BASELINE_API = DEFAULT_1T_BASELINE_API

GEMINI_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.innospark.cn/v1",
        api_key="",
        model="gemini-2.5-flash",
    ),
    url_env="GEMINI_BASE_URL",
    key_env="LLM_API_KEY",
    model_env="GEMINI_MODEL",
)
DEFAULT_JUDGE_API = GEMINI_API
Qwen27b_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://ea5maamppajpchmpmqk5obbpemep5p5k.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Qwen3.6-27B",
    ),
    url_env="QWEN27B_BASE_URL",
    key_env="QWEN27B_API_KEY",
    model_env="QWEN27B_MODEL",
)

GLM51FP8_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d46fd98b-a669-4acb-92b6-85966acbf383?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        base_url="https://pce5pjhmkeejckomjamehdjeekom5bhb.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="GLM-5.1-FP8",
    ),
    url_env="GLM51FP8_BASE_URL",
    key_env="GLM51FP8_API_KEY",
    model_env="GLM51FP8_MODEL",
)


def main():
    from llm_common.llm_infer.call import call_openai
    text = call_openai(
        api_key=DEFAULT_1T_BASELINE_API.api_key,
        base_url=DEFAULT_1T_BASELINE_API.base_url,
        model=DEFAULT_1T_BASELINE_API.model,
        prompt=os.environ.get("PROMPT", "write a paragraph with 3 words"),
        system_input=os.environ.get("SYSTEM_INPUT", "You're a helpful assistant"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        timeout=float(os.environ.get("TIMEOUT_SECONDS", "60")),
    )
    print(text)


if __name__ == '__main__':
    main()
