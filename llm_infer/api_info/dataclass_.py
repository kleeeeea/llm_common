import os
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from llm_common.llm_infer.load_env import ENV_FILE, load_env_file

load_env_file(ENV_FILE)
SII_API_KEY = os.environ.get("SII_API_KEY", "")


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    model_alias: Optional[str] = None
    # Whether the model accepts image inputs. Mirrors TeaCH's per-model
    # `multimodal` flag so callers no longer need that separate table.
    multimodal: bool = False

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
            # keep the display alias from defaults (env only overrides the real model).
            model_alias=defaults.model_alias,
            multimodal=defaults.multimodal,
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


DEFAULT_32B_INNOSPARK_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-8c4d814f-5b59-4db8-9e7f-0a652019c6c4?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        base_url="https://hocph9c5dmdjcpmhjqg58keda89joeoc.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Qwen3-32B-ceval",
    ),
    url_env="DEFAULT_1T_INNOSPARK_BASE_URL",
    key_env="DEFAULT_1T_INNOSPARK_API_KEY",
    model_env="DEFAULT_1T_INNOSPARK_MODEL",
)


DEFAULT_32B_OFFICIAL_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-1f4ffff2-be2a-43d5-bffd-e2dabcf6190e?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
        base_url="https://95c5555amqakcbpdm55pqapkmo5e9j8q.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="Qwen3-32B",
    ),
    url_env="DEFAULT_1T_INNOSPARK_BASE_URL",
    key_env="DEFAULT_1T_INNOSPARK_API_KEY",
    model_env="DEFAULT_1T_INNOSPARK_MODEL",
)


DEFAULT_1T_INNOSPARK_API = ApiConfig.from_env(
    defaults=ApiConfig(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-9833105f-d358-4cac-bc0e-eb9c95af3469?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        base_url="https://jhbb98d5pbdhcokomo8qqmjojdk5bcej.openapi-qb.sii.edu.cn/v1",
        api_key=SII_API_KEY,
        model="kimi",
        model_alias='Innospark-1T',
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
        multimodal=True,
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
# ---------------------------------------------------------------------------
# Configs ported from TeaCH (scripts/inference/api_configs.py).
# Entries with a concrete `api_base` become full ApiConfigs below; the proxy
# models (claude-*/gemini-*/gpt-*/doubao-*/mock) route through a vendor/gateway
# base_url supplied by the caller and are registered further down as zero-base
# ApiConfigs. `api_key_env` maps to from_env(key_env=...); where TeaCH set
# `api_model`, that becomes `model` and the dict key becomes `model_alias`; the
# `multimodal` flag is preserved on ApiConfig.multimodal.
# ---------------------------------------------------------------------------

DEEPSEEK_CHAT_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.deepseek.com",
        api_key="",
        model="deepseek-chat",
        multimodal=False,
    ),
    url_env="DEEPSEEK_BASE_URL",
    key_env="DEEPSEEK_API_KEY",
    model_env="DEEPSEEK_CHAT_MODEL",
)

DEEPSEEK_REASONER_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.deepseek.com",
        api_key="",
        model="deepseek-reasoner",
        multimodal=False,
    ),
    url_env="DEEPSEEK_BASE_URL",
    key_env="DEEPSEEK_API_KEY",
    model_env="DEEPSEEK_REASONER_MODEL",
)

KIMI_K25_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.agicto.cn/v1",
        api_key="",
        model="kimi-k2.5",
        model_alias="Kimi-K25",
        multimodal=True,
    ),
    url_env="KIMI_BASE",
    key_env="KIMI_KEY_PUBLIC",
    model_env="KIMI_MODEL",
)

MINIMAX_M27_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://bcedpgqjghjpcjkjjchem5oaopkoqbba.openapi-qb-ai.sii.edu.cn/v1",
        api_key="",
        model="MiniMax-M2.7",
        multimodal=True,
    ),
    url_env="MINIMAX_M27_BASE_URL",
    key_env="KIMI_KEY_PUBLIC",
    model_env="MINIMAX_M27_MODEL",
)

GLM5_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://5ach5c5dabhcceg5m8d8h5ahq9c8pmh5.openapi-qb-ai.sii.edu.cn/v1",
        api_key="",
        model="glm-5",
        multimodal=True,
    ),
    url_env="GLM5_BASE_URL",
    key_env="KIMI_KEY_PUBLIC",
    model_env="GLM5_MODEL",
)

GLM46V_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://d9mppg5ga5gcc8jkj85h88g8mghpjkbd.openapi-qb-ai.sii.edu.cn/v1",
        api_key="",
        model="glm-4.6v",
        multimodal=True,
    ),
    url_env="GLM46V_BASE_URL",
    key_env="GLM46V_KEY_PUBLIC",
    model_env="GLM46V_MODEL",
)

QWEN35_397B_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://cge8kkjh9jgqcmjqkgpdedqqaog8gbkb.openapi-qb-ai.sii.edu.cn/v1",
        api_key="",
        model="qwen3.5-397b",
        multimodal=True,
    ),
    url_env="QWEN35_397B_BASE_URL",
    key_env="QWEN_KEY_PUBLIC",
    model_env="QWEN35_397B_MODEL",
)

QWEN36_35B_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://8cm59gempbddcop8m5pmjbmheohoaohj.openapi-qb-ai.sii.edu.cn/v1",
        api_key="",
        model="qwen",
        model_alias="qwen3.6-35b",
        multimodal=False,
    ),
    url_env="QWEN36_35B_BASE_URL",
    key_env="QWEN_KEY_PUBLIC",
    model_env="QWEN36_35B_MODEL",
)

# Reverse mapping from a model name to its ApiConfig, so callers can look up the
# full config (base_url / api_key) given just a model string. Built by scanning
# this module's ApiConfig instances; on a model-name collision the first-defined
# config wins (globals() preserves definition order).
MODEL_TO_APICONFIG: Dict[str, ApiConfig] = {}
for _name, _value in list(globals().items()):
    if isinstance(_value, ApiConfig):
        MODEL_TO_APICONFIG.setdefault(_value.model, _value)

# ---------------------------------------------------------------------------
# Proxy / gateway models (claude-*/gpt-*/gemini-*/doubao-*/mock). These have no
# fixed base_url of their own — the caller supplies one (e.g. TeaCH's API_BASE)
# and falls back to it whenever ApiConfig.base_url is empty. Registering them
# here, together with the multimodal flag, makes this module the single source
# of truth for "which models exist" and "is this model multimodal", so callers
# (e.g. TeaCH run.py) no longer need their own MODEL_API_CONFIGS / MODELS table.
# ---------------------------------------------------------------------------
_PROXY_MULTIMODAL: Dict[str, bool] = {
    "mock": True,
    "claude-haiku-4-5": True,
    "claude-sonnet-4-5": True,
    "claude-sonnet-4-6": True,
    "claude-opus-4-5": True,
    "claude-opus-4-6": True,
    "gemini-2.0-flash": True,
    "gemini-2.5-flash": True,
    "gemini-2.5-pro": True,
    "gemini-3-flash-preview": True,
    "gemini-3-pro-preview": True,
    "gemini-3.1-pro-preview": True,
    "gpt-4o": True,
    "gpt-4.1": True,
    "gpt-5": True,
    "gpt-5.2": True,
    "doubao-seed-1-6-thinking-250715": False,
}

PROXY_MODEL_CONFIGS: Dict[str, ApiConfig] = {
    _m: ApiConfig(base_url="", api_key="", model=_m, multimodal=_mm)
    for _m, _mm in _PROXY_MULTIMODAL.items()
}

# Register proxy models too; setdefault keeps any concrete config already mapped
# under the same model name (e.g. gemini-2.5-flash → GEMINI_API wins).
for _m, _cfg in PROXY_MODEL_CONFIGS.items():
    MODEL_TO_APICONFIG.setdefault(_m, _cfg)

# Every name a caller may refer to: real model strings + display aliases.
KNOWN_MODELS: set = set(MODEL_TO_APICONFIG) | {
    cfg.model_alias for cfg in MODEL_TO_APICONFIG.values() if cfg.model_alias
}


def config_for_model(model: str) -> Optional[ApiConfig]:
    """Look up an ApiConfig by real model string first, then by model_alias."""
    cfg = MODEL_TO_APICONFIG.get(model)
    if cfg is not None:
        return cfg
    return next(
        (c for c in MODEL_TO_APICONFIG.values() if c.model_alias == model),
        None,
    )


def is_known_model(model: str) -> bool:
    """True if `model` is a registered model string or alias."""
    return model in KNOWN_MODELS


def model_is_multimodal(model: str) -> bool:
    """Whether `model` (by model string or alias) accepts image inputs.

    Unknown models default to False, matching TeaCH's ``.get("multimodal", False)``.
    """
    cfg = config_for_model(model)
    return bool(cfg.multimodal) if cfg is not None else False


def apiconfig_for_model(model: str) -> ApiConfig:
    """Return the ApiConfig registered for `model`, or raise KeyError."""
    try:
        return MODEL_TO_APICONFIG[model]
    except KeyError:
        raise KeyError(
            f"no ApiConfig registered for model {model!r}; "
            f"known models: {sorted(MODEL_TO_APICONFIG)}"
        )


def model_alias_for(model: str) -> str:
    """Display name for `model`: its registered ``model_alias`` if any, else `model`.

    Safe for arbitrary / unregistered model strings (returns them unchanged), so
    report code can call it on every model name without guarding.
    """
    cfg = MODEL_TO_APICONFIG.get(model)
    if cfg is not None and cfg.model_alias:
        return cfg.model_alias
    return model


def main():
    from llm_common.llm_infer.call_by_single_instance import call_openai
    from llm_common.llm_infer.instances import ChatCompletionRequest
    from llm_common.llm_infer.instances import LLMInferInputRecord
    text = call_openai(LLMInferInputRecord(
        prompt=os.environ.get("PROMPT", "write a paragraph with 3 words"),
        system_input=os.environ.get("SYSTEM_INPUT", "You're a helpful assistant"),
        api=DEFAULT_1T_BASELINE_API,
        chat_completion_request=ChatCompletionRequest(
            model=DEFAULT_1T_BASELINE_API.model,
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        ),
        timeout=float(os.environ.get("TIMEOUT_SECONDS", "60")),
    ))
    print(text)


if __name__ == '__main__':
    main()
