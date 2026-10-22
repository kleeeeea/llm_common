import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file

# When this file is executed directly, expose the current module under its
# package name too. Otherwise registry.py imports a second copy of ApiConfig,
# and its instances fail the isinstance check in register_api_configs.
if __name__ == "__main__":
    sys.modules.setdefault(
            "llm_common.llm_infer.api_info.dataclass_", sys.modules[__name__]
    )

load_env_file(ENV_FILE)
SII_API_KEY = os.environ.get("SII_API_KEY", "")
YJAPI_API_KEY = os.environ.get("YJAPI_KEY", "")
YJAPI_ZERO_DOLLAR_PURCHASE_KEY = os.environ.get("YJAPI_ZERO_DOLLAR_PURCHASE_KEY", "")
YJAPI_CLAUDE_KEY = os.environ.get("YJAPI_CLAUDE_KEY", "")
ECNU_API_KEY = os.environ.get("ECNU_API_KEY", "")
HCNSEC_CN_API_KEY = os.environ.get("HCNSEC_CN_API_KEY", "")
AGICTO_KEY_API_KEY = os.environ.get("AGICTO_KEY", "")
VOLCANO_API_KEY = os.environ.get("VOLCANO_API_KEY", "")


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    model_alias: Optional[str] = None
    # Whether the model accepts image inputs. Mirrors TeaCH's per-model
    # `multimodal` flag so callers no longer need that separate table.
    multimodal: bool = False
    notes_launchcmd: str | None = None

    @classmethod
    def from_env(
            cls,
            *,
            defaults: "ApiConfig",
            url_env: Optional[str] = None,
            key_env: Optional[str] = None,
            model_env: Optional[str] = None,
    ) -> "ApiConfig":
        def env_or_default(name: str, default: str) -> str:
            value = os.environ.get(name, "").strip() if name else None
            return value if value else default

        return cls(
                base_url=env_or_default(url_env, defaults.base_url),
                api_key=env_or_default(key_env, defaults.api_key),
                model=env_or_default(model_env, defaults.model),
                # keep the display alias from defaults (env only overrides the real model).
                model_alias=defaults.model_alias,
                multimodal=defaults.multimodal,
        )

    @classmethod
    def from_curl(
            cls,
            curl: str,
            *,
            model_alias: Optional[str] = None,
            multimodal: bool = False,
            notes_launchcmd=None,
    ) -> "ApiConfig":
        """Build an ApiConfig from a curl command (inverse of render_curl).

        Parses the three pieces render_curl emits:
          - the POST endpoint URL  → base_url (the trailing /chat/completions is
            stripped so it can drive an OpenAI-style client again);
          - ``Authorization: Bearer <key>``  → api_key;
          - the JSON ``-d`` body's ``"model"`` field  → model.
        """
        # ── URL → base_url ───────────────────────────────────────────────
        # Prefer the chat/completions endpoint; fall back to the first http(s)
        # URL in the command. Accepts single- or double-quoted or bare URLs.
        url_match = re.search(r'https?://\S*?/chat/completions', curl) \
                    or re.search(r'''https?://[^\s'"]+''', curl)
        if not url_match:
            raise ValueError("from_curl: 找不到请求 URL")
        url = url_match.group(0)
        base_url = re.sub(r'/chat/completions/?$', '', url)

        # ── Authorization header → api_key ───────────────────────────────
        key_match = re.search(
                r'''Authorization:\s*Bearer\s+([^\s'"]+)''', curl, re.IGNORECASE
        )
        api_key = key_match.group(1) if key_match else ""

        # ── -d / --data body → model ─────────────────────────────────────
        body_match = re.search(
                r'''(?:-d|--data(?:-raw|-binary)?)\s+(['"])(.*?)\1''',
                curl,
                re.DOTALL,
        )
        if not body_match:
            raise ValueError("from_curl: 找不到 -d 请求体")
        try:
            model = json.loads(body_match.group(2))["model"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"from_curl: 无法从请求体解析 model: {error}")

        return cls(
                base_url=base_url,
                api_key=api_key,
                model=model,
                model_alias=model_alias,
                multimodal=multimodal,
                notes_launchcmd=notes_launchcmd
        )


# @dataclass
# class APIInfo(object):
#     model_name: str
#     api_baseurl: str
#     apikey: str


# Reverse mapping from a model name to its ApiConfig, so callers can look up the
# full config (base_url / api_key) given just a model string. Built by scanning
# this module's ApiConfig instances; on a model-name collision the first-defined
# config wins (globals() preserves definition order).
MODEL_TO_APICONFIG: Dict[str, ApiConfig] = {}
# Separate alias index, so a config whose `model` collides with an earlier one
# (e.g. QWEN3_9B_API model="qwen" loses to QWEN36_35B_API in MODEL_TO_APICONFIG)
# is still reachable by its unique `model_alias` (here "qwen3.5-9b").
ALIAS_TO_APICONFIG: Dict[str, ApiConfig] = {}
# apply it to git_repos/llm_evals/llm_common/llm_infer/api_info/registry.py:22, but affect the ALIAS_TO_APICONFIG here
def register_api_configs(namespace: Dict[str, object]) -> None:
    """Register every ApiConfig in ``namespace`` in the shared indexes."""
    known_models = globals().get("KNOWN_MODELS")
    for _name, _value in list(namespace.items()):
        if isinstance(_value, ApiConfig):
            MODEL_TO_APICONFIG.setdefault(_value.model, _value)
            if _value.model_alias:
                ALIAS_TO_APICONFIG.setdefault(_value.model_alias, _value)
            if isinstance(known_models, set):
                known_models.add(_value.model)
                if _value.model_alias:
                    known_models.add(_value.model_alias)


register_api_configs(globals())

# ---------------------------------------------------------------------------
# Proxy / gateway models (claude-*/gpt-*/gemini-*/doubao-*/mock). These have no
# fixed base_url of their own — the caller supplies one (e.g. TeaCH's API_BASE)
# and falls back to it whenever ApiConfig.base_url is empty. Registering them
# here, together with the multimodal flag, makes this module the single source
# of truth for "which models exist" and "is this model multimodal", so callers
# (e.g. TeaCH run.py) no longer need their own MODEL_API_CONFIGS / MODELS table.
# ---------------------------------------------------------------------------
gemini_3_flash_preview = "gemini-3-flash-preview"
gemini___pro_preview = "gemini-3.1-pro-preview"
gpt__5_2 = "gpt-5.2"
_PROXY_MULTIMODAL: Dict[str, bool] = {
        "mock"                           : True,
        "claude-haiku-4-5"               : True,
        "claude-sonnet-4-5"              : True,
        "claude-sonnet-4-6"              : True,
        "claude-opus-4-5"                : True,
        # claude_opus_4_6                  : True,
        "gemini-2.0-flash"               : True,
        "gemini-2.5-flash"               : True,
        "gemini-2.5-pro"                 : True,
        gemini_3_flash_preview           : True,
        "gemini-3-pro-preview"           : True,
        gemini___pro_preview             : True,
        "gpt-4o"                         : True,
        "gpt-4.1"                        : True,
        "gpt-5"                          : True,
        gpt__5_2                         : True,
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
KNOWN_MODELS: set = set(MODEL_TO_APICONFIG) | set(ALIAS_TO_APICONFIG)

# ApiConfig declarations live in registry.py. Importing it here keeps the
# indexes populated for callers that only import this module; registry.py also
# calls register_api_configs after its declarations, which covers the reverse
# import order without relying on a partially initialized module.
from llm_common.llm_infer.api_info import registry as _registry

register_api_configs(vars(_registry))


def apiconfig_for_model(model: str) -> Optional[ApiConfig]:
    """Look up an ApiConfig by real model string first, then by model_alias."""
    cfg = MODEL_TO_APICONFIG.get(model)
    if cfg is not None:
        return cfg
    get = ALIAS_TO_APICONFIG.get(model)
    if get is None:
        raise ValueError(f"Unknown model: {model}")
    return get


def is_known_model(model: str) -> bool:
    """True if `model` is a registered model string or alias."""
    return model in KNOWN_MODELS


def model_is_multimodal(model: str) -> bool:
    """Whether `model` (by model string or alias) accepts image inputs.

    Unknown models default to False, matching TeaCH's ``.get("multimodal", False)``.
    """
    cfg = apiconfig_for_model(model)
    return bool(cfg.multimodal) if cfg is not None else False


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
    from llm_common.llm_infer.instances import LLMInferInputRecord
    from llm_common.llm_infer.api_info.registry import gemini3_flash_API
    from llm_common.llm_infer.api_info.registry import gpt6astra
    from llm_common.llm_infer.api_info.registry import GLM5_3_API_FROM_curl
    api = GLM5_3_API_FROM_curl
    text = call_openai(LLMInferInputRecord(
            prompt=os.environ.get("PROMPT", "你是什么模型"),
            api=api,
            # disable_thinking=True,
    ))
    print(text.llm_response)


if __name__ == '__main__':
    main()
