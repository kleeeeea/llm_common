from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.api_info.dataclass_ import register_api_configs

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

register_api_configs(globals())
