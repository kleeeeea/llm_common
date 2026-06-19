import json
import os
import re
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from llm_common.llm_infer.load_env import ENV_FILE
from llm_common.llm_infer.load_env import load_env_file

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
        url_env: Optional[str]=None,
        key_env: Optional[str]=None,
        model_env: Optional[str]=None,
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


claude_opus_4_6_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.innospark.cn/v1",
        api_key="",
        model='claude-opus-4-6',
        multimodal=True,
    ),
    url_env="GEMINI_BASE_URL",
    key_env="LLM_API_KEY",
    model_env="GEMINI_MODEL",
)

GEMINI_2_0_FLASH_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.innospark.cn/v1",
        api_key="",
        model="gemini-2.0-flash",
        multimodal=True,
    ),
    url_env="GEMINI_BASE_URL",
    key_env="LLM_API_KEY",
    model_env="GEMINI_MODEL",
)

GEMINI_2_5_FLASH_API = ApiConfig.from_env(
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

doubao_seed_2_0_lite_API = ApiConfig.from_env(
    defaults=ApiConfig(
        base_url="https://api.innospark.cn/v1",
        api_key="",
        model="doubao-seed-2-0-lite-260215",
    ),
    key_env="LLM_API_KEY",
)


DEFAULT_JUDGE_API = GEMINI_2_5_FLASH_API
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

KIMI_K25_API = ApiConfig.from_curl('''
curl -sS --fail -X POST "https://daapgkedka89cgphj5peoak85c5dmgkk.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
        "model": "kimi",
        "stream": true,
        "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'

''', model_alias='Kimi-K25')

KIMI_K251_API = ApiConfig.from_curl('''
curl -sS --fail -X POST "https://api.agicto.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-gkYa8UuT55XckVX3SGFsamidZMTzDfismSCU9i3YWJB8hs2S" \
    -d '{
        "model": "kimi-k2.5",
        "stream": true,
        "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'

''', model_alias='Kimi-K251')


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

'''
curl -sS --fail -X POST "https://qombekoaqjdbcaqqmh8k5h9makpdhpda.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "glm",
      "messages": [
        { "role": "user", "content": "你是什么模型" }
      ],
      "thinking": {"type": "disabled"},
      "chat_template_kwargs": {"enable_thinking": false},
      "enable_thinking": false
    }'
        '''

ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://ppag9oqb5bgqchqmmgqmg9goq98e9d9j.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "PaddleOCR-VL-0.9B",
      "messages": [
        { "role": "user", "content": "你是什么模型" }
      ],
      "enable_thinking": false
    }'
        ''',
    model_alias="qwen3.5-397b",
)

QWEN35_397B_API = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://dj5ghabgo8dkcqh5hhjmao58qbhaema5.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen397",
      "messages": [
        { "role": "user", "content": "hi" }
      ],
      "thinking": {"type": "disabled"},
      "chat_template_kwargs": {"enable_thinking": false},
      "enable_thinking": false
    }'

        ''',
    model_alias="qwen3.5-397b",
)

QWEN3_27B_API = ApiConfig.from_curl(

'''
curl -sS --fail -X POST "https://hadpekkpkjekcd8gk5jdmgmc9qkg9ppc.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen3.6-27b",
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
''',model_alias = 'qwen3.6-27b'
)

kimi_vl_a_3_b_instruct = ApiConfig.from_curl(

'''
curl -sS --fail -X POST "https://8ooeepbkqbe5cjgbmdajghm9peho8b8c.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Kimi-VL-A3B-Instruct",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
''',model_alias = 'Kimi-VL-A3B-Instruct'
)

mi_mo_vl_7_b_sft = ApiConfig.from_curl(

'''
curl -sS --fail -X POST "https://kjjmghgmdkeacmgqhpbocjjdkdmk5hpg.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "MiMo-VL-7B-SFT",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
''',model_alias = 'MiMo-VL-7B-SFT'
)
step_3_vl_10_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://bgamqga5gogjcb9gmqeaohh8hdhbgddm.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Step3-VL-10B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)


qwen_3_5_4_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://mopmqcgg8j9qccc5m8ppqchgpeadkcam.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Qwen3.5-4B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)


intern_vl_3_5_8_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://qod5gecmhdkmchm9kd95pqp5qpod89ga.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "InternVL3_5-8B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

intern_vl_3_4_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://ppagqmjmcbpmco5bhpgqhcomhqdje8hp.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "InternVL3_5-4B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)


intern_vl_3_2_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://8ghmddbph8gece8hkq5kbokhchcahoop.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "InternVL3_5-2B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

intern_vl_3_14_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://9hj8ceppcaemckkmjk8q98pj995e558a.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "InternVL3_5-14B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

gemma_4_a_2_b_it = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://5jj9m99phggaceqamhokhkh9mhka9qhh.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "gemma-4-E2B-it",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

gemma_4_31_b_it = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://mpeghck9aqm5cqcdjjooqecpo5cj99c5.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "gemma-4-31B-it",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

gemma_4_26_b_a_4_b_it = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://okap9codcoamcmgehjp8pgdpm5kchc8d.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "gemma-4-26B-A4B-it",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)
gemma_4_12_b_it = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://pkqjqjjpcoacckdbkbeae9medpcdd8cg.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "gemma-4-12B-it",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        '''
)

molmo_2_8_b = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://p8ajcc5dpg95c5pckmg8dmcapdkhghkc.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Molmo2-8B",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        ''')

phi_3_5_vision_instruct = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://pqaqocbbmme8cg9kjegbbdee8am98g5d.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Phi-3.5-vision-instruct",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        ''')


QWEN3_35B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-04bf176b-d355-4cd0-bfdb-93b86c8c2c89?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d8e1782e-7440-49c8-a14c-889423a29ae5?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
  curl -sS --fail -X POST "https://ohpmm8jhhkj8cqegjjd8c5bcqbb8ophc.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Qwen3.5-35B-A3B",
      "messages": [
        { "role": "user", "content": "write a poem in 3 words" }
      ]
    }'
        ''',model_alias = 'qwen3.6-35b'
)

QWEN3_9B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-aa43984b-be99-45f3-9651-bdd9ec9ce147?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        '''
  curl -sS --fail -X POST "https://kkcbjhcmmqjjcd5bjed9mppjqojoq9cg.openapi-qb-ai.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen",
      "messages": [
        { "role": "user", "content": "write a poem in 3 words" }
      ]
    }'
        ''',model_alias = 'qwen3.5-9b'
)
# Reverse mapping from a model name to its ApiConfig, so callers can look up the
# full config (base_url / api_key) given just a model string. Built by scanning
# this module's ApiConfig instances; on a model-name collision the first-defined
# config wins (globals() preserves definition order).
MODEL_TO_APICONFIG: Dict[str, ApiConfig] = {}
# Separate alias index, so a config whose `model` collides with an earlier one
# (e.g. QWEN3_9B_API model="qwen" loses to QWEN36_35B_API in MODEL_TO_APICONFIG)
# is still reachable by its unique `model_alias` (here "qwen3.5-9b").
ALIAS_TO_APICONFIG: Dict[str, ApiConfig] = {}
for _name, _value in list(globals().items()):
    if isinstance(_value, ApiConfig):
        MODEL_TO_APICONFIG.setdefault(_value.model, _value)
        if _value.model_alias:
            ALIAS_TO_APICONFIG.setdefault(_value.model_alias, _value)

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
KNOWN_MODELS: set = set(MODEL_TO_APICONFIG) | set(ALIAS_TO_APICONFIG)


def config_for_model(model: str) -> Optional[ApiConfig]:
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
    cfg = config_for_model(model)
    return bool(cfg.multimodal) if cfg is not None else False

apiconfig_for_model = config_for_model

# def apiconfig_for_model(model: str) -> ApiConfig:
#     """Return the ApiConfig registered for `model`, or raise KeyError."""
#     try:
#         return MODEL_TO_APICONFIG[model]
#     except KeyError:
#         raise KeyError(
#             f"no ApiConfig registered for model {model!r}; "
#             f"known models: {sorted(MODEL_TO_APICONFIG)}"
#         )


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
    api = config_for_model('qwen3.5-397b')
    print('*' * 50 + f'''\n{api.api_key}\n^^^(doubao_seed_2_0_lite_API.api_key)^^^\n''' + '''\nat:\ndependencies/llm_evals/llm_common/llm_infer/api_info/dataclass_.py:396\n''' + '*' * 50)


    text = call_openai(LLMInferInputRecord(
        prompt=os.environ.get("PROMPT", "write a paragraph with 3 words"),
        api=api,
            disable_thinking=True,
    ))
    print(text)


if __name__ == '__main__':
    main()
