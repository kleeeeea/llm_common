import sys
from dataclasses import replace

from llm_common.llm_infer.api_info.dataclass_ import AGICTO_KEY_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.api_info.dataclass_ import ECNU_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import HCNSEC_CN_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import SII_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import VOLCANO_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import YJAPI_API_KEY
from llm_common.llm_infer.api_info.dataclass_ import YJAPI_CLAUDE_KEY
from llm_common.llm_infer.api_info.dataclass_ import YJAPI_ZERO_DOLLAR_PURCHASE_KEY
from llm_common.llm_infer.api_info.dataclass_ import register_api_configs
ismacos = sys.platform == "darwin"

qwen397_hcnsec = 'qwen397_hcnsec'

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

# claude_opus_4_6_API = ApiConfig.from_env(
#         defaults=ApiConfig(
#                 base_url="https://api.innospark.cn/v1",
#                 api_key="",
#                 model='claude-opus-4-6',
#                 multimodal=True,
#         ),
#         url_env="GEMINI_BASE_URL",
#         key_env="LLM_API_KEY",
#         model_env="GEMINI_MODEL",
# )
claude_opus_4_6 = "claude-opus-4-6"

gpt6astra=ApiConfig.from_curl('''
curl -sS -N -X POST \
  --max-time 120 \
  https://api.anyint.ai/openai/v1/chat/completions \
  -H "Authorization: Bearer sk-qNAdvklUJkSnYDkpzCxp5vhaHPH8QIyLkTLNe5F8MHZHqo0TnbNF" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-6-astra",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'
''')
gemini3_flash_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://www.shenlanqaq.com//v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_API_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemini-3-flash",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')
gemini_3_8_flash_high_flash_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_API_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemini-3.8-flash-high",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')
gpt_5_6_sol_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_ZERO_DOLLAR_PURCHASE_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-5.6-sol",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')
claude_opus_5_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_CLAUDE_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "claude-opus-5",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')
claude_sonnet_5_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_CLAUDE_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "claude-sonnet-5",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')

gptluna_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_ZERO_DOLLAR_PURCHASE_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-5.6-terra",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''')

claude_opus_4_6_API = ApiConfig.from_curl(f'''
curl -sS -N -X POST \
  --max-time 120 \
  https://yjapi.manqiaotechnology.com/v1/chat/completions \
  -H "Authorization: Bearer {YJAPI_API_KEY}" \\ ''' + '''
  -H 'Content-Type: application/json' \
  -d '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "hi",
          "type": "text"
        }
      ]
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'

''', model_alias=claude_opus_4_6)
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

deepseekv__hcnsec = 'deepseekv4_hcnsec'

Deepseekv4flash_API_FROM_curl = ApiConfig.from_curl(
        f'''

curl https://api.hcnsec.cn/v1/chat/completions \
  -H "Authorization: Bearer {HCNSEC_CN_API_KEY}" \\ ''' + '''
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
    "messages": [
      {
        "role": "system",
        "content": "你是一个专业、简洁的助手。"
      },
      {
        "role": "user",
        "content": "hi。"
      }
    ],
    "temperature": 0.7,
    "stream": true
  }'

''',
        model_alias=deepseekv__hcnsec
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d46fd98b-a669-4acb-92b6-85966acbf383?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
)

GLM5_3_API_FROM_curl = ApiConfig.from_curl(
        f'''
curl https://ark.cn-beijing.volces.com/api/plan/v1/messages \
  -H "Authorization: Bearer {VOLCANO_API_KEY}" \\ ''' + '''
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "glm-5.2",
    "max_tokens": 1024,
    "messages": [
      {"role":"user","content":"你是什么模型"}
    ]
  }'


''',
        model_alias='glm_5_1'
)
# https://docs.volcengine.com/docs/82379/1928262?utm_source=chatgpt.com&lang=zh
# https://console.volcengine.com/ark/region:cn-beijing/subscription/agent-plan


kimik25_hcnsec = 'kimik25_hcnsec'
Kimik25_hcn_API_FROM_curl = ApiConfig.from_curl(
        f'''

curl https://api.hcnsec.cn/v1/chat/completions \
  -H "Authorization: Bearer {HCNSEC_CN_API_KEY}" \\ ''' + '''
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K2.5",
    "messages": [
      {
        "role": "system",
        "content": "你是一个专业、简洁的助手。"
      },
      {
        "role": "user",
        "content": "hi。"
      }
    ],
    "temperature": 0.7,
    "stream": true
  }'

''',
        model_alias=kimik25_hcnsec
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d46fd98b-a669-4acb-92b6-85966acbf383?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
)

qwen3_27_qz = '''
        curl -sS --fail -X POST "https://hadpekkpkjekcd8gk5jdmgmc9qkg9ppc.openapi-sj.sii.edu.cn/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
            -d '{
              "model": "qwen3.6-27b",
                "stream": true,
              "messages": [
                { "role": "user", "content": "hi" }
              ]
            }'
        '''
QWEN3_27B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-825d6aad-a96a-4507-af95-b7b41ea71dd4
        f'''
        curl https://chat.ecnu.edu.cn/open/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {ECNU_API_KEY}" \\ ''' + '''
  -d '{
    "model": "ecnu-plus",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己"
      }
    ],
    "temperature": 0.7
  }'
''', model_alias='qwen3.6-27b', notes_launchcmd='''
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export GLOO_SOCKET_IFNAME=lo

vllm serve /inspire/sj-ssd3/project/ai4education/public/models/Qwen3.6-27B \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --served-model-name qwen3.6-27b \
  --reasoning-parser qwen3 \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' 
'''
)
QWEN3_9B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-223a04a9-9b65-482a-9c32-c5df9709a552?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
  curl -sS --fail -X POST "https://aabk8ckjjmedchakjobaodp5jg8jkdda.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen3.5-9b",
      "stream": true,
      "messages": [
        { "role": "user", "content": "write a poem in 3 words" }
      ]
    }'
        ''', model_alias='qwen3.5-9b',
        notes_launchcmd='''
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

unset VLLM_USE_MODELSCOPE

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export GLOO_SOCKET_IFNAME=lo

vllm serve /inspire/sj-ssd3/project/ai4education/public/models/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --served-model-name qwen3.5-9b \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
'''
)



qwen397_hcn_API_FROM_curl = ApiConfig.from_curl(
        f'''

curl https://api.hcnsec.cn/v1/chat/completions \
  -H "Authorization: Bearer {HCNSEC_CN_API_KEY}" \\ ''' + '''
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-397B-A17B",
    "messages": [
      {
        "role": "system",
        "content": "你是一个专业、简洁的助手。"
      },
      {
        "role": "user",
        "content": "hi。"
      }
    ],
    "temperature": 0.7,
    "stream": true
  }'

''',
        model_alias=qwen397_hcnsec
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d46fd98b-a669-4acb-92b6-85966acbf383?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
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

KIMI_K25_QZ_API = ApiConfig.from_curl(
        '''
    curl -sS --fail -X POST "https://eojmhg88q8ojcg98kd5q8geoqeeeag95.openapi-sj.sii.edu.cn/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
        -d '{
            "model": "kimi",
            "stream": true,
            "messages": [
            { "role": "user", "content": "hi" }
          ]
        }'
''',
        model_alias='Kimi-K252'
)
'''
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH && \
source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
source /usr/local/Ascend/nnal/atb/set_env.sh && \
export PATH=/opt/mamba/envs/ascend-infer/bin:$PATH && \
cd /inspire/sj-ssd3/project/ai4education/public/run_inference_llama3_2_11B && \
export PROJECT_DIR=/inspire/sj-ssd3/project/ai4education/public/run_inference_llama3_2_11B && \
export MODEL_PATH=/inspire/sj-ssd3/project/ai4education/public/models/Llama-3.2-11B-Vision-Instruct && \
bash serve.sh
'''
'ascend-a2-ubuntu'
llama_3_2_11_b_vision_instruct = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-7af3f3cc-6883-4619-94d9-1616debf2755?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://gk5b5booehhbce9jjcpj9e9epqm5agdd.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
        "model": "Llama-3.2-11B-Vision-Instruct",
        "stream": true,
        "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'

''',
        model_alias='Llama-3.2-11B-Vision-Instruct'
)

kimi_k2_5_agicto = 'Kimi-K25-agicto'
KIMI_K25_AGICTO_API = ApiConfig.from_curl(f'''
curl -sS --fail -X POST "https://api.agicto.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
  -H "Authorization: Bearer {AGICTO_KEY_API_KEY}" \\ ''' + '''
    -d '{
        "model": "kimi-k2.5",
        "stream": true,
        "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'

''', model_alias=kimi_k2_5_agicto)

qwen_3_5_397_b = 'qwen3.5-397b'
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

GLM46V_OFFICIAL_API = ApiConfig.from_curl(
        #     支持这种v4的custom url
        '''
        
    curl -sS --fail -X POST "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ede4a85f26aa42e7a17a6379d0134dd7.WNvvHwlVkhqxg8Gt" \
        -d '{
          "model": "glm-4.6v",
            "stream": true,
          "messages": [
            { "role": "user", "content": "你是什么模型" }
          ]
        }'
    
        ''', model_alias="glm-4.6v1"
)

'''
source /opt/mamba/etc/profile.d/conda.sh
conda activate ascend-infer
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export GLOO_SOCKET_IFNAME=lo
vllm serve "/inspire/sj-ssd3/project/ai4education/ky26140/glm4.6V"   --tensor-parallel-size 4   --data-parallel-size 4 --enable-expert-parallel   --seed 1024   --served-model-name glm   --max-model-len 25600   --max-num-batched-tokens 8192   --max-num-seqs 16   --trust-remote-code   --port ${PORT}   --gpu-memory-utilization 0.9
'''
'vllm-ascend'
GLM46V_API = ApiConfig.from_curl(
        #     https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-3d53e366-2716-4a44-ac8e-0ba13ee808f2?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://bpdbqbeeag9eckcckohq8mk9bqqh8h5a.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "glm",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        ''', model_alias="glm-4.6v2",

)

GLM46V_CURRENT = replace(GLM46V_OFFICIAL_API if ismacos else GLM46V_API, model_alias="glm-4.6v")

qwen_397b_agitcto = "qwen3.5-397b-agitcto"
QWEN35_397B_AGICTO_API = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://api.agicto.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-gkYa8UuT55XckVX3SGFsamidZMTzDfismSCU9i3YWJB8hs2S" \
    -d '{
        "model": "qwen3.5-397b-a17b",
        "stream": true,
        "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
    '''
        ,
        model_alias=qwen_397b_agitcto,
)
# pdaj8d5bm9hcckmkjbpkoekpe8q5pedp
QWEN35_397B_QZ_API = ApiConfig.from_curl(
        '''
curl -sS --fail -X POST "https://cd.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen397",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        ''',
        model_alias="qwen3.5-397b2",
)
# qwen397_hcn_API_FROM_curl
# QWEN35_397B_AGICTO_API
QWEN35_397B_CURRENT = replace(qwen397_hcn_API_FROM_curl if ismacos else QWEN35_397B_QZ_API, model_alias="qwen3.5-397b")

kimik25_EDU = "Kimi-K25"
KIMI_K25_CURRENT = replace(KIMI_K25_AGICTO_API if ismacos else KIMI_K25_QZ_API, model_alias=kimik25_EDU)
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/Kimi-VL-A3B-Thinking" \
  --host 0.0.0.0 \
  --port ${PORT}\
  --served-model-name Kimi-VL-A3B-Thinking \
  --trust-remote-code
  '''
kimi_vl_a_3_b_instruct = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-990f27b2-0636-43e7-b4b2-0e3000594782
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
        ''', model_alias='Kimi-VL-A3B-Instruct'
)
'''
MODEL_PATH=/inspire/sj-ssd3/project/ai4education/public/models/MiMo-VL-7B-SFT
vllm serve "$MODEL_PATH" --port 8012 --served-model-name MiMo-VL-7B-SFT
'''
mi_mo_vl_7_b_sft = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-a99b787b-5a62-4b7e-8d3c-4995a5baf8d7
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
        ''', model_alias='MiMo-VL-7B-SFT'
)
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/Step3-VL-10B" \
  --host 0.0.0.0 \
  --port 8025 \
  --served-model-name Step3-VL-10B \
  --trust-remote-code
'''
step_3_vl_10_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-372be72f-7c79-4317-be62-c8808583755b?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
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
'''
MODEL_PATH=/inspire/sj-ssd3/project/ai4education/public/models/Qwen3.5-4B
vllm serve "$MODEL_PATH" --port 8000 --served-model-name Qwen3.5-4B
'''

qwen_3_5_4_b_local_modelname = 'Qwen3.5-4B-Local'
qwen_3_5_4_b_local_apiconfig = ApiConfig.from_curl(
        '''
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
    -H "Authorization: Bearer notused1123355" \
  -d '{
    "model": "qwen3.5:4b",
    "messages": [
      {
        "role": "user",
        "content": "hi"
      }
    ],
        "stream": true,
    "temperature": 0.7
  }'

        ''',
        model_alias=qwen_3_5_4_b_local_modelname
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-bcdbe477-091e-444f-b0dd-e487d625a42b
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
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-bcdbe477-091e-444f-b0dd-e487d625a42b
)
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/InternVL3_5-2B" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --served-model-name InternVL3_5-2B \
  --trust-remote-code
'''
intern_vl_3_2_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-078b553d-8076-4608-aefa-3ca848428ccd
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
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/InternVL3_5-8B" \
  --host 0.0.0.0 \
  --port 8024 \
  --served-model-name InternVL3_5-8B \
  --trust-remote-code
'''
intern_vl_3_5_8_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-ec5f764b-85ab-48b1-a51f-dedfedc50172
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
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/InternVL3_5-4B" \
  --host 0.0.0.0 \
  --port 8021 \
  --served-model-name InternVL3_5-4B \
  --trust-remote-code
'''
intern_vl_3_4_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-dd27eb9b-43db-4cf3-8d7e-84cb8e59608d
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
'''
vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/InternVL3_5-14B" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --served-model-name InternVL3_5-14B \
  --trust-remote-code
'''
intern_vl_3_14_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-189046dd-426b-4aab-9a0e-10d911f6ff02
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
'''
cd /workspace/vllm
git apply /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral/patches/vllm_pixtral_npu_vision.patch


cd /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral

export MISTRAL_MODEL_ROOT=/inspire/sj-ssd3/project/ai4education/public/models
export PORT=8001
MODEL_ROOT="${MISTRAL_MODEL_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality-ascend/public/models/mistralai}"
MODEL_PATH="$MODEL_ROOT/Ministral-3-3B-Instruct-2512-BF16"
SERVED_NAME="ministral-3-3b"


# ---- 昇腾环境 ----
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export GLOO_SOCKET_IFNAME=lo

# ---- ascend-infer (镜像自带, 绝对路径规避 conda activate 非交互 shell 问题) ----
VLLM="/opt/mamba/envs/ascend-infer/bin/vllm"

# ---- 推理性能环境变量 ----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV

echo "[serve] model=$SERVED_NAME  port=$PORT"
echo "[serve] model_path=$MODEL_PATH"
echo "[serve] engine=vllm + vllm-ascend"
echo "----------------------------------------"

exec "$VLLM" serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm- '{"image":1}' \
  "$@"
'''
'ascend-a2-ubuntu'
ministral_3_3_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-896b6b78-3fbd-48ba-913f-47b822f38539?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://hjememghb5mpchmkj9b5cgmhaa5bhdbj.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "ministral-3-3b",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        '''
)
'''
cd /workspace/vllm
git apply /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral/patches/vllm_pixtral_npu_vision.patch


cd /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral
export PORT=8003
export MISTRAL_MODEL_ROOT=/inspire/sj-ssd3/project/ai4education/public/models



# ---- 模型路径 (换分区时用环境变量 MISTRAL_MODEL_ROOT 覆盖) ----
MODEL_ROOT="${MISTRAL_MODEL_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality-ascend/public/models/mistralai}"
MODEL_PATH="$MODEL_ROOT/Ministral-3-8B-Instruct-2512-BF16"
SERVED_NAME="ministral-3-8b"

# ---- 昇腾环境 ----
set +u  # set_env.sh 引用未定义变量, 关闭 nounset
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export GLOO_SOCKET_IFNAME=lo

# ---- ascend-infer (镜像自带, 绝对路径规避 conda activate 非交互 shell 问题) ----
VLLM="/opt/mamba/envs/ascend-infer/bin/vllm"

# ---- 推理性能环境变量 ----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV

echo "[serve] model=$SERVED_NAME  port=$PORT"
echo "[serve] model_path=$MODEL_PATH"
echo "[serve] engine=vllm + vllm-ascend"
echo "----------------------------------------"

exec "$VLLM" serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm- '{"image":1}' \
  "$@"
'''
'ascend-a2-ubuntu'
ministral_3_8_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-e8993f53-b3d1-4c6f-95ae-2e590ca08384?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://qhjjb5dbmbp9coe5jdkqaqdj8ok8hbhc.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "ministral-3-8b",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        '''
)
'''
cd /workspace/vllm
git apply /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral/patches/vllm_pixtral_npu_vision.patch


cd /inspire/sj-ssd3/project/ai4education/public/run_inference_Ministral

export MISTRAL_MODEL_ROOT=/inspire/sj-ssd3/project/ai4education/public/models

#!/usr/bin/env bash
# ============================================================================
# Ministral-3-14B-Instruct-2512-BF16  NPU 推理服务 (vLLM)
# ----------------------------------------------------------------------------
# 推理路径: vllm serve (OpenAI 兼容 API, 原生支持多模态)
# 运行环境: ascend-infer (镜像自带, vllm + vllm-ascend)
# 端口:     8003
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 模型路径 (换分区时用环境变量 MISTRAL_MODEL_ROOT 覆盖) ----
MODEL_ROOT="${MISTRAL_MODEL_ROOT:-/inspire/sj-ssd3/project/embodied-multimodality-ascend/public/models/mistralai}"
MODEL_PATH="$MODEL_ROOT/Ministral-3-14B-Instruct-2512-BF16"
SERVED_NAME="ministral-3-14b"
PORT=8003

# ---- 昇腾环境 ----
set +u  # set_env.sh 引用未定义变量, 关闭 nounset
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export GLOO_SOCKET_IFNAME=lo

# ---- ascend-infer (镜像自带, 绝对路径规避 conda activate 非交互 shell 问题) ----
VLLM="/opt/mamba/envs/ascend-infer/bin/vllm"

# ---- 推理性能环境变量 ----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV

echo "[serve] model=$SERVED_NAME  port=$PORT"
echo "[serve] model_path=$MODEL_PATH"
echo "[serve] engine=vllm + vllm-ascend"
echo "----------------------------------------"

exec "$VLLM" serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm- '{"image":1}' \
  "$@"
'''
ministral_3_14_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-2f2a8f6a-a6c3-472e-bbd5-31d927caf3e3?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://a5akh5ekkeobcgaqmmpdb9pjagok5bqb.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "ministral-3-14b",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        '''
)

gemma_4_a_2_b_it = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-f991b11d-8772-46e2-b8ca-8a81b8a21e88
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
        
        ''',
        notes_launchcmd='''
        HCCL_OP_EXPANSION_MODE=AIV \
vllm serve /inspire/sj-ssd3/project/ai4education/public/models/gemma-4-E2B-it \
  --port ${PORT} \
  --served-model-name gemma-4-E2B-it \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        '''
)

gemma_4_31_b_it = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-7a3f0206-bdd8-4a75-85e3-2bd8c67cb195
        '''
curl -sS --fail -X POST "https://gkb58a8empphckgdhdqbamphoegom9jd.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "gemma-4-31B-it",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        ''',
        notes_launchcmd='''
        export HCCL_OP_EXPANSION_MODE=AIV
export GLOO_SOCKET_IFNAME=lo

vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/gemma-4-31B-it" \
  --port ${PORT} \
  --served-model-name gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' 

        '''
)

gemma_4_26_b_a_4_b_it = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-ca2d14c5-4c2b-4b76-89d0-d75dbfa48405
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
        
        ''',notes_launchcmd='''
        export HCCL_OP_EXPANSION_MODE=AIV
export GLOO_SOCKET_IFNAME=lo

HCCL_BUFFSIZE=256 \
vllm serve /inspire/sj-ssd3/project/ai4education/public/models/gemma-4-26B-A4B-it \
  --port ${PORT} \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 2 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' 
        '''
)
gemma_4_12_b_it = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-dae9e99d-4c24-4a53-a362-61b65d455047
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
        
        ''', notes_launchcmd='''
export HCCL_OP_EXPANSION_MODE=AIV
export GLOO_SOCKET_IFNAME=lo

vllm serve /inspire/sj-ssd3/project/ai4education/public/models//gemma-4-12B-it \
  --port ${PORT} \
  --served-model-name gemma-4-12B-it \
  --tensor-parallel-size 2 \
  --data-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        '''
)

molmo_2_8_b = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d02944f5-d71f-4fd4-ab70-5b42d5a878a4
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
        
        ''', notes_launchcmd='''
        vllm serve "/inspire/sj-ssd3/project/ai4education/public/models/Molmo2-8B" \
--host 0.0.0.0 \
--port ${PORT} \
--served-model-name Molmo2-8B \
--trust-remote-code
''')

phi_3_5_vision_instruct = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-de2ec433-bb60-471a-9673-356602b107f5?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
curl -sS --fail -X POST "https://mjkgp8mc5qdbckjgjb9om9ebmm9o85d9.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "Phi-3.5-vision-instruct",
        "stream": true,
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
        
        ''',
        notes_launchcmd='''
        vllm serve /inspire/sj-ssd3/project/ai4education/public/models/Phi-3.5-vision-instruct   --port ${PORT} \
--served-model-name Phi-3.5-vision-instruct --trust-remote-code
'''
)

QWEN3_5_35B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-c8dbb098-8a90-4e67-9950-4f34e330cd84?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d8e1782e-7440-49c8-a14c-889423a29ae5?spaceId=ws-803be1bb-da46-40d8-ae72-df77df9112ca
        '''
  curl -sS --fail -X POST "https://phq9ghaqgcc8cjohk9cohhjcabkajkqd.openapi-sj.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen3.5-35b-a3b",
      "messages": [
        { "role": "user", "content": "write a poem in 3 words" }
      ]
    }'
        ''', model_alias='qwen3.6-35b',
        notes_launchcmd='''
        # 加载 conda 环境
        CONDA_BASE=/opt/mamba
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        
        # 强制激活环境
        conda activate ascend-infer
        
        # Ascend 环境
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        
        export LD_LIBRARY_PATH=/usr/local/Ascend/driver/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
        export HCCL_INTRA_ROCE_ENABLE=1
        
        # 用 conda run 确保在环境内执行 vllm（最稳）
        conda run -n ascend-infer --no-capture-output vllm serve /inspire/sj-ssd3/project/ai4education/public/models/Qwen3.5-35B-A3B --port ${PORT} \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --seed 1024 \
        --served-model-name Qwen3.5-35B-A3B \
        --max-num-seqs 32 \
        --max-model-len 133000 \
        --max-num-batched-tokens 8096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --no-enable-prefix-caching \
        --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}' \
        --distributed-executor-backend mp \
        --async-scheduling 

        '''
)

QWEN3_6_35B_API = ApiConfig.from_curl(
        # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-04bf176b-d355-4cd0-bfdb-93b86c8c2c89?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
        '''
  curl -sS --fail -X POST "https://8cm59gempbddcop8m5pmjbmheohoaohj.openapi-qb-ai.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=" \
    -d '{
      "model": "qwen",
      "messages": [
        { "role": "user", "content": "write a poem in 3 words" }
      ]
    }'
        ''', model_alias='qwen3.6-35b-latest',
        notes_launchcmd='''
        SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
sglang serve \
  --model-path /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.6-35B-A3B \
--served-model-name qwen \
  --tp 4 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser  qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static 0.6 \
  --context-length 1010000\
  --host 0.0.0.0 \`
  --port 24314

        '''
)


register_api_configs(globals())
