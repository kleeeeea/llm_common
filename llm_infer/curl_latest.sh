#!/usr/bin/env bash
LLM_API_KEY=${LLM_API_KEY:-sk-yJdHSUrrZNYkBS5f5dHOPgoxRw5Q8qRJPFTbKh6jOqnAUZNF}
curl -sS -N -X POST \
  --max-time 60.0 \
  https://api.innospark.cn/v1/chat/completions \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemini-2.5-flash",
  "messages": [
    {
      "role": "system",
      "content": "You'"'"'re a helpful assistant\nTotal token budget including reasoning is: 16000. Reasoning budget can not be more than 12800 tokens"
    },
    {
      "role": "user",
      "content": "output a paragraph with 100 words"
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "max_tokens": 16000,
  "stream_options": {
    "include_usage": true
  }
}'
