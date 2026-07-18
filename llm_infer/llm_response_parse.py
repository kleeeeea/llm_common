import json
from typing import Any


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)
    return ""

def _extract_stream_text(payload: dict[str, Any]) -> str:
    from llm_common.llm_infer.call_by_single_instance import ChatCompletionChunk

    chunk = ChatCompletionChunk.from_dict(payload)
    if chunk.choices:
        first = chunk.choices[0]
        text = normalize_text_content(first.delta.content)
        if text:
            return text
        text = normalize_text_content(first.message.content)
        if text:
            return text
    return normalize_text_content(chunk.content)


def extract_stream_reasoning_text(payload: dict[str, Any]) -> str:
    from llm_common.llm_infer.call_by_single_instance import ChatCompletionChunk

    chunk = ChatCompletionChunk.from_dict(payload)
    if chunk.choices:
        first = chunk.choices[0]
        text = normalize_text_content(first.delta.reasoning_content)
        if text:
            return text
        text = normalize_text_content(first.message.reasoning_content)
        if text:
            return text
    text = normalize_text_content(chunk.reasoning_content)
    if text:
        return text
    return normalize_text_content(chunk.reasoning)

def main():
    all_payload = ['{"id":"chatcmpl-d11d2c28-1900-44e9-bc7e-ac90dd07965a","object":"chat.completion.chunk","created":1783934433,"model":"qwen3.5-35b-a3b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}',
 '{"id":"chatcmpl-d11d2c28-1900-44e9-bc7e-ac90dd07965a","object":"chat.completion.chunk","created":1783934433,"model":"qwen3.5-35b-a3b","choices":[{"index":0,"delta":{"reasoning":"A"},"logprobs":null,"finish_reason":null,"token_ids":null}]}',
 '{"id":"chatcmpl-d11d2c28-1900-44e9-bc7e-ac90dd07965a","object":"chat.completion.chunk","created":1783934433,"model":"qwen3.5-35b-a3b","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}]}',
 '{"id": "chatcmpl-d11d2c28-1900-44e9-bc7e-ac90dd07965a", "object": "chat.completion.chunk", "created": 1783934433, "model": "qwen3.5-35b-a3b", "choices": [], "usage": {"prompt_tokens": 100, "total_tokens": 102, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 0}}}']
    # reproduce the failure of failing to extract any info
    payloads = [json.loads(raw_payload) for raw_payload in all_payload]
    text = "".join(_extract_stream_text(payload) for payload in payloads)
    reasoning = "".join(
            extract_stream_reasoning_text(payload) for payload in payloads)

    print(f"extracted text: {text!r}")
    print(f"extracted reasoning: {reasoning!r}")

    assert not text
    assert reasoning == "A"

if __name__ == '__main__':
    main()
