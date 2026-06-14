import unittest
from dataclasses import FrozenInstanceError

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.call_by_single_instance import ChatCompletionChunk
from llm_common.llm_infer.call_by_single_instance import ChatCompletionContentPartImage
from llm_common.llm_infer.call_by_single_instance import ChatCompletionContentPartText
from llm_common.llm_infer.call_by_single_instance import ChatCompletionImageURL
from llm_common.llm_infer.call_by_single_instance import ChatCompletionRequest
from llm_common.llm_infer.call_by_single_instance import _build_chat_completion_request
from llm_common.llm_infer.call_by_single_instance import _extract_stream_text
from llm_common.llm_infer.call_by_single_instance import build_user_content
from llm_common.llm_infer.call_by_single_instance import extract_stream_reasoning_text
from llm_common.llm_infer.instances import LLMInferInputRecord
from llm_common.llm_infer.instances import LLMInferResultRecord


class ChatCompletionRequestTest(unittest.TestCase):
    def make_input(self, **kwargs):
        settings = ChatCompletionRequest(
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
            system_input=kwargs.pop("system_input", "system"),
            max_tokens=kwargs.pop("max_tokens", 128),
            stream=kwargs.pop("stream", True),
        )
        return LLMInferInputRecord(
            prompt=kwargs.pop("prompt", "hello"),
            image_data_urls=kwargs.pop("image_data_urls", None),
            model_settings=settings,
            **kwargs,
        )

    def test_builds_standard_streaming_payload(self):
        payload = _build_chat_completion_request(self.make_input()).to_dict()

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual([message["role"] for message in payload["messages"]], [
            "system",
            "user",
        ])
        self.assertTrue(payload["messages"][0]["content"].startswith("system"))
        self.assertEqual(payload["messages"][1]["content"], "hello")
        self.assertNotIn("thinking", payload)
        self.assertNotIn("enable_thinking", payload)

    def test_disable_thinking_adds_compatible_extensions(self):
        payload = _build_chat_completion_request(
            self.make_input(),
            disable_thinking=True,
        ).to_dict()

        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertIs(payload["enable_thinking"], False)

    def test_omits_stream_options_when_streaming_is_disabled(self):
        payload = _build_chat_completion_request(
            self.make_input(stream=False),
        ).to_dict()

        self.assertNotIn("stream_options", payload)

    def test_preserves_multimodal_user_content(self):
        data_url = "data:image/png;base64,AAAA"
        payload = _build_chat_completion_request(
            self.make_input(image_data_urls=(data_url,)),
        ).to_dict()

        self.assertEqual(payload["messages"][-1]["content"], [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])

    def test_build_user_content_returns_frozen_schema(self):
        data_url = "data:image/png;base64,AAAA"
        content = build_user_content("hello", (data_url,))

        self.assertIsInstance(content[0], ChatCompletionContentPartText)
        self.assertIsInstance(content[1], ChatCompletionContentPartImage)
        self.assertIsInstance(content[1].image_url, ChatCompletionImageURL)
        with self.assertRaises(FrozenInstanceError):
            content[1].image_url.url = "other"

    def test_request_is_frozen(self):
        request = _build_chat_completion_request(self.make_input())

        with self.assertRaises(FrozenInstanceError):
            request.model = "other-model"
        self.assertIsInstance(request, ChatCompletionRequest)

    def test_request_template_materializes_input(self):
        input_ = self.make_input()
        request = input_.model_settings.with_user_input(
            prompt=input_.prompt,
            image_data_urls=input_.image_data_urls,
            disable_thinking=True,
        )

        self.assertIsInstance(request, ChatCompletionRequest)
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.messages[-1].content, "hello")
        self.assertIs(request.enable_thinking, False)


class ChatCompletionChunkTest(unittest.TestCase):
    def test_parses_standard_stream_chunk(self):
        chunk = ChatCompletionChunk.from_dict({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": None,
            }],
        })

        self.assertEqual(chunk.choices[0].delta.role, "assistant")
        self.assertEqual(_extract_stream_text({
            "choices": [{"delta": {"content": "hello"}}],
        }), "hello")

    def test_extracts_compatible_reasoning_content(self):
        payload = {
            "choices": [{
                "delta": {"reasoning_content": [{"text": "reasoning"}]},
            }],
        }

        self.assertEqual(extract_stream_reasoning_text(payload), "reasoning")

    def test_supports_message_and_top_level_fallbacks(self):
        self.assertEqual(_extract_stream_text({
            "choices": [{"message": {"content": "message"}}],
        }), "message")
        self.assertEqual(_extract_stream_text({"content": "top-level"}), "top-level")

    def test_ignores_malformed_choices(self):
        chunk = ChatCompletionChunk.from_dict({"choices": [None, "bad", {}]})

        self.assertEqual(len(chunk.choices), 1)
        self.assertEqual(_extract_stream_text({"choices": "bad"}), "")


class LLMInferResultTest(unittest.TestCase):
    def test_from_dict_uses_supplied_input(self):
        request = ChatCompletionRequest(
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
        )
        input_ = LLMInferInputRecord(
            id="input-id",
            prompt="authoritative prompt",
            image_data_urls=("data:image/png;base64,AAAA",),
            model_settings=request,
            extra={"source": "input"},
        )

        result = LLMInferResultRecord.from_dict({
            "id": "stale-id",
            "prompt": "stale prompt",
            "llm_response": "answer",
            "total_tokens": 7,
        }, input_=input_)

        self.assertEqual(result.id, "input-id")
        self.assertEqual(result.prompt, "authoritative prompt")
        self.assertIs(result.model_settings, request)
        self.assertEqual(result.image_data_urls, input_.image_data_urls)
        self.assertEqual(result.llm_response, "answer")
        self.assertEqual(result.total_tokens, 7)
        self.assertEqual(result.extra["source"], "input")

    def test_from_dict_still_supports_row_only_loading(self):
        result = LLMInferResultRecord.from_dict({
            "id": "row-id",
            "prompt": "row prompt",
            "llm_response": "answer",
        }, model_settings=ChatCompletionRequest(
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
        ))

        self.assertEqual(result.id, "row-id")
        self.assertEqual(result.prompt, "row prompt")
        self.assertEqual(result.llm_response, "answer")


if __name__ == "__main__":
    unittest.main()
