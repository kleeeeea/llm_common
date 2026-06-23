import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from inspect import signature
from unittest.mock import patch

from llm_common.llm_infer.api_info.dataclass_ import ApiConfig
from llm_common.llm_infer.call_by_single_instance import ChatCompletionChunk
from llm_common.llm_infer.call_by_single_instance import ChatCompletionContentPartImage
from llm_common.llm_infer.call_by_single_instance import ChatCompletionContentPartText
from llm_common.llm_infer.call_by_single_instance import ChatCompletionImageURL
from llm_common.llm_infer.call_by_single_instance import ChatCompletionRequest
from llm_common.llm_infer.call_by_single_instance import _build_chat_completion_payload
from llm_common.llm_infer.call_by_single_instance import _build_chat_completion_request
from llm_common.llm_infer.call_by_single_instance import _dump_curl_command
from llm_common.llm_infer.call_by_single_instance import _extract_stream_text
from llm_common.llm_infer.call_by_single_instance import _check_server_health
from llm_common.llm_infer.call_by_single_instance import build_user_content
from llm_common.llm_infer.call_by_single_instance import extract_stream_reasoning_text
from llm_common.llm_infer.instances import LLMInferInputRecord
from llm_common.llm_infer.instances import LLMInferResultRecord


class ChatCompletionRequestTest(unittest.TestCase):
    def make_input(self, **kwargs):
        api = ApiConfig(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            model="test-model",
        )
        request = ChatCompletionRequest(
            model="test-model",
            max_tokens=kwargs.pop("max_tokens", 128),
            stream=kwargs.pop("stream", True),
        )
        return LLMInferInputRecord(
            prompt=kwargs.pop("prompt", "hello"),
            image_data_urls=kwargs.pop("image_data_urls", None),
            chat_completion_request=request,
            api=api,
            system_input=kwargs.pop("system_input", "system"),
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

    def test_request_model_defaults_from_api(self):
        input_ = LLMInferInputRecord(
            prompt="hello",
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="api-model",
            ),
            chat_completion_request=ChatCompletionRequest(max_tokens=128),
        )

        self.assertEqual(input_.chat_completion_request.model, "api-model")

    def test_disable_thinking_adds_compatible_extensions(self):
        input_ = self.make_input(disable_thinking=True)
        payload = _build_chat_completion_payload(
            input_,
            disable_thinking=input_.disable_thinking,
        )

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
        request = input_.chat_completion_request

        self.assertIsInstance(request, ChatCompletionRequest)
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.messages[-1].content, "hello")
        self.assertEqual(set(request.to_dict()), {
            "model",
            "messages",
            "temperature",
            "stream",
            "max_tokens",
            "stream_options",
        })

    def test_input_system_prompt_overrides_settings_template(self):
        input_ = self.make_input(system_input="template")
        input_ = LLMInferInputRecord(
            prompt=input_.prompt,
            system_input="record",
            chat_completion_request=input_.chat_completion_request,
            api=input_.api,
        )

        request = _build_chat_completion_request(input_)

        self.assertTrue(request.messages[0].content.startswith("record"))

    def test_call_openai_only_accepts_input_record(self):
        from llm_common.llm_infer.call_by_single_instance import call_openai

        self.assertEqual(list(signature(call_openai).parameters), ["input_"])


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


class ServerHealthCheckTest(unittest.TestCase):
    @patch("llm_common.llm_infer.call_by_single_instance.subprocess.run")
    def test_failure_prints_redacted_reproduction_command(self, run):
        run.return_value.stdout = "503"
        run.return_value.stderr = "unavailable"

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaisesRegex(Exception, r"check_command=.*LLM_API_KEY") as raised:
                _check_server_health(
                    "https://example.com/v1",
                    "secret-api-key",
                    timeout=3,
                )

        output = stderr.getvalue()
        self.assertIn("export LLM_API_KEY=", output)
        self.assertIn("https://example.com/v1/models", output)
        self.assertNotIn("secret-api-key", output)
        self.assertNotIn("secret-api-key", str(raised.exception))


class CurlDumpTest(unittest.TestCase):
    def test_dump_curl_command_uses_env_api_key_and_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "curl_latest.sh")
            body = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }

            written_path = _dump_curl_command(
                "http://localhost:8000/v1/chat/completions",
                "test-key",
                body,
                timeout=5.0,
                output_path=output_path,
            )

            self.assertEqual(written_path, output_path)
            with open(output_path, encoding="utf-8") as file:
                script = file.read()

        self.assertIn("LLM_API_KEY", script)
        self.assertIn("LLM_API_KEY=${LLM_API_KEY:-test-key}", script)
        self.assertIn("http://localhost:8000/v1/chat/completions", script)
        self.assertIn("--max-time 5.0", script)
        self.assertIn("  -d '{", script)
        self.assertNotIn("--data ", script)
        self.assertIn('"model": "test-model"', script)
        self.assertIn('"content": "hello"', script)
        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))


class CallOpenAIStreamingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dump_patcher = patch(
            "llm_common.llm_infer.call_by_single_instance._dump_curl_command",
            return_value=os.path.join(self.temp_dir.name, "curl_latest.sh"),
        )
        self.dump_mock = self.dump_patcher.start()

    def tearDown(self):
        self.dump_patcher.stop()
        self.temp_dir.cleanup()

    @patch("llm_common.llm_infer.call_by_single_instance._check_server_health")
    @patch("llm_common.llm_infer.call_by_single_instance.urllib.request.urlopen")
    def test_mock_model_returns_without_network(self, urlopen, health_check):
        input_ = LLMInferInputRecord(
            prompt="hello",
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="mock",
            ),
            chat_completion_request=ChatCompletionRequest(model="mock"),
        )

        from llm_common.llm_infer.call_by_single_instance import call_openai
        result = call_openai(input_)

        self.assertEqual(result.llm_response, "mock response with prompt: hello")
        self.assertIsNone(result.reasoning)
        self.assertEqual(result.latency_ms, 0.0)
        self.assertEqual(result.chat_completion_request, input_.chat_completion_request)
        self.assertEqual(result.api, input_.api)
        health_check.assert_not_called()
        urlopen.assert_not_called()

    @patch("llm_common.llm_infer.call_by_single_instance._check_server_health")
    @patch("llm_common.llm_infer.call_by_single_instance.urllib.request.urlopen")
    def test_mock_model_uses_first_text_part(self, urlopen, health_check):
        prompt = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        input_ = LLMInferInputRecord(
            prompt=prompt,
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="mock",
            ),
            chat_completion_request=ChatCompletionRequest(model="mock"),
        )

        from llm_common.llm_infer.call_by_single_instance import call_openai
        result = call_openai(input_)

        self.assertEqual(result.llm_response, "mock response with prompt: first")
        health_check.assert_not_called()
        urlopen.assert_not_called()

    @patch("llm_common.llm_infer.call_by_single_instance._check_server_health")
    @patch("llm_common.llm_infer.call_by_single_instance.urllib.request.urlopen")
    def test_per_line_stream_printing_uses_input_option(self, urlopen, health_check):
        class Response:
            status = 200
            headers = {"content-type": "text/event-stream"}

            def __init__(self):
                payloads = [
                    {"choices": [{"delta": {"reasoning_content": "reason"}}]},
                    {"choices": [{"delta": {"content": "answer"}}]},
                    {"choices": [], "usage": {"total_tokens": 2}},
                ]
                self.data = (
                    "".join(
                        f"data: {json.dumps(payload)}\n\n"
                        for payload in payloads
                    )
                    + "data: [DONE]\n\n"
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size=-1):
                data, self.data = self.data, b""
                return data

        urlopen.return_value = Response()
        input_ = LLMInferInputRecord(
            prompt="hello",
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
            chat_completion_request=ChatCompletionRequest(model="test-model"),
            do_print_one_response_per_line=True,
        )

        from llm_common.llm_infer.call_by_single_instance import call_openai
        result = call_openai(input_)

        self.assertEqual(result.llm_response, "answer")
        self.assertEqual(result.reasoning, "reason")
        self.assertEqual(result.total_tokens, 2)
        health_check.assert_called_once()
        self.dump_mock.assert_called_once()
        self.assertEqual(self.dump_mock.call_args.args[1], "test-key")

    @patch("llm_common.llm_infer.call_by_single_instance._check_server_health")
    @patch("llm_common.llm_infer.call_by_single_instance.urllib.request.urlopen")
    def test_default_stream_prints_reasoning_red_and_answer_green(
            self, urlopen, _health_check):
        class Response:
            status = 200
            headers = {"content-type": "text/event-stream"}

            def __init__(self):
                self.data = (
                    'data: {"choices":[{"delta":{"reasoning_content":"reason"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
                    "data: [DONE]\n\n"
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size=-1):
                data, self.data = self.data, b""
                return data

        urlopen.return_value = Response()
        input_ = LLMInferInputRecord(
            prompt="hello",
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
            chat_completion_request=ChatCompletionRequest(model="test-model"),
        )

        from llm_common.llm_infer.call_by_single_instance import call_openai
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            call_openai(input_)

        output = stdout.getvalue()
        self.assertIn("\033[31mreason\033[0m", output)
        self.assertIn("\033[32manswer\033[0m", output)

    @patch("llm_common.llm_infer.call_by_single_instance._check_server_health")
    @patch("llm_common.llm_infer.call_by_single_instance.urllib.request.urlopen")
    def test_non_streaming_json_response(self, urlopen, _health_check):
        class Response:
            status = 200
            headers = {"content-type": "application/json"}

            def __init__(self):
                self.data = json.dumps({
                    "choices": [{
                        "message": {
                            "content": "answer",
                            "reasoning_content": "reason",
                        },
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                }).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size=-1):
                data, self.data = self.data, b""
                return data

        urlopen.return_value = Response()
        input_ = LLMInferInputRecord(
            prompt="hello",
            api=ApiConfig(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
            ),
            chat_completion_request=ChatCompletionRequest(
                model="test-model",
                stream=False,
            ),
        )

        from llm_common.llm_infer.call_by_single_instance import call_openai
        result = call_openai(input_)

        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertIs(request_body["stream"], False)
        self.assertNotIn("stream_options", request_body)
        self.assertEqual(result.llm_response, "answer")
        self.assertEqual(result.reasoning, "reason")
        self.assertEqual(result.prompt_tokens, 1)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.total_tokens, 3)


class LLMInferResultTest(unittest.TestCase):
    def test_from_dict_uses_supplied_input(self):
        api = ApiConfig(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            model="test-model",
        )
        request = ChatCompletionRequest(
            model="test-model",
        )
        input_ = LLMInferInputRecord(
            id="input-id",
            prompt="authoritative prompt",
            image_data_urls=("data:image/png;base64,AAAA",),
            chat_completion_request=request,
            api=api,
            disable_thinking=True,
            system_input="record system",
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
        self.assertEqual(result.chat_completion_request.model, request.model)
        self.assertEqual(result.image_data_urls, input_.image_data_urls)
        self.assertTrue(result.disable_thinking)
        self.assertEqual(result.system_input, "record system")
        self.assertEqual(result.llm_response, "answer")
        self.assertEqual(result.total_tokens, 7)
        self.assertEqual(result.extra["source"], "input")

    def test_from_dict_still_supports_row_only_loading(self):
        result = LLMInferResultRecord.from_dict({
            "id": "row-id",
            "prompt": "row prompt",
            "llm_response": "answer",
        }, chat_completion_request=ChatCompletionRequest(model="test-model"),
           api=ApiConfig(
               base_url="http://localhost:8000/v1",
               api_key="test-key",
               model="test-model",
           ))

        self.assertEqual(result.id, "row-id")
        self.assertEqual(result.prompt, "row prompt")
        self.assertEqual(result.llm_response, "answer")

    def test_from_dict_restores_embedded_request(self):
        result = LLMInferResultRecord.from_dict({
            "id": "row-id",
            "prompt": "row prompt",
            "llm_response": "answer",
            "chat_completion_request": {
                "model": "embedded-model",
                "messages": [{"role": "user", "content": "row prompt"}],
                "temperature": 0.2,
                "stream": True,
            },
        }, api=ApiConfig(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            model="embedded-model",
        ))

        self.assertEqual(result.model, "embedded-model")
        self.assertEqual(result.chat_completion_request.temperature, 0.2)


if __name__ == "__main__":
    unittest.main()
