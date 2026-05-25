from by_function.llm_infer.call import CallOpenaiInput
from by_function.llm_infer.call import call_openai
from by_function.llm_infer.test.data.index import LLM_INFER_TEST_IMAGE_JPEG_PATH


def main() -> int:
    image_path = LLM_INFER_TEST_IMAGE_JPEG_PATH
    text = call_openai(
            input_=CallOpenaiInput(
                    prompt="describe the image using 100 words",
                    system_input="You're a image captioner",
                    image_paths=(image_path,),
            )
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
