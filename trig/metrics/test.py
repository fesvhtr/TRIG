from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                    },
                },
                {"type": "text", "text": "Is this a picture of a dog?"},
            ],
        },
    ],
    logprobs=True,
    top_logprobs=5,
)
print(completion.choices[0].message.content)
top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
print('top_logprobs:', top_logprobs)