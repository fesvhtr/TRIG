import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-3816bf9e709540598d239fc684fe0423",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen2.5-vl-72b-instruct", 
    messages=[{"role": "user", "content": [
            {"type": "text", "text": "这是什么"},
            {"type": "image_url",
             "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}}
            ]}]
    )

# 打印结果
print(completion.choices[0].message.content)
