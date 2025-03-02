from openai import OpenAI
import base64
import sys
from PIL import Image
from io import BytesIO
import requests
import json
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:10021/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
src_image = '/home/muzammal/Projects/TRIG/demo.jpg'
gen_image = '/home/muzammal/Projects/LLaVA-NeXT/docs/ov_chat_images/example1_tree.png'
def get_image_type(image_path):
    image_type = image_path.split('.')[-1]
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return image_type
def encode_image(image_path):
    image = {}
    with open(image_path, "rb") as image_file:
        image['base64'] = base64.b64encode(image_file.read()).decode('utf-8')
    image['type'] = get_image_type(image_path)
    return image

src_image = encode_image(src_image)
gen_image = encode_image(gen_image)
sys_msg = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }]
user_msg = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Differentiate between the two images.'},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/{src_image['type']};base64,{src_image['base64']}"}},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/{gen_image['type']};base64,{gen_image['base64']}"}}]
            }]
msg = sys_msg + user_msg
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=msg,
    logprobs=True,
    top_logprobs=5,
)
print(completion.choices[0].message.content)
# top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
# print('top_logprobs:', top_logprobs)