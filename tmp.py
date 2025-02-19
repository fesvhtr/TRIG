from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:10021/v1",
    api_key="EMPTY",
)

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

image = encode_image("/home/muzammal/Projects/TRIG/demo.jpg")
completion = client.chat.completions.create(
  model="Benasd/Qwen2.5-VL-72B-Instruct-AWQ",
  messages= [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image"},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/{image['type']};base64,{image['base64']}"}}]
            }]
)

print(completion.choices[0].message.content)