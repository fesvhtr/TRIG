import openai
from openai import OpenAI
from trig.models.base import BaseModel
from trig.config import API_KEY
from trig.utils import base64_to_image

class DALLE3Model(BaseModel):
    def __init__(self):
        self.model_name = "dalle3"
        self.pipe = OpenAI(api_key=API_KEY)

    def generate(self, prompt, **kwargs):
        cnt = 0
        while cnt < 3:
            try:
                response = self.pipe.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1,
                                                response_format='b64_json')
                image_b64 = response.data[0].b64_json
                image = base64_to_image(image_b64)
                return image
            except Exception:
                cnt += 1
                continue
        return None
