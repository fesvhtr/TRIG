import openai
from openai import OpenAI
import json
from io import BytesIO
import numpy as np
from trim.metrics.base import BaseMetric
from trim.utils.utils import encode_image
import torch
from trim.utils.config import gpt_logit_system_msg, default_answer_template


class GPTLogitMetric(BaseMetric):
    def __init__(self, API_KEY, top_logprobs, dimension, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.top_logprobs = top_logprobs
        self.client = openai.Client(api_key=API_KEY)
        self.model_name = "gpt-4o"

    def compute(self, image_path, prompt):
        image = encode_image(image_path)
        sys_msg = [{
            "role": "developer",
            "content": gpt_logit_system_msg.format(self.dimension)
        }]
        user_msg = [{
            "role": "user",
            "content": [
                 {"type": "text", "text": prompt},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/{image['type']};base64,{image['base64']}"}}]
        }]
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=sys_msg + user_msg,
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )
        except Exception as e:
            print(f"Error: {e}")
            return torch.Tensor([0.0])
        print(completion.choices[0].logprobs.content[0].top_logprobs)
        print(completion.usage.prompt_tokens)
        print(completion.usage.completion_tokens)
        print()
        # for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
        #     if top_logprob.token == answer:
        #         return torch.Tensor([top_logprob.logprob]).exp()
        #     else:
        #         return torch.Tensor([0.0])

    def compute_batch(self, images, prompts, dimension):
        results = []
        for idx, (image_path, prompt) in enumerate(zip(images, prompts)):
            results.append(self.compute(prompt, image_path))
        return results


if __name__ == "__main__":
    API_KEY = "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"
    # Example usage
    metric = GPTLogitMetric(API_KEY, top_logprobs=5, dimension='A')
    image_path = r"H:\ProjectsPro\TRIM\demo.jpg"
    prompt = "A historic building, probably the main building of some university"
    result = metric.compute(image_path, prompt)
    print(result)
