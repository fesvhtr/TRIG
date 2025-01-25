import openai
from openai import OpenAI
import json
from io import BytesIO
import numpy as np
from trig.metrics.base import BaseMetric
from trig.utils.utils import encode_image
import torch
from trig.utils.config import gpt_logit_system_msg
from tqdm import tqdm

class GPTLogitMetric(BaseMetric):
    def __init__(self, API_KEY, dimension,top_logprobs=5, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.top_logprobs = top_logprobs
        self.client = openai.Client(api_key=API_KEY)
        self.model_name = "gpt-4o"

    def compute(self, image_path, prompt):
        return 0.5
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
        usage_tokens = [completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.prompt_tokens + completion.usage.completion_tokens]
        print('usage_tokens:', usage_tokens)
        # for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
        #     if top_logprob.token == answer:
        #         return ([top_logprob.logprob]).exp()
        #     else:
        #         return torch.Tensor([0.0])

    def compute_batch(self, data_ids, images, prompts):
        results = {}
        for idx, (data_id, image_path, prompt) in tqdm(enumerate(zip(data_ids, images, prompts))):
            results[data_id] = self.compute(image_path, prompt['prompt'])
        return results


if __name__ == "__main__":
    API_KEY = "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"
    # Example usage
    metric = GPTLogitMetric(API_KEY, top_logprobs=5, dimension='IQ-R')
    image_path = r"H:\ProjectsPro\TRIM\demo.jpg"
    prompt = "A historic building, probably the main building of some university"
    result = metric.compute(image_path, prompt)
    print(result)
