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
import math


class GPTLogitMetric(BaseMetric):
    def __init__(self, API_KEY, dimension, top_logprobs=5, **kwargs):
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
            return 0.0
        print(completion.choices[0].message.content)
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        print('top_logprobs:', top_logprobs)
        usage_tokens = [completion.usage.prompt_tokens, completion.usage.completion_tokens,
                        completion.usage.prompt_tokens + completion.usage.completion_tokens]
        print('usage_tokens:', usage_tokens)
        score = self.logprobs_score(top_logprobs)
        return score

    def logprobs_score(self, top_logprobs):
        score = 0.0
        tokens = [item.token for item in top_logprobs]
        logprobs = [item.logprob for item in top_logprobs]

        # 转换为线性概率（防御性编程）
        try:
            linear_probs = [math.exp(float(lp)) for lp in logprobs]
        except OverflowError:
            linear_probs = [0.0] * len(logprobs)  # 处理极端数值

        # 概率归一化（防止除零）
        total = sum(linear_probs) + 1e-10
        normalized = [p / total for p in linear_probs]

        for token, prob in zip(tokens, normalized):
            if token in ["excellent", "Excellent"]:
                score += 1.0 * math.exp(prob)
            elif token == ["good", "Good"]:
                score += 0.75 * math.exp(prob)
            elif token == ["medium", "Medium"]:
                score += 0.5 * math.exp(prob)
            elif token == ["bad", "Bad"]:
                score += 0.25 * math.exp(prob)
            elif token == ["terr", "Terr"]:
                score += 0 * math.exp(prob)
            else:
                score += 0.5 * math.exp(prob)
        return score/5.0

    def compute_batch(self, data_ids, images, prompts):
        results = {}
        for idx, (data_id, image_path, prompt) in tqdm(enumerate(zip(data_ids, images, prompts))):
            results[data_id] = self.compute(image_path, prompt['prompt'])
        return results


if __name__ == "__main__":
    API_KEY = "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"
    # Example usage
    metric = GPTLogitMetric(API_KEY, top_logprobs=5, dimension='IQ-C')
    image_path = r"H:\ProjectsPro\TRIG\demo.jpg"
    prompt = ["A old building like a main building of a university",
              "A old building like a main building of a university with green grass and blue sky",
              "A dog swimming in the water",
              "A new techinical building with a grey sky",
              "A beautiful sunset with a beach"]

    for i in prompt:
        result = metric.compute(image_path, i)
        print(result)
