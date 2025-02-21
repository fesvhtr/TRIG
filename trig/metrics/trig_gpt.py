## This file could still be used but deprecated, please use the trig_api.py file instead and setup with gpt model

import openai
from openai import OpenAI
import json
from io import BytesIO
import numpy as np
from trig.metrics.base import BaseMetric
from trig.utils.utils import encode_image
import torch
from trig.config import gpt_logit_system_msg,gpt_logit_dimension_msg, API_KEY
from tqdm import tqdm
import math


class TRIGGPTMetric(BaseMetric):
    def __init__(self, API_KEY, top_logprobs=5, **kwargs):
        super().__init__(**kwargs)
        print("Initializing TRIGGPTMetric, params: API_KEY: {}, dimension: {}, top_logprobs: {}".format(API_KEY, self.dimension, top_logprobs))
        self.dimension = dimension
        self.top_logprobs = top_logprobs
        self.client = openai.Client(api_key=API_KEY)
        self.model_name = "gpt-4o"

    def compute(self, image_path, prompt):
        image = encode_image(image_path)
        sys_msg = [{
            "role": "developer",
            "content": gpt_logit_system_msg.format(gpt_logit_dimension_msg[self.dimension])
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
        # print(completion.choices[0].message.content)
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        # print('top_logprobs:', top_logprobs)
        usage_tokens = [completion.usage.prompt_tokens, completion.usage.completion_tokens,
                        completion.usage.prompt_tokens + completion.usage.completion_tokens]
        print('usage_tokens:', usage_tokens)
        score = self.logprobs_score(top_logprobs)
        return score

    import math

    def logprobs_score(self, top_logprobs):
        score = 0.0

        valid_tokens = ["excellent", "Excellent", "good", "Good", "medium", "Medium", "bad", "Bad", "terr", "Terr"]

        filtered_tokens = []
        filtered_logprobs = []

        for item in top_logprobs:
            if item.token in valid_tokens:
                filtered_tokens.append(item.token)
                filtered_logprobs.append(float(item.logprob))

        # If there is no matching token, directly return 0 points
        if not filtered_tokens:
            return 0.0

        # Convert log probabilities to linear probabilities, dealing with extreme values
        try:
            linear_probs = [math.exp(lp) for lp in filtered_logprobs]
        except OverflowError:
            linear_probs = [0.0] * len(filtered_logprobs)

        # Renormalise probabilities
        total = sum(linear_probs) + 1e-10
        normalized_probs = [p / total for p in linear_probs]

        for token, prob in zip(filtered_tokens, normalized_probs):
            # print(token, prob)
            if token in ["excellent", "Excellent"]:
                score += 1.0 * prob
            elif token in ["good", "Good"]:
                score += 0.75 * prob
            elif token in ["medium", "Medium"]:
                score += 0.5 * prob
            elif token in ["bad", "Bad"]:
                score += 0.25 * prob
            elif token in ["terr", "Terr"]:
                score += 0 * prob

        return score

    def compute_batch(self, data_ids, images, prompts):
        if data_ids is None:
            results = []
            for idx, (image_path, prompt) in tqdm(enumerate(zip(images, prompts))):
                results.append(self.compute(image_path, prompt['prompt'])) 
            return results
        
        else:
            results = {}
            for idx, (data_id, image_path, prompt) in tqdm(enumerate(zip(data_ids, images, prompts))):
                results[data_id] = self.compute(image_path, prompt['prompt'])
            return results


if __name__ == "__main__":
    API_KEY = "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"
    # Example usage
    metric = TRIGGPTMetric(API_KEY, top_logprobs=5, dimension='TA-C')
    image_path = r"/home/muzammal/Projects/TRIG/demo.jpg"
    prompt = ["A old building like a main building of a university",
              "A old building like a main building of a university with green grass and blue sky",
              "A dog swimming in the water",
              "A new techinical building with a grey sky",
              "A beautiful sunset with a beach"]

    for i in prompt:
        result = metric.compute(image_path, i)
        print(result)
