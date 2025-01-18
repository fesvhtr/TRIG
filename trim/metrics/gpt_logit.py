import openai
import json
import re
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from trim.metrics.base import BaseMetric
from trim.utils.utils import encode_image
import torch

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'

class GPTLogitMetric(BaseMetric):
    def __init__(self, API_KEY, top_logprobs, **kwargs):
        super().__init__(**kwargs)
        self.openai_key = API_KEY
        self.top_logprobs = top_logprobs
        self.client = openai
        self.model_name = "gpt-4"

    def compute_single(self, question, image):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image['type']};base64,{image['base64']}"
                                }
                            }
                        ]
                    }
                ],
                logprobs=True,
                top_logprobs=self.top_logprobs,
                # logit_bias={yes_token:50, no_token:50}
            )
        except Exception as e:
            print(f"Error: {e}")
            return torch.Tensor([0.0])
        is_generated = False
        for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
            if top_logprob.token == answer:
                is_generated = True
                return torch.Tensor([top_logprob.logprob]).exp()
        if not is_generated:
            print(
                f"Warning: answer not generated for image: {image['path']} and question: {question} and answer: {answer}")
            print(completion.choices[0].logprobs.content[0].top_logprobs)
            return torch.Tensor([0.0])

    def compute(self, images, prompts, dimension):
        results = []
        for idx, (image, prompt) in enumerate(zip(images, prompts)):
            question = default_question_template.format(dimension)
            answer = default_answer_template
            results.append(self.compute_single(prompt))