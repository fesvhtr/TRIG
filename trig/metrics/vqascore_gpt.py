# modified from t2v metric: https://github.com/linzhiqiu/t2v_metrics
from typing import List, Optional, Tuple, Union
import torch
from openai import OpenAI
from trig.metrics.base import BaseMetric

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

class VQAScoreGPTMetric(BaseMetric):
    """Implementation of t2v metric (GPT)"""
    def __init__(self,
                model_name='gpt-4-turbo',
                device='cuda',
                cache_dir=None,
                openai_key=None,
                top_logprobs=2):
        assert openai_key is not None, "Please provide an OpenAI API key"
        self.openai_key = openai_key
        self.top_logprobs = top_logprobs
        super().__init__(model_name=model_name,
                            device=device,
                            cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.client = OpenAI(api_key=self.openai_key)

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return the string
        """
        image = [{'path': img, 'type': get_image_type(img), 'base64': encode_image(img)} for img in image]
        return image
    
    def forward_single(self, image, question, answer):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages= [
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
        except:
            print(f"Warning: completion not generated for image: {image['path']} and question: {question} and answer: {answer}")
            print(f"Trying again with the same image")
            try:
                completion = self.client.chat.completions.create(model=self.model_name, messages= [{"role": "user","content": [{"type": "text", "text": question},{ "type": "image_url","image_url": {"url": f"data:image/{image['type']};base64,{image['base64']}"}}]}],logprobs=True,top_logprobs=self.top_logprobs,)
            except:
                print(f"Failed image: {image['path']} and question: {question} and answer: {answer}")
                return torch.Tensor([0.0])

        # print(completion.choices[0].message)
        # print(completion.choices[0].logprobs)
        # print(completion.choices[0].logprobs.content[0])
        is_generated = False
        for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
            if top_logprob.token == answer:
                is_generated = True
                return torch.Tensor([top_logprob.logprob]).exp()
        if not is_generated:
            print(f"Warning: answer not generated for image: {image['path']} and question: {question} and answer: {answer}")
            print(completion.choices[0].logprobs.content[0].top_logprobs)
            return torch.Tensor([0.0])

    # @torch.no_grad()
    # @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        for ans in answers:
            ans_tokens = self.tokenizer.encode(ans)
            assert len(ans_tokens) == 1, "Currently only support single token answers"

        images = self.load_images(images)
        
        lm_prob = torch.zeros(len(images))
        
        for idx, (image, question, answer) in enumerate(zip(images, questions, answers)):
            lm_prob[idx] = self.forward_single(image, question, answer)
        
        return lm_prob

    def compute(self, image_path, prompt):
        return self.forward(image_path, prompt)
    
    def compute_batch(self, data_ids, images, prompts):
        results = {}
        for idx, (data_id, image_path, prompt) in tqdm(enumerate(zip(data_ids, images, prompts))):
            results[data_id] = self.compute(image_path, prompt['prompt'])
        return results

if __name__ == '__main__':
    API_KEY = "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"
    # Example usage
    metric = VQAScoreGPTMetric(openai_key=API_KEY)
    image_path = r"/home/muzammal/Projects/TRIG/demo.jpg"
    prompt = ["A old building like a main building of a university",
              "A old building like a main building of a university with green grass and blue sky",
              "A dog swimming in the water",
              "A new techinical building with a grey sky",
              "A beautiful sunset with a beach"]

    for i in prompt:
        result = metric.compute(image_path, i)
        print(result)