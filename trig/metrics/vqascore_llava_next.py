# modified from t2v metric: https://github.com/linzhiqiu/t2v_metrics and LLaVA_NeXT: https://github.com/LLaVA-VL/LLaVA-NeXT
from typing import List
import torch
import copy
from transformers import AutoTokenizer
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from trig.metrics.base import BaseMetric

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

class VQAScoreLLaVANextMetric(BaseMetric):
    def __init__(self, model_name='lmms-lab/llama3-llava-next-8b',base_model= "llava_llama3", device='cuda:1'):
        self.model_name = model_name
        self.base_model = base_model
        self.device = device
        self.load_model()


    def load_model(self):
        """Load the model, tokenizer, and image processor."""
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_name, None, self.base_model, device_map="auto"
        )
        self.model.eval()
        self.model.tie_weights()

    def load_image(self, image_path):
        """Load and preprocess images."""
        image = Image.open(image_path)
        image_sizes = [image.size]
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        return image_tensor, image_sizes

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, images: List[str], texts: List[str]):
        """Compute similarity scores for image-text pairs."""
        assert len(images) == len(texts), "Number of images and texts must match"
        
        questions = [default_question_template.format(text) for text in texts]
        answers = [default_answer_template.format(text) for text in texts]
        
    
        
        for image, question, answer in zip(images, questions, answers):
            image_tensor, image_size = self.load_image(image)

            conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            outputs = self.model.model(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                return_dict=True
            )
            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)
            print(logits)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            shift_labels = shift_labels.to(shift_logits.device)
            lm_prob = torch.zeros(shift_logits.shape[0])
            for k in range(lm_prob.shape[0]):
                lm_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k])).exp()
            return lm_prob

    def compute(self, image_path, prompt):
        return self.forward(image_path, prompt)

    def compute_batch(self, data_ids, images, prompts):
        results = {}
        for idx, (data_id, image_path, prompt) in tqdm(enumerate(zip(data_ids, images, prompts))):
            results[data_id] = self.forward(image_path, prompt['prompt'])
        return results


if __name__ == '__main__':
    vqascore_llava_next_metric = VQAScoreLLaVANextMetric()
    images = ['/home/muzammal/Projects/TRIG/demo.jpg']
    texts = ['a classic white building']
    scores = vqascore_llava_next_metric.compute(images, texts)
    print(scores)
