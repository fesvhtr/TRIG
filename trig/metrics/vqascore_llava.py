# modified from t2v metric: https://github.com/linzhiqiu/t2v_metrics
from typing import List, Optional, Tuple, Union
import torch
import copy
from dataclasses import dataclass, field
from trig.metrics.llava.mm_utils import expand2square, tokenizer_image_token
from trig.metrics.llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from PIL import Image
from trig.metrics.base import BaseMetric
import os
import pdb

CONTEXT_LEN = 2048
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"
cache_dir = None


LLAVA_MODELS = {
    'llava-v1.5-13b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-13b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-13b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-v1.5-7b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-7b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-7b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
}

@dataclass
class ModelArguments:
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the second last layer in llava1.5
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_vision_select_feature: Optional[str] = field(default="patch")



class VQAScoreLLaVAMetric(BaseMetric):
    """Implementation of t2v metric (LLaVA-1.5)"""


    def __init__(self,
                 model_name='llava-v1.5-13b',
                 device='cuda:1'):
        assert model_name in LLAVA_MODELS
        self.model_name = model_name
        self.device = device
        self.load_model()

    def load_pretrained_model(self,
                            model_cls,
                          model_args,
                          model_path=None,
                          tokenizer_path=None,
                          model_max_length=None,
                          padding_side=None,
                          image_aspect_ratio='pad', # or 'square'
                          mmprojector_repo=None,
                          mmprojector_name=None,
                          device='cuda'):
        tokenizer_dict = {}
        if model_max_length:
            tokenizer_dict['model_max_length'] = model_max_length
        if padding_side:
            tokenizer_dict['padding_side'] = padding_side
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, **tokenizer_dict)
        # tokenizer.pad_token = tokenizer.unk_token # could be redundant

        model = model_cls.from_pretrained(model_path)
        
        if mmprojector_repo:
            from huggingface_hub import hf_hub_download
            model_base_name = mmprojector_repo.split('/')[-1]
            
            if cache_dir is not None:
                local_dir = os.path.join(cache_dir, model_base_name)
            elif os.environ.get('HF_HOME') is not None:
                local_dir = os.path.join(os.environ.get('HF_HOME'), model_base_name)
            else:
                local_dir = os.path.join(os.path.expanduser("~"), model_base_name)
            print(f"Downloading projector weights to {local_dir}")
            hf_hub_download(
                repo_id=mmprojector_repo,
                filename=mmprojector_name,
                local_dir=local_dir,
            )
            pretrain_mm_mlp_adapter = os.path.join(local_dir, mmprojector_name)
            model_args.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter # important to set to correct path
            
            model.get_model().initialize_vision_modules(model_args) # This will load the CLIP vision encoder and MLP projector
        else:
            model.resize_token_embeddings(len(tokenizer)) # perhaps not needed

        if not model.get_vision_tower().is_loaded:
            model.get_vision_tower().load_model()
        model.to(device=device, dtype=torch.bfloat16)
        image_processor = model.get_vision_tower().image_processor

        model.requires_grad_(False)
        
        
        # below might be redundant
        model.config.image_aspect_ratio = image_aspect_ratio
        model.config.use_cache = False
        model.config.image_grid_pinpoints = None
        model.config.freeze_mm_mlp_adapter = True

        model = model.eval()
        return tokenizer, model, image_processor
        
    def format_question(self,question, conversation_style='chat'):
        if conversation_style == 'plain':  # for 1st stage model
            question = DEFAULT_IMAGE_TOKEN + question
        elif conversation_style == 'chat':  # for 2nd stage model
            question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
        else:
            raise NotImplementedError()
        return question

    def format_answer(self,answer, conversation_style='chat'):
        if conversation_style == 'plain':  # for 1st stage model
            answer = answer + "\n"
        elif conversation_style == 'chat':  # for 2nd stage model
            answer = answer + "</s>"
        else:
            raise NotImplementedError()
        return answer

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_args = ModelArguments()
        model_max_length = LLAVA_MODELS[self.model_name]['tokenizer']['model_max_length'] \
            if 'model_max_length' in LLAVA_MODELS[self.model_name]['tokenizer'] else None
        padding_side = LLAVA_MODELS[self.model_name]['tokenizer']['padding_side'] \
            if 'padding_side' in LLAVA_MODELS[self.model_name]['tokenizer'] else None
        mmprojector_repo = LLAVA_MODELS[self.model_name]['model']['mmprojector_repo'] \
            if 'mmprojector_repo' in LLAVA_MODELS[self.model_name]['model'] else None
        mmprojector_name = LLAVA_MODELS[self.model_name]['model']['mmprojector_name'] \
            if 'mmprojector_name' in LLAVA_MODELS[self.model_name]['model'] else None
        
        # default is 'pad' (llava-1.5 says this reduces hallucination)
        # stage-1 models use 'square'
        self.image_aspect_ratio = LLAVA_MODELS[self.model_name]['model']['image_aspect_ratio'] \
            if 'image_aspect_ratio' in LLAVA_MODELS[self.model_name]['model'] else 'pad'
        
        self.conversational_style = LLAVA_MODELS[self.model_name]['model']['conversation']
        
        self.context_len = CONTEXT_LEN
        
        self.tokenizer, self.model, self.image_processor = self.load_pretrained_model(
            LlavaLlamaForCausalLM,
            model_args,
            model_path=LLAVA_MODELS[self.model_name]['model']['path'],
            tokenizer_path=LLAVA_MODELS[self.model_name]['tokenizer']['path'],
            model_max_length=model_max_length,
            padding_side=padding_side,
            image_aspect_ratio=self.image_aspect_ratio,
            mmprojector_repo=mmprojector_repo,
            mmprojector_name=mmprojector_name,
            device=self.device,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def image_loader(self, image_path):
        if image_path.split('.')[-1] == 'npy':
            return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
        else:
            return Image.open(image_path).convert("RGB")

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        if self.image_aspect_ratio == 'pad':
            image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
        image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
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
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [self.format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [self.format_answer(answer, conversation_style=self.conversational_style) for answer in answers]
        
        images = self.load_images(images)

        # format prompts
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in prompts]
        labels = copy.deepcopy(input_ids)
        for label, qs in zip(labels, questions):
            tokenized_len = len(tokenizer_image_token(qs, self.tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = IGNORE_INDEX
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
            
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            labels,
            images
        )
        
        assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': input_ids, # None for LLaVA-1.5
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': False,
        }
        
        outputs = self.model.model(
            **model_input_kwargs
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)

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
        score = self.forward([image_path], [prompt])
        return score

    def compute_batch(self, image_path_list, prompt_list):
        scores = self.forward(image_path_list, prompt_list)
        return scores

if __name__ == '__main__':
    llava_metric = VQAScoreLLaVAMetric()
    images = ['/home/muzammal/Projects/TRIG/demo.jpg','/home/muzammal/Projects/TRIG/demo.jpg']
    texts = ['a classic white building','a technical drawing of a building']
    scores = llava_metric.compute_batch(images, texts)
    print(scores)