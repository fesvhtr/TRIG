# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import os


warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-critic-7b"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/LLaVA-VL/blog/blob/main/2024-10-03-llava-critic/static/images/critic_img_seven.png?raw=True"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

# pairwise ranking
critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe first response: [The image is a black and white sketch of a line that appears to be in the shape of a cross. The line is a simple and straightforward representation of the cross shape, with two straight lines intersecting at a point.]\nThe second response: [This is a handwritten number seven.]\nASSISTANT:\n"

# pointwise scoring
# critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of answer answers provided by a Large Multimodal Model (LMM). Score the response out of 100 and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe LMM response: [This is a handwritten number seven.]\nASSISTANT:\n "

question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
