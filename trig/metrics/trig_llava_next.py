from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import pdb

pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda:1"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

model.eval()
model.tie_weights()

image_path = "/home/muzammal/Projects/TRIG/demo.jpg"
image = Image.open(image_path)
images = [image, image]  # batch size = 2

image_tensors = process_images(images, image_processor, model.config)
image_tensors = torch.stack(
    [_image.to(dtype=torch.float16, device=device) for _image in image_tensors],
    dim=0
)
image_sizes = [img.size for img in images]


conv_template = "qwen_1_5"  
question = DEFAULT_IMAGE_TOKEN + "\nIs this a building in this image, please answer with 'yes' or 'no'?"

conv1 = copy.deepcopy(conv_templates[conv_template])
conv1.append_message(conv1.roles[0], question)
conv1.append_message(conv1.roles[1], None)
prompt1 = conv1.get_prompt()

conv2 = copy.deepcopy(conv_templates[conv_template])
conv2.append_message(conv2.roles[0], question)
conv2.append_message(conv2.roles[1], None)
prompt2 = conv2.get_prompt()

prompts = [prompt1, prompt2]

token_tensors = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts]
input_ids = torch.stack(token_tensors, dim=0).to(device)


with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        images=image_tensors,     
        image_sizes=image_sizes,   
    )
    logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    last_token_logits = logits[:, -1, :]  # shape: [batch_size, vocab_size]
    predicted_token_ids = torch.argmax(last_token_logits, dim=-1)  # shape: [batch_size]
    predicted_tokens = [tokenizer.decode(token_id) for token_id in predicted_token_ids]
    print("Predicted tokens:", predicted_tokens)
