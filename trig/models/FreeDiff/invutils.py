import os
import cv2
import sys
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Callable
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0





@torch.no_grad()
def load_512(image_path, left=0, right=0, top=0, bottom=0, return_type='np', NH=512, NW=512):
    '''
        load the image and pad to 512x512
    '''
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    if return_type == 'np':
        image = np.array(Image.fromarray(image).resize((NH, NW)))
    else:
        image = Image.fromarray(image).resize((NH, NW), resample=Image.LANCZOS)
    return image



@torch.no_grad()
def latn2img(latents, model):
    '''
        Change VAE latent from gpu back to images(255 val-space) as numpy arrays in cpu
    '''
    latents = 1 / 0.18215 * latents
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    if image.requires_grad:
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    else:
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image



@torch.no_grad()
def img2latn(image, model, DEVICE):
    '''
        Change images(255 val-space) as numpy arrays in cpu ->  VAE latent in gpu
    '''
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    latent = model.vae.encode(image)['latent_dist'].mean
    latent = latent * 0.18215
    return latent



def encode_text(prompts, model):
    '''
        Encode the prompt to test embeddings. For inversion, the prompt should be null string "".
    '''
    text_input = model.tokenizer(prompts, padding="max_length",
        max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_embeddings



@torch.no_grad()
def ddim_inversion_null_fixpt(x0, model, uncond_emb, save_all=False, INV_STEPS=50, FP_STEPS=3, ):
    '''
        Invert the image latent to x_T using fixpoint iteration, which generally improves the reconstruction accuracy
        but still has chance to fail. For fail cases with default fix-point steps, change the number of fix-point steps
        may help sometimes.
    '''
    def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray],
                     t_index=None):
        input_timestep = timestep
        timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep

        alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        next_sample = ((alpha_prod_t_next/alpha_prod_t)**0.5)*sample + (beta_prod_t_next**0.5 - (alpha_prod_t_next*beta_prod_t/alpha_prod_t)**0.5)*model_output
        return next_sample

    latent = x0
    all_latents = [latent]
    latent = latent.clone().detach()

    for i in range(INV_STEPS):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]

        xt_cur = latent.clone().detach()
        xt_next = latent
        FIXPT_STEPS=FP_STEPS
        # perform fix-point iteration in a single timestep
        for j in range(FIXPT_STEPS):
            pred_noise = model.unet(xt_next, t, encoder_hidden_states=uncond_emb)["sample"]
            xt_next = next_step(pred_noise, t, xt_cur,  t_index=i)
        latent = xt_next.clone().detach()

        if save_all:
            all_latents.append(latent.clone().detach())

    return latent, all_latents
