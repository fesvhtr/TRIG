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
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

ETA = 0.90
KAPPA = 0.6


def record_params(save_dir="./subdir", FS=None, TS=None,  clear_low=True, guidance_scale=7.5,
                    prompt=None, HS=None, gen_MSK_THRS=None):
    '''
        Record the used parameter sets and prompts
    '''
    os.makedirs(save_dir, exist_ok=True)
    if TS is not None:
        with open(os.path.join(save_dir,"record.txt") , "w") as rf:
            if clear_low:
                rf.write("Clear low " + str(ETA) + "\n")
            else:
                rf.write("No clear low\n")
            if gen_MSK_THRS is not None:
                rf.write( "gen_MSK_THRS " + str(gen_MSK_THRS) + "\n")
            for pt in prompt:
                rf.write(pt)
            rf.write("\n")
            rf.write(str(guidance_scale)+"\n")
            rf.write(", ".join(str(tp) for tp in TS) + "\n")
            rf.write(", ".join(str(tp) for tp in FS) + "\n")
            rf.write(", ".join(str(tp) for tp in HS) + "\n")



@torch.no_grad()
def encode_prompt(model, prompt, batch_size=1):
    '''
        Encode the given prompt and default null prompt into embeddings
    '''
    text_input = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length,
        truncation=True, return_tensors="pt")
    cond_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    return cond_embeddings, uncond_embeddings



@torch.no_grad()
def init_latent(latent, model, height, width, generator, batch_size, ):
    '''
        Initilize latents as random values or expand them from given ones
    '''
    SZ = model.vae_scale_factor ### 8
    if latent is not None:
        latents = latent.expand(batch_size, model.unet.config.in_channels, height//SZ, width//SZ).to(model.device)
        return latent, latents

    latent = torch.randn((1, model.unet.config.in_channels, height//SZ, width//SZ), generator=generator,)
    latents = latent.expand(batch_size, model.unet.config.in_channels, height//SZ, width//SZ).to(model.device)

    return latent, latents



@torch.no_grad()
def latn2img(vae, latents):
    '''
        Change VAE latent from gpu back to images(255 val-space) as numpy arrays in cpu
    '''
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    if image.requires_grad:
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    else:
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image



# name should be step for intermediate results, and their prefix denoting their type, e.g. "ORI-5"
def latn_to_img(model, latn, save_dir='./outputs/tmp_test/', tstamp=None, name='TMP', suf="", pre=""):
    '''
        A warpper of latn2img, adding names
    '''
    if tstamp is None:
        tstamp = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now()) 
    image = latn2img(model.vae, latn) # latent, gpu torch tensor; -> image, cpu numpy array
    for i in range (image.shape[0]):
        Image.fromarray(image[i]).save(save_dir + tstamp + '-' + pre + name + suf + '-'+str(i)+'.png')




@torch.no_grad()
def get_frq(x_t, shift=True, absv=False, density=False, pow_density=False, float64=False,
            norm_typ="ortho"):
    if float64: ### 64 version
        with torch.autocast(device_type='cuda', dtype=torch.float64):
            x_freq = torch.fft.fftn(x_t, dim=(-2, -1), norm=norm_typ)
            if shift:
                x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
            if absv:
                x_freq = torch.abs(x_freq)
            elif density:
                x_freq = torch.abs(x_freq)
                # x_freq = x_freq / torch.sum(x_freq)
            elif pow_density:
                x_freq = torch.abs(x_freq)
                x_freq_pow = torch.pow(x_freq, 2)
                # x_freq_pow = x_freq_pow / torch.sum(x_freq_pow)
    else:
        x_freq = torch.fft.fftn(x_t, dim=(-2, -1), norm=norm_typ)
        if shift:
            x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        if absv:
            x_freq = torch.abs(x_freq)
        elif density:
            x_freq = torch.abs(x_freq)
            # x_freq = x_freq / torch.sum(x_freq)
        elif pow_density:
            x_freq = torch.abs(x_freq)
            x_freq_pow = torch.pow(x_freq, 2)
            # x_freq_pow = x_freq_pow / torch.sum(x_freq_pow)

    return x_freq




def get_diamond_mask(THRS, mask, H, W):
    '''
        You may use a diamond shape mask, rather than a square one
    '''
    x = torch.arange(H, device=mask.get_device()).unsqueeze(-1)
    y = torch.arange(W, device=mask.get_device())
    dist = torch.abs(x-(H//2)+0.5) + torch.abs(y-(W//2)+0.5)
    dist[dist < (THRS + 0.5)] = 0
    dist[dist > 0.5] = 1.0
    dist = dist.unsqueeze(0).unsqueeze(0)

    return dist




@torch.no_grad()
def mod_frq(x_t, thrs, high_thrs=None, cross_thrs=4, mod_sty=False,
            rx_t=None, guidance_scale=3.5, norm_typ="ortho",
            filter_shape="dm"):
    '''
        Apply filter to the guidance, currently we use only high-pass filters.
        Low-pass filters will be easy to implement.
    '''
    if thrs==0:
        return x_t
    elif thrs>=32:
        empty_guidance = torch.zeros_like(x_t).to(x_t.get_device())
        return empty_guidance
    bs, ch, H, W = x_t.shape

    x_freq = get_frq(x_t, norm_typ="ortho",)
    mask = torch.ones((1,ch,H,W), device=x_t.get_device())
    crow, ccol = H//2, W//2
    THRS = thrs

    # We can use diamond or square shape filter
    if filter_shape == "sq":
        mask[..., crow-THRS:crow+THRS, ccol-THRS:ccol+THRS] = 0 
    else: # 'dm'
        mask =  get_diamond_mask(THRS, mask, H, W)*mask

    x_freq = x_freq*mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2,-1))
    xt_mod = torch.fft.ifftn(x_freq, dim=(-2,-1), norm=norm_typ).real

    if THRS < 32 :
        chg_msk = (torch.abs(xt_mod - x_t) / torch.abs(x_t)) > KAPPA
        xt_mod[chg_msk] = 0

    return xt_mod




@torch.no_grad()
def diffusion_one_step(model, latn, context, t, guidance_scale, null_img=False, modi_frq=False,
                       clear_low=False, LOW_THRES=0.004, ori_guidance=None, 
                       remove=False, total_mask=None, guide_mask=None, record_steps=0,
                       filter_shape = 'dm', t_index=None,
                       TS = (881, 801, 501), FS=(1, 1, 1, 1), HS=(32, 32, 32, 32),
                       over_head=None, time_cost=None,):
    if not null_img: # text generation
        if over_head is not None:
            time_start = time.time()

        noise_pred_null, noise_pred_text = model.unet(torch.cat([latn, latn]), 
                                    t, encoder_hidden_states=context)["sample"].chunk(2)

        if over_head is not None:
            time_end = time.time()
            time_cost += time_end - time_start

        guidance = noise_pred_text - noise_pred_null
        raw_guidance= guidance.clone().detach()

        if modi_frq: 
            if guide_mask is not None: ### if this is not none, then we assume that we are in the second stage of two-step editing
                guidance = guidance*guide_mask
                noise_pred = noise_pred_null + guidance_scale*guidance 
            else:
                found = False
                for ti in range(len(TS)):
                    if t >= TS[ti]:
                        thrs_cur = FS[ti]
                        high_thrs_cur = HS[ti]
                        found = True
                        break
                if not found:
                    thrs_cur = FS[-1]
                    high_thrs_cur = HS[-1]
                
                if over_head is not None:
                    time_start = time.time()
                guidance = mod_frq(guidance, thrs=thrs_cur, high_thrs=high_thrs_cur, rx_t=noise_pred_null, guidance_scale=guidance_scale,
                                    norm_typ="ortho", filter_shape = filter_shape,)

                if clear_low:
                    # pass
                    B, C, H, W = guidance.shape
                    abs_gui = torch.abs(guidance).flatten(start_dim=1)
                    tmp = torch.quantile(abs_gui, ETA, dim=1).unsqueeze(0) # currently for all channel
                    mask = (abs_gui < tmp).reshape(B, C, H, W) | (torch.abs(guidance)<LOW_THRES) 
                    guidance[mask] = 0

                else:
                    mask = torch.abs(guidance)<LOW_THRES
                    guidance[mask] = 0
                if over_head is not None:
                    time_end = time.time()
                    over_head += time_end - time_start

                if total_mask is not None:
                    if t <= 961:
                        msk_guidance = torch.abs(guidance)
                        sum_msk_guidance = torch.sum(msk_guidance)

                        msk_thrs = 0.65
                        tmp_msk = msk_guidance.clone().detach().flatten()
                        tmp = torch.quantile(tmp_msk, msk_thrs, dim=0)

                        temp = torch.mean(msk_guidance)
                        tmp = max(temp, tmp)
                        msk_guidance[msk_guidance < tmp] = 0
                        
                        if sum_msk_guidance < 25:
                            msk_guidance *= 25/sum_msk_guidance

                        total_mask += msk_guidance 
                        if torch.sum(msk_guidance) > 0.01:
                            record_steps += 1
                    noise_pred = noise_pred_null  + guidance_scale*guidance 
                else:
                    if remove:
                        direction = -1
                    else:
                        direction = 1
                    noise_pred = noise_pred_null + guidance_scale*guidance * direction
        else:
            noise_pred = noise_pred_null + guidance_scale*guidance
    else: # null generation
        noise_pred_null = model.unet(latn, t, encoder_hidden_states=context[0].unsqueeze(0))["sample"]
        guidance = noise_pred_null
        noise_pred = noise_pred_null
        raw_guidance= guidance.clone().detach()

    if over_head is not None:
        time_start = time.time()

    out = model.scheduler.step(noise_pred, t, latn)
    pred_x0 = out["pred_original_sample"]
    latn = out["prev_sample"]

    if over_head is not None:
        time_end = time.time()
        time_cost += time_end - time_start

    return noise_pred_null, guidance, noise_pred, pred_x0, latn, total_mask, raw_guidance, over_head, time_cost, record_steps




@torch.no_grad()
def diffusion_step_frq(model, ori_latn, latn, context, t, guidance_scale, null_latn=None, 
                       mod_guidance_frq=False,
                       tstamp=None, save_dir="./outputs/tmp_test/", 
                       clear_low=True,  TS=None, FS=None, HS=None,
                       total_mask=None, guide_mask=None,
                       remove=False, filter_shape='dm',
                       t_index=None, record_steps=0,
                       over_head=None, time_cost=None,):

    ori_noise_pred_null, ori_guidance, ori_noise_pred, ori_pred_x0, ori_latn, _, _, _, _, _,= diffusion_one_step(
            model, ori_latn, context, t, guidance_scale, t_index=t_index,
            )

    noise_pred_null, guidance, noise_pred, pred_x0, latn, total_mask, raw_guidance, over_head, time_cost, record_steps = diffusion_one_step(model, latn, context, 
                t, guidance_scale, modi_frq=mod_guidance_frq, clear_low=clear_low,
                TS=TS, FS=FS, HS=HS, ori_guidance=ori_guidance, 
                total_mask=total_mask, guide_mask=guide_mask, remove=remove,
                filter_shape = filter_shape, t_index=t_index, record_steps=record_steps,
                over_head = over_head, time_cost=time_cost,)

    if null_latn is not None:
        null_noise_pred_null, _, _, null_pred_x0, null_latn, _, _, _, _, _,= diffusion_one_step(
            model, null_latn, context, t, guidance_scale, null_img=True, t_index=t_index,
            )

    return ori_latn, latn, null_latn, total_mask, over_head, time_cost, record_steps




@torch.no_grad()
def frq_img_gen(model, prompt: List[str],  num_infer_steps: int=50,
                 guidance_scale: float=7.5, generator: Optional[torch.Generator]=None, 
                 latent: Optional[torch.FloatTensor]=None,  save_dir='./outputs/test/',
                 g_seed=None, record_time = False,  
                 mod_guidance_frq=True,  TS=None, FS=None, HS=None, 
                 clear_low=True, filter_shape = 'sq', remove=False, 
                 generate_mask=False, record_steps=0, gen_MSK_THRS= 0.065, 
                 guide_mask=None, reverse_mask=False,
                ):
    '''

        3 kinds for images will be generated in a single run:
            1. image that reconstructed from inversion, to check if the inversion fails, termed with prefix null
            2. image that edited without FreeDiff, termed with prefix ori
            3. image that edited with FreeDiff, termed without prefix

        Params:
            model: the diffusion model
            prompt: the target prompt in a list
            num_infer_steps: number of inference steps, we gennerally use 50 as widely used for DDIN generation
            guidance_scale: default value set to 7.5, can try with other values when using FreeDiff
            latent: the inverted latent of image
            save_dir: the directory where the output images and parameter records will be saved
            g_seed: the seed for generator
            record_time: record the time used for modification generation, and the overhead of applying freediff

            FreeDiff 1 stage related params:
            TS: time steps for controlling corresponding filters, tuples
            FS: the filter size for the highpass filter, ranging from [0, 32]
            HS: the filter size for the lowpass filter, we do not really set it for editing and keep as default value 32
            clear_low: use the \eta_{0.8} as mentioned in the paper to clear the lower percentile
            filter_shape: the shape of the frequency filter, which can be set as square 'sq' or diamond 'dm'
            remove: when the editing type is remove, we can change the guidance direction for performing removal, 
                    or simply use a target prompt that describe the background but without containing the object to be removed.

            Two stage editing related params:
            generate_mask: set to True to generate a mask for refining guidance at the second stage
            gen_MSK_THRS: parameter to control the filteration of the generated mask
            guide_mask: the mask generated from first stage and used to guide the generation on the second stage
            reverse_mask: reverse_the guide_mask 

    '''
    # record params for generation
    # record_params(save_dir, TS, FS,  HS=HS, clear_low=clear_low, guidance_scale=guidance_scale, 
    #                 prompt=prompt, gen_MSK_THRS=gen_MSK_THRS)

    # record time and overhead for generation
    if record_time:
        over_head = 0
        time_cost = 0
    else:
        over_head = None
        time_cost = None

    tstamp = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now()) 
    if g_seed is not None:
        tstamp = str(g_seed) + "-" + tstamp

    for param in model.unet.parameters():
        param.requires_grad = False

    if guide_mask:
        guide_mask = torch.load(save_dir+"guide_mask.pt").to(model.device)
        if reverse_mask:
            guide_mask = 1 - guide_mask

    batch_size = len(prompt)

    height = width = 512
    cond_emb, uncond_emb = encode_prompt(model, prompt)
    context = torch.cat([uncond_emb, cond_emb], dim=0).to(model.device)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size,)
    model.scheduler.set_timesteps(num_infer_steps)

    # latents copy, ori_latn:guided generation w/o modification, null_latn: null generation
    ori_latn = latents.clone().detach()
    latn = latents
    null_latn = ori_latn.clone().detach()

    # generate an empty mask for the two stage editing's first stage
    if generate_mask:
        total_mask = torch.zeros(latn.shape).to(latn.get_device())
    else:
        total_mask = None

    current_timesteps = model.scheduler.timesteps
    t_index = 0

    for t in current_timesteps: 
        ori_latn, latn, null_latn,  total_mask, over_head, time_cost, record_steps = diffusion_step_frq(
                            model, ori_latn, latn, context, t, guidance_scale, null_latn, tstamp=tstamp,
                            save_dir=save_dir, mod_guidance_frq=mod_guidance_frq,
                            TS=TS, FS=FS, HS=HS, clear_low=clear_low, 
                            total_mask=total_mask, guide_mask=guide_mask, remove=remove, 
                            filter_shape = filter_shape, t_index = t_index, 
                            record_steps=record_steps,
                            over_head=over_head, time_cost=time_cost,)
        t_index += 1
    if over_head is not None:
        print("overhead: %.3f"%over_head, "time cost: %.3f"%time_cost, "overhead ratio: %.3f"%(over_head/time_cost))
    
    if total_mask is not None:
        total_mask = torch.sum(total_mask, dim=1, keepdim=True) 
        # quantile
        total_mask[total_mask<record_steps*gen_MSK_THRS] = 0

        _, N, H, W = latn.shape
        vmax = torch.max(total_mask)
        plt.figure()
        plt.axis("off")
        plt.imshow(total_mask[0,0].cpu().numpy(), extent=(-N//2, N//2, -N//2, N//2), vmin=0, vmax=vmax)
        plt.savefig(save_dir + 'mask_not_unified.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.cla()
        plt.clf()

        total_mask[total_mask>0] = 1
        torch.save(total_mask, save_dir + 'guide_mask.pt')

        _, N, H, W = latn.shape
        vmax = torch.max(total_mask)
        plt.figure()
        plt.axis("off")
        plt.imshow(total_mask[0,0].cpu().numpy(), extent=(-N//2, N//2, -N//2, N//2), vmin=0, vmax=vmax)
        plt.savefig(save_dir + 'mask.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.cla()
        plt.clf()

    # latn_to_img(model, ori_latn, save_dir, tstamp, name='ORI')
    # latn_to_img(model, latn, save_dir, tstamp, name='MOD')
    # latn_to_img(model, null_latn, save_dir, tstamp, name='NUL')

    res_latents = [ori_latn, latn, null_latn]
    return res_latents














