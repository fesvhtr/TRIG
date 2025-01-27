import argparse
import os
import shutil
import json
import base64
from io import BytesIO
from PIL import Image

import torch
from openai import OpenAI
from codebase.OmniGen import OmniGenPipeline
from codebase.onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from diffusers import SanaPipeline
from diffusers import DiffusionPipeline
from diffusers import Transformer2DModel, PixArtSigmaPipeline


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIM_DICT = {
    "IQ-R": ["IQ-O", "IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-O": ["IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-A": ["TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-C": ["TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-R": ["TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-S": ["D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-M":  ["D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-K":  ["D-A", "R-T", "R-B", "R-E"],
    "D-A":  ["R-T", "R-B", "R-E"],
    "R-T":  ["R-B", "R-E"],
    "R-B":  ["R-E"]
}
OD_NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"


def base64_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_prompts(prompt_file):
    with open(os.path.join('datasets/Trig/Trig-text-to-image/', prompt_file), 'r') as file:
        prompts = json.load(file)
    
    # check details
    total_len = len(prompts)
    uni_len = sum(
                    1
                    for prompt in prompts
                    if "parent_dataset" in prompt
                    and len(prompt["parent_dataset"]) == 2
                    and prompt["parent_dataset"][0].startswith("<")
                    and prompt["parent_dataset"][0].endswith(">")
                )
    print(total_len, uni_len)
    
    return prompts


def load_pipeline(model):
    
    if model == 'OmniGen':
        """"
        Arxiv 2024
        OmniGen: Unified Image Generation
        https://github.com/VectorSpaceLab/OmniGen
        """
        pipe = OmniGenPipeline.from_pretrained("models/OmniGen")
    
    elif model == 'OneDiffusion':
        """
        Arxiv 2024
        One Diffusion to Generate Them All
        https://github.com/lehduong/OneDiffusion
        """
        pipe = OneDiffusionPipeline.from_pretrained("models/One-Diffusion").to(device=device, dtype=torch.bfloat16)

    elif model == 'Sana':
        """
        Arxiv 2024
        Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer
        https://github.com/NVlabs/Sana
        """
        pipe = SanaPipeline.from_pretrained("models/Sana/Sana_1600M_1024px_MultiLing_diffusers", variant="fp16", torch_dtype=torch.float16,)
        pipe.to("cuda")

        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)

    elif model == 'Pixart-Sigma':
        """
        ECCV 2024
        PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation
        https://github.com/PixArt-alpha/PixArt-sigma
        """
        weight_dtype = torch.float16
        transformer = Transformer2DModel.from_pretrained(
            "models/PixArt-Sigma/PixArt-Sigma-XL-2-1024-MS", 
            subfolder='transformer', 
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            "models/PixArt-Sigma/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        pipe.to(device)

    elif model == 'SD-XL':
        """
        ICLR 2024
        SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
        https://github.com/Stability-AI/generative-models
        """

        pipe = DiffusionPipeline.from_pretrained(
            "models/stable-diffusion/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to(device)
        
    elif model == 'dalle3':
        api_key = "your-api-key"
        pipe = OpenAI(api_key=api_key)
        
    return pipe


def main(args):
    
    prompts = load_prompts(args.prompt_file)
    pipe = load_pipeline(args.model)

    
    for dim1, dim2_list in DIM_DICT.items():
        for dim2 in dim2_list:
            dim = f"{dim1}_{dim2}"
            prompt_list = [prompt for prompt in prompts if prompt["data_id"].startswith(dim)]
            
            output_path = os.path.join('./experiments', args.prompt_file.split('.')[0], args.model, dim)
            os.makedirs(output_path, exist_ok=True)
            
            # demo
            prompt_list = prompt_list[:5]
            
            for dim_prompt in prompt_list:
                if dim_prompt["parent_dataset"][0].startswith("<") and dim_prompt["parent_dataset"][0].endswith(">"):
                    parent_dim1 = dim_prompt["parent_dataset"][0].split(", ")[0]
                    parent_dim2 = dim_prompt["parent_dataset"][0].split(", ")[1]
                    parent_dim = f"{parent_dim1[1:]}_{parent_dim2[:-1]}"
                    idx = dim_prompt['data_id'].split('_')[-1]
                    
                    if os.path.exists(os.path.join(output_path.replace(dim, parent_dim), f'{parent_dim}_{idx}.png')):
                        shutil.copy(os.path.join(output_path.replace(dim, parent_dim), f'{parent_dim}_{idx}.png'), os.path.join(output_path, f"{dim_prompt['data_id']}.png"))
                    else:
                        continue

                else:
                    prompt = dim_prompt["prompt"]
                    if args.model == 'OmniGen':
                        image = pipe(prompt=prompt, height=1024, width=1024, guidance_scale=2.5, seed=0)[0]
                    elif args.model == 'OneDiffusion':
                        image = pipe(prompt=f"[[text2image]] {prompt}", negative_prompt=OD_NEGATIVE_PROMPT, num_inference_steps=50, guidance_scale=4 ,height=1024, width=1024,).images[0]
                    elif args.model == 'Sana':
                        image = pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20, generator=torch.Generator(device="cuda").manual_seed(42))[0]
                        image = image[0]
                    elif args.model == 'Pixart-Sigma':
                        image = pipe(prompt).images[0]
                    elif args.model == 'SD-XL':
                        image = pipe(prompt).images[0]
                    elif args.model == 'dalle3':
                        try:
                            response = pipe.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1, response_format='b64_json')                  
                            image_b64 = response.data[0].b64_json
                            image = base64_to_image(image_b64)
                        except Exception:
                            print(dim_prompt['data_id'])
                            continue

                    image.save(os.path.join(output_path, f"{dim_prompt['data_id']}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--prompt_file", type=str, default="text-to-imgae-v1.json")
    parser.add_argument("--model", type=str, default='SD-XL', choices=['OmniGen', 'OneDiffusion', 'Sana', 'Pixart-Sigma', 'SD-XL', 'flux-pro', 'dalle3'])

    args = parser.parse_args()
    main(args)


    
    