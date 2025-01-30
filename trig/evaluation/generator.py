import os
import json
import torch
from trig.utils.utils import load_config, base64_to_image
from openai import OpenAI
from pathlib import Path

from codebase.OmniGen import OmniGenPipeline
from codebase.onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from diffusers import SanaPipeline
from diffusers import DiffusionPipeline
from diffusers import Transformer2DModel, PixArtSigmaPipeline

project_root = Path(__file__).resolve().parents[2]

class Generator:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)

    def load_prompts(self,prompt_file):
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

    def generate(self,args):

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
                    if dim_prompt["parent_dataset"][0].startswith("<") and dim_prompt["parent_dataset"][0].endswith(
                            ">"):
                        parent_dim1 = dim_prompt["parent_dataset"][0].split(", ")[0]
                        parent_dim2 = dim_prompt["parent_dataset"][0].split(", ")[1]
                        parent_dim = f"{parent_dim1[1:]}_{parent_dim2[:-1]}"
                        idx = dim_prompt['data_id'].split('_')[-1]

                        if os.path.exists(
                                os.path.join(output_path.replace(dim, parent_dim), f'{parent_dim}_{idx}.png')):
                            shutil.copy(os.path.join(output_path.replace(dim, parent_dim), f'{parent_dim}_{idx}.png'),
                                        os.path.join(output_path, f"{dim_prompt['data_id']}.png"))
                        else:
                            continue

                    else:
                        prompt = dim_prompt["prompt"]
                        if args.model == 'OmniGen':
                            image = pipe(prompt=prompt, height=1024, width=1024, guidance_scale=2.5, seed=0)[0]
                        elif args.model == 'OneDiffusion':
                            image = pipe(prompt=f"[[text2image]] {prompt}", negative_prompt=OD_NEGATIVE_PROMPT,
                                         num_inference_steps=50, guidance_scale=4, height=1024, width=1024, ).images[0]
                        elif args.model == 'Sana':
                            image = \
                            pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20,
                                 generator=torch.Generator(device="cuda").manual_seed(42))[0]
                            image = image[0]
                        elif args.model == 'Pixart-Sigma':
                            image = pipe(prompt).images[0]
                        elif args.model == 'SD-XL':
                            image = pipe(prompt).images[0]
                        elif args.model == 'dalle3':
                            try:
                                response = pipe.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024",
                                                                quality="standard", n=1, response_format='b64_json')
                                image_b64 = response.data[0].b64_json
                                image = base64_to_image(image_b64)
                            except Exception:
                                print(dim_prompt['data_id'])
                                continue

                        image.save(os.path.join(output_path, f"{dim_prompt['data_id']}.png"))
