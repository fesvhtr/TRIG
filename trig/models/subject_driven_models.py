import torch
from PIL import Image
from diffusers.utils import load_image
from trig.models.base import BaseModel
from trig.config import gpt_logit_dimension_msg, DIM_NAME_DICT
import time
from openai import OpenAI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BlipDiffusionModel(BaseModel):
    """
    NIPS 2023
    Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing    
    https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion
    """
    def __init__(self):
        self.model_name = "Blip-diffusion"
        from diffusers import BlipDiffusionPipeline
        self.pipe = BlipDiffusionPipeline.from_pretrained(
            "Salesforce/blipdiffusion", 
            torch_dtype=torch.float16
        )
        self.pipe.to(device)
        self.negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    def generate_s2p(self, prompt, item, input_image):
        if isinstance(input_image, str):
            cond_image = load_image(input_image)
        else:
            cond_image = input_image
        image = self.pipe(
            prompt,
            cond_image,
            item,
            item,
            guidance_scale=7.5,
            num_inference_steps=25,
            neg_prompt=self.negative_prompt,
            height=512,
            width=512,
        ).images[0]

        return image


class SSREncoderModel(BaseModel):
    """
    CVPR 2024
    SSR-Encoder: Encoding Selective Subject Representation for Subject-Driven Generation
    https://github.com/Xiaojiu-z/SSR_Encoder
    """
    def __init__(self):
        self.model_name = "SSREncoder"
        from trig.models.SSREncoder.utils.pipeline_t2i import StableDiffusionPipeline
        from trig.models.SSREncoder.ssr_encoder import SSR_encoder
        from diffusers import StableDiffusionPipeline
        from diffusers import UniPCMultistepScheduler

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            torch_dtype=torch.float32).to("cuda")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.ssr_model = SSR_encoder(self.pipe.unet, "openai/clip-vit-large-patch14", 
                                     "cuda", dtype=torch.float32)
        self.ssr_model.get_pipe(self.pipe)
        
        # FIXME: give a script to download the model
        base_ssr = "/home/muzammal/Projects/TRIG/data/ssrencoder"
        ssr_ckpt = [base_ssr+"/pytorch_model.bin",
                    base_ssr+"/pytorch_model_1.bin"]
        self.ssr_model.load_SSR(ssr_ckpt[0], ssr_ckpt[1])

    def generate_s2p(self, prompt, item, input_image):
        pil_img = load_image(input_image)
        image = self.ssr_model.generate(
            pil_image=pil_img,
            concept=item,
            uncond_concept="",
            prompt=prompt,
            negative_prompt="bad quality",
            num_samples=1,
            seed=None,
            guidance_scale=5,
            scale=0.65,
            num_inference_steps=30,
            height=512,
            width=512,
        )[0]

        return image


class OminiControlModel(BaseModel):
    """
    Arxiv 2024
    OminiControl: Minimal and Universal Control for Diffusion Transformer
    https://github.com/Yuanshi9815/OminiControl
    """
    def __init__(self):
        self.model_name = "OminiControl"
        from trig.models.OminiControl.flux.condition import Condition
        from trig.models.OminiControl.flux.generate import generate as omini_generate
        from trig.models.OminiControl.flux.generate import seed_everything
        from diffusers import FluxPipeline
        self.Condition = Condition
        self.omini_generate = omini_generate
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        self.pipe.load_lora_weights(
            "Yuanshi/OminiControl",
            weight_name=f"omini/subject_512.safetensors",
            adapter_name="subject",
        )
        seed_everything(0)

    def generate_s2p(self, prompt, item, input_image):
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        
        condition = self.Condition("subject", input_image, position_delta=(0, 32))
        image = self.omini_generate(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            num_inference_steps=8,
            height=512,
            width=512,
        ).images[0]

        return image


class XFluxModel(BaseModel):
    """
    Huggingface: https://huggingface.co/XLabs-AI/flux-ip-adapter-v2
    A IP-Adapter checkpoint for FLUX.1-dev model by Black Forest Labs
    https://github.com/XLabs-AI/x-flux
    """
    def __init__(self):
        self.model_name = "xflux"
        from trig.models.xflux.flux.xflux_pipeline import XFluxPipeline
        self.pipe = XFluxPipeline("flux-dev", "cuda", False)
        self.pipe.set_ip(None, "XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors")

    def generate_s2p(self, prompt, item, input_image):
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        
        image = self.pipe(
            prompt=prompt,
            controlnet_image=None,
            width=512,
            height=512  ,
            guidance=4,
            num_steps=25,
            seed=42,
            true_gs=3.5,
            control_weight=0.0,
            neg_prompt="",
            timestep_to_start_cfg=1,
            image_prompt=input_image,
            neg_image_prompt=None,
            ip_scale=1.0,
            neg_ip_scale=1.0,
        )

        return image

class XFluxDTMDimModel(BaseModel):
    """
    Stable Diffusion 3.5 with Dimensional Fine-tuning with DTM
    https://github.com/Stability-AI/sd3.5
    """
    def __init__(self):
        self.model_name = "xflux"
        from trig.models.xflux.flux.xflux_pipeline import XFluxPipeline
        self.pipe = XFluxPipeline("flux-dev", "cuda", False)
        self.pipe.set_ip(None, "XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors")
        
        self.DTM_3d35 = {
            "synergy": [
                ["TA-C", "IQ-O"],   
                ["TA-S", "IQ-A"],
            ],
            "bottleneck": [
                ["IQ-R", "R-T"],
                ["TA-S", "R-T"],
            ],
            "tilt": [
                ["IQ-R", "TA-S"],
            ],
            "dispersion": [
                ["IQ-O", "R-B"]
            ]
        }

        # uncomment if you want to use less GPU memory
        # self.enable_model_cpu_offload()

    def generate_s2p(self, prompt, item, input_image, dimension):
        prompt = self.change(prompt, self.DTM_3d35, dimension)
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        
        image = self.pipe(
            prompt=prompt,
            controlnet_image=None,
            width=512,
            height=512  ,
            guidance=4,
            num_steps=25,
            seed=42,
            true_gs=3.5,
            control_weight=0.0,
            neg_prompt="",
            timestep_to_start_cfg=1,
            image_prompt=input_image,
            neg_image_prompt=None,
            ip_scale=1.0,
            neg_ip_scale=1.0,
        )

        return image

    def send_request_with_retry(self, msg, max_retries=3, delay=2):
        SYSTEM_MSG = """You are a prompt engineering expert specializing in multi-dimensional image generation optimization. I have discovered some tradeoff relationships in pairwise dimension.
        You need to modify the original prompt so that the final result of the image generation model performs best in multiple dimensions.
        You will only receive one of the following relationships, and you will know the relationship between the two Dimensions and the trade-off changes you need to make

        **Optimization Rules:**

        - Synergy (X & Y): Enhance both dimensions simultaneously
        - Bottleneck (X-Y): Break limitations between conflicting dimensions
        - Tilt (X↑Y): Prioritize X while maintaining minimum Y
        - Dispersion (X~Y): Balance unstable dimension relationships
        - No trade-off: No trade-off relationship between the two dimensions, no need to modify
         
        **Modify Requirements:**
        1. What you need to do is to understand the original prompt and balance and modify the content of the different dimensions according to the Optimisation Rules only, 
        but not to add direct relational indications of the dimensions and not to change the content massively. Note that you are balancing and trade-off.
        2. If no relationship exists, then no change is needed,
        3. if the prompt is very long, streamline as required.Retain important information to enhance results
        4. If the relationship exists, it's the detail, not the content, that you need to make changes to; 
        you can't have the generated image have an overall change in content, you have only help the model make the trade-offs by making changes to the detail.
        5. Important!!!!: You can't add direct words like 'balancing dimension A and dimension B', it's not allowed, you can only go into reasonable detail later, you can't refer to any specific dimension.
        
        **Output Requirements:**
        1. Very Important: Keep prompts pithy and concise (must <50 tokens, you must follow this)
        2. Maintain original artistic intent
        3. Use natural language phrasing"""
        for attempt in range(max_retries):
            try:
                # api_key = 'sk-mqUwZI8bhIv746rG6f3fE830D8B146E789Fd11717aD8C4B1'
                api_key = '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'
                # client = OpenAI(api_key=api_key, base_url="https://api.bltcy.ai/v1")
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[ {"role": "system", "content": SYSTEM_MSG},
                                {"role": "user", "content": msg}],
                    max_tokens=500
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"Retry {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retry limit reached, returning original prompt")
                    return None


    def change(self, prompt, DTM, dimension):
        
        dim_def_dict = gpt_logit_dimension_msg['t2i']
        # id
        dim1 = dimension[0]
        dim2 = dimension[1]
        dim_def = '1. ' + dim_def_dict[dim1] + '\n' + '2. ' + dim_def_dict[dim2]
        
        
        instructions = []
        tradeoff = 'No Tradeoff'
        for key, value in DTM.items():
            for dim_pair in value:
                if (dim_pair[0] == dim1 and dim_pair[1] == dim2) or (dim_pair[0] == dim2 and dim_pair[1] == dim1):
                    tradeoff = key
                    break
        # now name
        dim1_name = DIM_NAME_DICT[dim1]
        dim2_name = DIM_NAME_DICT[dim2]

        if {dim1, dim2} == {"TA-C", "IQ-0"} and key == "synergy":
            instructions.append(
                "Enhance content alignment (TA-C) while ensuring high image originality (IQ-0). "
                "Add detail to the subject content and also use this way to enrich originality"
            )
        elif {dim1, dim2} == {"TA-S", "IQ-A"} and key == "synergy":
            instructions.append(
                "Balance Style (TA-S) with Aesthetics (IQ-A). "
                "The stylisation will be enhanced along with the aesthetics, and you can achieve both by refining the details of the style"
            )
        elif {dim1, dim2} == {"IQ-R", "R-T"} and key == "bottleneck":
            instructions.append(
                "Optimize image realism (IQ-R) while maintaining Relation (R-T)." 
                "You don't have to sacrifice realism to avoid toxicity and add realistic details to improve it, you can cut and compress the toxic content as appropriate"
            )
        elif {dim1, dim2} == {"TA-S", "R-T"} and key == "bottleneck":
            instructions.append(
                "Enhance Style Alignment (TA-S) while preserving Toxicity (R-T). "
                "You don't have to sacrifice stylisation to avoid toxicity and add stylistic details to improve the effect, you can cut and compress toxic content as appropriate"
            )
        elif {dim1, dim2} == {"IQ-R", "TA-S"} and key == "tilt":
            instructions.append(
                "Prioritize image realism (IQ-R) while keeping a sufficient level of Style Alignment (TA-S). "
                "Prioritise or add authentic detail, but ensure basic and clear spatial relationship"
            )
        elif {dim1, dim2} == {"IQ-O", "R-B"} and key == "dispersion":
            instructions.append(
                "Stabilize image originality (IQ-O) and prevent Bias (R-B). "
                "Improve the originality of your images by adding details"
            )
        else:
            instructions.append(
                "No tradeoff relationship between the two dimensions, no need to modify"
                "If the prompt itself isn't very long, you can add slightly more detail on each of the two dimensions. But don't get into other Dimension"
                )
        
        msg = '''

        Original prompt: {str1}\n
        Dimension Definition: {str0}\n
        Modify instrctions: {str2}\n
        Output the final prompt:'''.format(str1=prompt, str0=dim_def, str2=" ".join(instructions))
            
        print('Sending request for DTM')
        # print(msg)

        print('Original prompt:', prompt)
        modified_prompt = self.send_request_with_retry(msg)
        print('Modified prompt:', modified_prompt)
        if ':' in modified_prompt:
            modified_prompt = modified_prompt.split(':')[1].strip('"')
        return modified_prompt if modified_prompt else prompt


