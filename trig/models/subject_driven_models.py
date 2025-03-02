import torch
from PIL import Image
from diffusers.utils import load_image
from trig.models.base import BaseModel

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
        cond_image = load_image(input_image)
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
