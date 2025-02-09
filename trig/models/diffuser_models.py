import torch
from trig.models.base import BaseModel
from diffusers import DiffusionPipeline
from trig.config import OD_NEGATIVE_PROMPT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SDXLModel(BaseModel):
    """
    ICLR 2024
    SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
    https://github.com/Stability-AI/generative-models
    """
    def __init__(self):
        self.model_name = "SDXL"
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to(device)

    def generate(self, prompt):
        try:
            image = self.pipe(prompt).images[0]
            return image
        except Exception as e:
            print(f"Error generating image with {self.model_name}: {e}")
            return None

class OmniGenModel(BaseModel):
    """
    Arxiv 2024
    OmniGen: Unified Image Generation
    https://github.com/VectorSpaceLab/OmniGen
    """
    def __init__(self):
        self.model_name = "OmniGen"
        from trig.models.OmniGen import OmniGenPipeline
        self.pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=2.5, seed=0)[0]
        return image
    
    def generate_p2p(self, prompt):
        pass

class PixartSigmaModel(BaseModel):
    """
    ECCV 2024
    PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation
    https://github.com/PixArt-alpha/PixArt-sigma
    """
    def __init__(self):
        self.model_name = "PixArt_Sigma"
        from diffusers import Transformer2DModel, PixArtSigmaPipeline

        weight_dtype = torch.float16
        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            subfolder='transformer',
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        self.pipe.to(device)

    def generate(self, prompt):
        image = self.pipe(prompt).images[0]
        return image

class OneDiffusionModel(BaseModel):
    """
    Arxiv 2024
    One Diffusion to Generate Them All
    https://github.com/lehduong/OneDiffusion
    """
    def __init__(self):
        self.model_name = "OneDiffusion"
        from trig.models.onediffusion import OneDiffusionPipeline
        self.pipe = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)

    def generate(self, prompt):
        image = self.pipe(prompt=f"[[text2image]] {prompt}", negative_prompt=OD_NEGATIVE_PROMPT, num_inference_steps=50,
                     guidance_scale=4, height=1024, width=1024, ).images[0]
        return image

    def generate_p2p(self, prompt):
        pass
        
class SanaModel(BaseModel):
    """
    Arxiv 2024
    Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer
    https://github.com/NVlabs/Sana
    """
    def __init__(self):
        self.model_name = "Sana"
        from diffusers import SanaPipeline
        self.pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers", variant="fp16", torch_dtype=torch.float16,)
        self.pipe.to(device)

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20,
                     generator=torch.Generator(device="cuda").manual_seed(42))[0]
        image = image[0]
        return image

class FLUXModel(BaseModel):
    """
    FLUX from Black Forest Labs
    https://github.com/black-forest-labs/flux
    """
    def __init__(self):
        self.model_name = "FLUX"
        self.model_id = "black-forest-labs/FLUX.1-dev"
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)

        # uncomment if you want to use less GPU memory
        # self.pipe.enable_model_cpu_offload()
    
    def generate(self, prompt):
        image = self.pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator(device="cpu").manual_seed(0)
        ).images[0]
        return image
    
class SD35Model(BaseModel):
    """
    Stable Diffusion 3.5
    https://github.com/Stability-AI/sd3.5
    """
    def __init__(self):
        self.model_name = "SD3.5"
        self.model_id = "stabilityai/stable-diffusion-3.5-large"
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(device)

        # uncomment if you want to use less GPU memory
        # self.enable_model_cpu_offload()

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, prompt_3=prompt, num_inference_steps=28, guidance_scale=4.5,max_sequence_length=512).images[0]
        return image
