import argparse
import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import DiffusionPipeline

class SDXLModel(BaseDModel):
    def __init__(self):
        super().__init__("SD-XL")
        self.pipe = DiffusionPipeline.from_pretrained(
            "models/stable-diffusion/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to(device)

    def generate(self, prompt, **kwargs):
        try:
            return self.pipe(prompt).images[0]
        except Exception as e:
            print(f"Error generating image with {self.model_name}: {e}")
            return None

class OmniGen(BaseModel):
    def __init__(self):
        super().__init__("OmniGen")
        from codebase.OmniGen import OmniGenPipeline
        self.pipe = OmniGenPipeline.from_pretrained("models/OmniGen")

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt=prompt, **kwargs)[0]

class PixartSigma(BaseModel):
    def __init__(self):
        super().__init__("Pixart-Sigma")
        from diffusers import Transformer2DModel, PixArtSigmaPipeline

        transformer = Transformer2DModel.from_pretrained(
            "models/PixArt-Sigma/PixArt-Sigma-XL-2-1024-MS",
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "models/PixArt-Sigma/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt).images[0]

class OneDiffusion(BaseModel):
    def __init__(self):
        super().__init__("OneDiffusion")
        from codebase.OneDiffusion import OneDiffusionPipeline
        self.pipe = OneDiffusionPipeline.from_pretrained("models/One-Diffusion").to(device=device, dtype=torch.bfloat16)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt=prompt, **kwargs)[0]