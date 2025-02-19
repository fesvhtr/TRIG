import torch
from PIL import Image

from trig.models.base import BaseModel
from trig.models.OmniGen import OmniGenPipeline
from trig.models.onediffusion import OneDiffusionPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OmniGenModel(BaseModel):
    """
    Arxiv 2024
    OmniGen: Unified Image Generation
    https://github.com/VectorSpaceLab/OmniGen
    """
    def __init__(self):
        self.model_name = "OmniGen"
        self.pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    def generate(self, prompt, task='t2i', input_image=None, item=None):
        if task == 't2i':
            image = self.generate_t2i(prompt)
        elif task == 'p2p':
            image = self.generate_p2p(prompt, input_image)
        elif task == 's2p':
            image = self.generate_s2p(prompt, item, input_image)
        else:
            raise ValueError("Invalid task")
        return image
    
    def generate_t2i(self, prompt):
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=2.5, seed=0)[0]
        return image
    
    def generate_p2p(self, prompt, input_image):
        prompt=f"Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n {prompt}"
        images = self.pipe(prompt=prompt, input_images=[input_image], height=1024, idth=1024, guidance_scale=2.5, 
                           img_guidance_scale=1.6, seed=0)[0]
        return images
    
    def generate_s2p(self, prompt, item, input_image):
        prompt = f"The {item} is in <img><|image_1|></img>. {prompt}"
        images = self.pipe(prompt=prompt, input_images=[input_image], height=1024, idth=1024, guidance_scale=2.5, 
                           img_guidance_scale=1.6, seed=0)[0]
        return images


class OneDiffusionModel(BaseModel):
    """
    Arxiv 2024
    One Diffusion to Generate Them All
    https://github.com/lehduong/OneDiffusion
    """
    def __init__(self):
        self.model_name = "OneDiffusion"
        self.pipe = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)
        self.OD_NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"

    def generate(self, prompt, task='t2i', input_image=None, item=None):
        if task == 't2i':
            image = self.generate_t2i(prompt)
        elif task == 'p2p':
            image = self.generate_p2p(prompt, input_image)
        elif task == 's2p':
            image = self.generate_s2p(prompt, item, input_image)
        else:
            raise ValueError("Invalid task")
        return image
    
    def generate_t2i(self, prompt):
        image = self.pipe(prompt=f"[[text2image]] {prompt}", negative_prompt=self.OD_NEGATIVE_PROMPT, num_inference_steps=50,
                     guidance_scale=4, height=1024, width=1024, ).images[0]
        return image
    
    def generate_p2p(self, prompt, input_image):
        input_image = Image.open(input_image)
        image = self.pipe.img2img(image=input_image, prompt=f"[[image_editing]] {prompt}", negative_prompt=self.OD_NEGATIVE_PROMPT, 
                                  num_inference_steps=60, denoise_mask=[1, 0], guidance_scale=4, height=1024, width=1024).images[0]
        return image
    
    def generate_s2p(self, prompt, item, input_image):
        input_image = Image.open(input_image)
        image = self.pipe.img2img(image=input_image, prompt=f"[[subject_driven]] <item: {item}> [[img0]] {prompt}", negative_prompt=self.OD_NEGATIVE_PROMPT, 
                                  num_inference_steps=60, denoise_mask=[1, 0], guidance_scale=4, height=1024, width=1024).images[0]
        return image
 
 