import torch
import tempfile
import os
from PIL import Image

from trig.models.base import BaseModel



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OmniGenModel(BaseModel):
    """
    Arxiv 2024
    OmniGen: Unified Image Generation
    https://github.com/VectorSpaceLab/OmniGen
    """
    def __init__(self):
        super().__init__()
        self.model_name = "OmniGen"
        self.default_model_id = "Shitao/OmniGen-v1"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "OMNIGEN_MODEL_PATH")
        
        from trig.models.OmniGen import OmniGenPipeline
        self.pipe = OmniGenPipeline.from_pretrained(model_path)

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
        # Handle both path and PIL Image object
        temp_file = None
        try:
            if isinstance(input_image, str):
                # input_image is a path
                image_path = input_image
                pil_image = Image.open(input_image)
            else:
                # input_image is a PIL Image object, save to temporary file
                pil_image = input_image
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                input_image.save(temp_file.name, 'PNG')
                image_path = temp_file.name
                temp_file.close()
            
            width, height = pil_image.size
            width = int((width/2) // 16) * 16
            height = int((height/2) // 16) * 16
            prompt=f"Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n {prompt}"
            images = self.pipe(prompt=prompt, input_images=[image_path], height=height, width=width, guidance_scale=2.5, 
                               img_guidance_scale=1.6, seed=0)[0]
            return images
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def generate_s2p(self, prompt, item, input_image):
        # Handle both path and PIL Image object
        temp_file = None
        try:
            if isinstance(input_image, str):
                # input_image is a path
                image_path = input_image
            else:
                # input_image is a PIL Image object, save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                input_image.save(temp_file.name, 'PNG')
                image_path = temp_file.name
                temp_file.close()
            
            prompt = f"The {item} is in <img><|image_1|></img>. {prompt}"
            images = self.pipe(prompt=prompt, input_images=[image_path], height=512, width=512, guidance_scale=2.5, 
                               img_guidance_scale=1.6, seed=0)[0]
            return images
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


class OneDiffusionModel(BaseModel):
    """
    Arxiv 2024
    One Diffusion to Generate Them All
    https://github.com/lehduong/OneDiffusion
    """
    def __init__(self):
        super().__init__()
        self.model_name = "OneDiffusion"
        self.default_model_id = "lehduong/OneDiffusion"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "ONEDIFFUSION_MODEL_PATH")
        
        from trig.models.onediffusion import OneDiffusionPipeline
        self.pipe = OneDiffusionPipeline.from_pretrained(model_path).to(device=device, dtype=torch.bfloat16)
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
        if isinstance(input_image, str):
            input_image = Image.open(input_image)
        width, height = input_image.size
        width = int((width/2) // 16) * 16
        height = int((height/2) // 16) * 16
        image = self.pipe.img2img(image=input_image, prompt=f"[[image_editing]] {prompt}", negative_prompt=self.OD_NEGATIVE_PROMPT, 
                                  num_inference_steps=60, denoise_mask=[1, 0], guidance_scale=4, height=height, width=width).images[0]
        return image
    
    def generate_s2p(self, prompt, item, input_image):
        if isinstance(input_image, str):
            input_image = Image.open(input_image)
        width, height = input_image.size
        width = int((width/2) // 16) * 16
        height = int((height/2) // 16) * 16
        image = self.pipe.img2img(image=input_image, prompt=f"[[subject_driven]] <item: {item}> [[img0]] {prompt}", negative_prompt=self.OD_NEGATIVE_PROMPT, 
                                  num_inference_steps=60, denoise_mask=[1, 0], guidance_scale=4, height=height, width=width).images[0]
        return image
 
 