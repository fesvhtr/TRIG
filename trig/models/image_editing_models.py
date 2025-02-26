import torch
from PIL import Image
from diffusers.utils import load_image
from trig.models.base import BaseModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InstructPix2PixModel(BaseModel):
    """
    CVPR 2023
    InstructPix2Pix: Learning to Follow Image Editing Instructions
    https://github.com/timothybrooks/instruct-pix2pix
    """
    def __init__(self):
        self.model_name = "InstructPix2Pix"
        self.model_id = "timbrooks/instruct-pix2pix"
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
        self.pipe.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def generate_p2p(self, prompt, input_image):
        # print(input_image)
        input_image = load_image(input_image)
        image = self.pipe(
            prompt, 
            image=input_image, 
            num_inference_steps=100
        ).images[0]
        return image


class FreeDiff(BaseModel):
    """
    ECCV 2024
    FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models
    https://github.com/thermal-dynamics/freediff
    """
    def __init__(self):
        self.model_name = "FreeDiff"
        self.num_infer_steps = 50
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from trig.models.FreeDiff import invutils, frq_ptputils 


        self.invutils = invutils
        self.frq_ptputils = frq_ptputils

        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                           beta_schedule="scaled_linear", clip_sample=False, 
                           set_alpha_to_one=False, steps_offset=1)
        self.model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=self.scheduler).to(device)
        self.model.scheduler.set_timesteps(self.num_infer_steps)
    
    def generate_p2p(self, prompt, input_image):
        xT, xts = self.get_latents(input_image)

        res_latents = self.frq_ptputils.frq_img_gen(
            self.model, [prompt], g_seed=8888, latent=xT, mod_guidance_frq =True, guidance_scale=7.5, 
            TS=(801, 781, 581), FS=(32, 32, 10, 10), HS=(32, 32, 32, 32, 32, 32),
            filter_shape = 'sq', num_infer_steps = self.num_infer_steps, clear_low = True, record_time = True
        )
        image = self.frq_ptputils.latn2img(self.model.vae, res_latents[1])
        image = Image.fromarray(image[0])
        
        return image

    def get_latents(self, ipath):
        img_np = self.invutils.load_512(ipath)
        img_latn = self.invutils.img2latn(img_np, self.model, device)
        uncond_emb = self.invutils.encode_text("", self.model)
        xT, xts = self.invutils.ddim_inversion_null_fixpt(img_latn, self.model, uncond_emb, save_all=True, FP_STEPS=5, INV_STEPS=self.num_infer_steps)

        return xT, xts



class FlowEdit(BaseModel):
    def __init__(self, model_type='FLUX',):
        self.model_name = "FlowEdit"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        from trig.models.FlowEdit_utils import FlowEditSD3, FlowEditFLUX

        if model_type == 'FLUX':
            from diffusers import FluxPipeline
            # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
            self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        elif model_type == 'SD3':
            from diffusers import StableDiffusion3Pipeline
            self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
    
        self.scheduler = pipe.scheduler
        self.pipe = self.pipe.to(self.device)

    def generate_p2p(self, prompt, input_image):
        src_prompt = prompt
        tar_prompts = prompt
        negative_prompt =  "" # optionally add support for negative prompts (SD3)
   
        # load image
        image = Image.open(input_imageh)
        # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
        image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
        image_src = self.pipe.image_processor.preprocess(image)
        # cast image to half precision
        image_src = image_src.to(device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
        x0_src = (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        # send to cuda
        x0_src = x0_src.to(device)
        
        if self.model_type == 'SD3':
            x0_tar = FlowEditSD3(pipe,scheduler,x0_src,src_prompt,tar_prompt,negative_prompt,T_steps,n_avg,src_guidance_scale,tar_guidance_scale,n_min,n_max,)
            
        elif self.model_type == 'FLUX':
            x0_tar = FlowEditFLUX(pipe,scheduler,x0_src,src_prompt,tar_prompt,negative_prompt, T_steps,n_avg, src_guidance_scale,tar_guidance_scale,n_min,n_max,)
        else:
            raise NotImplementedError(f"Sampler type {model_type} not implemented")


        x0_tar_denorm = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        image_tar = self.pipe.image_processor.postprocess(image_tar)

        return image_tar[0]

class HQEdit(BaseModel):
    def __init__(self):
        self.model_name = "HQEdit"
        self.model_id = "MudeHui/HQ-Edit"
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "MudeHui/HQ-Edit", 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
        self.pipe.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
    
    def generate_p2p(self, prompt, input_image):
        image_guidance_scale = 1.5
        guidance_scale = 7.0
        height, width = Image.open(input_image).size
        image = load_image(input_image).resize((height, width))

        edit_instruction = "Turn sky into a cloudy one"
        edited_image = pipe(
            prompt=edit_instruction,
            image=image,
            height=height,
            width=width,
            guidance_scale=image_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=30,
        ).images[0]
        return edited_image

if __name__ == "__main__":
    prompt = "Transform the woman's earrings into small, glowing orbs that illuminate her face subtly, casting gentle reflections on her skin. Retain the realistic lighting while introducing an ethereal glow that suggests a unique, otherworldly origin."
    input_image = "./task_style_121254.png"
    
    model = FreeDiff()
    image = model.generate_p2p(prompt, input_image)
    image.save("output.png")
    
    model = InstructPix2PixModel()
    image = model.generate_p2p(prompt, input_image)
    image.save("output.png")
