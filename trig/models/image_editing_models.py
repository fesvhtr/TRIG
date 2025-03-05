import torch
from PIL import Image
from diffusers.utils import load_image
from base import BaseModel


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


class FreeDiffModel(BaseModel):
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


class HQEditModel(BaseModel):
    """
    Arxiv 2024
    HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing
    https://github.com/UCSC-VLAA/HQ-Edit
    """
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
        res = 512
        image = load_image(input_image).resize((res, res))

        edited_image = self.pipe(
            prompt=prompt,
            image=image,
            height=res,
            width=res,
            guidance_scale=image_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=30,
        ).images[0]
        return edited_image


class RFInversionModel(BaseModel):
    """
    Arxiv 2024
    Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations
    https://github.com/LituRout/RF-Inversion
    """
    def __init__(self):
        self.model_name = "RFInversion"
        self.model_id = "black-forest-labs/FLUX.1-dev"
        from trig.models.RFInversion.pipeline_flux_rf_inversion import RFInversionFluxPipeline
        self.pipe = RFInversionFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")

    def generate_p2p(self, prompt, input_image):
        image = load_image(input_image)
        inverted_latents, image_latents, latent_image_ids = self.pipe.invert(
            image=image, 
            num_inversion_steps=28, 
            gamma=0.5
        )
        image = self.pipe(
            prompt=prompt,
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            start_timestep=0,
            stop_timestep=7/28,
            num_inference_steps=28,
            eta=0.9,    
        ).images[0]
        
        return image


class RFSolverEditModel(BaseModel):
    """
    Arxiv 2024
    Taming Rectified Flow for Inversion and Editing
    https://github.com/wangjiangshan0725/RF-Solver-Edit
    """
    def __init__(self):
        self.model_name = "RFSolverEdit"
        self.model_id = "black-forest-labs/FLUX.1-dev"
        from RF_Solver_Edit.edit import main as RFEditPipeline
        self.pipe = RFEditPipeline

    def generate_p2p(self, prompt, input_image):
        image = self.pipe(
            source_img_dir=input_image,
            source_prompt='',
            target_prompt=prompt,   
        )
        
        return image


if __name__ == "__main__":
    prompt =  "Add a subtle layer of transparent digital rain falling in the background, ensuring it accentuates the metallic shine and depth of the robotic skull's features without obscuring the headphones or altering the lighting."
    input_image = "/home/yanzhonghao/datasets/MOGAI/dataset/Trig-image-editing/images/task_attr_mod_color_97708.png"
    
    model = RFSolverEditModel()
    image = model.generate_p2p(prompt, input_image)
    image.save("output.png")
