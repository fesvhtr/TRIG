import torch
from PIL import Image

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline, 
)
from trig.models.base import BaseModel
from trig.models.FreeDiff import invutils, frq_ptputils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FreeDiff(BaseModel):
    """
    ECCV 2024
    FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models
    https://github.com/thermal-dynamics/freediff
    """
    def __init__(self):
        self.model_name = "FreeDiff"
        self.num_infer_steps = 50
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                           beta_schedule="scaled_linear", clip_sample=False, 
                           set_alpha_to_one=False, steps_offset=1)
        self.model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=self.scheduler).to(device)
        self.model.scheduler.set_timesteps(self.num_infer_steps)
    
    def generate(self, prompt, input_image):
        xT, xts = self.get_latents(input_image)

        res_latents = frq_ptputils.frq_img_gen(
            self.model, [prompt], g_seed=8888, latent=xT, mod_guidance_frq =True, guidance_scale=7.5, 
            TS=(801, 781, 581), FS=(32, 32, 10, 10), HS=(32, 32, 32, 32, 32, 32),
            filter_shape = 'sq', num_infer_steps = self.num_infer_steps, clear_low = True, record_time = True
        )
        image = frq_ptputils.latn2img(self.model.vae, res_latents[1])
        image = Image.fromarray(image[0])
        
        return image

    def get_latents(self, ipath):
        img_np = invutils.load_512(ipath)
        img_latn = invutils.img2latn(img_np, self.model, device)
        uncond_emb = invutils.encode_text("", self.model)
        xT, xts = invutils.ddim_inversion_null_fixpt(img_latn, self.model, uncond_emb, save_all=True, FP_STEPS=5, INV_STEPS=self.num_infer_steps)

        return xT, xts


if __name__ == "__main__":
    prompt = "Transform the woman's earrings into small, glowing orbs that illuminate her face subtly, casting gentle reflections on her skin. Retain the realistic lighting while introducing an ethereal glow that suggests a unique, otherworldly origin."
    input_image = "./task_style_121254.png"
    
    model = FreeDiff()
    image = model.generate(prompt, input_image)
    image.save("output.png")
    
