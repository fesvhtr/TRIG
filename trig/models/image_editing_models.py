import torch
from PIL import Image
from diffusers.utils import load_image
from trig.models.base import BaseModel
from openai import OpenAI
import time
from trig.config import gpt_logit_dimension_msg, DIM_NAME_DICT
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

class HQEditDTMDimModel(BaseModel):
    """
    Stable Diffusion 3.5 with Dimensional Fine-tuning with DTM
    https://github.com/Stability-AI/sd3.5
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

    def generate_p2p(self, prompt, input_image, dimension):
        prompt = self.change(prompt, self.DTM_3d35, dimension)
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

if __name__ == "__main__":
    prompt =  "Add a subtle layer of transparent digital rain falling in the background, ensuring it accentuates the metallic shine and depth of the robotic skull's features without obscuring the headphones or altering the lighting."
    input_image = "/home/yanzhonghao/datasets/MOGAI/dataset/Trig-image-editing/images/task_attr_mod_color_97708.png"
    
    model = RFSolverEditModel()
    image = model.generate_p2p(prompt, input_image)
    image.save("output.png")
