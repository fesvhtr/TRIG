import base64
from io import BytesIO
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
from trig.models.base import BaseModel
import time
from openai import OpenAI
from trig.config import gpt_logit_dimension_msg, DIM_NAME_DICT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DALLE3Model(BaseModel):
    """
    DALLE3 from OpenAI
    """
    def __init__(self):
        self.model_name = "dalle3"
        api_key = 'TBD'
        from openai import OpenAI
        # self.pipe = OpenAI(api_key=API_KEY)
        self.pipe = OpenAI(api_key=api_key, base_url="TBD")

    def generate(self, prompt, **kwargs):
        cnt = 0
        while cnt < 1:
            try:
                response = self.pipe.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

                url = response.data[0].url
                print(url)
                return url
            except Exception as e:
                cnt += 1
                print(f"Error generating image with: {e}")
                continue
        return None
    
    def base64_to_image(base64_string):
        try:
            image_data = base64.b64decode(base64_string)
            image_buffer = BytesIO(image_data)
            image = Image.open(image_buffer)
            return image
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


class SDXLModel(BaseModel):
    """
    ICLR 2024
    SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
    https://github.com/Stability-AI/generative-models
    """
    def __init__(self):
        super().__init__()
        self.model_name = "SDXL"
        self.default_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SDXL_MODEL_PATH")
        
        from diffusers import DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path, 
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

class SD15Model(BaseModel):
    """
    ICLR 2024
    SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
    https://github.com/Stability-AI/generative-models
    """
    def __init__(self):
        super().__init__()
        self.model_name = "SD1.5"
        self.default_model_id = "sd-legacy/stable-diffusion-v1-5"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SD15_MODEL_PATH")
        
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
        )
        
        self.pipe.to(device)

    def generate(self, prompt):
        try:
            image = self.pipe(prompt).images[0]
            return image
        except Exception as e:
            print(f"Error generating image with {self.model_name}: {e}")
            return None
        
class SD15DDPOModel(BaseModel):
    """
    ICLR 2024
    SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
    https://github.com/Stability-AI/generative-models
    """
    def __init__(self):
        super().__init__()
        self.model_name = "SD15_DDPO"
        self.default_model_id = "sd-legacy/stable-diffusion-v1-5"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SD15_DDPO_MODEL_PATH")
        
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
        )
        self.pipe.load_lora_weights("/home/muzammal/Projects/TRIG/scripts/save/checkpoints/checkpoint_13/pytorch_lora_weights.safetensors")
        self.pipe.to(device)

    def generate(self, prompt):
        try:
            image = self.pipe(prompt).images[0]
            return image
        except Exception as e:
            print(f"Error generating image with {self.model_name}: {e}")
            return None
            
class PixartSigmaModel(BaseModel):
    """
    ECCV 2024
    PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation
    https://github.com/PixArt-alpha/PixArt-sigma
    """
    def __init__(self):
        super().__init__()
        self.model_name = "PixArt_Sigma"
        self.default_transformer_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        self.default_pipeline_id = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        transformer_path = self.get_model_path(self.default_transformer_id, "PIXART_TRANSFORMER_PATH")
        pipeline_path = self.get_model_path(self.default_pipeline_id, "PIXART_PIPELINE_PATH")
        
        weight_dtype = torch.float16
        from diffusers import Transformer2DModel, PixArtSigmaPipeline
        transformer = Transformer2DModel.from_pretrained(
            transformer_path,
            subfolder='transformer',
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            pipeline_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        self.pipe.to(device)

    def generate(self, prompt):
        image = self.pipe(prompt).images[0]
        return image


class SanaModel(BaseModel):
    """
    ICLR 2025
    Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer
    https://github.com/NVlabs/Sana
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Sana"
        self.default_model_id = "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SANA_MODEL_PATH")
        
        from diffusers import SanaPipeline
        self.pipe = SanaPipeline.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16,
        )
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
        super().__init__()
        self.model_name = "FLUX"
        self.default_model_id = "black-forest-labs/FLUX.1-dev"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "FLUX_MODEL_PATH")
        
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(device)

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

class FLUXFTModel(BaseModel):
    """
    FLUX from Black Forest Labs
    https://github.com/black-forest-labs/flux
    """
    def __init__(self):
        super().__init__()
        self.model_name = "FLUX_FT"
        self.default_model_id = "black-forest-labs/FLUX.1-dev"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "FLUX_FT_MODEL_PATH")
        
        lora_path = "/home/muzammal/Projects/TRIG/trig/ft/flux_ft/pytorch_lora_weights.safetensors"
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe.load_lora_weights(lora_path)

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
        super().__init__()
        self.model_name = "SD3.5"
        self.default_model_id = "stabilityai/stable-diffusion-3.5-large"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SD35_MODEL_PATH")
        
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe = self.pipe.to(device)

        # uncomment if you want to use less GPU memory
        # self.enable_model_cpu_offload()

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, prompt_3=prompt, num_inference_steps=28, guidance_scale=4.5,max_sequence_length=512).images[0]
        return image


class JanusFlowModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "janus-flow"
        self.default_model_id = "deepseek-ai/JanusFlow-1.3B"
        self.default_vae_id = "stabilityai/sdxl-vae"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "JANUS_FLOW_MODEL_PATH")
        vae_path = self.get_model_path(self.default_vae_id, "JANUS_FLOW_VAE_PATH")
        
        from trig.models.janusflow.models import VLChatProcessor
        from diffusers.models import AutoencoderKL
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae = self.vae.to(torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def janus_generate(
        self,
        vl_gpt,
        vl_chat_processor,
        prompt: str,
        cfg_weight: float = 5.0,
        num_inference_steps: int = 30,
        batchsize: int = 1
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        
        tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
        tokens[batchsize:, 1:] = vl_chat_processor.pad_id
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

        # we remove the last <bog> token and replace it with t_emb later
        inputs_embeds = inputs_embeds[:, :-1, :] 
        
        # generate with rectified flow ode
        # step 1: encode with vision_gen_enc
        z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
        
        dt = 1.0 / num_inference_steps
        dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt
        
        # step 2: run ode
        attention_mask = torch.ones((2*batchsize, inputs_embeds.shape[1]+577)).to(vl_gpt.device)
        attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
        attention_mask = attention_mask.int()
        for step in range(num_inference_steps):
            # prepare inputs for the llm
            z_input = torch.cat([z, z], dim=0) # for cfg
            t = step / num_inference_steps * 1000.
            t = torch.tensor([t] * z_input.shape[0]).to(dt)
            z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
            z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
            z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
            z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
            llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

            # input to the llm
            # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
            if step == 0:
                outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                                use_cache=True, 
                                                attention_mask=attention_mask,
                                                past_key_values=None)
                past_key_values = []
                for kv_cache in past_key_values:
                    k, v = kv_cache[0], kv_cache[1]
                    past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
                past_key_values = tuple(past_key_values)
            else:
                outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                                use_cache=True, 
                                                attention_mask=attention_mask,
                                                past_key_values=past_key_values)
            hidden_states = outputs.last_hidden_state
            
            # transform hidden_states back to v
            hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
            hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
            v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
            z = z + dt * v
            
        # step 3: decode with vision_gen_dec and sdxl vae
        decoded_image = self.vae.decode(z / self.vae.config.scaling_factor).sample
        image = decoded_image.to(torch.float32).clip_(-1.0, 1.0) * 0.5 + 0.5
        image = image.squeeze(0)
        image = TF.to_pil_image(image)
        
        return image
    
    def generate(self, prompt):
        conversation = [
            {
                "role": "User",
                "content": prompt,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_gen_tag
        image = self.janus_generate(self.vl_gpt, self.vl_chat_processor, prompt)
        return image


class JanusProModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "janus-pro"
        self.default_model_id = "deepseek-ai/Janus-Pro-7B"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "JANUS_PRO_MODEL_PATH")
        
        from trig.models.janus.models import MultiModalityCausalLM, VLChatProcessor
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def janus_generate(
        self,
        mmgpt,
        vl_chat_processor,
        prompt: str,
        temperature: float = 1,
        parallel_size: int = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)


        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        image = Image.fromarray(visual_img[0])
        return image

    def generate(self, prompt):
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        image = self.janus_generate(self.vl_gpt, self.vl_chat_processor, prompt)
        return image


class SD35DTMModel(BaseModel):
    """
    Stable Diffusion 3.5 with Finu-tuning with DTM
    https://github.com/Stability-AI/sd3.5
    """
    def __init__(self):
        super().__init__()
        self.model_name = "SD3.5_DTM"
        self.default_model_id = "stabilityai/stable-diffusion-3.5-large"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SD35_DTM_MODEL_PATH")
        
        from diffusers import StableDiffusion3Pipeline
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe = self.pipe.to(device)
        self.DTM_3d35 = {
        "synergy": [
            ("Content Alignment", "Originality"),   
            ("Style Alignment", "Aesthetic")
        ],
        "bottleneck": [
            ("Realism", "Toxic"), 
            ("Style Alignment", "Toxic")
        ],
        "tilt": [
            ("Realism", "Style Alignment") 
        ],
        "dispersion": [
            ("Originality", "Bias")
        ]
        }

        # uncomment if you want to use less GPU memory
        # self.enable_model_cpu_offload()

    def generate(self, prompt):
        prompt = self.change(prompt, self.DTM_3d35)
        image = self.pipe(prompt=prompt, prompt_3=prompt, num_inference_steps=28, guidance_scale=4.5,max_sequence_length=512).images[0]
        return image

    def send_request_with_retry(self, msg, max_retries=3, delay=2):
        SYSTEM_MSG = """You are a prompt engineering expert specializing in multi-dimensional image generation optimization. I have discovered some tradeoff relationships in pairwise dimension, and you need to modify the original prompts to make the final result perform best in multiple dimensions.
        **Optimization Rules:**
        - Synergy (X & Y): Enhance both dimensions simultaneously
        - Bottleneck (X-Y): Break limitations between conflicting dimensions
        - Tilt (X↑Y): Prioritize X while maintaining minimum Y
        - Dispersion (X~Y): Balance unstable dimension relationships
         
        **Modify Requirements:**
        1. Firstly, understand the prompt, confirm whether the prompt involves multiple dimensions, and if it is found that two dimensions are involved that match the dimension pairs in the Optimisation Rules, then modify them according to the rule.
        2. If multiple dimensions are not involved, then it is sufficient to optimise the details simply to the extent that they increase the effect.
        3. Your modifications are to the content, you can refine the suppressed dimensions reasonably well, and you can reduce the dominant dimensions somewhat
        4. However, you have to make sure that you don't add completely new content, the main content of your modified prompt can't be changed in any way, you can only fine-tune the details
        
        !!!!!!!!
        You can't add words like ‘balancing dimension A and dimension B’, it's not allowed, you can only go into reasonable detail later, you can't refer to any specific dimension.

        **Output Requirements:**
        1. Keep prompts pithy and concise (<70 tokens, you must follow this)
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


    def change(self, prompt: str, DTM: dict) -> str:
        instructions = []
        for dim1, dim2 in DTM.get("synergy", []):
            instructions.append(
                f"Improve {dim1} & {dim2} jointly"
            )
        
        # Bottleneck关系处理  
        for dim1, dim2 in DTM.get("bottleneck", []):
            instructions.append(
                f"Break {dim1}-{dim2} limitation"
            )
        
        # Tilt关系处理
        for src_dim, target_dim in DTM.get("tilt", []):
            instructions.append(
                f"Prioritize {src_dim}↑ while maintaining {target_dim}"
            )
        
        # Dispersion关系处理
        for dim1, dim2 in DTM.get("dispersion", []):
            instructions.append(
                f"Stabilize {dim1}-{dim2} balance"
            )

        msg = '''
        original input: {str1}
        modify instrctions: {str2}
        Output the final prompt (No more content than the original prompt)'''.format(str1=prompt, str2=" ".join(instructions))
            
        print('Original prompt:', prompt)
        modified_prompt = self.send_request_with_retry(msg)
        print('Modified prompt:', modified_prompt)
        if ':' in modified_prompt:
            modified_prompt = modified_prompt.split(':')[1].strip('"')
        return modified_prompt if modified_prompt else prompt

class FLUXDTMModel(BaseModel):
    """
    FLUX with DTM optimization
    https://github.com/black-forest-labs/flux
    """
    def __init__(self):
        super().__init__()
        self.model_name = "FLUX_DTM"
        self.default_model_id = "black-forest-labs/FLUX.1-dev"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "FLUX_DTM_MODEL_PATH")
        
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(device)
        self.DTM_3d35 = {
        "synergy": [
            ("Content Alignment", "Originality"),   
            ("Style Alignment", "Aesthetic")
        ],
        "bottleneck": [
            ("Realism", "Toxic"), 
            ("Style Alignment", "Toxic")
        ],
        "tilt": [
            ("Realism", "Style Alignment") 
        ],
        "dispersion": [
            ("Originality", "Bias")
        ]
        }


    
    def generate(self, prompt):
        prompt = self.change(prompt, self.DTM_3d35)
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


    def send_request_with_retry(self, msg, max_retries=3, delay=2):
        SYSTEM_MSG = """You are a prompt engineering expert specializing in multi-dimensional image generation optimization. I have discovered some tradeoff relationships in pairwise dimension, and you need to modify the original prompts to make the final result perform best in multiple dimensions.
        **Optimization Rules:**
        - Synergy (X & Y): Enhance both dimensions simultaneously
        - Bottleneck (X-Y): Break limitations between conflicting dimensions
        - Tilt (X↑Y): Prioritize X while maintaining minimum Y
        - Dispersion (X~Y): Balance unstable dimension relationships
         
        **Modify Requirements:**
        1. Firstly, understand the prompt, confirm whether the prompt involves multiple dimensions, and if it is found that two dimensions are involved that match the dimension pairs in the Optimisation Rules, then modify them according to the rule.
        2. If multiple dimensions are not involved, then it is sufficient to optimise the details simply to the extent that they increase the effect.
        3. Your modifications are to the content, you can refine the suppressed dimensions reasonably well, and you can reduce the dominant dimensions somewhat
        4. However, you have to make sure that you don't add completely new content, the main content of your modified prompt can't be changed in any way, you can only fine-tune the details
        
        !!!!!!!!
        You can't add words like ‘balancing dimension A and dimension B’, it's not allowed, you can only go into reasonable detail later, you can't refer to any specific dimension.

        **Output Requirements:**
        1. Keep prompts pithy and concise (<70 tokens, you must follow this)
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


    def change(self, prompt: str, DTM: dict) -> str:
        instructions = []
        for dim1, dim2 in DTM.get("synergy", []):
            instructions.append(
                f"Improve {dim1} & {dim2} jointly"
            )
        
        # Bottleneck关系处理  
        for dim1, dim2 in DTM.get("bottleneck", []):
            instructions.append(
                f"Break {dim1}-{dim2} limitation"
            )
        
        # Tilt关系处理
        for src_dim, target_dim in DTM.get("tilt", []):
            instructions.append(
                f"Prioritize {src_dim}↑ while maintaining {target_dim}"
            )
        
        # Dispersion关系处理
        for dim1, dim2 in DTM.get("dispersion", []):
            instructions.append(
                f"Stabilize {dim1}-{dim2} balance"
            )

        msg = '''
        original input: {str1}
        modify instrctions: {str2}
        Output the final prompt (No more content than the original prompt)'''.format(str1=prompt, str2=" ".join(instructions))
            
        print('Original prompt:', prompt)
        modified_prompt = self.send_request_with_retry(msg)
        print('Modified prompt:', modified_prompt)
        if ':' in modified_prompt:
            modified_prompt = modified_prompt.split(':')[1].strip('"')
        return modified_prompt if modified_prompt else prompt

class SanaDTMModel(BaseModel):
    """
    Sana with DTM optimization
    https://github.com/NVlabs/Sana
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Sana_DTM"
        self.default_model_id = "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SANA_DTM_MODEL_PATH")
        
        from diffusers import SanaPipeline
        self.pipe = SanaPipeline.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16,
        )
        self.pipe.to(device)

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.DTM_3d35 = {
        "synergy": [
            ("Content Alignment", "Originality"),   
            ("Style Alignment", "Aesthetic")
        ],
        "bottleneck": [
            ("Realism", "Toxic"), 
            ("Style Alignment", "Toxic")
        ],
        "tilt": [
            ("Realism", "Style Alignment") 
        ],
        "dispersion": [
            ("Originality", "Bias")
        ]
        }


    
    def generate(self, prompt):
        prompt = self.change(prompt, self.DTM_3d35)
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20,
                     generator=torch.Generator(device="cuda").manual_seed(42))[0]
        image = image[0]
        return image


    def send_request_with_retry(self, msg, max_retries=3, delay=2):
        SYSTEM_MSG = """You are a prompt engineering expert specializing in multi-dimensional image generation optimization. I have discovered some tradeoff relationships in pairwise dimension, and you need to modify the original prompts to make the final result perform best in multiple dimensions.
        **Optimization Rules:**
        - Synergy (X & Y): Enhance both dimensions simultaneously
        - Bottleneck (X-Y): Break limitations between conflicting dimensions
        - Tilt (X↑Y): Prioritize X while maintaining minimum Y
        - Dispersion (X~Y): Balance unstable dimension relationships
         
        **Modify Requirements:**
        1. Firstly, understand the prompt, confirm whether the prompt involves multiple dimensions, and if it is found that two dimensions are involved that match the dimension pairs in the Optimisation Rules, then modify them according to the rule.
        2. If multiple dimensions are not involved, then it is sufficient to optimise the details simply to the extent that they increase the effect.
        3. Your modifications are to the content, you can refine the suppressed dimensions reasonably well, and you can reduce the dominant dimensions somewhat
        4. However, you have to make sure that you don't add completely new content, the main content of your modified prompt can't be changed in any way, you can only fine-tune the details
        
        !!!!!!!!
        You can't add words like ‘balancing dimension A and dimension B’, it's not allowed, you can only go into reasonable detail later, you can't refer to any specific dimension.

        **Output Requirements:**
        1. Keep prompts pithy and concise (<70 tokens, you must follow this)
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


    def change(self, prompt: str, DTM: dict) -> str:
        instructions = []
        for dim1, dim2 in DTM.get("synergy", []):
            instructions.append(
                f"Improve {dim1} & {dim2} jointly"
            )
        
        # Bottleneck关系处理  
        for dim1, dim2 in DTM.get("bottleneck", []):
            instructions.append(
                f"Break {dim1}-{dim2} limitation"
            )
        
        # Tilt关系处理
        for src_dim, target_dim in DTM.get("tilt", []):
            instructions.append(
                f"Prioritize {src_dim}↑ while maintaining {target_dim}"
            )
        
        # Dispersion关系处理
        for dim1, dim2 in DTM.get("dispersion", []):
            instructions.append(
                f"Stabilize {dim1}-{dim2} balance"
            )

        msg = '''
        original input: {str1}
        modify instrctions: {str2}
        Output the final prompt (No more content than the original prompt)'''.format(str1=prompt, str2=" ".join(instructions))
            
        print('Original prompt:', prompt)
        modified_prompt = self.send_request_with_retry(msg)
        print('Modified prompt:', modified_prompt)
        if ':' in modified_prompt:
            modified_prompt = modified_prompt.split(':')[1].strip('"')
        return modified_prompt if modified_prompt else prompt


class SD35DTMDimModel(BaseModel):
    """
    Stable Diffusion 3.5 with Dimensional Fine-tuning with DTM
    https://github.com/Stability-AI/sd3.5
    """
    def __init__(self):
        super().__init__()
        self.model_name = "SD3.5_DTM_DIM"
        self.default_model_id = "stabilityai/stable-diffusion-3.5-large"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SD35_DTM_DIM_MODEL_PATH")
        
        from diffusers import StableDiffusion3Pipeline
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe = self.pipe.to(device)
        
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

    def generate(self, prompt, dimension):
        prompt = self.change(prompt, self.DTM_3d35, dimension)
        image = self.pipe(prompt=prompt, prompt_3=prompt, num_inference_steps=28, guidance_scale=4.5,max_sequence_length=512).images[0]
        return image

    def send_request_with_retry(self, msg, max_retries=3, delay=2):
        SYSTEM_MSG = """You are a prompt engineering expert specializing in multi-dimensional image generation optimization. I have discovered some tradeoff relationships in pairwise dimension.
        You need to modify the original prompt so that the final result of the image generation model performs best in multiple dimensions.

        **Optimization Rules:**

        - Synergy (X & Y): Enhance both dimensions simultaneously
        - Bottleneck (X-Y): Break limitations between conflicting dimensions
        - Tilt (X↑Y): Prioritize X while maintaining minimum Y
        - Dispersion (X~Y): Balance unstable dimension relationships
        - No trade-off: No trade-off relationship between the two dimensions, no need to modify
         
        **Modify Requirements:**
        1. What you need to do is to understand the original prompt and balance and modify the content of the different dimensions according to the Optimisation Rules only, 
        but not to add direct relational indications of the dimensions and not to change the content massively. Note that you are balancing and trade-off.
        2. If no relationship exists, then no change is needed, if the prompt is long, streamline as required
        3. If the relationship exists, it's the detail, not the content, that you need to make changes to; 
        you can't have the generated image have an overall change in content, you have only help the model make the trade-offs by making changes to the detail.
        4. Important!!!!: You can't add direct words like 'balancing dimension A and dimension B', it's not allowed, you can only go into reasonable detail later, you can't refer to any specific dimension.
        
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
        dim1 = DIM_NAME_DICT[dim1]
        dim2 = DIM_NAME_DICT[dim2]

        if key == 'synergy':
            print('synergy')
            instructions.append(
                f"Improve {dim1} & {dim2} jointly"
            )
        elif key == 'bottleneck':
            print('bottleneck')
            instructions.append(
                f"Break {dim1} - {dim2} limitation"
            )
        elif key == 'tilt':
            print('tilt')
            instructions.append(
                f"Prioritize {dim1}↑ while maintaining {dim2}"
            )
        elif key == 'dispersion':
            print('dispersion')
            instructions.append(
                f"Stabilize {dim1} - {dim2} balance"
            )

        
        msg = '''

        Original input: {str1}\n
        Dimension Definition: {str0}\n
        Modify instrctions: {str2}\n
        Output the final prompt:'''.format(str1=prompt, str0=dim_def, str2=" ".join(instructions))
            
        print('Sending request for DTM')
        # print(msg)

        modified_prompt = self.send_request_with_retry(msg)
        if ':' in modified_prompt:
            modified_prompt = modified_prompt.split(':')[1].strip('"')
        return modified_prompt if modified_prompt else prompt

class SanaDTMDimModel(BaseModel):
    """
    Sana with Dimensional Fine-tuning with DTM
    https://github.com/NVlabs/Sana
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Sana_DTM_DIM"
        self.default_model_id = "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"
        
        # 加载配置文件
        self.load_local_config()
        
        # 获取模型路径（配置文件或HuggingFace）
        model_path = self.get_model_path(self.default_model_id, "SANA_DTM_DIM_MODEL_PATH")
        
        from diffusers import SanaPipeline
        self.pipe = SanaPipeline.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16,
        )
        self.pipe.to(device)

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        
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

    def generate(self, prompt, dimension):
        prompt = self.change(prompt, self.DTM_3d35, dimension)
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20,
                generator=torch.Generator(device="cuda").manual_seed(42))[0]
        image = image[0]
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


