import base64
from io import BytesIO
from openai import OpenAI

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

from transformers import AutoModelForCausalLM
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    StableDiffusion3Pipeline,
    Transformer2DModel,
    PixArtSigmaPipeline,
    SanaPipeline,
    FluxPipeline
)
from trig.models.base import BaseModel
from trig.models.janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from trig.models.janus.models import MultiModalityCausalLM, VLChatProcessor
from trig.config import API_KEY

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DALLE3Model(BaseModel):
    """
    DALLE3 from OpenAI
    """
    def __init__(self):
        self.model_name = "dalle3"
        self.pipe = OpenAI(api_key=API_KEY)

    def generate(self, prompt, **kwargs):
        cnt = 0
        while cnt < 3:
            try:
                response = self.pipe.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1,
                                                response_format='b64_json')
                image_b64 = response.data[0].b64_json
                image = self.base64_to_image(image_b64)
                return image
            except Exception:
                cnt += 1
                continue
        return None
    
    @staticmethod
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


class PixartSigmaModel(BaseModel):
    """
    ECCV 2024
    PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation
    https://github.com/PixArt-alpha/PixArt-sigma
    """
    def __init__(self):
        self.model_name = "PixArt_Sigma"
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
   

class SD35Model(BaseModel):
    """
    Stable Diffusion 3.5
    https://github.com/Stability-AI/sd3.5
    """
    def __init__(self):
        self.model_name = "SD3.5"
        self.model_id = "stabilityai/stable-diffusion-3.5-large"
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(device)

        # uncomment if you want to use less GPU memory
        # self.enable_model_cpu_offload()

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, prompt_3=prompt, num_inference_steps=28, guidance_scale=4.5,max_sequence_length=512).images[0]
        return image


class FLUXModel(BaseModel):
    """
    FLUX from Black Forest Labs
    https://github.com/black-forest-labs/flux
    """
    def __init__(self):
        self.model_name = "FLUX"
        self.model_id = "black-forest-labs/FLUX.1-dev"
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


class SanaModel(BaseModel):
    """
    ICLR 2025
    Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer
    https://github.com/NVlabs/Sana
    """
    def __init__(self):
        self.model_name = "Sana"
        self.pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers", variant="fp16", torch_dtype=torch.float16,)
        self.pipe.to(device)

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

    def generate(self, prompt):
        image = self.pipe(prompt=prompt, height=1024, width=1024, guidance_scale=4.5, num_inference_steps=20,
                     generator=torch.Generator(device="cuda").manual_seed(42))[0]
        image = image[0]
        return image


class JanusFlowModel(BaseModel):
    def __init__(self):
        self.model_name = "janus-flow"
        self.model_id = "deepseek-ai/JanusFlow-1.3B"
        self.vae_id = "stabilityai/sdxl-vae"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

        self.vae = AutoencoderKL.from_pretrained(self.vae_id)
        self.vae = self.vae.to(torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def janus_generate(
        self,
        vl_gpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
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
        self.model_name = "janus-pro"
        self.model_id = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def janus_generate(
        self,
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
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

