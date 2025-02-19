# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# copied from JanusFlow-1.3B by DeepSeek https://github.com/deepseek-ai/Janus

import torch
import torchvision.transforms.functional as TF

from transformers import AutoModelForCausalLM
from diffusers.models import AutoencoderKL

from trig.models.base import BaseModel
from trig.models.janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor


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
