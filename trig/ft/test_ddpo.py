import torch
from diffusers import StableDiffusionPipeline

base_model_id = "sd-legacy/stable-diffusion-v1-5"
lora_path = "/home/muzammal/Projects/TRIG/scripts/save/checkpoints/checkpoint_13/pytorch_lora_weights.safetensors"
device = "cuda:3"
prompt = "A photo of a an astronaut riding a horse on the moon, highly detailed"
seed = 42

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe.to(device)

pipe.load_lora_weights(lora_path)
generator_with_lora = torch.Generator(device=device).manual_seed(seed)
image_with_lora = pipe(
    prompt,
    generator=generator_with_lora
).images[0]
image_with_lora.save("generated_image_with_lora.png")

pipe.unload_lora_weights()
generator_without_lora = torch.Generator(device=device).manual_seed(seed)
image_without_lora = pipe(
    prompt,
    generator=generator_without_lora
).images[0]
image_without_lora.save("generated_image_without_lora.png")