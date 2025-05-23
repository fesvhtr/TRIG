import torch
from diffusers import FluxPipeline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

pipe.load_lora_weights(
    "/home/muzammal/Projects/TRIG/trig/ft/flux_ft/checkpoint-1442",
    weight_name="pytorch_lora_weights.safetensors"
)
pipe.to(device)

prompt = "a 3dicon, two black and pink boxes with the word uber on them"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
image.save("flux_dev_with_lora.png")
print("Image with LoRA saved as flux_dev_with_lora.png")

pipe.unload_lora_weights()
image_without_lora = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
image_without_lora.save("flux_dev_without_lora.png")
print("Image without LoRA saved as flux_dev_without_lora.png")