import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
prompt = "a photo of a cat"
for id in range(195, 200):
    image = pipe(prompt).images[0]  
    image.save(f"./real_reg/my_samples_cat/images/{id}.png")
