import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    "outputs_cat_dog", weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion("outputs_cat_dog", weight_name="<new1>.bin")
pipe.load_textual_inversion("outputs_cat_dog", weight_name="<new2>.bin")

image = pipe(
    "a <new1> cat and a <new2> dog is sitting under a tree",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat_dog_under_a_tree.png")

