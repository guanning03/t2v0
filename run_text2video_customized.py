import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)

prompt = "A sks dog running on the street"
params = {"t0": 45, "t1": 48 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 24, "chunk_size": 12}

out_path, fps = f"./cus_text2video_{prompt.replace(' ','_')}.mp4", 8
model.process_customized_text2video(prompt, fps = fps, path = out_path, model_name="dreambooth/outputs_dog",  **params)