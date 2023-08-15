import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)

prompt = "A sks dog running on the street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./cus_text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_customized_text2video(prompt, fps = fps, path = out_path, model_name="/root/autodl-tmp/Text2Video-Zero/dreambooth/outputs_dog2",  **params)