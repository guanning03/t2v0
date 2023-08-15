# 注意，这份代码只能在 diffusers 0.14.0 才能运行成功
import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)

prompt = "A dog running on the street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)