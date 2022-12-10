import os
import torch
from torch import autocast
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    repo = 'stabilityai/stable-diffusion-2-1-base'
    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler", prediction_type="v_prediction")
    model = DiffusionPipeline.from_pretrained(repo, 
                                              torch_dtype=torch.float16, 
                                              revision="fp16",
                                              scheduler=scheduler,
                                              use_auth_token=HF_AUTH_TOKEN).to("cuda")   
    

if __name__ == "__main__":
    download_model()
