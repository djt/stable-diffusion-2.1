import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    repo = 'stabilityai/stable-diffusion-2-1-base'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
    model = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, revision="fp16", scheduler=scheduler, use_auth_token=HF_AUTH_TOKEN) 
    
if __name__ == "__main__":
    download_model()
