from diffusers import StableDiffusionPipeline
import os

def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    model = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base', 
                                              torch_dtype=torch.float16, 
                                              revision="fp16",
                                              use_auth_token=HF_AUTH_TOKEN)
    

if __name__ == "__main__":
    download_model()
