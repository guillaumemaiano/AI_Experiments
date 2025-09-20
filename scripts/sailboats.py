from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
from datetime import datetime
from pathlib import Path
import re

# --- Base SDXL model (you must download it once into this folder)
base_model = "models/sdxl_base_1.0"

# --- Your LoRA file
lora_path = "models/style/"
lora_name = "watercolor_sdxl.safetensors"

# --- Prompt
prompt = (
    "A beautiful Italian port town, watercolor painting style, "
    "harbor with sailboats, pastel colors, sunlight, picturesque architecture"
)

negative = "blurry, low quality, distorted, cartoonish, text, watermark"

# --- Output file
safe_prompt = re.sub(r'[^a-zA-Z0-9_-]+', '_', prompt)[:50]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("outputs").mkdir(exist_ok=True)
outfile = f"outputs/{timestamp}_{safe_prompt}.png"

# --- Load base SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16
).to("cuda")

# --- Attach LoRA
pipe.load_lora_weights(lora_path, weight_name=lora_name)
pipe.fuse_lora(lora_scale=0.8)  # Adjust 0.5–1.0 to taste

# --- Scheduler & safety checker
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# --- Generate
image = pipe(
    prompt,
    negative_prompt=negative,
    guidance_scale=7.5,
    num_inference_steps=30,
    height=768,
    width=1024
).images[0]

image.save(outfile)
print(f"Saved to {outfile}")
