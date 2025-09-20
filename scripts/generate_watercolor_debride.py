from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch, sys, re
from datetime import datetime
from pathlib import Path

# --- Model path (point to your actual file under AI_Models)
model_path = "models/base/realisticVisionV60B1_v51HyperVAE.safetensors"

# --- Handle args: prompt + optional outfile
prompt = sys.argv[1] if len(sys.argv) > 1 else (
    "Korean babe in compromising situation (miniskirt:1.3)"
)

outfile = sys.argv[2] if len(sys.argv) > 2 else None

# --- Default outfile naming
if outfile is None:
    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]+', '_', prompt)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("outputs").mkdir(exist_ok=True)
    outfile = f"outputs/{timestamp}_{safe_prompt}.png"

# --- Load pipeline
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

# Replace scheduler for sharper results
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# --- Generate image
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
image.save(outfile)

print(f"Prompt: {prompt}")
print(f"Saved to {outfile}")
