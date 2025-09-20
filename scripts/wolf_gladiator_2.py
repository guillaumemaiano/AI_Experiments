from dotenv import load_dotenv
load_dotenv()

import os, re, torch
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# --- Settings ---
model_path = "models/base/counterfeitV30_v30.safetensors"

prompt = (
    "masterpiece, best quality, highly detailed, "
    "a majestic anthropomorphic wolf gladiator, tall, muscular, heroic, "
    "wearing ornate fantasy gladiator armor with metallic and leather details, "
    "standing proudly as a protector. At his leg, a small scruffy dirty bear "
    "with an evil face is punching weakly, comical contrast, epic fantasy illustration, "
    "dramatic lighting, digital painting."
)

negative_prompt = (
    "bad anatomy, deformed, extra limbs, blurry, lowres, text, logo, watermark, "
    "out of frame, cropped, ugly, poorly drawn, mutated hands"
)

guidance = 8.0
steps = 30
size = 512

# --- File prep ---
safe_prompt = re.sub(r'[^a-zA-Z0-9_-]+', '_', prompt)[:50]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("outputs").mkdir(exist_ok=True)
outfile = f"outputs/{timestamp}_{safe_prompt}.png"

# --- Load pipeline ---
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
).to("cuda")

# Replace scheduler with Euler A (sharper images)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Safety checker override (don’t break on NSFW flagging)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Optional: clip skip tweak (many models benefit from 2)
if hasattr(pipe.config, "clip_skip"):
    pipe.config.clip_skip = 2

# --- Generate image ---
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance,
    num_inference_steps=steps,
    height=size,
    width=size,
).images[0]

image.save(outfile)

print(f"✅ Image saved to {outfile}")
