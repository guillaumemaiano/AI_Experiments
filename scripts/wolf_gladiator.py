from dotenv import load_dotenv
load_dotenv()
import os
if not os.environ.get("HF_HUB_OFFLINE"):
    raise RuntimeError("HF_HUB_OFFLINE is not set! Refusing to run online.")
from diffusers import StableDiffusionPipeline
import torch, re
from datetime import datetime
from pathlib import Path

# --- Model path (pick one of your installed models)
# Try Counterfeit or RealisticVision for best results
model_path = "models/base/counterfeitV30_v30.safetensors"

# --- Default prompt for wolf gladiator vs scruffy bear
prompt = (
        "A majestic anthropomorphic wolf gladiator, tall, muscular, heroic, "
        "wearing heavy fantasy gladiator armor with metallic and leather details, "
        "standing proudly as a protector. At his leg, a small scruffy dirty bear "
        "with an evil face is punching weakly, comical contrast, epic fantasy illustration, "
        "dramatic lighting, digital painting."
)

# --- Output filename
safe_prompt = re.sub(r'[^a-zA-Z0-9_-]+', '_', prompt)[:50]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("outputs").mkdir(exist_ok=True)
outfile = f"outputs/{timestamp}_{safe_prompt}.png"

# --- Load pipeline
pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16
).to("cuda")

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# --- Generate image
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
image.save(outfile)

print(f"Prompt: {prompt}")
print(f"Saved to {outfile}")