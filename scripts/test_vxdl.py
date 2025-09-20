from diffusers import StableDiffusionXLPipeline
import torch

# --- Paths ---
base_model = "./models/base/sdxl-base-1.0"
lora_path = "./loras/watercolor"  # folder or .safetensors file

# --- Load pipeline ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# --- Test 1: Base model ---
prompt_base = "A realistic photo of a sailboat in an Italian harbor"
image_base = pipe(prompt_base, num_inference_steps=30).images[0]
image_base.save("outputs/base_test.png")

# --- Test 2: Base + Watercolor LoRA ---
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
prompt_lora = "A watercolor painting of an Italian port town with sailboats"
image_lora = pipe(prompt_lora, num_inference_steps=30).images[0]
image_lora.save("outputs/watercolor_port.png")

print("✅ Done. Images saved in outputs/: base_test.png and watercolor_port.png")

