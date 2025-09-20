# TODO

## Environment
- [x] Ubuntu 22.04
- [x] Python 3.10.12
- [x] venv setup: `python3 -m venv venv`
- [x] Installed: torch 2.5.1+cu121, diffusers, transformers, accelerate, safetensors, huggingface-hub

---

## Image Generation
- [ ] Finish downloading **SDXL base 1.0**  
  - Only need: `model.safetensors`, `vae.safetensors`, `model_index.json`, `tokenizer/*`
- [ ] Generate test: *Italian port town, watercolor LoRA*
- [ ] Store all models in `models/`
- [ ] Store LoRAs in `loras/`
- [ ] Store datasets in `dataset/`
- [ ] Train first LoRA with **kohya-ss**
- [ ] Test combining **style LoRA (watercolor)** + **subject LoRA**

---

## Language Models
- [ ] Install **Ollama** (done, runs in low VRAM mode)
- [ ] Pull **nomic-embed-text** for embeddings
- [ ] Build minimal **JSON vault prototype**
- [ ] Integrate **vault + mistral** for persistent memory
- [ ] Connect Ollama to **Weather API** (test integration)
- [ ] Pull **Mistral-7B raw** from Hugging Face
- [ ] Build into Ollama with custom **Modelfile**
- [ ] Expand vault → **multi-user memory system**

---

## Longer-Term
- [ ] Voice I/O (speech-to-text + TTS)
- [ ] Optional avatar/embodiment
- [ ] Generalize API integrations beyond weather
