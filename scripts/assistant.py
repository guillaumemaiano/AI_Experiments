# assistant.py  — Vault (texts) + Vec cache (separate) + Weather + Ollama
import json, os, requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME  = "george"
VAULT_FILE  = "vault.json"        # source of truth (text+id only)
VEC_FILE    = "vault.vec.json"    # derived cache (id -> vector)
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"  # as decided
embedder    = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

def load_vault():
    with open(VAULT_FILE) as f:
        vault = json.load(f)
    return vault

def load_vec_cache():
    if os.path.exists(VEC_FILE):
        with open(VEC_FILE) as f:
            return json.load(f)
    return {"model": EMBED_MODEL, "vectors": {}}  # id(str) -> list[float]

def save_vec_cache(cache):
    with open(VEC_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def ensure_vectors(vault, cache):
    # Recompute if model changed
    if cache.get("model") != EMBED_MODEL:
        cache = {"model": EMBED_MODEL, "vectors": {}}
    changed = False
    for e in vault:
        k = str(e["id"])
        if k not in cache["vectors"]:
            cache["vectors"][k] = embedder.encode(e["text"]).tolist()
            changed = True
    if changed:
        save_vec_cache(cache)
    return cache

def retrieve_vault(query, vault, cache, k=2):
    q = embedder.encode(query)
    scored = []
    for e in vault:
        v = np.array(cache["vectors"][str(e["id"])])
        score = util.cos_sim(
    torch.tensor(q, dtype=torch.float32).unsqueeze(0),
    torch.tensor(v, dtype=torch.float32).unsqueeze(0)
)[0][0].item()
        scored.append((score, e["text"]))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]

def call_weather(city="Paris"):
    r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
    return r.text.strip()

def call_ollama(messages):
    with requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "messages": messages},
        stream=True
    ) as r:
        r.raise_for_status()
        content = []
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                content.append(data["message"]["content"])
        return "".join(content).strip()


def append_to_vault(text):
    vault = load_vault()
    new_id = (max([e["id"] for e in vault]) + 1) if vault else 1
    vault.append({"id": new_id, "text": text})
    with open(VAULT_FILE, "w") as f:
        json.dump(vault, f, indent=2)
    # update vec cache for the new entry (surjection)
    cache = load_vec_cache()
    cache = ensure_vectors(vault, cache)

def main():
    vault = load_vault()
    cache = load_vec_cache()
    cache = ensure_vectors(vault, cache)  # builds vecs without touching vault.json

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        context = []
        if "weather" in query.lower():
            context.append("Weather info: " + call_weather("Paris"))
        # memory
        context += retrieve_vault(query, vault, cache, k=2)

        prompt = "Context:\n" + "\n".join(context) + "\n\nUser: " + query
        answer = call_ollama([{"role": "user", "content": prompt}])
        print("George:", answer)

        # Surjection: write new Q and A to vault.json (texts only), vecs go to cache
        append_to_vault(query)
        append_to_vault(answer)
        # refresh in-memory copies for next turn
        vault = load_vault()
        cache  = ensure_vectors(vault, load_vec_cache())

if __name__ == "__main__":
    main()
