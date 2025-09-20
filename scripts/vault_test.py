import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load vault
with open("vault.json") as f:
    vault = json.load(f)

# Encode texts
for entry in vault:
    entry["embedding"] = model.encode(entry["text"]).tolist()

# Query
query = "How do I remember things across sessions?"
query_vec = model.encode(query)

# Similarity search
scores = []
for entry in vault:
    score = util.cos_sim(query_vec, np.array(entry["embedding"]))[0][0].item()
    scores.append((score, entry["text"]))

# Sort & print best match
scores.sort(reverse=True)
print("Query:", query)
print("Best match:", scores[0][1])

