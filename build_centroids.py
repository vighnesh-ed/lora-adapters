import glob, os, pickle, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
centroids = {}
for path in sorted(glob.glob("data/*.jsonl")):
    domain = os.path.splitext(os.path.basename(path))[0]
    data = load_dataset("json", data_files=path, split="train")
    samples = [ex['instruction'] for ex in data]
    embeddings = embedder.encode(samples)
    centroids[domain] = np.mean(embeddings, axis=0)

with open("centroids.pkl", "wb") as f:
    pickle.dump(centroids, f)
print("Saved centroids.pkl")