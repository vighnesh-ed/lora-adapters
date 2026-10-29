import pickle, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(model, "adapters/programming_adapter", adapter_name="programming")
model.load_adapter("adapters/finance_adapter", adapter_name="finance")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open("centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

def route_query(query, temperature=0.1):
    query_emb = embedder.encode(query)
    sims = {d: np.dot(query_emb, c) / (np.linalg.norm(query_emb) * np.linalg.norm(c)) for d, c in centroids.items()}
    vals = np.array(list(sims.values())) / temperature
    exp = np.exp(vals - np.max(vals))
    weights = exp / exp.sum()
    return dict(zip(sims.keys(), weights))

while True:
    query = input("\nEnter query (or 'quit'): ")
    if query.lower() == "quit":
        break
    weights = route_query(query)
    print("Routing weights:", weights)
    if "combined" in model.peft_config:
        model.delete_adapter("combined")
    model.add_weighted_adapter(adapters=["programming", "finance"], weights=[weights["programming"], weights["finance"]],
                                adapter_name="combined", combination_type="linear")
    model.set_adapter("combined")
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))