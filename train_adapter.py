import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--domain", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=2)
args = parser.parse_args()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

dataset = load_dataset("json", data_files=args.data, split="train")
dataset = dataset.map(lambda ex: {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"})

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(output_dir=f"./runs/{args.domain}", num_train_epochs=3, per_device_train_batch_size=args.batch_size,
                           gradient_accumulation_steps=args.grad_accum, learning_rate=2e-4, logging_steps=10, save_strategy="epoch", fp16=True)
trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)
trainer.train()
model.save_pretrained(f"adapters/{args.domain}_adapter")
print(f"Saved adapters/{args.domain}_adapter")