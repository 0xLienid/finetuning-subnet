import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "silvainrichou/gemma-3b-002",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/gemma-7b-it-tokenizer"
)

model.save_pretrained("local-models/downloads/gemma-3b")
tokenizer.save_pretrained("local-models/downloads/gemma-3b")
