# -*- coding: utf-8 -*-
"""
Step04_llm_c.py
Created on Sat Mar 14 15:51:14 2026

"""
import torch
import GPTModel
import os
import requests


BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# begin with the small model
file_name = "gpt2-small-124M.pth"
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

if not os.path.exists(file_name): # usually takes less than a minute
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        f.write(response.content) # 124M is 622 MB (652,874,815 bytes)
    print(f"Downloaded to {file_name}")

# Load weights

gpt = GPTModel.GPTModel(BASE_CONFIG)
gpt.load_state_dict(torch.load(file_name, weights_only=True))
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device);

# Generate text
import tiktoken
from methods_from_b4 import generate, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate(
    model=gpt.to(device),
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=BASE_CONFIG["context_length"],
    top_k=1,
    temperature=1.0
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# -------------- medium sized model ----------------------
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
file_name = "gpt2-medium-355M.pth"

# insert code here

# -------------- large sized model ----------------------
CHOOSE_MODEL = "gpt2-large (774M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
file_name = "gpt2-large-774M.pth"

# insert code here


# -------------- Extra large sized model ----------------
CHOOSE_MODEL = "gpt2-xl (1558M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
file_name = "gpt2-xl-1558M.pth"

# insert code here



