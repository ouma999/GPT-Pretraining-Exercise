# -*- coding: utf-8 -*-
"""
PolexOumaOtieno_trainer_ex.py
Student: Polex Ouma Otieno
"""

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

fpath = "/Users/polexouma/Desktop/GPT_Exercise"

import os
os.getcwd()
os.chdir(fpath)

import tiktoken
import torch
import GPTModel
from methods_from_b4 import create_dataloader_v1
from methods_from_b4 import train_model_simple2
from methods_from_b4 import generate
from methods_from_b4 import text_to_token_ids
from methods_from_b4 import token_ids_to_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# plot_losses - defined locally to avoid NumPy conflicts
# ============================================================
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, filename="loss_plot.png"):
    epochs_list = [float(e) for e in epochs_seen]
    tokens_list = [int(t) for t in tokens_seen]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_list, train_losses, label="Training loss")
    ax1.plot(epochs_list, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_list, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

# ============================================================
# trainIt function
# ============================================================
def trainIt(model, gpt_config, optimizer, settings,
            file_path="the-verdict.txt",
            start_context="Every effort moves you",
            loadIt=False):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loadIt:
        checkpoint = torch.load("model_and_optimizer.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.80
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple2(
        model, train_loader, val_loader, optimizer, device, start_context,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        tokenizer=tokenizer
    )

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
               "model_and_optimizer.pth")

    return train_losses, val_losses, tokens_seen, model


# ============================================================
# Base configuration
# ============================================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1
}

# ============================================================
# PART B - Train from scratch
# ============================================================
print("\n" + "="*60)
print("PART B: Training 124M model from scratch")
print("="*60)

gpt_model = GPTModel.GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)
torch.manual_seed(123)

optimizer = torch.optim.AdamW(
    gpt_model.parameters(),
    lr=OTHER_SETTINGS["learning_rate"],
    weight_decay=OTHER_SETTINGS["weight_decay"]
)

print("\nPart B - First training run:")
train_losses, val_losses, tokens_seen, model = trainIt(
    gpt_model, GPT_CONFIG_124M, optimizer, OTHER_SETTINGS
)

epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
            filename="PartB_first_run_loss.png")

print("\nPart B - Second training run (continuing from checkpoint):")
train_losses, val_losses, tokens_seen, model = trainIt(
    gpt_model, GPT_CONFIG_124M, optimizer, OTHER_SETTINGS, loadIt=True
)

epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
            filename="PartB_second_run_loss.png")


# ============================================================
# PART C - Generate text
# ============================================================
print("\n" + "="*60)
print("PART C: Text generation with scratch-trained model")
print("="*60)

tokenizer = tiktoken.get_encoding("gpt2")
model.eval()

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=1024,
    top_k=50,
    temperature=1.5
)
print("Part C Output text:\n", token_ids_to_text(token_ids, tokenizer))


# ============================================================
# PART D - Pretrained 124M weights
# ============================================================
print("\n" + "="*60)
print("PART D: Using pretrained GPT-2 Small (124M) weights")
print("="*60)

gpt_model = GPTModel.GPTModel(GPT_CONFIG_124M)
gpt_model.to(device)

file_name = "gpt2-small-124M.pth"
gpt_model.load_state_dict(torch.load(file_name, weights_only=True))
gpt_model.eval()

print("\nPart D - Generation with pretrained weights ONLY (before fine-tuning):")
token_ids = generate(
    model=gpt_model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=50, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

optimizer = torch.optim.AdamW(
    gpt_model.parameters(),
    lr=OTHER_SETTINGS["learning_rate"],
    weight_decay=OTHER_SETTINGS["weight_decay"]
)

print("\nPart D - Fine-tuning pretrained 124M on the-verdict.txt:")
train_losses, val_losses, tokens_seen, model = trainIt(
    gpt_model, GPT_CONFIG_124M, optimizer, OTHER_SETTINGS
)

epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
            filename="PartD_finetuned_loss.png")

model.eval()

print("\nPart D - Generate: max_new_tokens=25, top_k=10, temperature=1.5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart D - Generate: temperature raised to 5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=5.0
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart D - Generate: temperature lowered to 0.1")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=0.1
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart D - Generate: max_new_tokens=50, temperature=1.5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=50, context_size=1024, top_k=10, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart D - Generate: max_new_tokens=30, top_k=20")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=30, context_size=1024, top_k=20, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# ============================================================
# PART E - Medium (355M) model
# ============================================================
print("\n" + "="*60)
print("PART E: Using pretrained GPT-2 Medium (355M) weights")
print("="*60)

model_name = "gpt2-medium (355M)"
GPT_CONFIG_355M = GPT_CONFIG_124M.copy()
GPT_CONFIG_355M.update(model_configs[model_name])
OTHER_355M = OTHER_SETTINGS.copy()

gpt_model = GPTModel.GPTModel(GPT_CONFIG_355M)
gpt_model.to(device)

optimizer = torch.optim.AdamW(
    gpt_model.parameters(),
    lr=OTHER_355M["learning_rate"],
    weight_decay=OTHER_355M["weight_decay"]
)

file_name = "gpt2-medium-355M.pth"
gpt_model.load_state_dict(torch.load(file_name, weights_only=True))
gpt_model.eval()

print("\nPart E - Generation with pretrained Medium weights ONLY:")
token_ids = generate(
    model=gpt_model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart E - Fine-tuning pretrained 355M on the-verdict.txt:")
train_losses, val_losses, tokens_seen, model = trainIt(
    gpt_model, GPT_CONFIG_355M, optimizer, OTHER_355M
)

epochs_tensor = torch.linspace(0, OTHER_355M["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
            filename="PartE_medium_loss.png")

model.eval()

print("\nPart E - Generate: max_new_tokens=25, top_k=10, temperature=1.5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart E - Generate: temperature raised to 5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=5.0
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart E - Generate: temperature lowered to 0.1")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=1024, top_k=10, temperature=0.1
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart E - Generate: max_new_tokens=50, temperature=1.5")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=50, context_size=1024, top_k=10, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nPart E - Generate: max_new_tokens=30, top_k=20")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=30, context_size=1024, top_k=20, temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\n" + "="*60)
print("All parts complete! Add outputs and plots to your Word document.")
print("Plots saved as PNG files in your GPT_Exercise folder.")
print("="*60)
