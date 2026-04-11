# -*- coding: utf-8 -*-
"""
Step04_llm_a.py
Created on Thu Mar 12 15:37:37 2026
"""
import torch


torch.manual_seed(123)

# fpath="C:/Users/tom.tiahrt/Documents/729/W/"
#fpath="C:/Users/tom.tiahrt/OneDrive - The University of South Dakota/Documents/Documents/729/W"
#fpath="C:/Users/Thoma/OneDrive - The University of South Dakota/Documents/Documents/729/W"
fpath="C:/Users/tom.tiahrt/OneDrive - The University of South Dakota/Documents/Documents/729/Step04"

# set the working directory
import os
os.getcwd()
os.chdir(fpath)


import GPTModel
from methods_from_b4 import GPT_CONFIG_124M
model = GPTModel.GPTModel(GPT_CONFIG_124M)
model.eval()


import tiktoken

from methods_from_b4 import generate_text_simple, text_to_token_ids
from methods_from_b4 import token_ids_to_text


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]


targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]


with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# easier to read that scientific notation
torch.set_printoptions(sci_mode=False,  precision=7)

# output the probabilities for batch 1 and 2
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# The goal is to make this average log probability as large as possible
#   by optimizing the model weights
# Due to the log, the largest possible value is 0, and we are
#   currently far away from 0
# In deep learning, instead of maximizing the average log-probability,
#   it's a standard convention to minimize the negative average
#   log-probability value; in our case, instead of maximizing -10.7722
#   so that it approaches 0, in deep learning, we would minimize 10.7722
#   so that it approaches 0
# The value negative of -10.7722, i.e., 10.7722, is also called
#   cross-entropy loss in deep learning

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# Note that the targets are the token IDs, which also represent the
#   index positions in the logits tensors that we want to maximize
# The cross_entropy function in PyTorch will automatically take care
#   of applying the softmax and log-probability computation internally
#   over those token indices in the logits that are to be maximized

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

# Note that the targets are the token IDs, which also represent the
#   index positions in the logits tensors that we want to maximize
# The cross_entropy function in PyTorch will automatically take care
#   of applying the softmax and log-probability computation internally
#   over those token indices in the logits that are to be maximized
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# A concept related to the cross-entropy loss is the perplexity of an LLM
#   The perplexity is the exponential of the cross-entropy loss
# The perplexity is often considered more interpretable because it can
#   be understood as the effective vocabulary size that the model is
#   uncertain about at each step (in the example above, that would be
#   48,725 words or tokens)
# In other words, perplexity provides a measure of how well the
#   probability distribution predicted by the model matches the actual
#   distribution of the words in the dataset
# Similar to the loss, a lower perplexity indicates that the model
#  predictions are closer to the actual distribution
perplexity = torch.exp(loss)
print(perplexity)


file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# first characters
print(text_data[:264])

# last characters
print(text_data[-536:])

total_characters = len(text_data)
print("Characters:", total_characters)

total_tokens = len(tokenizer.encode(text_data))
print("Tokens:", total_tokens, " minus 256 =", total_tokens-256)

print(text_data[(total_tokens-256):])

train_ratio = 0.80
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from methods_from_b4 import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# make sure the settings are within the bounds needed
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

# Another check to see if the loaders are in order
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# Yet another check to ensure that the tokens are in the neighborhood
#  of what we expect
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

from methods_from_b4 import calc_loss_loader
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

from methods_from_b4 import train_model_simple

import time
start_time = time.time()

# 
torch.manual_seed(123)
model = GPTModel.GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)


num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Looking at the results above, we can see that the model starts out
#  generating incomprehensible strings of words, towards the end, 
#  it's able to produce grammatically more or less correct sentences
# However, based on the training and validation set losses, we can see
#  that the model starts overfitting
# If we were to check a few passages it writes towards the end, 
#  we would find that they are contained in the training set verbatim
#  -- it simply memorizes the training data
# Later, we will cover decoding strategies that can mitigate this
#  memorization by a certain degree
# Note that the overfitting here occurs because we have a very,
#  very small training set, and we iterate over it so many times
# The LLM training here primarily serves educational purposes; 
#  we mainly want to see that the model can learn to produce coherent text
# Instead of spending weeks or months on training this model on vast
#  amounts of expensive hardware, we load pretrained weights later


