# -*- coding: utf-8 -*-
"""
Step04_llm_b.py
Created on Fri Mar 13 16:06:04 2026

"""
import torch
import tiktoken

from methods_from_b4 import generate_text_simple
from methods_from_b4 import text_to_token_ids
from methods_from_b4 import token_ids_to_text

import GPTModel
from methods_from_b4 import GPT_CONFIG_124M

model = GPTModel.GPTModel(GPT_CONFIG_124M)
model.eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)


# use the CPU setting here because inference is cheap with 
# the CPU model
inference_device = torch.device("cpu")

model.to(inference_device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Even if we execute the generate_text_simple function above multiple
#   times, the LLM will always generate the same outputs

# Previously, we always sampled the token with the highest probability
#   as the next token using torch.argmax
# To add variety, we can sample the next token using the
#   torch.multinomial(probs, num_samples=1),
#     sampling from a probability distribution
# Here, each index's chance of being picked corresponds to
#   its probability in the input tensor
# Let us recap generating the next token, assuming a very small
#   vocabulary for illustration purposes:

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

inverse_vocab = {v: k for k, v in vocab.items()}

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()

# The next generated token is then as follows:
print(inverse_vocab[next_token_id]) # forward

# Instead of determining the most likely token via torch.argmax,
#   we use torch.multinomial(probas, num_samples=1) to determine
#   the most likely token by sampling from the softmax distribution
# For illustration purposes, let's see what happens when we sample
#  the next token 1,000 times using the original softmax probabilities:
def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# We can control the distribution and selection process via 
#   temperature scaling
# "Temperature scaling" divides the logits by a number greater than 0
#    Temperatures greater than 1 will result in more uniformly
#      distributed token probabilities after applying the softmax
#    Temperatures smaller than 1 will result in more confident
#      (sharper or more peaked) distributions after applying the softmax
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, & lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]

# Plotting
import matplotlib.pyplot as plt

x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.savefig("temperature-plot.pdf")
plt.show()

# the rescaling via temperature 0.1 results in a sharper distribution,
#   approaching torch.argmax, such that the most likely word
#   is almost always selected:
print_sampled_tokens(scaled_probas[1])

# The rescaled probabilities via temperature 5 are
#   more uniformly distributed:
print_sampled_tokens(scaled_probas[2])
# Assuming an LLM input "every effort moves you", using the approach
#   above can sometimes result in nonsensical texts, such as
#  "every effort moves you pizza", 3.2% of the time (32 out of 1000 times)

# Top-k sampling
#  To be able to use higher temperatures to increase output variety
#  and to reduce the probability of nonsensical sentences,
#  we can restrict the sampled tokens to the top-k most likely tokens:
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

# Modifying the text generation function with 
#   temperature sampling and top-k sampling

from methods_from_b4 import generate

torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Loading and saving model weights in PyTorch
#  Training LLMs is computationally expensive, so it is important
#  to be able to save and load LLM weights
#
# The recommended way in PyTorch is to save the model weights,
#   using the state_dict structure,
#   by applying the torch.save function to the .state_dict() method:
torch.save(model.state_dict(), "model.pth")

# Then we can load the model weights into a new GPTModel model
#  instance as follows:
model = GPTModel.GPTModel(GPT_CONFIG_124M)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()



# It's common to train LLMs with adaptive optimizers like Adam or AdamW
#   instead of regular SGD
# These adaptive optimizers store additional parameters
#   for each model weight, so it makes sense to save them as
#   well in case we plan to continue the pretraining later:
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)
