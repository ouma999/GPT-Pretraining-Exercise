# -*- coding: utf-8 -*-
"""
trainer_ex.py
  Use Save As... to place your first name in front of the file name,
  followed by the underscore (_) character to separate it from the existing. 
  file name. For example, if your first name happened to be 'Alice' 
  would be 'Alice_trainer_ex.py'
"""
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

fpath = "path_to_your_files"

# import the packages needed to perform the tasks below
# set the working directory
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

# After training, plot the results
from methods_from_b4 import plot_losses

# train It is the function that will open the file named in the 
#  actual arguments (the formal arguments appear just below this
#  comment). The start context is the phrase that the LLM will extend
#  as part of the training process. The settings and gpt configuration
#  are as before, but can be modified. train It assumes that one of
#  gpt2-small-124M.pth, gpt2-medium-355M.pth, gpt2-large-774M.pth, or
#  gpt2-xl-1558M.pth has already been set up
# ===============================================================
def trainIt(model, gpt_config, optimizer, settings,
            file_path = "the-verdict.txt",
            start_context = "Every effort moves you",
            loadIt = False):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if there is a prior training run, read that in
    if loadIt:
        checkpoint = torch.load("model_and_optimizer.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Set up dataloaders
    train_ratio = 0.80 # Train/validation ratio 80%/20%
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size = settings["batch_size"],
        max_length = gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last = True,
        shuffle = True,
        num_workers = 0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size = settings["batch_size"],
        max_length = gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last = False,
        shuffle = False,
        num_workers = 0
    )

    # Train model
    tokenizer = tiktoken.get_encoding("gpt2")

    # train_model_simple2 is needed to avoid sending a positional 
    #  argument after a keyword argument 
    train_losses, val_losses, tokens_seen = train_model_simple2(
        model, train_loader, val_loader, optimizer, device, start_context,
        num_epochs = settings["num_epochs"], eval_freq=5, eval_iter=1,
        tokenizer = tokenizer
    )
    torch.save({"model_state_dict": gpt_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),},
               "model_and_optimizer.pth")

    return train_losses, val_losses, tokens_seen, model

# ------------------------------------------------------------------------
# Conduct training for the verdict
GPT_CONFIG_124M = {
	"vocab_size": 50257,    # Vocabulary size
	"context_length": 1024, # context length (orig: 1024)
	"emb_dim": 768,         # Embedding dimension
	"n_heads": 12,          # Number of attention heads
	"n_layers": 12,         # Number of layers
	"drop_rate": 0.1,       # Dropout rate
	"qkv_bias": True       # Query-key-value bias
}
OTHER_SETTINGS = {
	"learning_rate": 5e-4,
	"num_epochs": 10,
	"batch_size": 2,
	"weight_decay": 0.1
}

# Train without using the Open AI weights 
gpt_model = GPTModel.GPTModel(GPT_CONFIG_124M) # Initialize model with hypers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)  

torch.manual_seed(123)

optimizer = torch.optim.AdamW(
    gpt_model.parameters(), 
    lr = OTHER_SETTINGS["learning_rate"], 
    weight_decay = OTHER_SETTINGS["weight_decay"]
)


# default is "the-verdict.txt"
train_losses, val_losses, tokens_seen, model = trainIt(gpt_model,
                                                       GPT_CONFIG_124M,
                                                       optimizer,
                                                       OTHER_SETTINGS)

# Plot results
epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

train_losses, val_losses, tokens_seen, model = trainIt(gpt_model,
                                                       GPT_CONFIG_124M,
                                                       optimizer,
                                                       OTHER_SETTINGS,
                                                       loadIt = True)
# Plot results
epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# try generating a sequence with the trained model
tokenizer = tiktoken.get_encoding("gpt2")
model.eval()
token_ids = generate(model=model,
                     idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
                     max_new_tokens=25, 
                     context_size=1024,
                     top_k=50, temperature=1.5
                     ) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# --------------------------------------------------------------------
# Train with the Open AI weights 
#  Create a new instance of the model in which to load the values
#   so that we have a clean slate to start with
gpt_model = GPTModel.GPTModel(GPT_CONFIG_124M) # Initialize model with hypers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)  

# find your file in whatever directory you have placed it
file_name = "..\LLM_Dictionaries\gpt2-small-124M.pth"
gpt_model.load_state_dict(torch.load(file_name, weights_only=True))
gpt_model.eval()

# using only the pretrained numbers, try text generation,
#  that is, call the generate function (see the example code above)
#  (place your new code just beneath this comment block)


# next, try training using the verdict and add again it to the gpt-2 model 
train_losses, val_losses, tokens_seen, model = trainIt(gpt_model,
                                                       GPT_CONFIG_124M,
                                                       optimizer,
                                                       OTHER_SETTINGS)
# Plot results
epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# ------------- generate text ----------------------------------------
# note that we are using model here and not gpt model
# set the max new token count to 25
# set the top k to 10
# set the temperature to 1.5
model.eval()
# place your new code calling generate() just beneath this line
#  be sure to output the print the output text

# raise the temperature to 5 then call generate with the new temp setting
#  be sure to output the print the output text

# lower the temperature to 0.1 then call generate
#  be sure to output the print the output text

# set the max new token count to 50 and set the temperature to 1.5
#  be sure to output the print the output text

# set the max new token count to 30 and set the top k to 20
#  be sure to output the print the output text



# ===== repeat the above actions for the medium model ==============
# ===============================================================
# Try the medium sized model
#  begin with the original settings from the 124M model

# Copy the base configuration and update with specific model settings,
#  "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
model_name = "gpt2-medium (355M)"  # medium model name
GPT_CONFIG_355M = GPT_CONFIG_124M.copy()
GPT_CONFIG_355M.update(model_configs[model_name])

OTHER_355M = OTHER_SETTINGS.copy()
# OTHER_355M.update({"num_epochs":20}) # would 20 work better?

gpt_model = GPTModel.GPTModel(GPT_CONFIG_355M) # Initialize medium with hypers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)  

optimizer = torch.optim.AdamW(
    gpt_model.parameters(), 
    lr = OTHER_SETTINGS["learning_rate"], 
    weight_decay = OTHER_SETTINGS["weight_decay"]
)

file_name = "..\LLM_Dictionaries\gpt2-medium-355M.pth"
gpt_model.load_state_dict(torch.load(file_name, weights_only=True))

model.eval()
# place your new code calling generate() just beneath this line
#  be sure to output the print the output text

# raise the temperature to 5 then call generate with the new temp setting
#  be sure to output the print the output text

# lower the temperature to 0.1 then call generate
#  be sure to output the print the output text

# set the max new token count to 50 and set the temperature to 1.5
#  be sure to output the print the output text

# set the max new token count to 30 and set the top k to 20
#  be sure to output the print the output text


