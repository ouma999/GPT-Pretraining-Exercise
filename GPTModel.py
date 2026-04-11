# -*- coding: utf-8 -*-
"""
GPTModel.py
Created on Thu Feb 19 16:23:42 2026
"""
import torch
import torch.nn as nn
import TransformerBlock
import LayerNorm

'''
 Insert the transformer block into the architecture we coded at the
  very beginning of step 3 so that we obtain a usable GPT architecture
 The transformer block is repeated multiple times;
  in the case of the smallest 124M GPT-2 model, we repeat it 12 times
'''

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock.TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


