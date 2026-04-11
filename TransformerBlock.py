# -*- coding: utf-8 -*-
"""
TransformerBlock.py
Created on Thu Feb 19 15:59:14 2026

Combine some of the previous concepts into a transformer block

A transformer block combines the causal multi-head attention module
 from Step 2 with the linear layers, the feed forward neural network
 and dropout and shortcut connections

"""
import torch.nn as nn

import MultiHeadAttention
import FeedForward
import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward.FeedForward(cfg)
        self.norm1 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
