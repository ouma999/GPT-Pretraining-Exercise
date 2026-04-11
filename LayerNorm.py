# -*- coding: utf-8 -*-
"""
LayerNorm.py
Created on Thu Feb 19 15:18:25 2026

 In addition to performing the normalization by subtracting
  the mean and dividing by the variance, we add two trainable parameters,
  a scale and a shift parameter
 The initial scale (multiplying by 1) and shift (adding 0) values have
  no effect; however, scale and shift are trainable parameters that the
  LLM automatically adjusts during training if it determines that doing so
  would improve the model's performance on its training task
 This allows the model to learn appropriate scaling and shifting that
  best suit the data it is processing
 We also add a smaller value (eps) before computing the square root
  of the variance; this is to avoid division-by-zero errors
  if the variance is 0

"""
import torch
import torch.nn as nn

"""
In the variance calculation below, setting unbiased=False means using
 the formula sum((x_i - xBar)/n) instead of sum((x_i - xBar)/(n-1))
 (Bessel's correction); hence the estimate is biased. 
For LLMs, where the embedding dimension n is very large, the difference
 between using n and n-1 is negligible
However, GPT-2 was trained *with* a biased variance in the
 normalization layers, which is why we also adopted this setting
 for compatibility reasons with the pretrained weights that
 we will load in a later step
"""

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
