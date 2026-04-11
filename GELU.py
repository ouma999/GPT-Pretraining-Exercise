# -*- coding: utf-8 -*-
"""
GELU.py
Created on Thu Feb 19 15:41:40 2026

 In deep learning, ReLU (Rectified Linear Unit) activation functions
  are commonly used due to their simplicity and effectiveness in various
   neural network architectures
 In LLMs, various other types of activation functions are used beyond
  the traditional ReLU; two notable examples are
   GELU (Gaussian Error Linear Unit) and SwiGLU (Swish-Gated Linear Unit)
 GELU and SwiGLU are more complex, smooth activation functions
  incorporating Gaussian and sigmoid-gated linear units, respectively,
   offering better performance for deep learning models,
   unlike the simpler, piecewise linear function of ReLU
 GELU can be implemented in several ways; the exact version is defined as
  GELU(x)=x⋅Φ(x), where Φ(x) is the cumulative distribution function of
  the standard Gaussian distribution.
 In practice, it's common to implement a computationally cheaper approximation: 
  GELU(x) = 0.5(x)(1 + tanh[sqrt(2/pi)(x + 0.44715(x^3))])
  (the original GPT-2 model was also trained with this approximation)
"""
import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

