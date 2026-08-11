"""Sinusoidal time embeddings used by the flow-matching models."""
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the flow-matching time scalar `t`, used by CondFlow."""

    def __init__(self, dim, output_dim=None):
        super().__init__()
        self.register_buffer("freqs", torch.exp(
            -torch.arange(0, dim // 2).float() * torch.log(torch.tensor(10000.0)) / (dim // 2)
        ))
        out_dim = output_dim if output_dim is not None else dim
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, t):
        args = t * self.freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class MoDLTimeEmbedding(nn.Module):
    """Sinusoidal time embedding used by the MoDL/FlowMatchingMaskGenerator model."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freqs", torch.exp(
            -torch.arange(0, dim // 2).float() * torch.log(torch.tensor(10000.0)) / (dim // 2)
        ))
        self.proj = nn.Linear(dim, 64)

    def forward(self, t):
        args = t * self.freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)
