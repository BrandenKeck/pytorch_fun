# Imports
import torch
import torch.nn as nn
from typing import Optional

# Module Imports
from .multihead_attention import MultiheadAttention

# Encoder Layer for Graphormer
class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ff_dim: int = 3072,
        nhead: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.nhead = nhead

        # Initialize blocks
        self.relu = nn.ReLU() # Replace with GeLU?
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        # Multihead Attention Layer
        residual = x
        x = self.layernorm(x)
        x = self.self_attn(x, padding_mask, attn_bias)
        x = self.dropout(x)
        x = residual + x

        # Feed Forward Layer
        residual = x
        x = self.layernorm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        return x
