# Imports
import sys, math, torch
from torch import Tensor, nn
from typing import Optional, Tuple


# Multihead Attention from Graphormer Model
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        # Parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Layers
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.softmax = nn.Softmax()
        self.reset_parameters()

    # From Graphformer Model - Adjusted for equivalent K, Q, V
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x,
        padding_mask,
        attn_bias: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        # Calculate Attention Matrices
        tgt_len, bsz, embed_dim = x.size()
        src_len = tgt_len
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q *= self.scaling

        # Reformat
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # Apply Attention
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float(-1E-16)
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = self.softmax(attn_weights)
        attn_probs = self.dropout(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn
