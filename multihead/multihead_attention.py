# Basic Multihead Attention Implementation
# See: https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
# See: https://github.com/pytorch/pytorch/blob/a1a2023eb86805b1a3867dbda9c89be3cd63dd27/torch/nn/functional.py#L6081


# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multihead Attention Module
class MultiheadAttention(nn.Module):

    def __init__(self,
                 embed_size,
                 nheads,
                 activation=F.relu,
                 device = "cpu"):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.nheads = nheads
        self.activation = activation
        self.head_dim = embed_size // nheads
        self.device = device
        self.linear_q = nn.Linear(embed_size, embed_size, bias=True)
        self.linear_k = nn.Linear(embed_size, embed_size, bias=True)
        self.linear_v = nn.Linear(embed_size, embed_size, bias=True)
        self.linear_o = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, q, k, v):
        
        # Get a standard mask from class function
        mask = self.get_mask(q)
        mask = mask.repeat(self.nheads, 1, 1).to(self.device)
        
        # Create Q, K, V matrices and reshape
        bsz, seq_len, _ = q.shape
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = self.activation(q), self.activation(k), self.activation(v)
        q = q.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)

        # Apply attention
        q_scaled = q * math.sqrt(1.0 / float(self.embed_size))
        attn_output_weights = torch.baddbmm(
            mask, q_scaled, k.transpose(-2, -1)
        )
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seq_len * bsz, self.embed_size)
        )
        attn_output = self.linear_o(attn_output)
        attn_output = self.activation(attn_output)
        attn_output = attn_output.view(bsz, seq_len, self.embed_size)

        # Return Outputs
        return attn_output, attn_output_weights

    # Method to generate a standard mask
    @staticmethod
    def get_mask(x):
        batch_size, seq_len, _ = x.shape
        return torch.tril((1e9)*torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1) - 1e9
