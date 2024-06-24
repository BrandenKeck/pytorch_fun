import torch, math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from multihead_attention import MultiHeadAttention

# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerCustom(nn.Module):

    def __init__(self,
                 d_model=16,
                 nhead=2,
                 max_len=512,
                 dim_feedforward=128,
                 dropout=0,
                 layer_norm_eps=1E-5,
                 bias=True,
                 activation=F.relu,
                 device="cpu"):
        super(TransformerCustom, self).__init__()
        self.max_len = max_len
        self.device=device
        self.activation=activation
        self.self_attn = MultiHeadAttention(d_model, nhead, bias, activation)

        # Input Embedding Component
        self.input_embedding  = nn.Linear(1, d_model)

        # Positional Encoding Component
        self.position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        # Normalization and Dropout Components
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask.to(self.device)
        x = self.input_embedding(x)
        x = self.position_encoder(x)
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, mask):
        x = self.self_attn(x, x, x, mask=mask)#[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    # Masking Function
    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.max_len, self.max_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )
