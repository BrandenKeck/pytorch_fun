# Imports
import torch, math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer

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

# A forcasting model
class PFamModel(torch.nn.Module):
    def __init__(self, 
                 max_len=200,
                 embed_size = 1024,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 num_classes=0,
                 device = "cuda"):
        super(PFamModel, self).__init__()
        self.device = device
        self.max_len = max_len
        self.embed_size = embed_size
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = self.embed_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )
        self.input_embedding = nn.Embedding(30, embed_size, padding_idx=0)
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                   dropout=dropout,
                                                   max_len=max_len)
        self.conv_out = nn.Conv1d(max_len, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear_softmax = nn.Linear(embed_size, num_classes)
        self.softmax = nn.Softmax()
    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask.to(self.device)
        x = self.input_embedding(x)
        x = self.position_encoder(x)
        x = self.transformer_encoder(x, src_mask=src_mask)
        x = self.conv_out(x)
        x = self.linear(x)
        x = x.reshape(-1, self.embed_size) 
        x = self.linear_softmax(x)
        return self.softmax(x)

    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.max_len, self.max_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )
