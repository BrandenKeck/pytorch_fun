# Imports
import torch, math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from modules.graph_encoder import GraphormerGraphEncoder


# A Graphformer Regression model
class GraphRegressionModel(torch.nn.Module):
    def __init__(self, 
                num_atoms: int,
                num_degrees: int,
                num_edges: int,
                num_spatial: int,
                nhead: int = 4,
                embed_dim: int = 128,
                ff_dim: int = 256,
                num_enc_layers: int = 4,
                num_reg_layers: int = 4,
                dropout: float = 0.1):
        super(GraphRegressionModel, self).__init__()

        # Transformer Encoder Layer Component
        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms, 
            num_degrees, 
            num_edges, 
            num_spatial,
            num_enc_layers,
            embed_dim,
            ff_dim,
            nhead,
            dropout,
        )

        # Build Regression Layers
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                nn.Linear(int(embed_dim/2**i), int(embed_dim/2**(i+1)))
                for i in range(num_reg_layers-1)
            ]
        )
        self.layers.extend(
            [nn.Linear(int(embed_dim/2**(num_reg_layers-1)), 1)]
        )

        # Basic Components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Model Forward Pass
    def forward(self, x):
        x = self.graph_encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.relu(x)
                x = self.dropout(x)
        return x.reshape(-1,)