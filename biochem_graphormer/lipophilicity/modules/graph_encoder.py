# Imports
import torch
import torch.nn as nn
from typing import Tuple


# Import Modules
from .multihead_attention import MultiheadAttention
from .features import GraphNodeFeature, GraphAttnBias
from .graph_encoder_layer import GraphormerGraphEncoderLayer


# Initialize parameters
def init_graphormer_params(module):

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


# Graphformer Model
class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_degrees: int,
        num_edges: int,
        num_spatial: int,
        num_layers: int = 4,
        embed_dim: int = 128,
        ff_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.graph_node_feature = GraphNodeFeature(
            num_atoms=num_atoms,
            num_degrees=num_degrees,
            embed_dim=embed_dim,
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=nhead,
            num_edges=num_edges,
            num_spatial=num_spatial,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerGraphEncoderLayer(
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.apply(init_graphormer_params)

    def forward(
        self,
        batched_data,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Extract Data
        data_x = batched_data["x"]

        # Generate a Padding Mask
        n_graph = data_x.size()[0]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        # Forware pass encoders
        attn_bias = self.graph_attn_bias(batched_data)
        x = self.graph_node_feature(batched_data)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(
                x,
                padding_mask,
                attn_bias=attn_bias,
            )
        graph_rep = x[0, :, :]
        return graph_rep
