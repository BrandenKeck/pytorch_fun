# Imports
import torch
import torch.nn as nn

# Randomly initialize embedding weights
def init_params(module):
    module.weight.data.normal_(mean=0.0, std=0.02)

# Compute Node Features
class GraphNodeFeature(nn.Module):
    def __init__(self, num_atoms, num_degrees, embed_dim):
        super(GraphNodeFeature, self).__init__()

        # Define Encoder Layers
        self.atom_encoder = nn.Embedding(num_atoms+1, embed_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degrees+1, embed_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, embed_dim)
        # self.apply(lambda module: init_params(module))

    def forward(self, batched_data):
        x, degree = (
            batched_data["x"],
            batched_data["degree"],
        )
        n_graph = x.size()[0]
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        node_feature = node_feature + self.degree_encoder(degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature


class GraphAttnBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_edges,
        num_spatial,
    ):
        super(GraphAttnBias, self).__init__()

        # Store parameters
        self.num_heads = num_heads

        # Spacial Encoder Layers
        self.edge_encoder = nn.Embedding(num_edges+1, num_heads, padding_idx=0)
        self.spatial_pos_encoder = nn.Embedding(num_spatial+1, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # self.apply(lambda module: init_params(module))

    def forward(self, batched_data):

        # Batch data extraction
        attn_bias, spatial_pos = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # Handle edge encoding
        edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # Return result
        return graph_attn_bias
