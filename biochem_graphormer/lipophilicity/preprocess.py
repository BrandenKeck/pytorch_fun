# Get More Data:
#
# https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
# 
# wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
# md5sum pcqm4m-v2-train.sdf.tar.gz # fd72bce606e7ddf36c2a832badeec6ab
# tar -xf pcqm4m-v2-train.sdf.tar.gz # extracted pcqm4m-v2-train.sdf

# imports
import torch, pickle
import numpy as np
import pandas as pd
from torch_geometric.utils.smiles import from_smiles


# custom imports
from modules.algos import gen_edge_input, floyd_warshall


# padding and other preprocessing
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


# converts a torch_geometric datastructure to a graphformer format
def preprocess_item(item):

    # extract data
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    # shortest path information
    shortest_path_result, path = floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # +1 with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.degree = adj.long().sum(dim=1).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


# create a data structure
data = pd.read_csv("/mnt/data/lipophilicity.csv")
Y = list(data["exp"])
X = list(data["smiles"])
X = [from_smiles(x) for x in X]
X = [preprocess_item(x) for x in X]
for i, x in enumerate(X): 
    x.idx = i
    x.y = torch.tensor([Y[i]])

# Dump Data
with open("/mnt/data/preprocessed.pkl", "wb") as f:
    pickle.dump(X, f)
