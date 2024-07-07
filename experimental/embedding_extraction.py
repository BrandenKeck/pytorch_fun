import torch
import torch.nn as nn
from transformers import BertModel

class ProtEmbedding(nn.Module):
    def __init__(self, embed=nn.Embedding(30, 1024, padding_idx=0)):
        super(ProtEmbedding, self).__init__()
        self.embed = embed
    def forward(self, x):
        return self.embed(x)

input_embedding = BertModel.from_pretrained("Rostlab/prot_bert").embeddings.word_embeddings
model = ProtEmbedding(input_embedding)
torch.save(model.state_dict(), "protembed.h5")
