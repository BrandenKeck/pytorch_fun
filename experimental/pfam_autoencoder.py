import torch
import torch.nn as nn

class PFamEncode(nn.Module):
    def __init__(self, embed_size=1024):
        super(PFamEncode, self).__init__()
        self.activation = nn.LeakyReLU()
        self.lin1 = nn.Linear(1, int(embed_size/32))
        self.lin2 = nn.Linear(int(embed_size/32), int(embed_size/16))
        self.lin3 = nn.Linear(int(embed_size/16), int(embed_size/8))
        self.lin4 = nn.Linear(int(embed_size/8), int(embed_size/4))
        self.lin5 = nn.Linear(int(embed_size/4), embed_size)
    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.activation(self.lin4(x))
        x = self.activation(self.lin5(x))
        return x


class PFamDecode(nn.Module):
    def __init__(self, embed_size=1024):
        super(PFamDecode, self).__init__()
        assert embed_size%32==0, "Embedding size must be a multiple of 32"
        self.activation = nn.LeakyReLU()
        self.lin1 = nn.Linear(embed_size, int(embed_size/4))
        self.lin2 = nn.Linear(int(embed_size/4), int(embed_size/8))
        self.lin3 = nn.Linear(int(embed_size/8), int(embed_size/16))
        self.lin4 = nn.Linear(int(embed_size/16), int(embed_size/32))
        self.lin5 = nn.Linear(int(embed_size/32), 1)
    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.activation(self.lin4(x))
        x = self.activation(self.lin5(x))
        return x
    
class PFamAutoencoder(nn.Module):
    def __init__(self, num_classes=1000, embed_size=1024):
        super(PFamAutoencoder, self).__init__()
        self.encoder = PFamEncode(embed_size)
        self.decoder = PFamDecode(embed_size)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten()
    