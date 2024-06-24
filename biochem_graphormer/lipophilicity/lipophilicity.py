# Imports
import torch, pickle
import numpy as np
import pandas as pd
from random import shuffle
from modules.collator import collator
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from modules.graph_regression_model import GraphRegressionModel


# Get Batches of Data
def get_batches(X, n=32):
    shuffle(X)
    return (X[i:i+n] for i in range(0, len(X), n))


# Load Batched Data from preprocessed dataset
with open("./data/preprocessed.pkl", "rb") as f:
    X = pickle.load(f)
Xtrain, Xtest, _, _ = train_test_split(X, len(X)*[0], test_size=0.1, random_state=42)


# Define Parameters
EPOCHS = 500
EARLY_STOPPING = 100
BATCH_SIZE = 16
LEARNING_RATE = 1E-5
LR_GAMMA = 0.999
MAX_NODE = 64
MAX_DIST = 16
MAX_SPAC = 16
NHEAD = 8
EMBED_DIM = 512
FF_DIM = 512
NUM_ENC_LAYERS = 4
NUM_REG_LAYERS = 6
DROPOUT = 0.1
DEVICE = "cuda:1"


# Calclulate Max Embedding Numbers
NUM_ATOMS = int(torch.max(torch.tensor([xx for x in X for xx in x.x.tolist()])))+1
NUM_DEGREES = int(torch.max(torch.tensor([xx for x in X for xx in x.degree.tolist()])))+1
NUM_EDGES = int(torch.max(torch.tensor([xxx for x in X for xx in x.attn_edge_type.tolist() for xxx in xx])))+1
NUM_SPACIAL = int(torch.max(torch.tensor([xxx for x in X for xx in x.spatial_pos.tolist() for xxx in xx])))+1


# Create a Model
model = GraphRegressionModel(
    num_atoms = NUM_ATOMS,
    num_degrees = NUM_DEGREES,
    num_edges = NUM_EDGES,
    num_spatial = NUM_SPACIAL,
    nhead = NHEAD,
    embed_dim = EMBED_DIM,
    ff_dim = FF_DIM,
    num_enc_layers = NUM_ENC_LAYERS,
    num_reg_layers = NUM_REG_LAYERS,
    dropout = DROPOUT
)
model.to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)


# Training Loop
status = []
best_vloss = np.inf
stop_epochs = 0
for epoch in range(EPOCHS):

    # Handle Train Component
    model.train()
    train_loss = 0
    batches = get_batches(Xtrain, n=BATCH_SIZE)
    for batch in batches:
        optimizer.zero_grad()
        data = collator(batch, max_node=MAX_NODE, max_dist=MAX_DIST, spatial_pos_max=MAX_SPAC)
        data = {k:v.to(DEVICE) for k, v in data.items()}
        Y_hat = model(data)
        loss = criterion(Y_hat, data["y"].to(DEVICE))
        train_loss = train_loss + loss.item()
        loss.backward()
        optimizer.step()

    # Handle Validation Component
    model.eval()
    test_loss = 0
    batches = get_batches(Xtest, n=BATCH_SIZE)
    for batch in batches:
        data = collator(batch, max_node=MAX_NODE, max_dist=MAX_DIST, spatial_pos_max=MAX_SPAC)
        data = {k:v.to(DEVICE) for k, v in data.items()}
        Y_hat = model(data)
        loss = criterion(Y_hat, data["y"].to(DEVICE))
        test_loss = test_loss + loss.item()

    # Update Status
    lastyhats = Y_hat.detach().cpu().tolist()[:3]
    current_status = {"Loss": train_loss, "Val. Loss": test_loss, "Last Y_Hats": lastyhats}
    status.append(current_status)
    print(f"Epoch {epoch+1}/{EPOCHS}: LR={scheduler.get_last_lr()} STATUS={current_status}")
    scheduler.step()
    if test_loss < best_vloss:
        stop_epochs = 0
        best_vloss = test_loss
        torch.save(model.state_dict(), f"./model/lipophilicity.h5")
    else:
        stop_epochs = stop_epochs + 1
        if stop_epochs > EARLY_STOPPING:
            break


# Evaluate the model
from sklearn.metrics import mean_squared_error, mean_absolute_error
Ys = []
Yhats = []
model = GraphRegressionModel(
    num_atoms = NUM_ATOMS,
    num_degrees = NUM_DEGREES,
    num_edges = NUM_EDGES,
    num_spatial = NUM_SPACIAL,
    nhead = NHEAD,
    embed_dim = EMBED_DIM,
    ff_dim = FF_DIM,
    num_enc_layers = NUM_ENC_LAYERS,
    num_reg_layers = NUM_REG_LAYERS,
    dropout = DROPOUT
)
model.load_state_dict(torch.load(f"./model/lipophilicity.h5"))
model.to(DEVICE)
model.eval()
predictions = []
batches = get_batches(Xtest, n=BATCH_SIZE)
for batch in batches:
    data = collator(batch, max_node=MAX_NODE, max_dist=MAX_DIST, spatial_pos_max=MAX_SPAC)
    Ys = Ys + data["y"].tolist()
    data = {k:v.to(DEVICE) for k, v in data.items()}
    Y_hat = model(data)
    Yhats = Yhats + Y_hat.detach().cpu().tolist()


# Metrics
print(f"Mean Square Error: {mean_squared_error(Ys, Yhats)}")
print(f"Mean Absolute Error: {mean_absolute_error(Ys, Yhats)}")
pd.DataFrame({"Y": Ys, "Yhat": Yhats}).to_csv(f"results/results-{EPOCHS}EPOCH-{EMBED_DIM}EMBED.csv")

