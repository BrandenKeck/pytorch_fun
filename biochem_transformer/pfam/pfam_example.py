# Imports
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pfam_model import PFamModel
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset, DataLoader


# Start Timer
import time
start = time.time()


# Create a Dictionary of Amino Acids
amino_dict = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
    "[SEP]": 3, "[MASK]": 4,
    "A":6, "B":27, "C":23, "D":14, "E":9, "F":19,
    "G":7, "H":22, "I":11, "J":1, "K":12, "L":5,
    "M":21, "N":17, "O":29, "P":16, "Q":18, "R":13,
    "S":10, "T":15, "U":26, "V":8, "W":24, "X":25,
    "Y":20, "Z":28
}


# A naive padding / truncate function
def process_seqs(X, max_len):
    for idx, xx in enumerate(X):
        xx = [2] + xx
        if len(xx) >= max_len:
            xx = xx[:max_len]
            xx[max_len-1] = 3
            X[idx] = xx
        else:
            xx = xx + (max_len-len(xx))*[0]
            xx[max_len-1] = 3
            X[idx] = xx
    return X


# One-Hot Encoding For Labels
def to_onehot(Y, maxlen):
    ohvecs = []
    for y in Y:
        onehot = np.zeros(maxlen)
        onehot[y] = 1
        ohvecs.append(onehot.tolist())
    return ohvecs


# Set Model Parameters
LEARNING_RATE = 1E-6        # Learning Rate of Model
LR_GAMMA = 1.000            # Learning Rate Decay Of Model
MIN_LEN = 0                 # Minimum Sequence Length
MAX_LEN = 512               # Maximum Sequence Length
EMBED_SIZE = 64            # Embedding Size
NHEAD = 8                   # Number of Multi-Attention Heads
DIM_FF = 512                # Feed Forward Layer of Transformer
DROPOUT = 0.1               # Transformer Dropout
EPOCHS = 100                # Number of Epochs
BATCH_SIZE = 1              # Batch Size
NUMCLASSES = 30             # Number of Classes to Attempt (out of 50)
EARLY_STOPPING = 200        # Number of Epochs before Early Stopping is invoked
FROM_CHECKPOINT = False     # Load Model from Checkpoint
DEVICE="cuda:0"             # Device for Primary Model


# Process Train Data
data = pd.read_csv("data/pfam_data.csv")
data = data[data["Y"]<NUMCLASSES]
Y = list(data["Y"])
X = list(data["X"])
X = [list(x) for x in X]
X = [[amino_dict[xx] for xx in x] for x in X]
data = pd.DataFrame({"X": X, "Y": Y})
data["length"] = data.apply(lambda x: len(x["X"]), axis=1)
data = data[data["length"]>MIN_LEN]
data = data[data["length"]<MAX_LEN]
Y = list(data["Y"])
X = list(data["X"])
X = process_seqs(X, MAX_LEN)


# Training Setup / Model Definition
model = PFamModel(MAX_LEN, 
            embed_size=EMBED_SIZE,
            nhead=NHEAD, 
            dim_feedforward=DIM_FF, 
            dropout=DROPOUT,
            num_classes=NUMCLASSES,
            device=DEVICE)
model.to(DEVICE)
if FROM_CHECKPOINT:
    model.load_state_dict(torch.load(f"./models/pfam_model-{NUMCLASSES}.h5"))
ros = RandomOverSampler()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)
X_train, Y_train = ros.fit_resample(X_train, Y_train)
train_dataset = TensorDataset(torch.Tensor(X_train).type(torch.LongTensor).to(DEVICE), 
                        torch.Tensor(to_onehot(Y_train, NUMCLASSES)).to(DEVICE))
test_dataset = TensorDataset(torch.Tensor(X_test).type(torch.LongTensor).to(DEVICE), 
                    torch.Tensor(to_onehot(Y_test, NUMCLASSES)).to(DEVICE))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=1)


# Training Loop
status = []
best_vloss = np.inf
stop_epochs = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xx, yy in train_dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        train_loss = train_loss + loss.item()
        loss.backward()
        optimizer.step()
    model.eval()
    test_loss = 0
    for xx, yy in test_dataloader:
        out = model(xx)
        loss = criterion(out, yy)
        test_loss = test_loss + loss.item()
    current_status = {"Loss": train_loss, "Val. Loss": test_loss}
    status.append(current_status)
    print(f"Epoch {epoch+1}/{EPOCHS}: LR={scheduler.get_lr()} STATUS={current_status}")
    scheduler.step()
    if test_loss < best_vloss:
        stop_epochs = 0
        best_vloss = test_loss
        torch.save(model.state_dict(), f"./models/pfam_model-{NUMCLASSES}.h5")
    else:
        stop_epochs = stop_epochs + 1
        if stop_epochs > EARLY_STOPPING:
            break


# Evaluate the model
model = PFamModel(MAX_LEN, 
            embed_size=EMBED_SIZE,
            nhead=NHEAD, 
            dim_feedforward=DIM_FF, 
            dropout=DROPOUT,
            num_classes=NUMCLASSES,
            device=DEVICE)
model.load_state_dict(torch.load(f"./models/pfam_model-{NUMCLASSES}.h5"))
model.to(DEVICE)
model.eval()
predictions = []
for xx, yy in test_dataloader:
    out = model(xx)
    true_out = out.detach().cpu().numpy()
    true_out = [np.argmax(y) for y in true_out]
    predictions = predictions + true_out


# Output Results
results = pd.DataFrame({"Y": Y_test, "Y_hat": predictions})
results["Correct"] = results.apply(lambda x: 1 if round(x["Y_hat"])==x["Y"] else 0, axis=1)
results.to_csv(f"./results/pfam_results-{NUMCLASSES}class-{EMBED_SIZE}embed-{EPOCHS}epoch.csv", index=False)
pd.DataFrame(status).to_csv(f"./results/pfam_losses-{NUMCLASSES}class-{EMBED_SIZE}embed-{EPOCHS}epoch.csv", index=False)


# End Timer
end = time.time()
print(f"Total Time: {(end - start)/3_600}")
pd.DataFrame({"Total Time": [(end - start)/3_600]}).to_csv(f"./results/pfam_timer-{NUMCLASSES}class-{EMBED_SIZE}embed-{EPOCHS}epoch.csv", index=False)
