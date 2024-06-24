#Imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting_model import ForecastingModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error)


# Create a dataset
seq_len = 200
data = list(pd.read_csv("sunspots.csv")["Monthly Mean Total Sunspot Number"])[1000:]
x = np.array(data[:2000])
forcast = np.array(data[2000:])
X = np.array([x[ii:ii+seq_len] for ii in range(0, x.shape[0]-seq_len)]).reshape((-1, seq_len, 1))
Y = np.array([x[ii+seq_len] for ii in range(0, x.shape[0]-seq_len)]).reshape((-1, 1))


# Training Loop
device = "cuda:0"
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 6.6E-6
model = ForecastingModel(seq_len, embed_size=8, nhead=2, 
                         dim_feedforward=1024, dropout=0, 
                         conv1d_emb=False, conv1d_kernel_size=5, device=device).to(device)
model.train()
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)
dataset = TensorDataset(torch.Tensor(X).to(device), torch.Tensor(Y).to(device))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
for epoch in range(EPOCHS):
    for xx, yy in dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")


# Prediction Loop
model.eval()
for ff in range(len(forcast)):
    xx = x[len(x)-seq_len:len(x)]
    yy = model(torch.Tensor(xx).reshape((1, seq_len, 1)).to(device))
    x = np.concatenate((x, yy.detach().cpu().numpy().reshape(1,)))


# Plot Predictions
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6))
plt.plot(range(2000), data[:2000], label="Training")
plt.plot(range(2000, len(data)), forcast, 'g-', label="Actual")
plt.plot(range(2000, len(data)), x[2000:], 'r--', label="Predicted")
plt.legend()
fig.savefig("./img/sunspots_example.png")


# Export Metrics
print(f"MSE: {mean_squared_error(x[2000:], forcast)}")
print(f"MAE: {mean_absolute_error(x[2000:], forcast)}")
