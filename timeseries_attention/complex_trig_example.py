# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from forecasting_model import ForecastingModel
from torch.utils.data import TensorDataset, DataLoader


# Get a noisy, more complex wave
DATA_SIZE = 6000
x = torch.linspace(0, 30, DATA_SIZE, requires_grad = True)
y = torch.sin(x)-torch.cos(x)**2
Y = torch.sum(y)
Y.backward()
x = x.grad + np.random.normal(0, 0.05, DATA_SIZE)
x = x.detach().numpy()


# Create a dataset
seq_len = 1600
X = np.array([x[ii:ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])
Y = np.array([x[ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])


# Training Loop
EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-6
model = ForecastingModel(seq_len).to("cuda")
model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = TensorDataset(torch.Tensor(X).to("cuda"), torch.Tensor(Y).to("cuda"))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
for epoch in range(EPOCHS):
    for xx, yy in dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")


# Prediction Loop
FORCAST = 6000
model.eval()
for ff in range(FORCAST):
    xx = x[len(x)-seq_len:len(x)]
    yy = model(torch.Tensor(xx).reshape(1, xx.shape[0]).to("cuda"))
    x = np.concatenate((x, yy.detach().cpu().numpy().reshape(1,)))


# Plot Predictions
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6))
x_new = torch.linspace(30, 60, FORCAST, requires_grad = True)
y_new = torch.sin(x_new)-torch.cos(x_new)**2
Y_new = torch.sum(y_new)
Y_new.backward()
plt.plot(range(x[:DATA_SIZE].shape[0]), x[:DATA_SIZE], label="Training")
plt.plot(range(x[:DATA_SIZE].shape[0], x.shape[0]), x[DATA_SIZE:DATA_SIZE+FORCAST], 'r--', label="Predicted")
plt.plot(range(x[:DATA_SIZE].shape[0], x.shape[0]), x_new.grad.detach().numpy(), 'g-', label="Actual")
plt.legend()
fig.savefig("./img/complex_trig_example.png")
