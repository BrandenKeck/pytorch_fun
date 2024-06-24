import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 10, 1000, requires_grad = True)
y = torch.sin(x)-torch.cos(x)**2
Y = torch.sum(y)
Y.backward()

fig = plt.figure(figsize=(6, 6))
plt.plot(x.detach().numpy(), y.detach().numpy(), label="f(x)")
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="f'(x)")
plt.legend()
fig.savefig("./img/simple_derivative.png")