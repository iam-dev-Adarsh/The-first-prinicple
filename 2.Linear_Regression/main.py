# Linear regression - PyTorch implementation

import torch
import torch.nn as nn

xs_data = [1,2,3,4,5,6,7,8,9,10]
ys_data = [2.1,4.0,5.9,8.1,9.8,12.0,14.1,15.9,18.0,20.1]

xs = torch.tensor(xs_data, dtype=torch.float32)
ys = torch.tensor(ys_data, dtype=torch.float32)

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
    def forward(self,x):
        return x * self.w + self.b

model = LinearRegression()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(xs)
    loss = (pred - ys).square().mean()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: w= {model.w.item():.4f}, b= {model.b.item():.4f}")

print(f"Training complete: w= {model.w.item():.4f}, b= {model.b.item():.4f}")        