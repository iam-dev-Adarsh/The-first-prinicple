# Logistic Regression - PyTorch implementation
import torch
import torch.nn as nn

X = torch.tensor([[1,1], [2,2], [3,1], [6,7], [7,8],[8,6]] , dtype = torch.float32)
y = torch.tensor([0,0,0,1,1,1], dtype = torch.float32)

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X).squeeze()
    loss = nn.functional.binary_cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

print("Training complete!")