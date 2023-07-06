import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 80),
            nn.Tanh(),
            nn.Linear(80, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 80),
            nn.Tanh(),
            nn.Linear(80, 40),
            nn.Tanh(),
            nn.Linear(40, 3)
        )


    def forward(self, x):
        x = self.model(x)
        return x