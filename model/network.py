import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self,hparam):
        super().__init__()
        self.hparam = hparam
        self.model = nn.Sequential(
            nn.Linear(3, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )


    def forward(self, x):
        x = self.model(x)
        return x