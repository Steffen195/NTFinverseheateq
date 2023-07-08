import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self,hparam):
        super().__init__()
        self.hparam = hparam
        self.model = nn.Sequential(
            nn.Linear(3, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )


    def forward(self, x):
        x = self.model(x)
        return x