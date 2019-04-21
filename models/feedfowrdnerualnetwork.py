import torch
import torch.nn as nn

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(9, 20)
        self._fc2 = nn.Linear(20, 40)
        self._fc3 = nn.Linear(40, 9)

    def forward(self, x):
        x = torch.sigmoid(self._fc1(x))
        x = torch.sigmoid(self._fc2(x))
        x = torch.sigmoid(self._fc3(x))

        return x