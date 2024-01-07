import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

