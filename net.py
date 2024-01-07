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


class ConvNet(nn.Module):
    def __init__(self,output_shape):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,8,(3,3),1,padding="same")
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*224*224, output_shape)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
