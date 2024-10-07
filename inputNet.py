import torch.nn as nn

class SimpleNet(nn.Module): 
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc1(x)