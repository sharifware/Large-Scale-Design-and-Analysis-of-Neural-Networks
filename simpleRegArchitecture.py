import torch.nn as nn

class SimpleNet(nn.Module):
    # 2 because a, b
    input_size = 2
    # num of hidden neurons, should change in future but for now 10
    hidden_size = 10
    output_size = 1
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # Right now RELU but we can change later
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

