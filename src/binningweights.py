import importlib.util
import os
import json
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from standardRegArchitercture import SimpleNet

load_dotenv() 

# Path to the directory containing the saved networks
directory = "/Users/dani/Documents/GitHub/Large-Scale-Design-and-Analysis-of-Neural-Networks/Working Networks"

# List all model files in the directory (assuming .pt or .pth extensions)
model_files = [f for f in os.listdir(directory) if f.endswith('.pt') or f.endswith('.pth')]

# Load models into a dictionary
models = {}
for model_file in model_files:
    model_path = os.path.join(directory, model_file)
    model_name = os.path.splitext(model_file)[0]  # Use the file name (without extension) as the key
    models[model_name] = torch.load(model_path)

# Print the loaded models (or access them by name)
for name, model in models.items():
    print(f"Loaded model: {name}")
    print(model)
    
class SimpleNet(nn.Module):
    # 2 because a, b
    input_size = 2
    # num of hidden neurons
    hidden_size = 5
    output_size = 1
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # Right now RELU but we can change later
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
