#!/usr/bin/env python
# coding: utf-8

# This file takes in a user-defined class with a certain name. Instantiating the network is simply for testing purposes

# In[1]:


import importlib.util
import os
import json
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
load_dotenv()


# In[2]:


class ArchitectureImporter:
    def __init__(self, config):
        self.config = config
        self.net_class = None

    # Purpose: This method finds where the class of the Neural Network is defined
    # Data: Reads the file path and class name from the config
    def import_architecture(self):
        # Get the file path and class name from the config
        user_file = self.config.get("architecture_file")
        user_class_name = self.config.get("network_class_name")

        # Check if the provided file path is valid
        if not os.path.isfile(user_file):
            raise FileNotFoundError(f"File '{user_file}' does not exist.")

        # Get the file name without extension
        module_name = os.path.splitext(os.path.basename(user_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, user_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, user_class_name):
            raise AttributeError(f"The file '{user_file}' does not have a class named '{user_class_name}'.")

        self.net_class = getattr(module, user_class_name)

    # Purpose: Allows instantiation of the Network
    # Returns: Class inputted by the user
    def get_architecture(self):
        return self.net_class


# In[86]:


# # This whole Block of code is all testing

# # Load the configuration from the JSON file
# config_file_path = os.getenv('CONFIG_FILE_PATH')
# print(config_file_path)
# with open(config_file_path, "r") as f:
#     config = json.load(f)

# importer = ArchitectureImporter(config)
# importer.import_architecture()

# nn_class = importer.get_architecture()

# # Instantiating the network
# nn1 = nn_class()
# all_attributes = dir(nn1)
# print(all_attributes)


# In[3]:



# Read data
#make constant in future
csv_data = pd.read_csv('data/simpleReg.csv')

a = csv_data['a'].values.reshape(-1, 1)
b = csv_data['b'].values.reshape(-1, 1)
y = csv_data['y'].values.reshape(-1, 1)

# Convert to Tensors
a_tensor = torch.tensor(a, dtype=torch.float32)
b_tensor = torch.tensor(b, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

X_tensor = torch.cat((a_tensor, b_tensor), dim=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# In[5]:


# Load the configuration from the JSON file
config_file_path = os.getenv('CONFIG_FILE_PATH')
print(config_file_path)
with open(config_file_path, "r") as f:
    config = json.load(f)

# Get the architecture
importer = ArchitectureImporter(config)
importer.import_architecture()
architecture = importer.get_architecture()

num_networks = config.get("num_networks")

networks = []
for i in range(num_networks):
    networks.append(architecture())


# In[6]:


def train_network(X_train, y_train, X_test, y_test, network, num_epochs):
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr = 0.01)
    for epoch in range(num_epochs):
        # Forward Pass
        outputs = network(X_train)
        loss = criterion(outputs, y_train)

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print the loss every 100 epochs
        if (epoch+1) % 10 == 0:
            # Evaluate on validation set
            network.eval()
            with torch.no_grad():
                test_outputs = network(X_test)
                test_loss = criterion(test_outputs, y_test)
            network.train()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')


# In[90]:


#train
num_epochs = config.get("num_epochs")

networks_state_dict = {}
for i, network in enumerate(networks):
    train_network(X_train, y_train, X_test, y_test, network, num_epochs)
    print(f"finished training network: {i}")
    networks_state_dict[f'network_{i}'] = network.state_dict()

# Save the state dict as a file
output_dir = config.get("output_dir")
torch.save(networks_state_dict, f'{output_dir}/trainedNetworks.pt')

