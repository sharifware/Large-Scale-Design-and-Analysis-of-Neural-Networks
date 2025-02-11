from nnanalyzer import WeightBinning, NetworkAnalyzer
from standardRegArchitercture import SimpleNet
import torch
from torch.utils.data import DataLoader
import torch
from largerNetArchitecture import LargerNet
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from largerNetArchitecture import LargerNet
from standardRegArchitercture import SimpleNet
from torch.nn import MSELoss

# Read data
csv_data = pd.read_csv('data/simpleReg.csv')

a = csv_data['a'].values.reshape(-1, 1)
b = csv_data['b'].values.reshape(-1, 1)
y = csv_data['y'].values.reshape(-1, 1)

# Convert to Tensors
a_tensor = torch.tensor(a, dtype=torch.float32)
b_tensor = torch.tensor(b, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Combine input features
X_tensor = torch.cat((a_tensor, b_tensor), dim=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Define batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

loss_fn = MSELoss()
networks = NetworkAnalyzer(model_architecture=SimpleNet, amount_to_produce=2000, success_loss=0.1, convergence_threshold=1, max_attempts=10)
networks.generate_networks(train_loader=train_loader, test_loader=test_loader, num_epochs=200, loss_fn=loss_fn, learning_rate=0.05)


analyzer = WeightBinning(architecture=SimpleNet)
#networks = NetworkAnalyzer()
save_dir = "directory here"



weights, layer_bin_ranges = analyzer.store_weights()
normalized_distributions = analyzer.normalize_distributions()
fit_params = analyzer.fit_gaussians_to_weight_distributions(normalized_distributions, layer_bin_ranges)
kl_indices, ce_indices = analyzer.kl_ce_indices()
#print(weights)

analyzer.plot_weight_bins(weights, layer=0, weight_position=(0, 0), bin_edges=layer_bin_ranges[0])
#analyzer.return_weight_bins_plot(weight_distributions=weights, layer=0, weight_position=(0,0), bin_edges=layer_bin_ranges[0])
#analyzer.plot_index(weights, layer_bin_ranges=layer_bin_ranges)

#analyzer.plot_weight_bins_with_fit(weights, layer_bin_ranges[0], fit_params, 0, weight_position=(0,0))

analyzer.plot_unique_distributions(kl_indices, 0, normalized_distributions, layer_bin_ranges, fit_params)
analyzer.plot_unique_distributions(kl_indices, 1, normalized_distributions, layer_bin_ranges, fit_params)
analyzer.plot_unique_distributions(kl_indices, 2, normalized_distributions, layer_bin_ranges, fit_params)