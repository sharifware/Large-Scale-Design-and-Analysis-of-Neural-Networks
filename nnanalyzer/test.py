from nnanalyzer import WeightBinning, NetworkAnalyzer
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
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


#alright

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Define batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Declare Loss Function To use
loss_fn = MSELoss()

#Declare a directory to save all files 
save_dir = "./Histograms"

#Create Network ananlyser object and generate networks
networks = NetworkAnalyzer(model_architecture=SimpleNet, amount_to_produce=100, success_loss=0.3, convergence_threshold=(0.1), max_attempts=15)
networks.generate_networks(train_loader=train_loader, test_loader=test_loader, num_epochs=200, loss_fn=loss_fn, learning_rate=0.05)


#Create Weight binning object
analyzer = WeightBinning(architecture=SimpleNet, save_dir = save_dir)

'''

for each fully connected layer in the netwoek, create matrix of shape (N, M, B) where
N = num neurons in layer
M = num neurons in previous layer
B = number of bins, the Generate matrix of shape layer, network, neuron, incoming weight

Returns:
weight distributions: a lst that stores the weight distributions fr each layer
layer_bin_ranges: A list that stores the weights of each fully connected layer for all networks
then a matrix is generated of shape layer, network, neuron and incoming weight. saved as a matrix for layer weight distributions and layer bin_ranges

'''

layer_weight_distributions , layer_bin_ranges = analyzer.store_weights() 

'''
this function plots the weight of a specific weight in a layer in the network using binned weights
'''

analyzer.plot_weight_bins(layer_weight_distributions , layer =0, weight_position=(3,0), bin_edges = layer_bin_ranges[0])

'''
If you would prefer to use normalized distributios and counts for layer weight distributions use following function:
'''

normalized_weight_distributions, normalized_counts = analyzer.normalize_distributions(layer_weight_distributions)


analyzer.save_normalized_distributions()

'''
Can also load npreviously saved normalized distributions
'''
# analyzer.load_normalized_distributions()
'''
The following functionn is used to pass the plot of the weght. does not save the plot
'''

analyzer.return_weight_bins_plot(layer_weight_distributions , layer =0, weight_position=(3,1), bin_edges = layer_bin_ranges[0])

'''
This function plots the index of networks
'''

#analyzer.plot_index(layer_weight_distributions, layer_bin_ranges)

'''
Fits Gaussian curves to all weight histograms and save the fit parameters.
'''
        
gaussian_fit_params = analyzer.fit_gaussians_to_weight_distributions(layer_weight_distributions, layer_bin_ranges)

'''
Plot the histogram of weights and the fitted Gaussian curve. you must first call fit gaussians function first
'''
analyzer.plot_weight_bins_with_fit(layer_weight_distributions, bin_edges=layer_bin_ranges[0], fit_params=gaussian_fit_params, layer=0, weight_position=(0,0))

'''
        Fits a Gaussian Mixture Model (GMM) to each layer's weight histogram
        by replicating bin_centers replicate_factor * normalized_count times (if > 0).

'''
# gmm_models = analyzer.fit_gmm_to_weight_distributions(weight_distributions=layer_weight_distributions, bin_edges_list=layer_bin_ranges, peaks_per_layer=1)

'''
Plot the histogram of weights and the fitted GMM for a specified (layer, neuron_idx, from_weight_idx).
'''

# analyzer.plot_weight_bins_with_gmm(layer_weight_distributions, bin_edges_list=layer_bin_ranges[0], gmm_models=gmm_models, layer=0, weight_position=(4,2))

'''
have fit params
'''

fit_params = analyzer.fit()

'''
The following are for computing variances
'''
kl_threshold = 0.001
ce_threshold = 0.555

kl_indices = analyzer.cluster_gaussians_by_kl_divergence(fit_params=fit_params, threshold=kl_threshold)
print(kl_indices)

ce_indices = analyzer.cluster_gaussians_by_cross_entropy(fit_params=fit_params, threshold=ce_threshold)
print(ce_indices)

# analyzer.compute_kl_divergence_gaussians()
# analyzer.compute_cross_entropy_gaussians()


#analyzer.cluster_gaussians_by_kl_divergence(fit_params, )
#analyzer.cluster_gaussians_by_cross_entropy(fit_params, )

'''
        Plot unique distributions for a specified layer based on cluster indices.

        Description:
        - This function plots unique Gaussian weight distributions for a specified layer.
        - A distribution is considered "unique" if it belongs to a cluster that has not yet been plotted
        (as determined by the cluster indices in `indices`).
        - The function iterates through the neurons and weights of the specified layer, identifies
        unique distributions (based on their cluster index), and plots them using the provided
        `plot_weight_bins_with_fit` function.
'''
 

analyzer.plot_unique_distributions(indices=kl_indices, layer=0, normalized_distributions=normalized_weight_distributions, all_bin_edges=layer_bin_ranges, fit_params=fit_params)

