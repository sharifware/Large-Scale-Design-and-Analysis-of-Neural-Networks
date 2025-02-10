import importlib.util
import os 
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import norm



class WeightBinning():

    def __init__(self, directory, architecture):
        
        self.NUM_BINS = 30
        self.directory = directory
        self.architecture = architecture
        self.load_models()
      #  self.store_weights()
       # self.populate_bins()
        
       

    def load_models(self):

    # Load models into a dictionary
        # Path to the directory containing the saved networks
   #     directory = "/Users/dani/Documents/GitHub/Large-Scale-Design-and-Analysis-of-Neural-Networks/Working Networks"
   

        # List all model files in the directory (assuming .pt or .pth extensions)
        model_files = [f for f in os.listdir(self.directory) if f.endswith('.pt') or f.endswith('.pth')]


        models = {}
        for model_file in model_files:
            model_path = os.path.join(self.directory, model_file)
            model_name = os.path.splitext(model_file)[0]  # Use the file name (without extension) as the key
            models[model_name] = torch.load(model_path, weights_only=True)

        # Print the loaded models (or access them by name)
        for name, model in models.items():
            print(f"Loaded model: {name}")
            print(model)
        
        working_networks_path = "./working_networks.pt"
        broken_networks_path = "BrokenNetworks/broken_networks.pt"

        #Load in the networks
        loaded_state_dict = torch.load(working_networks_path, weights_only=True)

        #create new instantiations to contain the saved networks
        num_networks = len(loaded_state_dict)
        self.networks = []
        for n in range(num_networks):
            self.networks.append(self.architecture())

        for i, network in enumerate(self.networks):
            state_dict_key = f'network_{i+1}'
            network.load_state_dict(loaded_state_dict[state_dict_key])

        return self.networks

    ##Get min and max weight values in each layer of the networks


    def __getMinOrMax__(self, networks, layerNum, getMin):
        """
        Parameters:
        - networks: an array of trained pytorch networks where the layers are nn.Linear
        - layer: an int representing the fully connected layer to access
        - getMin: a boolean indicating wheter to get the min or max
        Outputs: the minimum or maximum weight in some array of neural networks
        """
        weights = []
        curLayer = -1
        for network in networks:
            for layer in network.children():
                if isinstance(layer, nn.Linear):
                    curLayer += 1
                    if curLayer == layerNum:
                        #.cpu moves the tensor from the gpu to cpu if it is there in order to use .numpy()
                        weights.extend(layer.weight.data.cpu().numpy().flatten())
            curLayer = -1
        
        if getMin:
            return min(weights) - 1e-6 #small adjustment to account for floating point rounding error, issue occured with getting the upper bound of the max bin
        else:
            return max(weights) + 1e-6

    def store_weights(self):
        print("Str")

        #for each fully connected layer, create matrix of shape (N, M, B) where
        #N = num neurons in layer
        #M = num neurons in previous layer
        #B = number of bins
        self.layer_weight_distributions = []
        self.layer_bin_ranges = []
        network_fc_indices = []
            
        layers = self.networks[0].children()
        fcLayerNum = -1
        for index, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                fcLayerNum += 1
                layerMin = self.__getMinOrMax__(self.networks, fcLayerNum, True)
                layerMax = self.__getMinOrMax__(self.networks, fcLayerNum, False)
                #for each layer make an array of shape (num of neurons in layer, num of inputs to layer, num of bins)
                network_fc_indices.append(index)
                layer_shape = layer.weight.shape
                self.layer_weight_distributions.append(np.zeros(layer_shape + (self.NUM_BINS,), dtype=int))
                #subtract 1 to make 0-indexed
                bin_edges = np.histogram_bin_edges(a=[], bins=(self.NUM_BINS), range=(layerMin, layerMax))
                self.layer_bin_ranges.append(bin_edges)

                
                
        #Generate matrix of shape layer, network, neuron, incoming weight
        self.network_weights = []
        for index, layer_index in enumerate(network_fc_indices):
            network_weights_per_layer = []
            for network in self.networks:
                net_layers = list(network.children())
                network_weights_per_layer.append(net_layers[layer_index].weight.data.numpy())
            self.network_weights.append(network_weights_per_layer)

        # print(layer_weight_distributions[0].shape)
        print(self.layer_bin_ranges)
        #print the first weight of the first fully connected layer of the first network
        print(self.network_weights)

        for layer_num, layer_distribution in enumerate(self.layer_weight_distributions):
            #iterate over neurons in layer
            for i in range(layer_distribution.shape[0]):  
                #iterate over incoming weights to neuron
                for j in range(layer_distribution.shape[1]):  # incoming weights
                    # for this weight, iterate over each network to add data to corresponding bin
                    for network in self.network_weights[layer_num]:
                        #select network[neuron i, incoming weight j]
                        weight = network[i][j]
                        corresponding_bin = np.digitize(weight, self.layer_bin_ranges[layer_num], right=False) - 1
                        if corresponding_bin >= self.NUM_BINS or corresponding_bin < 0:
                            print(weight)
                            print(corresponding_bin)
                        
                        self.layer_weight_distributions[layer_num][i][j][corresponding_bin] += 1
                        print(self.layer_weight_distributions[0].shape)
                        print(self.layer_weight_distributions[0])

        return self.layer_weight_distributions, self.layer_bin_ranges


    def plot_weight_bins(self, weight_distributions, layer, weight_position, bin_edges):
            """
            Parameters:
            - weight_distributions: weight counts in the shape (layer, neuron, incoming weight, bin)
            - layer: Index of the layer to plot (0-indexed).
            - weight_position: Tuple indicating the neuron index and incoming weight index in the layer.
            - bin_edges: The edges of the bins for the specified layer, used to label the x-axis with actual values.
            """
            # get the bin data for the specified weight
            weight_distribution = weight_distributions[layer][weight_position[0]][weight_position[1]][:]

            num_bins = len(weight_distribution)

            bin_labels = [f"{bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}" for i in range(num_bins)]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(num_bins), weight_distribution, width=0.8, align='center', edgecolor='black')
            plt.xlabel('Bin Range')
            plt.ylabel('Count')
            plt.title(f'Weight Histogram for Layer {layer}, Position {weight_position}')
            plt.xticks(range(num_bins), bin_labels, rotation=45, ha='right')  # Rotate for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()  # Adjust layout to prevent label cut-off
            plt.show()

            # neuron 10 in layer 1(0) coming from input neuron 2
            print(self.layer_bin_ranges)
            #currently bin_edges are global and aren't specific to each layer
            # plot_weight_bins(self.layer_weight_distributions, layer=0, weight_position=(0, 0), bin_edges=self.layer_bin_ranges[0])

    def normalize_distributions(self):
        self.normalized_distributions = []
        for layer in self.layer_weight_distributions:
            layer_counts = []
            for neuron in layer:
                neuron_counts = []
                for from_weight in neuron:
                    #total should be num of networks
                    total = sum(from_weight)
                    normalized_bin_counts = [bin_count / total for bin_count in from_weight]
                    neuron_counts.append(np.array(normalized_bin_counts))
                layer_counts.append(np.array(neuron_counts))
            self.normalized_distributions.append(np.array(layer_counts))


        # print(len(normalized_counts))
        # print(sum(normalized_counts[0]))
        # print(normalized_counts[0])

        print(self.layer_bin_ranges[0])
        return self.normalized_distributions

    def normal_pdf(x, mu, sigma, amplitude):
        return amplitude * norm.pdf(x, mu, sigma)

    def return_weight_bins_plot(self, weight_distributions, layer, weight_position, bin_edges):
        """
        Parameters:
        - weight_distributions: weight counts in the shape (layer, neuron, incoming weight, bin)
        - layer: Index of the layer to plot (0-indexed).
        - weight_position: Tuple indicating the neuron index and incoming weight index in the layer.
        - bin_edges: The edges of the bins for the specified layer, used to label the x-axis with actual values.
        """
        weight_distribution = weight_distributions[layer][weight_position[0]][weight_position[1]][:]

        num_bins = len(weight_distribution)

        bin_labels = [f"{bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}" for i in range(num_bins)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(num_bins), weight_distribution, width=0.8, align='center', edgecolor='black')
        ax.set_xlabel('Bin Range')
        ax.set_ylabel('Count')
        ax.set_title(f'Weight Histogram for Layer {layer}, Position {weight_position}')
        ax.set_xticks(range(num_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')  # Rotate for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()  # Adjust layout to prevent label cut-off
        
        return fig, ax



    def plot_index(self, layer_weight_distributions, layer_bin_ranges):

        for layer_index in range(len(layer_weight_distributions)):
            counts = np.array(self.normalized_counts[layer_index])

            min_val_layer = np.min(layer_bin_ranges[layer_index])
            max_val_layer = np.max(layer_bin_ranges[layer_index])

            bin_centers = (layer_bin_ranges[layer_index][:-1] + layer_bin_ranges[layer_index][1:]) / 2
            print(np.max(counts))

            initial = [np.mean(bin_centers), np.std(bin_centers), np.max(counts)]

            fits, covariance = curve_fit(self.normal_pdf, bin_centers, counts, p0=initial)

            mu_fit, sigma_fit, amplitude_fit = fits
            print(f"Fitted parameters:\nMu = {mu_fit}\nSigma = {sigma_fit}\nAmplitude = {amplitude_fit}")

            y_fit = self.normal_pdf(bin_centers, mu_fit, sigma_fit, amplitude_fit)
            print(y_fit)

            fig, ax = self.return_weight_bins_plot(layer_weight_distributions, layer_index, (0, 0), layer_bin_ranges[layer_index])

            ax.plot(bin_centers, y_fit, color='red', linewidth=2, label='Fitted Normal Distribution')

            ax.legend()
            plt.show()


    def gaussian(self,  x, A, mu, sigma):
            return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

    def fit_gaussians_to_weight_distributions(self, weight_distributions, bin_edges_list):
        """
        Fit Gaussian curves to all weight histograms and save the fit parameters.

        Parameters:
        - weight_distributions: List of weight counts, where each element corresponds to a layer and is structured
                                as a NumPy array (neuron, incoming weight, bin).
        - bin_edges_list: List of bin edges for each layer.

        Returns:
        - fit_params: List of fit parameters, mirroring the structure of weight_distributions.
                    Each layer's fit parameters are stored as a NumPy array (neuron, incoming weight, 3).
        """
        self.fit_params = []
        for layer_idx, (layer_data, bin_edges) in enumerate(zip(weight_distributions, bin_edges_list)):
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            num_neurons, num_from_weights, _ = layer_data.shape
            layer_fit_params = np.zeros((num_neurons, num_from_weights, 3))

            for neuron_idx in range(num_neurons):
                for from_weight_idx in range(num_from_weights):
                    weight_distribution = layer_data[neuron_idx, from_weight_idx, :]
                    #check for existence
                    if np.any(weight_distribution):
                        initial_guess = [
                            np.max(weight_distribution),
                            bin_centers[np.argmax(weight_distribution)],
                            np.std(bin_centers)
                        ]
                        try:
                            #fitting
                            popt, _ = curve_fit(self.gaussian, bin_centers, weight_distribution, p0=initial_guess)
                        except RuntimeError:
                            print(f"Error - curve_fit failed for layer {layer_idx}, neuron {neuron_idx}, from_weight {from_weight_idx}")
                            popt = initial_guess
                    else:
                        popt = [0, 0, 0]
                    layer_fit_params[neuron_idx, from_weight_idx, :] = popt
            self.fit_params.append(layer_fit_params)
        return self.fit_params

    def plot_weight_bins_with_fit(self, weight_distributions, bin_edges, fit_params, layer, weight_position):
        """
        Plot the histogram of weights and the fitted Gaussian curve.

        Parameters:
        - weight_distributions: List of weight counts, where each element corresponds to a layer and is structured
                                as a NumPy array (neuron, incoming weight, bin).
        - bin_edges: Bin edges for the specified layer.
        - fit_params: List of fit parameters, mirroring the structure of weight_distributions.
                    Each layer's fit parameters are stored as a NumPy array (neuron, incoming weight, 3).
        - layer: Index of the layer to plot.
        - weight_position: Tuple indicating the neuron index and incoming weight index in the layer.
        """
        neuron_idx, from_weight_idx = weight_position

        #get data
        weight_distribution = weight_distributions[layer][neuron_idx, from_weight_idx, :]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        #get fit params
        fit_params_layer = fit_params[layer]
        fit_params_for_weight = fit_params_layer[neuron_idx, from_weight_idx, :]


        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        y_fit = self.gaussian(x_fit, *fit_params_for_weight)

        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, weight_distribution, width=np.diff(bin_edges), align='center',
                edgecolor='black', alpha=0.6, label='Histogram')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted Gaussian Curve')
        plt.xlabel('Bin Range')
        plt.ylabel('Count')
        plt.title(f'Weight Histogram and Fitted Curve for Layer {layer}, Position {weight_position}')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def fit(self):
        fit_params = self.fit_gaussians_to_weight_distributions(
        self.normalized_distributions, bin_edges_list=self.layer_bin_ranges
    )

    def fit_gmm_to_weight_distributions(weight_distributions, bin_edges_list, peaks_per_layer, random_state=None, replicate_factor=1000):
        """
        Fits a Gaussian Mixture Model (GMM) to each layer's weight histogram
        by replicating bin_centers replicate_factor * normalized_count times (if > 0).

        Parameters:
        weight_distributions (list of np.ndarray): 
            Each element is shape (num_neurons, num_from_weights, num_bins).
        bin_edges_list (list of np.ndarray): 
            Bin edges for each layer.
        peaks_per_layer (list of int): 
            For each layer, how many Gaussian components to fit.
        random_state (int or None): 
            Random seed for reproducibility.
        replicate_factor (int): 
            Multiplier for normalized counts to avoid int truncation to 0.

        Returns:
        gmm_models (list): 
            A list of length = number of layers, where gmm_models[layer_idx] is
            a 2D array (num_neurons, num_from_weights) of fitted GaussianMixture objects.
            None indicates an empty distribution.
        """
        gmm_models = []

        for layer_idx, (layer_data, bin_edges) in enumerate(zip(weight_distributions, bin_edges_list)):
            n_components = peaks_per_layer[layer_idx]
            num_neurons, num_from_weights, num_bins = layer_data.shape
            layer_models = np.empty((num_neurons, num_from_weights), dtype=object)

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            for neuron_idx in range(num_neurons):
                for from_weight_idx in range(num_from_weights):
                    counts = layer_data[neuron_idx, from_weight_idx, :]
                    if np.any(counts):
                        X_list = []
                        for center, c in zip(bin_centers, counts):
                            #scaling up normalized count
                            if c < 1:
                                rep_count = int(round(c * replicate_factor))
                            else:
                                rep_count = int(c)
                                
                            if rep_count > 0:
                                X_list.extend([center] * rep_count)

                        if len(X_list) == 0:
                            layer_models[neuron_idx, from_weight_idx] = None
                            continue

                        X = np.array(X_list).reshape(-1, 1)
                        gmm = GaussianMixture(
                            n_components=n_components,
                            random_state=random_state
                        )
                        gmm.fit(X)
                        layer_models[neuron_idx, from_weight_idx] = gmm
                    else:
                        layer_models[neuron_idx, from_weight_idx] = None

            gmm_models.append(layer_models)

        return gmm_models

    def plot_weight_bins_with_gmm(self, weight_distributions, bin_edges_list, gmm_models, layer, weight_position,
                                scale_to_hist=True, show_individual=True):
        """
        Plot the histogram of weights and the fitted GMM for a specified (layer, neuron_idx, from_weight_idx).
        """
        neuron_idx, from_weight_idx = weight_position
        gmm = gmm_models[layer][neuron_idx, from_weight_idx]

        if gmm is None:
            print("No data or GMM not fitted for this distribution.")
            return

        weight_distribution = weight_distributions[layer][neuron_idx, from_weight_idx, :]
        bin_edges = bin_edges_list[layer]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Evaluate the GMM PDF on a smooth grid
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000).reshape(-1, 1)
        log_probs = gmm.score_samples(x_fit)
        pdf = np.exp(log_probs)

        #scale the fit st it matches the height of histogram, optional
        counts_sum = np.sum(weight_distribution)
        bin_width = (bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1) if len(bin_edges) > 1 else 1.0
        if scale_to_hist:
            pdf *= (counts_sum * bin_width)

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, weight_distribution, width=np.diff(bin_edges),
                align='center', edgecolor='black', alpha=0.6, label='Histogram')
        plt.plot(x_fit, pdf, 'r-', label='GMM PDF')

        # Plot each component
        if show_individual:
            for i, (w, m, c) in enumerate(zip(gmm.weights_, gmm.means_, gmm.covariances_)):
                mu = m[0]
                sigma = np.sqrt(c[0, 0])
                component_pdf = w * norm.pdf(x_fit.flatten(), mu, sigma)
                if scale_to_hist:
                    component_pdf *= (counts_sum * bin_width)
                plt.plot(x_fit, component_pdf, '--', label=f'Component {i+1}')

        plt.xlabel('Bin Range')
        plt.ylabel('Count')
        plt.title(f'Layer {layer}, Position {weight_position} - GMM Fit')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def sk_gaussian(self):
        peaks_per_layer = [3, 2, 3]
        gmm_models = self.fit_gmm_to_weight_distributions(
            weight_distributions=self.normalized_distributions,
            bin_edges_list=self.layer_bin_ranges,
            peaks_per_layer=peaks_per_layer,
            random_state=42
        )

        layer_to_plot = 0
        num_neurons, num_from_weights, _ =  self.normalized_distributions[layer_to_plot].shape

        for i in range(num_neurons):
            for j in range(num_from_weights):
                self.plot_weight_bins_with_gmm(
                    weight_distributions=self.normalized_distributions,
                    bin_edges_list=self.layer_bin_ranges,
                    gmm_models=gmm_models,
                    layer=layer_to_plot,
                    weight_position=(i, j),
                    scale_to_hist=True,
                    show_individual=True
                )


    def compute_kl_divergence_gaussians(self, mu1, sigma1, mu2, sigma2):
        sigma1_adj = max(sigma1, 1e-8)
        sigma2_adj = max(sigma2, 1e-8)
        kl_divergence = np.log(sigma2_adj / sigma1_adj) + \
            (sigma1_adj ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2_adj ** 2) - 0.5
        return kl_divergence

    def compute_cross_entropy_gaussians(self, mu1, sigma1, mu2, sigma2):
        sigma2_adj = max(sigma2, 1e-8)
        cross_entropy = 0.5 * np.log(2 * np.pi * sigma2_adj ** 2) + \
            ((sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2_adj ** 2))
        return cross_entropy

    def cluster_gaussians_by_kl_divergence(self, fit_params, threshold):
        """
        Cluster Gaussians within each layer based on KL divergence.

        Parameters:
        - fit_params: List of fit parameters for each layer. Each layer is a NumPy array of shape (neurons, from_weights, 3).
        - threshold: KL divergence threshold for clustering.

        Returns:
        - cluster_indices_list: List over layers. Each layer is a NumPy array of shape (neurons, from_weights) containing cluster indices.
        """
        cluster_indices_list = []

        for layer_idx, layer_params in enumerate(fit_params):
            num_neurons, num_from_weights, _ = layer_params.shape
            total_gaussians = num_neurons * num_from_weights

            #flatten params
            mu_sigma_list = []
            for i in range(num_neurons):
                for j in range(num_from_weights):
                    A, mu, sigma = layer_params[i, j, :]
                    mu_sigma_list.append((mu, sigma))

            #make adjacecny matrix where mat[i, j] is the distance between i and j 
            kl_matrix = np.zeros((total_gaussians, total_gaussians))
            for idx1 in range(total_gaussians):
                mu1, sigma1 = mu_sigma_list[idx1]
                for idx2 in range(idx1 + 1, total_gaussians):
                    mu2, sigma2 = mu_sigma_list[idx2]
                    kl = self.compute_kl_divergence_gaussians(mu1, sigma1, mu2, sigma2)
                    kl_matrix[idx1, idx2] = kl
                    kl_matrix[idx2, idx1] = kl 

            #filter adjacency matrix by the threshold. replaces wit hbools
            adjacency_matrix = kl_matrix < threshold

            #replace with inedexes
            cluster_indices = np.full(total_gaussians, -1, dtype=int)
            current_cluster = 0
            for idx in range(total_gaussians):
                if cluster_indices[idx] == -1:
                    cluster_indices[idx] = current_cluster
                    queue = [idx]
                    while queue:
                        current_idx = queue.pop(0)
                        neighbors = np.where(adjacency_matrix[current_idx])[0]
                        for neighbor_idx in neighbors:
                            if cluster_indices[neighbor_idx] == -1:
                                cluster_indices[neighbor_idx] = current_cluster
                                queue.append(neighbor_idx)
                    current_cluster += 1
            #resahpe
            cluster_indices_layer = cluster_indices.reshape((num_neurons, num_from_weights))
            cluster_indices_list.append(cluster_indices_layer)

        return cluster_indices_list

    def cluster_gaussians_by_cross_entropy(self, fit_params, threshold):
        """
        Cluster Gaussians within each layer based on cross-entropy.

        Parameters:
        - fit_params: List of fit parameters for each layer. Each layer is a NumPy array of shape (neurons, from_weights, 3).
        - threshold: Cross-entropy threshold for clustering.

        Returns:
        - cluster_indices_list: List over layers. Each layer is a NumPy array of shape (neurons, from_weights) containing cluster indices.
        """
        cluster_indices_list = []

        for layer_idx, layer_params in enumerate(fit_params):
            num_neurons, num_from_weights, _ = layer_params.shape
            total_gaussians = num_neurons * num_from_weights

            mu_sigma_list = []
            for i in range(num_neurons):
                for j in range(num_from_weights):
                    A, mu, sigma = layer_params[i, j, :]
                    mu_sigma_list.append((mu, sigma))

            ce_matrix = np.zeros((total_gaussians, total_gaussians))
            for idx1 in range(total_gaussians):
                mu1, sigma1 = mu_sigma_list[idx1]
                for idx2 in range(idx1 + 1, total_gaussians):
                    mu2, sigma2 = mu_sigma_list[idx2]
                    ce = self.compute_cross_entropy_gaussians(mu1, sigma1, mu2, sigma2)
                    ce_matrix[idx1, idx2] = ce
                    ce_matrix[idx2, idx1] = ce

            adjacency_matrix = ce_matrix < threshold

            cluster_indices = np.full(total_gaussians, -1, dtype=int)
            current_cluster = 0
            for idx in range(total_gaussians):
                if cluster_indices[idx] == -1:
                    cluster_indices[idx] = current_cluster
                    queue = [idx]
                    while queue:
                        current_idx = queue.pop(0)
                        neighbors = np.where(adjacency_matrix[current_idx])[0]
                        for neighbor_idx in neighbors:
                            if cluster_indices[neighbor_idx] == -1:
                                cluster_indices[neighbor_idx] = current_cluster
                                queue.append(neighbor_idx)
                    current_cluster += 1

            cluster_indices_layer = cluster_indices.reshape((num_neurons, num_from_weights))
            cluster_indices_list.append(cluster_indices_layer)

        return cluster_indices_list


    def kl_ce_indices(self):
        kl_threshold = 0.001
        ce_threshold = 0.555

        kl_indices = self.cluster_gaussians_by_kl_divergence(fit_params=self.fit_params, threshold=kl_threshold)
        #print(kl_indices)

        ce_indices = self.cluster_gaussians_by_cross_entropy(fit_params=self.fit_params, threshold=ce_threshold)
        #print(ce_indices)
        return kl_indices, ce_indices
                


    def plot_unique_distributions(self, indices, layer, normalized_distributions, all_bin_edges, fit_params):
        """
        Plot unique distributions for a specified layer based on cluster indices.

        Parameters:
        - indices: A 3D matrix (list of 2D NumPy arrays) where each element corresponds to a layer,
                and each value in the innermost array represents the cluster index for a Gaussian.
                Shape: (layers, neurons, from_weights).
        - layer: The layer (index) for which to plot unique distributions.
        - normalized_distributions: A 4D matrix representing the normalized weight distributions.
                                    Shape: (layers, neurons, from_weights, bins).
        - all_bin_edges: A list of bin edges for each layer, used for plotting the histogram bins.
                        Shape: (layers, bin_edges_per_layer).
        - fit_params: A list of fitted Gaussian parameters for each layer.
                    Shape: (layers, neurons, from_weights, 3).

        Description:
        - This function plots unique Gaussian weight distributions for a specified layer.
        - A distribution is considered "unique" if it belongs to a cluster that has not yet been plotted
        (as determined by the cluster indices in `indices`).
        - The function iterates through the neurons and weights of the specified layer, identifies
        unique distributions (based on their cluster index), and plots them using the provided
        `plot_weight_bins_with_fit` function.
        """
        layer_indices = indices[layer]

        seen_indices = set()

        for i, unique_distribution_neuron_indices in enumerate(layer_indices):
            for j, unique_distribution_index in enumerate(unique_distribution_neuron_indices):
                if unique_distribution_index not in seen_indices:
                    self.plot_weight_bins_with_fit(
                        normalized_distributions, self.layer_bin_ranges[layer], 
                        fit_params=fit_params, layer=layer, weight_position=(i, j)
                    )
                    seen_indices.add(unique_distribution_index)