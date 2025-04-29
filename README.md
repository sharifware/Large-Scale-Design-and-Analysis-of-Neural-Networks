# Large-Scale Design and Analysis of Neural Networks

This project provides a Python package for the large-scale training, analysis, and visualization of neural network weight distributions. The framework supports generating diverse neural networks, analyzing weight distributions across models, fitting statistical models, clustering distributions based on KL divergence, and visualizing key properties to inform architecture and training design decisions.

## Features

- Train and collect large numbers of neural networks across GPUs.
- Save successfully converged networks based on customizable loss thresholds.
- Load and process multiple saved PyTorch networks.
- Bin layer weights into histograms across networks.
- Fit single-Gaussian or Gaussian Mixture Models (GMM) to weight distributions.
- Cluster distributions using symmetrized KL divergence.
- Visualize weight histograms, fitted distributions, and conditional probability matrices.
- Interactive walkthroughs of neuron-level distributions.

## Installation
Install the package directly from TestPyPI:
pip install -i https://test.pypi.org/simple/ nnanalyzer==0.5.5

Or install from source:
pip install .

# Main Components

## Network Analyser
- generate_networks(): Generates multiple networks meeting convergence criteria.
- check_loss(): Trains and evaluates a single network, visualizing loss.
- __train_network__(): Internal method for model training with early stopping.
- __test_network__(): Internal evaluation of model test loss.
- __visualize_loss__(): Plots training and testing loss across epochs.
- __save_networks__(): Saves working and non-working models separately.
- __get_device__(): Automatically detects CPU, CUDA, or MPS devices.

# NeuralNetworkTrainer
- load_data(): Loads and splits the dataset for training/testing.
- train_single_network(): Trains a single network on a specific GPU.
- worker(): Worker process for parallelized network training.
- parallel_train_networks(): Manages concurrent network training across GPUs.
- save_networks(): Saves successful networks to disk.
- load_existing_networks(): Loads previously saved networks.
- create_experiment_metadata(): Records detailed metadata for experiments.
- run(): Executes the complete network training pipeline

## WeightBinning
- store_weights(): Extracts and bins weights into histograms.
- normalize_distributions(): Converts bin counts into probability distributions.
- fit_distributions(multi_peak=False): Fits single or multi-peak distributions (GMMs) to weight data.
- cluster_distributions(threshold=0.001): Clusters distributions using KL divergence.
- plot_distribution(): Plots weight distributions with model fits.
- plot_conditional_weight_matrix(): Visualizes conditional dependencies between weights.
- analyze_distributions(): Full pipeline for fitting, clustering, and visualization.
- weights_walkthrough(): Step-by-step visualization of neuron-level weight distributions.


## Typical Workflow
1. Use NeuralNetworkTrainer to mass-produce trained models. 
2. Alternatively, use NetworkAnalyzer for streamlined single-network generation
3. Initialize WeightBinning with saved network models and architecture.
4. Store and bin weights across the model ensemble.
5. Normalize weight distributions.
6. Fit statistical models (Gaussian or GMM) to distributions.
7. Cluster distributions using KL divergence.
8. Visualize representative distributions and conditional relationships.


## Requirements
- Python >= 3.8
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- pandas

## License
This project is licensed under the MIT License.






