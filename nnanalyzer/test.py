from nnanalyzer import WeightBinning, NetworkAnalyzer
from standardRegArchitercture import SimpleNet


#
#networks = NetworkAnalyzer(SimpleNet, 100)
#networks.generate_networks()
analyzer = WeightBinning(directory="/home/sharifadepetu/Senior Design/Large-Scale-Design-and-Analysis-of-Neural-Networks/nnanalyzer", architecture=SimpleNet)
#networks = NetworkAnalyzer()
save_dir = "directory here"



weights, layer_bin_ranges = analyzer.store_weights()
normalized_distributions = analyzer.normalize_distributions()
fit_params = analyzer.fit_gaussians_to_weight_distributions(normalized_distributions, layer_bin_ranges)
kl_indices, ce_indices = analyzer.kl_ce_indices()
#print(weights)

#analyzer.plot_weight_bins(weights, layer=0, weight_position=(0, 0), bin_edges=layer_bin_ranges[0])
#analyzer.return_weight_bins_plot(weight_distributions=weights, layer=0, weight_position=(0,0), bin_edges=layer_bin_ranges[0])
#analyzer.plot_index(weights, layer_bin_ranges=layer_bin_ranges)

#analyzer.plot_weight_bins_with_fit(weights, layer_bin_ranges[0], fit_params, 0, weight_position=(0,0))

analyzer.plot_unique_distributions(kl_indices, 0, normalized_distributions, layer_bin_ranges, fit_params)
#analyzer.plot_unique_distributions(kl_indices, 1, normalized_distributions, layer_bin_ranges, fit_params)
#analyzer.plot_unique_distributions(kl_indices, 2, normalized_distributions, layer_bin_ranges, fit_params)