
'''
This function is unused in later versions of program. delete before completion
'''
    def plot_index(self, layer_weight_distributions, layer_bin_ranges):

        for layer_index in range(len(layer_weight_distributions)):
            counts = np.array(self.normalized_counts[layer_index])

            min_val_layer = np.min(layer_bin_ranges[layer_index])
            max_val_layer = np.max(layer_bin_ranges[layer_index])

            bin_centers = (layer_bin_ranges[layer_index][:-1] + layer_bin_ranges[layer_index][1:]) / 2
            #print(np.max(counts))

            initial = [np.mean(bin_centers), np.std(bin_centers), np.max(counts)]

            fits, covariance = curve_fit(self.normal_pdf, bin_centers, counts, p0=initial)

            mu_fit, sigma_fit, amplitude_fit = fits
            #print(f"Fitted parameters:\nMu = {mu_fit}\nSigma = {sigma_fit}\nAmplitude = {amplitude_fit}")

            y_fit = self.normal_pdf(bin_centers, mu_fit, sigma_fit, amplitude_fit)
            #print(y_fit)

            fig, ax = self.return_weight_bins_plot(layer_weight_distributions, layer_index, (0, 0), layer_bin_ranges[layer_index])

            ax.plot(bin_centers, y_fit, color='red', linewidth=2, label='Fitted Normal Distribution')

            ax.legend()
            #plt.show()
            plt.savefig(self.save_dir+"/index_plot")
