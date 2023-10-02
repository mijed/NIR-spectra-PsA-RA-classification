import os
import numpy as np
from src.data_src.data_manager import DataManager
import matplotlib.pyplot as plt

"""
Simple script to visualize the NIR spectra (second derivative) data_src
"""
plot_save_path = "../../plots/nir_eda_plots"
nir_data, labels = DataManager().load_data()
rand_idx = np.random.choice(len(nir_data))


plt.plot(nir_data[rand_idx])
plt.title(f"Label: {labels[rand_idx]}")
plt.savefig(os.path.join(plot_save_path,"nir_spectra_3"))
plt.show()
