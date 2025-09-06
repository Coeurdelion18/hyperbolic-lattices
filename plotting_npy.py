# file: plot_results.py
import numpy as np
import matplotlib.pyplot as plt

# --- Load results ---
rs_mean = np.load("rs_mean_nobins.npy")    # shape (N-2, len(Ws))
ipr_mean = np.load("ipr_mean_nobins.npy")  # shape (N-2, len(Ws))

# --- Axis setup ---
# Ws must match what you used in the simulation
Ws = 4 + np.arange(1, 30) * 4              # 29 disorder strengths
E = np.linspace(0, 1, rs_mean.shape[0])    # normalized energies

# --- Plot r heatmap ---
plt.figure(figsize=(6, 5))
plt.pcolormesh(E, Ws, rs_mean.T, cmap="hot", shading="auto")
plt.colorbar(label="⟨r⟩")
plt.gca().invert_yaxis()  # W increases from top to bottom
plt.xlabel("Normalized Energy (E)")
plt.ylabel("Disorder Strength (W)")
plt.title("Mean spacing ratio ⟨r⟩")
plt.tight_layout()
plt.show()

# --- Plot IPR heatmap ---
plt.figure(figsize=(6, 5))
plt.pcolormesh(E, Ws, ipr_mean.T, cmap="hot", shading="auto")
plt.colorbar(label="⟨IPR⟩")
plt.gca().invert_yaxis()
plt.xlabel("Normalized Energy (E)")
plt.ylabel("Disorder Strength (W)")
plt.title("Mean Inverse Participation Ratio (IPR)")
plt.tight_layout()
plt.show()
