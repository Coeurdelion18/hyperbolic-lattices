#This is the implementation which should give results similar to the paper. However, this code has not been parallelized.
#Furthermore, the energy spectrum has not been divided into bins in this implementation.

import math
from pbc_adj import build_hyperbolic_lattice
import numpy as np

p, q, n = 3, 7, 4

adj, G, circles = build_hyperbolic_lattice(Nc=n, k=q, m=p, seed=1)
adj_matrix = adj
size = adj_matrix.shape[0]
N = size
C_E = np.arange(1, N + 1) 
normalized_energies = C_E / N
normalized_energies = normalized_energies[1:-1]

Nds = 100 #Averaging over 100 disorder realizations
N = 29 #Number of disorder strengths
Ws = 4 + np.arange(1, N+1) * 4


IPRs_cumulative = []
rs_cumulative = []

for Nd in range(Nds):
    IPRs = []
    rs = []
    for ind, W in enumerate(Ws):
        H = adj_matrix.copy()
        rng = np.random.default_rng()
        for i in range(size):
            H[i, i] += rng.uniform(-W/2.0, W/2.0)

        eigenvalues, eigenvectors = np.linalg.eig(H)
        # Each column is an eigenvector
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i]) #Normalize the eigenvectors
            
        
        sort_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices] 

        gd = np.diff(sorted_eigenvalues)       # spacing array of length N-1
        r_vals = gd[:-1] / gd[1:]              # ratio of consecutive spacings
        r_vals = np.minimum(r_vals, 1.0/r_vals) # min(r, 1/r)      
        rs.append(r_vals)

        iprs = []
        sorted_eigenvectors = sorted_eigenvectors.T
        for eigenvector in sorted_eigenvectors:
            ipr = np.sum(np.abs(eigenvector)**4)
            iprs.append(ipr)
        # Trim to match r_vals length
        iprs = iprs[1:-1] 
        IPRs.append(iprs)

    IPRs_cumulative.append(IPRs) #This should be 3d
    rs_cumulative.append(rs) #This should be 3d

IPRs_mean = np.mean(np.array(IPRs_cumulative), axis = 0) #This should now be a 2d array
rs_mean = np.mean(np.array(rs_cumulative), axis = 0) #This should also be 2d, ready for a heatmap.

# Now, we plot the heatmap

import matplotlib.pyplot as plt

# --- Prepare axis arrays ---
E = np.linspace(0, 1, rs_mean.shape[1] + 2)[1:-1]  # normalized energies (trimmed to match N-2)
W = Ws  # disorder strengths

# --- Plot r heatmap ---
plt.figure(figsize=(6,5))
plt.pcolormesh(E, W, rs_mean, cmap="hot", shading="auto")
plt.colorbar(label="⟨r⟩")
plt.gca().invert_yaxis()   # W increases from top to bottom
plt.xlabel("Normalized Energy (E)")
plt.ylabel("Disorder Strength (W)")
plt.title("Mean spacing ratio ⟨r⟩")
plt.show()

# --- Plot IPR heatmap ---
plt.figure(figsize=(6,5))
plt.pcolormesh(E, W, IPRs_mean, cmap="hot", shading="auto")
plt.colorbar(label="⟨IPR⟩")
plt.gca().invert_yaxis()
plt.xlabel("Normalized Energy (E)")
plt.ylabel("Disorder Strength (W)")
plt.title("Mean Inverse Participation Ratio (IPR)")
plt.show()
