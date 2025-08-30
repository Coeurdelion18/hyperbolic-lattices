from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import matplotlib.pyplot as plt
import math
from pbc_adj import build_hyperbolic_lattice
import numpy as np

p, q, n = 3, 7, 4
"""
P = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')
neighbours = {}

for i in range(len(P)):
	neighbours[i] = find_radius_optimized_single(P, i, radius=None, eps=1e-05)

#Now that we have the list of neighbours, we construct the adjacency matrix
size = max(neighbours.keys()) + 1
adj_matrix = np.zeros((size, size))

for node, neighs in neighbours.items():
	for neighbour in neighs:
		adj_matrix[node, neighbour] = 1
"""
adj, G, circles = build_hyperbolic_lattice(Nc=n, k=q, m=p, seed=1)
adj_matrix = adj
size = adj_matrix.shape[0]

#Loop over disorder realizations
heat_r = []
heat_ipr = []
normalized_energies = []
Ws = np.arange(10, 120, 5)

for W in Ws:
	H = adj_matrix.copy()
	rng = np.random.default_rng()
	for i in range(size):
		H[i, i] += rng.uniform(-W/2.0, W/2.0)

	eigenvalues, eigenvectors = np.linalg.eig(H)
	# Each column is an eigenvector
	for i in range(eigenvectors.shape[1]):
		eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])
		
	N = len(eigenvalues)

	sort_indices = np.argsort(eigenvalues)
	sorted_eigenvalues = eigenvalues[sort_indices]
	sorted_eigenvectors = eigenvectors[:, sort_indices]

	# Calculate normalized energies
	C_E = np.arange(1, N + 1) 
	normalized_energies = C_E / N

	# --- FIXED: ratio of level spacings ---
	gd = np.diff(sorted_eigenvalues)       # spacing array of length N-1
	r_vals = gd[:-1] / gd[1:]              # ratio of consecutive spacings
	r_vals = np.minimum(r_vals, 1.0/r_vals) # min(r, 1/r)
	heat_r.append(r_vals)

	# Calculate the IPRs
	iprs = []
	sorted_eigenvectors = sorted_eigenvectors.T
	for eigenvector in sorted_eigenvectors:
		ipr = np.sum(np.abs(eigenvector)**4)
		iprs.append(ipr)
	# Trim to match r_vals length
	iprs = iprs[1:-1]
	heat_ipr.append(iprs)

heat_ipr = np.array(heat_ipr).T   # shape (N-2, len(Ws))
heat_r   = np.array(heat_r).T     # shape (N-2, len(Ws))

x = Ws
y = normalized_energies[1:-1]     # match length N-2

# --- Plot heatmap with 'r' ---
W_grid, e_grid = np.meshgrid(x, y)
plt.figure(figsize=(6,5))
plt.pcolormesh(W_grid, e_grid, heat_r, cmap="hot", shading="auto")
plt.colorbar(label = "r")
plt.gca().invert_yaxis() 
plt.xlabel("W")
plt.ylabel("normalized energy (k/N)")
plt.title("Level spacing ratio ⟨r⟩")
plt.show()

# --- Plot heatmap with 'IPR' ---
plt.figure(figsize=(6,5))
plt.pcolormesh(W_grid, e_grid, heat_ipr, cmap="hot", shading="auto")
plt.colorbar(label = "IPR")
plt.gca().invert_yaxis() 
plt.xlabel("W")
plt.ylabel("normalized energy (k/N)")
plt.title("Inverse Participation Ratio (IPR)")
plt.show()

