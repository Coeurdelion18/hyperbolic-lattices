from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import numpy as np
import matplotlib.pyplot as plt
import math

p, q, n = 3, 8, 2

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

#Loop over disorder realizations
heat_r = []
heat_ipr = []
normalized_energies = []
Ws = np.arange(10, 120, 5)

for W in np.arange(10, 120, 5):
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

	# Calculate normalized energies using the paper's definition
	# For the k-th state (0-indexed), C(E_k) = k + 1
	# np.arange(1, N + 1) creates the array of counts [1, 2, 3, ..., N]
	C_E = np.arange(1, N + 1) 
	normalized_energies = C_E / N

	#So far, we have normalized the energies and calculated the IPRs. We want the ratio of level spacings for one disorder realization
	r_i = [] #Set of ratios of level spacings
	for i in range(0, len(sorted_eigenvalues)) :
		delta_i = sorted_eigenvalues[i] - sorted_eigenvalues[i-1]
		delta_i_minus_1 = sorted_eigenvalues[i-1] - sorted_eigenvalues[i-2]
		r_i.append(min(delta_i, delta_i_minus_1)/max(delta_i, delta_i_minus_1))
		
	r_i_mean = np.mean(np.array(r_i))
	heat_r.append(r_i)

	#Calculate the IPRs
	iprs = []
	sorted_eigenvectors = sorted_eigenvectors.T
	for eigenvector in sorted_eigenvectors:
		ipr = 0
		for element in eigenvector:
			ipr += np.pow(np.abs(element), 4)
		iprs.append(ipr)
	heat_ipr.append(iprs)

heat_ipr = np.array(heat_ipr).T
heat_r = np.array(heat_r).T

x = Ws
y = normalized_energies

#Plot heatmap with 'r'
W_grid, e_grid = np.meshgrid(x, y)
plt.pcolormesh(W_grid, e_grid, heat_r, cmap="hot", shading="auto")
plt.colorbar(label = "r")
plt.gca().invert_yaxis() 
plt.xlabel("W")
plt.ylabel("e")
plt.show()

plt.pcolormesh(W_grid, e_grid, heat_ipr, cmap="hot", shading="auto")
plt.colorbar(label = "IPR")
plt.gca().invert_yaxis() 
plt.xlabel("W")
plt.ylabel("e")
plt.show()