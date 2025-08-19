from hypertiling import HyperbolicTiling
from hypertiling.graphics.svg import *
from hypertiling.graphics.plot import plot_tiling, quick_plot, plot_geodesic
import matplotlib.pyplot as plt
from hypertiling.operators import adjacency
from hypertiling.neighbors import find_radius_optimized_single
import numpy as np
import seaborn as sns
import matplotlib.cm as cmap

p, q, n = 10, 3, 2

P = HyperbolicTiling(p, q, n, kernel="SRG", center="cell") #This is the actual lattice.
quick_plot(P, dpi=220)
T = HyperbolicTiling(q, p, n, kernel="SRG", center="vertex") # This is the dual lattice, which is the one we use to number the vertices.

neighbours = {}
for i in range(len(T)):
    neighbours[i] = find_radius_optimized_single(T, i, radius=None, eps=1e-05)


plot_tiling(T, dpi=220)
for i in range(len(T)):
    z = T.get_center(i)
    l = T.get_layer(i)
    t = plt.text(z.real, z.imag, str(i), fontsize=10-4*l, ha="center", va="center")

#print(adj)   
plt.show()

# Convert to adjacency matrix
size = max(neighbours.keys()) + 1  # Assuming nodes are indexed from 0
adj_matrix = np.zeros((size, size))

for node, neighbors in neighbours.items():
    for neighbor in neighbors:
        adj_matrix[node, neighbor] = 1

# Plot heatmap
plt.figure(figsize=(6,6))
sns.heatmap(adj_matrix, cmap="viridis", linewidths=0.5, square=True, cbar=True)
plt.title("Neighbourhood Heatmap")
plt.xlabel("Nodes")
plt.ylabel("Nodes")
plt.show()

#Now, we diagonalize the adjacency matrix


eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
print(eigenvectors)

#Now, we select only the real eigenvalues
real_eigenvalues = [ev for ev in eigenvalues if np.isreal(ev)]

#Now, we sort the real eigenvalues
real_eigenvalues.sort()

#Now, we scatter plot the real eigenvalues versus their index
indices = np.arange(len(real_eigenvalues))
plt.scatter(indices, real_eigenvalues, color='blue', s=2)
plt.title("Energy Eigenvalues of the Adjacency Matrix")
plt.show()

# Find the smallest eigenvalue and its corresponding eigenvector
smallest_eigenvalue_index = np.argmin(real_eigenvalues)
smallest_eigenvector = eigenvectors[:, smallest_eigenvalue_index]

#Find the largest eigenvalue and its corresponding eigenvector
largest_eigenvalue_index = np.argmax(real_eigenvalues)
largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]

# Square each element of the eigenvector
squared_values = np.square(np.abs(largest_eigenvector))

# Normalize the squared values for better visualization
for i in range(eigenvectors.shape[1]):
    eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

squared_values_normalized = np.square(largest_eigenvector)
plot_tiling(T, colors = squared_values_normalized, cmap=cmap.RdBu, edgecolor="k", lw=0.2, plot_colorbar=True, linewidth = 0)
plt.show()


#Now, we want the density of states. real_eigenvalues is the list of energies, sorted.

# Density of states (Lorentzian broadening)
eta = 0.3 #this is the value of the broadening parameter which gives the closest results to the ones in the paper.
num_points = len(real_eigenvalues*10)
E_min = np.min(real_eigenvalues)
E_max = np.max(real_eigenvalues)
E_grid = np.linspace(E_min, E_max, num_points)
dos = np.zeros_like(E_grid)

for i, E in enumerate(E_grid):
    dos[i] = np.sum(eta / ((E - real_eigenvalues) ** 2 + eta ** 2)) / np.pi

# Normalize DOS
dos /= np.trapz(dos, E_grid)  # Area under curve = 1

# Plot
plt.plot(-E_grid, dos)
plt.xlabel("Energy")
plt.ylabel("Normalized Density of States (DOS)")
plt.title("Normalized DOS with Lorentzian Broadening")
plt.show()

