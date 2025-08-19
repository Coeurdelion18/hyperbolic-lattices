from hypertiling import HyperbolicTiling
from hypertiling.graphics.svg import *
from hypertiling.graphics.plot import plot_tiling, quick_plot, plot_geodesic
import matplotlib.pyplot as plt
from hypertiling.operators import adjacency
from hypertiling.neighbors import find_radius_optimized_single
import numpy as np

p, q, n = 7, 3, 2 #This has for centers, the vertices of the 3, 7, 2 graph

T = HyperbolicTiling(p, q, n, kernel="SRG", center="vertex")


neighbours = {}
for i in range(len(T)):
    neighbours[i] = find_radius_optimized_single(T, i, radius=None, eps=1e-05)

#To get next-nearest neighbor adjacency, we loop through the dictionary, and for each node, we add the neighbors of its neighbors to a set.

next_nearest_neighbors = {}

for node, neighbors in neighbours.items():
    temp = []
    for neighbor in neighbors:
        temp.extend(neighbours[int(neighbor)])
    next_nearest_neighbors[node] = set(temp) #set() to avoid duplicates 

print(type(next_nearest_neighbors))
print(next_nearest_neighbors)

#Now we can create the adjacency matrix for the next-nearest neighbors.

adj = np.array([[0]*len(next_nearest_neighbors) for _ in range(len(next_nearest_neighbors))])
for i in range(len(next_nearest_neighbors)):
    for j in next_nearest_neighbors[i]:
        adj[i][j] = 1

#Now we plot the adjacency matrix.
plt.figure(figsize=(6,6))
plt.imshow(adj, cmap="gray", interpolation="none")
plt.title("Next-Nearest Adjacency Matrix of the Tiling")
plt.xlabel("Polygon Index")
plt.ylabel("Polygon Index")
plt.colorbar(label="Adjacency (1 = connected, 0 = not connected)")
plt.show()