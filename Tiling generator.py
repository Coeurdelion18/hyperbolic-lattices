"""
from hypertiling import HyperbolicTiling
from hypertiling.graphics.plot import quick_plot

T = HyperbolicTiling(8, 3, 2, kernel="SRG")
quick_plot(T)
"""
from hypertiling import HyperbolicTiling
from hypertiling.graphics.svg import *
from hypertiling.graphics.plot import plot_tiling, quick_plot, plot_geodesic
import matplotlib.pyplot as plt
from hypertiling.operators import adjacency
import numpy as np

p, q = 3, 8 #This has for centers, the vertices of the 8, 3, 2 graph
tiles = []

for n in range(8):
    T = HyperbolicTiling(p, q, n, kernel="SRG", center="vertex")

    neighbours = T.get_nbrs_list()

    adj = adjacency(neighbours, weights=None, boundary=None)

    tiles.append(adj.shape[0])    

print(tiles)