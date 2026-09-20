import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix

from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import hypertiling as ht

from randomwalk import simulate_random_walk_sparse

# DEFINE PARAMETERS

p = 10
q = 3
n = 10 #This needs to be changed based on the results fro random_walk_by_gen.py. The n value should be the maximum n for which the linear region is observed for the given p and q. If the linear region just keeps increasing, then n can be set to a large value like 10 or 20. The goal is to capture the behavior of the quantum walk in the linear region before it saturates or localizes.
t_max = 10

Nds = 50  # Number of disorder realizations to average over
N = 20    # Number of disorder strengths to test
Ws = np.linspace(1, 40, N)

# ==========================================
# 2. PRECOMPUTE CLEAN GEOMETRY (Runs once)
# ==========================================
print(f"Generating pure hyperbolic lattice {{p={p}, q={q}, n={n}}}...")
T = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')
size = len(T)

# Build Adjacency Matrix
A_clean = lil_matrix((size, size), dtype=np.float64)
for i in range(size):
    neighbors = find_radius_optimized_single(T, i, radius=None, eps=1e-5)
    for j in neighbors:
        A_clean[i, j] = 1
        A_clean[j, i] = 1

# Base Tight-Binding Hamiltonian (Negative Hopping)
H_clean_csr = csr_matrix(-1.0 * A_clean)

# Precompute coordinates and squared distances from the center
print("Precomputing distances...")
coords = [T.get_center(i) for i in range(size)]
site = 0 # Starting site
r2 = np.zeros(size)
for i in range(size):
    d = ht.distance.disk_distance(coords[i], coords[site])
    r2[i] = d**2

# ==========================================
# 3. DYNAMIC SIMULATION FUNCTION
# ==========================================
def std_dev_with_t_sparse_fast(H_disordered, T, t_max, r2_array):
    """Calculates wavepacket spread over time for a given Hamiltonian."""
    times = np.arange(t_max)
    stds = np.zeros(t_max)

    for t in times:
        prob = simulate_random_walk_sparse(T, t, H_disordered)
        sigma = np.sqrt(np.sum(prob * r2_array))
        stds[t] = sigma
    
    return stds

# ==========================================
# 4. MAIN DISORDER LOOP
# ==========================================
# Matrix to hold the average spread for every W at every time t
# Shape: (Rows = len(Ws), Columns = t_max)
all_spreads_matrix = np.zeros((len(Ws), t_max))
rng = np.random.default_rng()

print("Starting dynamic localization simulations...")
for w_idx, W in enumerate(Ws):
    spreads_for_this_W = [] 
    
    for Nd in range(Nds):
        # Add random onsite disorder to the diagonal
        disorder_vector = rng.uniform(-W/2.0, W/2.0, size=size)
        H_disordered = H_clean_csr + sp.diags(disorder_vector, format='csr')
        
        # Run quantum walk and get the full time evolution array
        stds = std_dev_with_t_sparse_fast(H_disordered, T, t_max, r2)
        spreads_for_this_W.append(stds)
        
    # Average the wavepacket spread across all universes for this W
    mean_time_evolution = np.mean(spreads_for_this_W, axis=0)
    
    # Store it in our final heatmap matrix
    all_spreads_matrix[w_idx, :] = mean_time_evolution
    print(f"Completed W = {W:.2f} ({w_idx + 1}/{N})")

# ==========================================
# 5. HEATMAP PLOTTING
# ==========================================
print("Plotting heatmap...")
plt.figure(figsize=(10, 8))

# Create the heatmap using seaborn
ax = sns.heatmap(
    all_spreads_matrix,
    xticklabels=np.arange(t_max),
    yticklabels=np.round(Ws, 1),
    cmap="magma", 
    cbar_kws={'label': 'Wavepacket Spread $\sigma(t)$'}
)

# Invert Y-axis so lowest disorder (W=1) is at the bottom
ax.invert_yaxis() 

plt.title(f"Dynamical Localization: Time vs Disorder on {{{p}, {q}}} Lattice", pad=20, fontsize=14)
plt.xlabel("Time Step ($t$)", fontsize=12)
plt.ylabel("Disorder Strength ($W$)", fontsize=12)

plt.tight_layout()
plt.show()