import numpy as np
import matplotlib.pyplot as plt
from pbc_adj import build_hyperbolic_lattice

# --- Parameters ---
p, q, Nc = 3, 8, 5        # {p,q} lattice, number of circles
Nd = 500                  # disorder realizations (set smaller for testing, increase later)
N = 29                    # number of disorder strengths
W_vals = 4 + 4 * np.arange(1, N+1)   # W = 8, 12, ..., 120

# --- Build lattice adjacency matrix ---
adj, G, circles = build_hyperbolic_lattice(Nc=Nc, k=q, m=p, seed=1)
H0 = adj.copy()
N_lattice = H0.shape[0]
print(f"Lattice size: {N_lattice}")

# --- Storage arrays (Nd x N) ---
mean_r10m = np.zeros((Nd, N))
IPR10m   = np.zeros((Nd, N))
mean_r10l = np.zeros((Nd, N))
IPR10l   = np.zeros((Nd, N))
mean_r10r = np.zeros((Nd, N))
IPR10r   = np.zeros((Nd, N))

rng = np.random.default_rng()

for dis_N in range(Nd):  # loop disorder realizations
    for td, W in enumerate(W_vals):

        # Add onsite disorder
        disorder = rng.uniform(-W/2, W/2, size=N_lattice)
        H_tot = H0 + np.diag(disorder)

        # Eigen-decomposition
        eigenvals, eigenvecs = np.linalg.eigh(H_tot)
        sort_idx = np.argsort(eigenvals)
        eigenvals = eigenvals[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]

        # Level spacings
        gd = np.diff(eigenvals)   # length N_lattice-1
        r1 = gd[:-1] / gd[1:]
        r_vals = np.minimum(r1, 1.0/r1)

        # IPR for each eigenstate
        MIPR = np.sum(np.abs(eigenvecs)**4, axis=0)

        # --- Divide spectrum into windows ---
        mid = np.argmin(np.abs(eigenvals))   # index closest to 0
        ran = N_lattice // 20                # ~ 1/10 of spectrum

        # Middle 1/10
        idx_mid = np.arange(max(0, mid-ran), min(N_lattice, mid+ran+1))
        mean_r10m[dis_N, td] = np.mean(r_vals[idx_mid[:-1]])
        IPR10m[dis_N, td]    = np.mean(MIPR[idx_mid])

        # Left edge 1/10
        idx_left = np.arange(0, 2*ran)
        mean_r10l[dis_N, td] = np.mean(r_vals[idx_left[:-1]])
        IPR10l[dis_N, td]    = np.mean(MIPR[idx_left])

        # Right edge 1/10
        idx_right = np.arange(N_lattice-2*ran-1, N_lattice-1)
        mean_r10r[dis_N, td] = np.mean(r_vals[idx_right[:-1]])
        IPR10r[dis_N, td]    = np.mean(MIPR[idx_right])

    if (dis_N+1) % 50 == 0:
        print(f"Finished {dis_N+1}/{Nd} disorder realizations")

# --- Average over disorder realizations ---
mean_r10m_avg = np.mean(mean_r10m, axis=0)
IPR10m_avg    = np.mean(IPR10m, axis=0)
mean_r10l_avg = np.mean(mean_r10l, axis=0)
IPR10l_avg    = np.mean(IPR10l, axis=0)
mean_r10r_avg = np.mean(mean_r10r, axis=0)
IPR10r_avg    = np.mean(IPR10r, axis=0)

# --- Save results ---
np.savez(f"results_3_{q}_N{N_lattice}_W{W_vals[-1]}.npz",
         W_vals=W_vals,
         mean_r10m=mean_r10m_avg, IPR10m=IPR10m_avg,
         mean_r10l=mean_r10l_avg, IPR10l=IPR10l_avg,
         mean_r10r=mean_r10r_avg, IPR10r=IPR10r_avg)

# --- Plotting ---
plt.figure(figsize=(6,5))
plt.plot(W_vals, mean_r10m_avg, "ro-", label="middle 1/10")
plt.plot(W_vals, mean_r10l_avg, "bo-", label="left 1/10")
plt.plot(W_vals, mean_r10r_avg, "go-", label="right 1/10")
plt.axhline(0.53, color="k", linestyle="--", label="GOE")
plt.axhline(0.386, color="k", linestyle=":", label="Poisson")
plt.xlabel("Disorder strength W")
plt.ylabel("⟨r⟩")
plt.legend()
plt.title(f"Level spacing ratio ⟨r⟩ for {{3,{q}}}, N={N_lattice}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(W_vals, IPR10m_avg * N_lattice, "ro-", label="middle 1/10")
plt.plot(W_vals, IPR10l_avg * N_lattice, "bo-", label="left 1/10")
plt.plot(W_vals, IPR10r_avg * N_lattice, "go-", label="right 1/10")
plt.xlabel("Disorder strength W")
plt.ylabel("⟨IPR⟩ × N")
plt.legend()
plt.title(f"Inverse Participation Ratio for {{3,{q}}}, N={N_lattice}")
plt.tight_layout()
plt.show()
