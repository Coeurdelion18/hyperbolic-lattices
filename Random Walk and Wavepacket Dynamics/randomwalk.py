import numpy as np
import matplotlib.pyplot as plt
hbar = 1.0
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import hypertiling as ht

import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a', filename='randomwalk_nonlinear.log')

logger = logging.getLogger(__name__)

PLOT_DIR = "plots_0.05"
os.makedirs(PLOT_DIR, exist_ok=True)

def hyperbolic_adjacency_sparse(p, q, n):
    T = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')

    size = len(T)
    A = lil_matrix((size, size), dtype=np.float64)

    for i in range(size):
        neighbors = find_radius_optimized_single(T, i, radius=None, eps=1e-5)
        for j in neighbors:
            A[i, j] = 1
            A[j, i] = 1

    return csr_matrix(A), T  # return tiling too


def timeEvolve_sparse(psi0, H, t):
    #Trace: timeEvolve_sparse -> simulate_random_walk_sparse -> std_dev_with_t_sparse -> p_q_slope
    psi_t = expm_multiply((-1j * H * t / hbar), psi0)
    psi_t /= np.sqrt(np.vdot(psi_t, psi_t))
    E_t = np.vdot(psi_t, H @ psi_t)
    return psi_t, np.real(E_t)


def simulate_random_walk_sparse(T, t, H):
    num_sites = H.shape[0]

    psi0 = np.zeros(num_sites, dtype=complex)
    site = 0  # start at origin
    psi0[site] = 1.0

    psi_t, E_t = timeEvolve_sparse(psi0, H, t)

    prob = np.abs(psi_t)**2
    prob /= prob.sum()

    return prob, E_t


# --- Standard deviation vs time ---
def std_dev_with_t_sparse(p, q, n, t_max):
    times = np.arange(0, t_max, 0.05)
    stds = []
    energies = []

    # Build geometry + Hamiltonian
    A, T = hyperbolic_adjacency_sparse(p, q, n)
    H = -1.0 * A  # tight-binding Hamiltonian

    # Precompute coordinates
    coords = [T.get_center(i) for i in range(len(T))]
    site = 0

    # Precompute squared distances
    r2 = np.zeros(len(coords))
    for i in range(len(coords)):
        d = ht.distance.poincare_distance(coords[i], coords[site])
        r2[i] = d**2

    for t in times:
        prob, E_t = simulate_random_walk_sparse(T, t, H)

        sigma = np.sqrt(np.sum(prob * r2))
        stds.append(sigma)
        energies.append(E_t)

    # --- plot spread vs time and save ---
    plt.figure()
    plt.plot(times, stds, marker='o')
    plt.xlabel("t")
    plt.ylabel("std dev of spread")
    plt.title(f"Spread vs t (p={p}, q={q}, n={n})")
    plt.savefig(os.path.join(PLOT_DIR, f"spread_vs_t_p{p}_q{q}_n{n}.png"))
    plt.close()

    plt.figure()
    plt.plot(times, energies, marker='o')
    plt.xlabel("t")
    plt.ylabel("energy profile")
    plt.title(f"Energy vs t (p={p}, q={q}, n={n})")
    plt.savefig(os.path.join(PLOT_DIR, f"energy_vs_t_p{p}_q{q}_n{n}.png"))
    plt.close()

    return times, stds, energies


from scipy.stats import linregress

def linear_region_study(times, stds, r2_threshold=0.99):
    times = np.array(times)
    stds = np.array(stds)
    
    best_slope = None
    best_intercept = None
    limit_idx = 2
    
    for i in range(3, len(times)):
        slope, intercept, r_value, p_value, std_err = linregress(times[:i], stds[:i])
        r2 = r_value**2
        
        if r2 >= r2_threshold:
            best_slope = slope
            best_intercept = intercept
            limit_idx = i - 1
        else:
            break
    
    if best_slope is None:
        # Nothing cleared the threshold — fall back to the first two points
        slope, intercept, r_value, p_value, std_err = linregress(times[:2], stds[:2])
        best_slope, best_intercept = slope, intercept
        limit_idx = 1

    return limit_idx, times[limit_idx], best_slope, best_intercept

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def nonlinear_study(times, stds, p, q, save_dir="nonlinear_plots", poly_degree=3):
    times = np.array(times)  # x axis
    stds = np.array(stds)    # y axis
    os.makedirs(save_dir, exist_ok=True)

    # --- Locate first maxima and the first trough after it ---
    max_idx, _ = find_peaks(stds)
    if len(max_idx) == 0:
        raise ValueError("No local maxima found in stds.")
    first_max_idx = max_idx[0]

    min_idx, _ = find_peaks(-stds)
    min_idx_after = min_idx[min_idx > first_max_idx]
    if len(min_idx_after) == 0:
        raise ValueError("No local minima found after the first maxima.")
    first_min_idx = min_idx_after[0]

    height_diff = stds[first_max_idx] - stds[first_min_idx]

    # --- Fit log(stds) as a polynomial in log(times), up to the first maxima ---
    t_fit = times[:first_max_idx + 1]
    s_fit = stds[:first_max_idx + 1]
    mask = (t_fit > 0) & (s_fit > 0)   # power law needs strictly positive t, s
    t_fit, s_fit = t_fit[mask], s_fit[mask]

    log_t, log_s = np.log(t_fit), np.log(s_fit)

    degree = min(poly_degree, len(log_t) - 1)  # can't fit degree >= n_points
    if degree < 1:
        raise ValueError("Not enough positive points before the first maxima to fit.")

    coeffs = np.polyfit(log_t, log_s, degree)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()  # d(log s)/d(log t) = local scaling exponent, as a function of log t

    # Leading-order power: the local exponent at the smallest available t
    # (i.e. alpha as t -> the start of your data, the true "leading" behavior)
    alpha_leading = dpoly(log_t.min())

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, stds, "o-", color="steelblue", ms=3, lw=1, label="data")

    t_dense = np.linspace(t_fit.min(), t_fit.max(), 200)
    s_dense = np.exp(poly(np.log(t_dense)))
    ax.plot(t_dense, s_dense, "--", color="darkorange", lw=2,
             label=fr"poly(log-log) fit, deg={degree}, $\alpha_{{leading}}$={alpha_leading:.3f}")

    ax.plot(times[first_max_idx], stds[first_max_idx], "^", color="green", ms=10, label="first maxima")
    ax.plot(times[first_min_idx], stds[first_min_idx], "v", color="red", ms=10, label="first minima (after maxima)")
    ax.vlines(times[first_min_idx], stds[first_min_idx], stds[first_max_idx], color="gray", linestyle=":", lw=1.5)
    ax.annotate(f"Δ = {height_diff:.3f}",
                xy=(times[first_min_idx], (stds[first_max_idx] + stds[first_min_idx]) / 2),
                xytext=(10, 0), textcoords="offset points", va="center")

    ax.set_xlabel("time"); ax.set_ylabel("std")
    ax.set_title(f"alpha_leading={alpha_leading:.3f}, height diff={height_diff:.3f} (p={p}, q={q})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"p{p}_q{q}.png"), dpi=110)
    plt.close(fig)

    return alpha_leading, height_diff


#Now, we sweep across a {p, q} grid and compute the slopes
def p_q_slope(p_min=3, p_max=8, q_min=3, q_max=8, t_max=7):
    # Initialize an array of NaNs to store the slopes
    # Sized to allow direct indexing: slopes[p, q]
    slopes = np.full((p_max+1, q_max+1), np.nan)
    lin_times = np.full((p_max+1, q_max+1), np.nan)
    energies = np.full((p_max+1, q_max+1, len(np.arange(0, t_max, 0.05))), np.nan)

    for p in range(p_min, p_max+1):
        for q in range(q_min, q_max+1):
            
            # Ensure the geometry is strictly hyperbolic
            if (p - 2) * (q - 2) > 4:
                try:
                    # n (layer count) fixed at 4 for quick calculations
                    times, stds, e_t = std_dev_with_t_sparse(p, q, 4, t_max)
                    logger.info(f"Computed energy profile for p = {p}, q={q}: {e_t}")
                    
                    # Ensure this unpacking matches your actual linear_region_study function
                    _, lin_time, m, _ = linear_region_study(times, stds)
                    
                    # Store the slope
                    slopes[p, q] = m
                    lin_times[p, q] = lin_time
                    energies[p, q] = np.array(e_t)
                    logger.info(f"Computed slope for p={p}, q={q}: {m}, linear time: {lin_time}")
                    
                except Exception as e:
                    logger.error(f"Simulation failed for p={p}, q={q} due to: {e}")
                    pass # Leaves the value as NaN
    return slopes, lin_times, energies

def p_q_nonlinear(p_min=7, p_max=8, q_min=7, q_max=8, t_max=7):
    alphas = np.full((p_max+1, q_max+1), np.nan)
    height_diffs = np.full((p_max+1, q_max+1), np.nan)
    
    for p in range(p_min, p_max+1):
        for q in range(q_min, q_max+1):
            # Ensure the geometry is strictly hyperbolic
            if (p - 2) * (q - 2) > 4:
                try:
                    # n (layer count) fixed at 4 for quick calculations
                    times, stds = std_dev_with_t_sparse(p, q, 4, t_max)
                    
                    alpha, height_diff = nonlinear_study(times, stds, p, q)
                    alphas[p, q] = alpha
                    height_diffs[p, q] = height_diff

                    logger.info(f"Computed info for p={p}, q={q}: Alpha: {alpha}, Height Diff: {height_diff}")
                    
                except Exception as e:
                    logger.error(f"Simulation failed for p={p}, q={q} due to: {e}")
                    pass # Leaves the value as NaN
    
    return alphas, height_diffs

if __name__ == "__main__":
    slopes, lin_times, energies = p_q_slope(3, 3, 8, 8, 7)