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
import scipy.sparse as sp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', filename='randomwalk_disordered.log')

logger = logging.getLogger(__name__)

PLOT_DIR = "plots_disordered"
os.makedirs(PLOT_DIR, exist_ok=True)
Nds = 100

def hyperbolic_adjacency_sparse(p, q, n):
    T = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')

    size = len(T)
    A = lil_matrix((size, size), dtype=np.float64)

    for i in range(size):
        neighbors = find_radius_optimized_single(T, i, radius=None, eps=1e-5)
        for j in neighbors:
            A[i, j] = 1
            A[j, i] = 1

    H_clean = csr_matrix(-1.0 * A)
    
    return H_clean, T  # return tiling too

def add_disorder(H_clean, T, W):
    rng = np.random.default_rng()
    size = len(T)
    disorder_vector = rng.uniform(-W/2.0, W/2.0, size=size)
    H_disordered = H_clean + sp.diags(disorder_vector, format='csr')
    return H_disordered

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
    ipr = (np.abs(psi_t)**4).sum()
    prob /= prob.sum()

    return prob, E_t, ipr


# --- Standard deviation vs time ---
# def std_dev_with_t_sparse(p, q, n, t_max, H, T):
#     # Make this return IPR as well 
#     times = np.arange(0, t_max, 0.05)
#     stds = []
#     energies = []
#     iprs = []

#     # Precompute coordinates
#     coords = [T.get_center(i) for i in range(len(T))]
#     site = 0

#     # Precompute squared distances
#     r2 = np.zeros(len(coords))
#     for i in range(len(coords)):
#         d = ht.distance.poincare_distance(coords[i], coords[site])
#         r2[i] = d**2

#     for t in times:
#         prob, E_t, ipr = simulate_random_walk_sparse(T, t, H)

#         sigma = np.sqrt(np.sum(prob * r2))
#         stds.append(sigma)
#         energies.append(E_t)
#         iprs.append(ipr)

#     return times, stds, energies, iprs

def std_dev_with_t_sparse(p, q, n, t_max, H, T, dt=0.05):
    times = np.arange(0, t_max, dt)
    num = len(times)
    # Precompute coordinates
    coords = [T.get_center(i) for i in range(len(T))]
    site = 0

    # Precompute squared distances
    r2 = np.zeros(len(coords))
    for i in range(len(coords)):
        d = ht.distance.poincare_distance(coords[i], coords[site])
        r2[i] = d**2

    psi0 = np.zeros(H.shape[0], dtype=complex)
    psi0[0] = 1.0

    # ONE call for the whole trajectory, instead of 140 separate calls
    psi_traj = expm_multiply(-1j * H / hbar, psi0, start=0, stop=t_max,
                              num=num, endpoint=False)

    norms = np.linalg.norm(psi_traj, axis=1, keepdims=True)
    psi_traj = psi_traj / norms

    prob = np.abs(psi_traj) ** 2
    prob /= prob.sum(axis=1, keepdims=True)

    iprs = (prob ** 2).sum(axis=1)
    stds = np.sqrt((prob * r2).sum(axis=1))

    Hpsi = (H @ psi_traj.T).T
    energies = np.real(np.einsum('ti,ti->t', psi_traj.conj(), Hpsi))

    return times, stds, energies, iprs


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


#Now, we sweep across a {p, q} grid and compute the slopes
def p_q_slope(p, q, n=8, W = 40.0, t_max=7):
    # Initialize an array of NaNs to store the slopes
    # Sized to allow direct indexing: slopes[p, q]
    # slopes = np.full((p_max+1, q_max+1), np.nan)
    # lin_times = np.full((p_max+1, q_max+1), np.nan)
    # energies = np.full((p_max+1, q_max+1, len(np.arange(0, t_max, 0.05))), np.nan)
    energies = []
    stds = []
    times = []
    iprs = []

    # Ensure the geometry is strictly hyperbolic
    if (p - 2) * (q - 2) > 4:
        try:
            H_clean, T = hyperbolic_adjacency_sparse(p, q, n)

            for _ in tqdm(range(Nds)):
                H_disordered = add_disorder(H_clean, T, W)
                time, std, e_t, ipr = std_dev_with_t_sparse(p, q, n, t_max, H_disordered, T)
                times.append(time)
                iprs.append(ipr)
                energies.append(e_t)
                stds.append(std)

            # Average over realisations
            ave_energy = np.mean(energies, axis=0)
            ave_std = np.mean(stds, axis=0)
            ave_time = np.mean(times, axis=0)
            ave_ipr = np.mean(iprs, axis=0)

            _, lin_time, m, _ = linear_region_study(ave_time, ave_std)

            # --- plot spread vs time and save ---
            plt.figure()
            plt.plot(ave_time, ave_std, marker='o')
            plt.xlabel("t")
            plt.ylabel("std dev of spread")
            plt.title(f"Spread vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"spread_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()
        
            plt.figure()
            plt.plot(ave_time, ave_energy, marker='o')
            plt.xlabel("t")
            plt.ylabel("energy profile")
            plt.title(f"Energy vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"energy_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            plt.figure()
            plt.plot(ave_time, ave_ipr, marker='o')
            plt.xlabel("t")
            plt.ylabel("ipr profile")
            plt.title(f"IPR vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"ipr_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            logger.info(f"Computed slope for p={p}, q={q}: {m}, linear time: {lin_time}")
            return m, lin_time, ave_energy, ave_ipr
            
        except Exception as e:
            logger.error(f"Simulation failed for p={p}, q={q} due to: {e}")
            pass # Leaves the value as NaN

    return None

"""
We need a way to estimate the power law exponent in the initial sublinear region. Once we get that, we can vary W and get the plots of W vs alpha
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fit_powerlaw_exponent(x, y, t_min=None, t_max=None, auto_range=True,
                            min_points=5, r2_threshold=0.995, plot=True):
    """
    Estimate the power-law exponent alpha in y ~ x^alpha over the initial
    growth region of a curve that later saturates (e.g. spread vs t).

    Parameters
    ----------
    x, y : array-like
        Data points (e.g. t and std-dev-of-spread). Must be > 0 for log fit.
    t_min, t_max : float, optional
        If given, fit only points with t_min <= x <= t_max (manual mode).
    auto_range : bool
        If t_min/t_max are not given, automatically find the largest
        contiguous initial window (starting from the smallest x) over which
        log(y) vs log(x) is linear (R^2 >= r2_threshold).
    min_points : int
        Minimum number of points required in the fit window.
    r2_threshold : float
        R^2 cutoff used to decide where the power-law region ends, when
        auto_range=True.
    plot : bool
        If True, show a two-panel plot (linear + log-log) with the fit
        overlaid. Set to False (or comment out the plotting block below)
        to disable visualization entirely.

    Returns
    -------
    dict with keys:
        alpha, intercept, r2, x_fit_range, n_points
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    order = np.argsort(x)
    x, y = x[order], y[order]

    logx, logy = np.log(x), np.log(y)

    if not auto_range:
        if t_min is None: t_min = x.min()
        if t_max is None: t_max = x.max()
        sel = (x >= t_min) & (x <= t_max)
        lx, ly = logx[sel], logy[sel]
        slope, intercept, r, _, _ = stats.linregress(lx, ly)
        result = dict(alpha=slope, intercept=intercept, r2=r**2,
                     x_fit_range=(x[sel].min(), x[sel].max()),
                     n_points=sel.sum())
    else:
        best = None
        n = len(x)
        for end in range(min_points, n + 1):
            lx, ly = logx[:end], logy[:end]
            slope, intercept, r, _, _ = stats.linregress(lx, ly)
            r2 = r**2
            if r2 >= r2_threshold:
                best = dict(alpha=slope, intercept=intercept, r2=r2,
                            x_fit_range=(x[0], x[end-1]), n_points=end)
            else:
                if best is not None:
                    break

        if best is None:
            lx, ly = logx[:min_points], logy[:min_points]
            slope, intercept, r, _, _ = stats.linregress(lx, ly)
            best = dict(alpha=slope, intercept=intercept, r2=r**2,
                        x_fit_range=(x[0], x[min_points-1]), n_points=min_points)
        result = best

    # ---------------- PLOTTING BLOCK (comment out to disable) ----------------
    if plot:
        alpha, intercept = result['alpha'], result['intercept']
        x_start, x_end = result['x_fit_range']
        fit_mask = (x >= x_start) & (x <= x_end)

        x_fit_line = np.linspace(x_start, x_end, 100)
        y_fit_line = np.exp(intercept) * x_fit_line**alpha

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale view
        ax = axes[0]
        ax.plot(x, y, 'o', color='tab:blue', ms=4, label='data')
        ax.plot(x[fit_mask], y[fit_mask], 'o', color='tab:red', ms=5,
                label='fit region')
        ax.plot(x_fit_line, y_fit_line, '-', color='black', lw=1.5,
                label=f'fit: y ~ t^{alpha:.3f}')
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title('Linear scale')
        ax.legend()

        # Log-log view (where the power-law shows as a straight line)
        ax = axes[1]
        ax.loglog(x, y, 'o', color='tab:blue', ms=4, label='data')
        ax.loglog(x[fit_mask], y[fit_mask], 'o', color='tab:red', ms=5,
                   label='fit region')
        ax.loglog(x_fit_line, y_fit_line, '-', color='black', lw=1.5,
                   label=f'alpha={alpha:.3f}, R2={result["r2"]:.4f}')
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title('Log-log scale')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"alpha_fit_p{p}_q{q}_n{n}.png"))
        plt.close()
    # ---------------------------------------------------------------------

    return result


def p_q_alpha(p, q, n=8, W=40.0, t_max=7, plot_fit=False):
    stds = []
    times = []
    iprs = []

    if (p - 2) * (q - 2) > 4:
        try:
            H_clean, T = hyperbolic_adjacency_sparse(p, q, n)

            for _ in tqdm(range(Nds)):
                H_disordered = add_disorder(H_clean, T, W)
                time, std, e_t, ipr = std_dev_with_t_sparse(p, q, n, t_max, H_disordered, T)
                times.append(time)
                iprs.append(ipr)
                stds.append(std)

            ave_std = np.mean(stds, axis=0)
            ave_time = np.mean(times, axis=0)
            ave_ipr = np.mean(iprs, axis=0)

             # --- plot spread vs time and save ---
            plt.figure()
            plt.plot(ave_time, ave_std, marker='o')
            plt.xlabel("t")
            plt.ylabel("std dev of spread")
            plt.title(f"Spread vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"spread_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            plt.figure()
            plt.plot(ave_time, ave_ipr, marker='o')
            plt.xlabel("t")
            plt.ylabel("ipr profile")
            plt.title(f"IPR vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"ipr_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            # plot_fit=False during sweeps to avoid blocking plt.show() calls
            fit_result = fit_powerlaw_exponent(ave_time, ave_std, plot=plot_fit)
            alpha = fit_result['alpha']

            logger.info(f"Computed alpha for p={p}, q={q}, W={W}: {alpha} (R2={fit_result['r2']:.4f})")

            return alpha, ave_time, ave_ipr

        except Exception as e:
            logger.error(f"Simulation failed for p={p}, q={q}, W={W} due to: {e}")
            return None

    return None

if __name__ == "__main__":
    p, q, n = 3, 8, 4
    W_values = np.arange(0, 40, 2)

    alphas = []
    valid_Ws = []

    for W in tqdm(W_values, desc="W sweep"):
        print(f"Running W = {W}")
        result = p_q_alpha(p, q, n=n, W=W, t_max=7, plot_fit=True)
        if result is not None:
            alpha, ave_time, ave_ipr = result
            alphas.append(alpha)
            valid_Ws.append(W)
        else:
            logger.warning(f"Skipping W={W}: simulation failed")

    plt.figure()
    plt.plot(valid_Ws, alphas, marker='o')
    plt.xlabel("W (disorder strength)")
    plt.ylabel("power-law exponent alpha")
    plt.title(f"alpha vs W (p={p}, q={q}, n={n})")
    plt.savefig(os.path.join(PLOT_DIR, f"alpha_vs_W_p{p}_q{q}_n{n}.png"))
    plt.close()

# if __name__ == "__main__":
    # p, q, n = 3, 8, 4
    # W_values = np.arange(0, 41, 5)  # 0, 5, 10, ..., 40 — adjust step for resolution vs. runtime

    # alphas = []
    # valid_Ws = []
    # ipr_curves = {}  # W -> (time, ipr)

    # for W in W_values:
    #     print(f"Running W = {W}")
    #     result = p_q_alpha(p, q, n=n, W=W)
    #     if result is not None:
    #         alpha, ave_time, ave_ipr = result
    #         alphas.append(alpha)
    #         valid_Ws.append(W)
    #         ipr_curves[W] = (ave_time, ave_ipr)
    #     else:
    #         logger.warning(f"Skipping W={W}: simulation failed")

    # # --- Plot 1: alpha vs W ---
    # plt.figure()
    # plt.plot(valid_Ws, alphas, marker='o')
    # plt.xlabel("W (disorder strength)")
    # plt.ylabel("power-law exponent alpha")
    # plt.title(f"alpha vs W (p={p}, q={q}, n={n})")
    # plt.show()

    # # --- Plot 2: IPR vs t, one curve per W ---
    # plt.figure()
    # for W, (time, ipr) in ipr_curves.items():
    #     plt.plot(time, ipr, marker='o', label=f"W={W}")
    # plt.xlabel("t")
    # plt.ylabel("IPR")
    # plt.title(f"IPR vs t (p={p}, q={q}, n={n})")
    # plt.legend()
    # plt.show()
    # alpha, ave_time, ave_ipr = p_q_alpha(p, q, n, W=40.0, t_max=7, plot_fit=True)