import os

# Must be set BEFORE numpy/scipy are imported, or their BLAS backend
# (OpenBLAS/MKL) may already have initialized its own thread pool.
# Without this, each joblib worker process spawns its own internal BLAS
# threads too, oversubscribing the allocated cores and silently slowing
# things down instead of speeding them up.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import logging
import contextlib

import numpy as np
import joblib
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import linregress
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # headless backend -- no DISPLAY on a compute node
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import hypertiling as ht

hbar = 1.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                     filemode='w', filename='randomwalk_disordered.log')
logger = logging.getLogger(__name__)

PLOT_DIR = "plots_disordered"
os.makedirs(PLOT_DIR, exist_ok=True)

SPREAD_PLOT_DIR = os.path.join(PLOT_DIR, "spread_vs_t")
ENERGY_PLOT_DIR = os.path.join(PLOT_DIR, "energy_vs_t")
IPR_PLOT_DIR = os.path.join(PLOT_DIR, "ipr_vs_t")
ALPHA_PLOT_DIR = os.path.join(PLOT_DIR, "alpha_vs_W")
FIT_PLOT_DIR = os.path.join(PLOT_DIR, "powerlaw_fits")

for plot_dir in [SPREAD_PLOT_DIR, ENERGY_PLOT_DIR, IPR_PLOT_DIR,
                 ALPHA_PLOT_DIR, FIT_PLOT_DIR]:
    os.makedirs(plot_dir, exist_ok=True)

Nds = 100          # number of disorder realizations

# Read the actual SLURM core allocation rather than autodetecting the node's
# full core count (-1 in joblib can overshoot what was actually granted by
# `-c` if the physical node has more cores than the job's allocation).
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))


# ---------------------------------------------------------------------------
# Geometry / Hamiltonian construction (unchanged math, built once and reused)
# ---------------------------------------------------------------------------

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

    return H_clean, T


def compute_r2(T, site=0):
    """
    Squared Poincare distance of every site from `site` (the walk's origin).
    Depends only on the tiling geometry T -- identical for every disorder
    realization and every W, so this is computed once and reused everywhere,
    instead of being recomputed inside the Nds loop like in the original code.
    """
    coords = [T.get_center(i) for i in range(len(T))]
    r2 = np.empty(len(coords))
    for i, c in enumerate(coords):
        d = ht.distance.poincare_distance(c, coords[site])
        r2[i] = d**2
    return r2


def add_disorder(H_clean, W, rng=None):
    """
    Note: only needs H_clean.shape (not the HyperbolicTiling object T) to know
    the system size, so worker processes only ever need to receive H_clean,
    not T -- keeps inter-process pickling cheap during parallel sweeps.
    """
    if rng is None:
        rng = np.random.default_rng()
    size = H_clean.shape[0]
    disorder_vector = rng.uniform(-W / 2.0, W / 2.0, size=size)
    H_disordered = H_clean + sp.diags(disorder_vector, format='csr')
    return H_disordered


# ---------------------------------------------------------------------------
# Time evolution -- batched over the whole time grid in one expm_multiply call
# instead of one independent expm_multiply call per timestep (was O(num_t)
# redundant matrix-exponential computations, now O(1))
# ---------------------------------------------------------------------------

def evolve_trajectory_sparse(psi0, H, t_max, dt=0.05):
    times = np.arange(0, t_max, dt)
    num = len(times)

    psi_traj = expm_multiply(-1j * H / hbar, psi0, start=0, stop=t_max,
                              num=num, endpoint=False)

    norms = np.linalg.norm(psi_traj, axis=1, keepdims=True)
    psi_traj = psi_traj / norms

    return times, psi_traj


def std_dev_with_t_sparse(t_max, H, r2, dt=0.05):
    num_sites = H.shape[0]
    psi0 = np.zeros(num_sites, dtype=complex)
    psi0[0] = 1.0  # start at origin

    times, psi_traj = evolve_trajectory_sparse(psi0, H, t_max, dt=dt)

    prob = np.abs(psi_traj) ** 2
    prob /= prob.sum(axis=1, keepdims=True)

    iprs = (prob ** 2).sum(axis=1)
    stds = np.sqrt((prob * r2).sum(axis=1))

    Hpsi = (H @ psi_traj.T).T
    energies = np.real(np.einsum('ti,ti->t', psi_traj.conj(), Hpsi))

    return times, stds, energies, iprs


# ---------------------------------------------------------------------------
# tqdm + joblib: by default tqdm wrapping a joblib input generator only
# tracks task *dispatch*, not task *completion* (joblib pre-dispatches
# batches ahead of actual results), so the bar jumps to roughly 2*n_jobs
# almost instantly and then crawls. This patches joblib's completion
# callback so the bar advances once a task actually finishes -- gives a
# true wall-clock progress readout, which matters for judging how far
# into a multi-day job you are.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ---------------------------------------------------------------------------
# One disorder realization -- this is the unit of work parallelized via joblib
# ---------------------------------------------------------------------------

def run_one_realization(H_clean, r2, W, t_max, seed=None):
    rng = np.random.default_rng(seed)
    H_disordered = add_disorder(H_clean, W, rng=rng)
    return std_dev_with_t_sparse(t_max, H_disordered, r2)


def run_disorder_ensemble(H_clean, r2, W, t_max, n_jobs=N_JOBS, show_progress=True):
    """
    Runs Nds independent disorder realizations in parallel and returns the
    ensemble averages -- replaces the original serial `for _ in tqdm(range(Nds))`
    loop. Each worker gets an independent, non-overlapping RNG stream via
    SeedSequence.spawn, so parallel randomness stays statistically sound.
    """
    seed_seq = np.random.SeedSequence()
    child_seeds = seed_seq.spawn(Nds)

    # disable=not sys.stderr.isatty(): shows a live bar interactively, but
    # skips the constant \r-laden output when stderr is redirected to a
    # SLURM .err file (where it can't render in place anyway).
    pbar = tqdm(desc=f"W={W}", total=Nds, disable=not (show_progress and sys.stderr.isatty()))

    with tqdm_joblib(pbar):
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one_realization)(H_clean, r2, W, t_max, seed=s)
            for s in child_seeds
        )

    times_list, stds_list, energies_list, iprs_list = zip(*results)

    ave_time = np.mean(times_list, axis=0)
    ave_std = np.mean(stds_list, axis=0)
    ave_energy = np.mean(energies_list, axis=0)
    ave_ipr = np.mean(iprs_list, axis=0)

    return ave_time, ave_std, ave_energy, ave_ipr


# ---------------------------------------------------------------------------
# Linear-region slope estimate (unchanged)
# ---------------------------------------------------------------------------

def linear_region_study(times, stds, r2_threshold=0.99):
    times = np.array(times)
    stds = np.array(stds)

    best_slope = None
    best_intercept = None
    limit_idx = 2

    for i in range(3, len(times)):
        slope, intercept, r_value, p_value, std_err = linregress(times[:i], stds[:i])
        r2 = r_value ** 2

        if r2 >= r2_threshold:
            best_slope = slope
            best_intercept = intercept
            limit_idx = i - 1
        else:
            break

    if best_slope is None:
        slope, intercept, r_value, p_value, std_err = linregress(times[:2], stds[:2])
        best_slope, best_intercept = slope, intercept
        limit_idx = 1

    return limit_idx, times[limit_idx], best_slope, best_intercept


# ---------------------------------------------------------------------------
# Power-law exponent fit in the initial (pre-saturation) region
# ---------------------------------------------------------------------------

def fit_powerlaw_exponent(x, y, t_min=None, t_max=None, auto_range=True,
                           min_points=5, r2_threshold=0.995, plot=True,
                           plot_tag=""):
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

        ax = axes[0]
        ax.plot(x, y, 'o', color='tab:blue', ms=4, label='data')
        ax.plot(x[fit_mask], y[fit_mask], 'o', color='tab:red', ms=5, label='fit region')
        ax.plot(x_fit_line, y_fit_line, '-', color='black', lw=1.5, label=f'fit: y ~ t^{alpha:.3f}')
        ax.set_xlabel('t'); ax.set_ylabel('y'); ax.set_title('Linear scale'); ax.legend()

        ax = axes[1]
        ax.loglog(x, y, 'o', color='tab:blue', ms=4, label='data')
        ax.loglog(x[fit_mask], y[fit_mask], 'o', color='tab:red', ms=5, label='fit region')
        ax.loglog(x_fit_line, y_fit_line, '-', color='black', lw=1.5,
                  label=f'alpha={alpha:.3f}, R2={result["r2"]:.4f}')
        ax.set_xlabel('t'); ax.set_ylabel('y'); ax.set_title('Log-log scale'); ax.legend()

        plt.tight_layout()
        fname = f"powerlaw_fit{('_' + plot_tag) if plot_tag else ''}.png"
        plt.savefig(os.path.join(FIT_PLOT_DIR, fname))
        plt.close()
    # ---------------------------------------------------------------------

    return result


# ---------------------------------------------------------------------------
# High-level sweep functions
# ---------------------------------------------------------------------------

def p_q_slope(p, q, n=8, W=40.0, t_max=7, H_clean=None, T=None, r2=None):
    """
    Kept for backward compatibility with the original interface. Accepts
    precomputed H_clean/T/r2 to avoid rebuilding tiling geometry on every
    call (e.g. across a W sweep, where geometry doesn't depend on W).
    """
    if (p - 2) * (q - 2) > 4:
        try:
            if H_clean is None or T is None:
                H_clean, T = hyperbolic_adjacency_sparse(p, q, n)
            if r2 is None:
                r2 = compute_r2(T)

            ave_time, ave_std, ave_energy, ave_ipr = run_disorder_ensemble(
                H_clean, r2, W, t_max)

            _, lin_time, m, _ = linear_region_study(ave_time, ave_std)

            plt.figure()
            plt.plot(ave_time, ave_std, marker='o')
            plt.xlabel("t"); plt.ylabel("std dev of spread")
            plt.title(f"Spread vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(SPREAD_PLOT_DIR, f"spread_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            plt.figure()
            plt.plot(ave_time, ave_energy, marker='o')
            plt.xlabel("t"); plt.ylabel("energy profile")
            plt.title(f"Energy vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(ENERGY_PLOT_DIR, f"energy_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            plt.figure()
            plt.plot(ave_time, ave_ipr, marker='o')
            plt.xlabel("t"); plt.ylabel("ipr profile")
            plt.title(f"IPR vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(IPR_PLOT_DIR, f"ipr_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            logger.info(f"Computed slope for p={p}, q={q}: {m}, linear time: {lin_time}")
            return m, lin_time, ave_energy, ave_ipr

        except Exception as e:
            logger.error(f"Simulation failed for p={p}, q={q} due to: {e}")
            pass

    return None


def p_q_alpha(p, q, n=8, W=40.0, t_max=7, plot_fit=False, H_clean=None, T=None, r2=None):
    """
    Accepts precomputed H_clean/T/r2 so a W-sweep can build the tiling
    geometry once and reuse it across every W value, instead of rebuilding
    it inside every call like the original.
    """
    if (p - 2) * (q - 2) > 4:
        try:
            if H_clean is None or T is None:
                H_clean, T = hyperbolic_adjacency_sparse(p, q, n)
            if r2 is None:
                r2 = compute_r2(T)

            ave_time, ave_std, ave_energy, ave_ipr = run_disorder_ensemble(
                H_clean, r2, W, t_max)

            fit_result = fit_powerlaw_exponent(ave_time, ave_std, plot=plot_fit,
                                                plot_tag=f"p{p}_q{q}_n{n}_W{W}")
            alpha = fit_result['alpha']

            logger.info(f"Computed alpha for p={p}, q={q}, W={W}: {alpha} "
                        f"(R2={fit_result['r2']:.4f})")

            plt.figure()
            plt.plot(ave_time, ave_ipr, marker='o')
            plt.xlabel("t"); plt.ylabel("ipr profile")
            plt.title(f"IPR vs t (p={p}, q={q}, n={n}, W={W})")
            plt.savefig(os.path.join(IPR_PLOT_DIR, f"ipr_vs_t_p{p}_q{q}_n{n}_W{W}.png"))
            plt.close()

            return alpha, ave_time, ave_ipr

        except Exception as e:
            logger.error(f"Simulation failed for p={p}, q={q}, W={W} due to: {e}")
            return None

    return None


CHECKPOINT_PATH = os.path.join(PLOT_DIR, "checkpoint.npz")


def load_checkpoint():
    """
    Returns (valid_Ws, alphas, ipr_curves) from a prior run, or empty
    containers if no checkpoint exists yet. Lets the sweep resume after a
    wall-time kill instead of starting over.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return [], [], {}

    data = np.load(CHECKPOINT_PATH, allow_pickle=True)
    valid_Ws = list(data["valid_Ws"])
    alphas = list(data["alphas"])
    ipr_curves = {}
    for W in valid_Ws:
        ipr_curves[W] = (data[f"time_W{W}"], data[f"ipr_W{W}"])
    return valid_Ws, alphas, ipr_curves


def save_checkpoint(valid_Ws, alphas, ipr_curves):
    payload = {"valid_Ws": np.array(valid_Ws), "alphas": np.array(alphas)}
    for W, (time, ipr) in ipr_curves.items():
        payload[f"time_W{W}"] = time
        payload[f"ipr_W{W}"] = ipr
    # write to a temp file then rename -- atomic on the same filesystem, so a
    # job killed mid-write never leaves a corrupted checkpoint behind.
    # NOTE: np.savez silently appends ".npz" if the filename doesn't already
    # end in it -- so the temp name must end in ".npz" itself, or the file
    # np.savez actually writes won't match the path os.replace looks for.
    tmp_path = CHECKPOINT_PATH.replace(".npz", ".tmp.npz")
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, CHECKPOINT_PATH)


def save_alpha_vs_W_plot(valid_Ws, alphas, p, q, n):
    plt.figure()
    plt.plot(valid_Ws, alphas, marker='o')
    plt.xlabel("W (disorder strength)")
    plt.ylabel("power-law exponent alpha")
    plt.title(f"alpha vs W (p={p}, q={q}, n={n})")
    plt.savefig(os.path.join(ALPHA_PLOT_DIR, f"alpha_vs_W_p{p}_q{q}_n{n}.png"))
    plt.close()


def save_combined_ipr_plot(ipr_curves, p, q, n):
    plt.figure()
    for W, (time, ipr) in sorted(ipr_curves.items()):
        plt.plot(time, ipr, marker='o', label=f"W={W}")
    plt.xlabel("t"); plt.ylabel("IPR")
    plt.title(f"IPR vs t (p={p}, q={q}, n={n})")
    plt.legend()
    plt.savefig(os.path.join(IPR_PLOT_DIR, f"ipr_vs_t_all_W_p{p}_q{q}_n{n}.png"))
    plt.close()


if __name__ == "__main__":
    p, q, n = 4, 5, 8
    W_values = np.arange(5, 201, 5)  # 5, 10, ..., 200

    logger.info(f"Using N_JOBS={N_JOBS}")

    # Build tiling geometry ONCE -- reused across every W value in the sweep,
    # instead of being rebuilt inside p_q_alpha on every call.
    H_clean, T = hyperbolic_adjacency_sparse(p, q, n)
    r2 = compute_r2(T)

    valid_Ws, alphas, ipr_curves = load_checkpoint()
    if valid_Ws:
        logger.info(f"Resuming from checkpoint -- already have W={valid_Ws}")

    remaining_Ws = [W for W in W_values if W not in valid_Ws]

    for W in remaining_Ws:
        logger.info(f"Running W = {W}")
        result = p_q_alpha(p, q, n=n, W=W, H_clean=H_clean, T=T, r2=r2)
        if result is not None:
            alpha, ave_time, ave_ipr = result
            alphas.append(alpha)
            valid_Ws.append(W)
            ipr_curves[W] = (ave_time, ave_ipr)

            # Save everything as soon as this W's result is available, so a
            # wall-time kill or crash partway through the sweep loses at most
            # the W currently in flight, not the whole run.
            save_checkpoint(valid_Ws, alphas, ipr_curves)
            save_alpha_vs_W_plot(valid_Ws, alphas, p, q, n)
            save_combined_ipr_plot(ipr_curves, p, q, n)
        else:
            logger.warning(f"Skipping W={W}: simulation failed")

    logger.info("Sweep complete.")