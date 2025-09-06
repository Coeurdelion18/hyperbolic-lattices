# file: run_gpu_parallel_nobins.py
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from pbc_adj import build_hyperbolic_lattice

# Parameters
p, q, n = 3, 7, 5
Nc = n
k = q
m = p
seed = 1

Nds = 1500             # number of disorder realizations
Ws = 4 + np.arange(1, 30) * 4  # as you used earlier (29 values)
# Ws = np.arange(10,120,5)      # alternative

N_WORKERS = 1         # set to number of GPUs you will assign
GPU_IDS = list(range(N_WORKERS))  # map worker -> GPU id (0..NGPUS-1)

# Build adjacency (on CPU)
adj, G, circles = build_hyperbolic_lattice(Nc=Nc, k=k, m=m, seed=seed)
adj_matrix = np.array(adj, dtype=np.float64)
size = adj_matrix.shape[0]

# Worker function (runs in separate process)
def single_realization_worker(args):
    """
    args = (seed_int, worker_index)
    Each worker sets CUDA_VISIBLE_DEVICES to its assigned GPU id,
    uses cupy for computations and returns arrays of shape (len(Ws), N-2)
    """
    seed_int, worker_idx = args

    # assign GPU to this process
    gpu_id = GPU_IDS[worker_idx % len(GPU_IDS)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # import cupy inside the worker after setting environment
    import cupy as cp

    rng = cp.random.RandomState(seed_int)

    rs_local = []
    ipr_local = []

    # Move adjacency to GPU once
    A_gpu = cp.asarray(adj_matrix)

    for W in Ws:
        # construct H on GPU
        diag_vals = rng.uniform(-W/2.0, W/2.0, size).astype(A_gpu.dtype)
        H_gpu = A_gpu.copy()
        # add diagonal
        H_gpu += cp.diag(diag_vals)

        # eigendecomposition on GPU (symmetric real)
        # cupy.linalg.eigh returns w (eigs) and v (columns = eigenvectors)
        w_gpu, v_gpu = cp.linalg.eigh(H_gpu)

        # convert to CPU numpy for further consistent aggregation (or keep in GPU and cp.asnumpy at end)
        w = cp.asnumpy(w_gpu)
        v = cp.asnumpy(v_gpu)

        # sort
        idx = np.argsort(w)
        w = w[idx]
        v = v[:, idx]

        # spacings and r-values
        gd = np.diff(w)               # length N-1
        r_vals = gd[:-1] / gd[1:]
        r_vals = np.minimum(r_vals, 1.0 / r_vals)  # length N-2

        # IPRs
        # eigenvectors are columns; compute IPR per eigenvector
        iprs = np.sum(np.abs(v)**4, axis=0)  # length N
        iprs = iprs[1:-1]  # trim to length N-2

        rs_local.append(r_vals)
        ipr_local.append(iprs)

    rs_local = np.array(rs_local)   # shape (len(Ws), N-2)
    ipr_local = np.array(ipr_local)

    return rs_local, ipr_local

# Launch pool: parallel over disorder realizations
if __name__ == "__main__":
    args_list = [ (1000 + i, i % N_WORKERS) for i in range(Nds) ]  # seeds and worker idx
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(single_realization_worker, args_list)

    # results is list of (rs_local, ipr_local) for each realization
    rs_stack = np.stack([res[0] for res in results], axis=0)   # shape (Nds, len(Ws), N-2)
    ipr_stack = np.stack([res[1] for res in results], axis=0)

    # Average over disorder realizations (axis=0)
    rs_mean = np.mean(rs_stack, axis=0)    # shape (len(Ws), N-2)
    ipr_mean = np.mean(ipr_stack, axis=0)

    # Save (transpose to shape (N-2, len(Ws)) for plotting convention)
    np.save("rs_mean_nobins.npy", rs_mean.T)
    np.save("ipr_mean_nobins.npy", ipr_mean.T)

    print("Saved rs_mean_nobins.npy and ipr_mean_nobins.npy")
