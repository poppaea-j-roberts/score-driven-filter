"""
run_heatmap_rep.py is one (nu_eta, nu_eps) grid point for the 2D Kalman heatmap

Usage:
    python run_heatmap_rep.py <grid_idx>

Called via PBS array job with $PBS_ARRAY_INDEX as grid_idx.
Grid: 8x8 = 64 points indexed 0-63.
    grid_idx = i * 8 + j  →  nu_eta = NU_GRID[i], nu_eps = NU_GRID[j]

Results saved to figs/heatmap/heatmap_<grid_idx>.npz for aggregation by
aggregate_heatmap.py.
"""

import sys
import numpy as np
from dataclasses import replace
import robust_ssm as ss

GRID_IDX = int(sys.argv[1])

NU_GRID   = [3, 5, 8, 12, 20, 50, 100, 300]
N_GRID    = len(NU_GRID)              # 8
N_REPS    = 100                       # replications per grid point
T         = 1000                      # series length
N_PART    = 500                       # SMC particles
SEED_BASE = 7777

i        = GRID_IDX // N_GRID
j        = GRID_IDX  % N_GRID
nu_eta   = float(NU_GRID[i])
nu_eps   = float(NU_GRID[j])

p_base = ss.DualNuParams(
    phi=0.8, sigma=1.0, w=1.0,
    nu_eta=nu_eta, nu_eps=nu_eps,
    T=T, N=N_PART,
)

smc_reps, sd_reps, kal_reps = [], [], []

for rep in range(N_REPS):
    seed = SEED_BASE + GRID_IDX * 1000 + rep
    p    = replace(p_base, seed=seed)

    rng     = np.random.default_rng(seed)
    mu, y   = ss.simulate_dual(p, rng)

    rng_smc = np.random.default_rng(seed + 100_000)
    mu_smc, _, _ = ss.bootstrap_filter_dual(y, p, rng_smc)
    mu_kal, _    = ss.kalman_filter_corrected(y, p)

    mp_sd  = ss.ModelParams(
        phi=p.phi, sigma=p.sigma, w=p.w, mu_0=p.mu_0,
        nu=p.nu_eps, T=p.T, N=p.N, seed=seed,
    )
    kappa_ml = ss.estimate_sd_params_ml(y, mp_sd, free=('kappa',))['kappa']
    mu_sd    = ss.score_driven_filter(y, replace(mp_sd, kappa=kappa_ml))

    mu_true = mu[1:]
    smc_reps.append(np.sqrt(np.mean((mu_smc - mu_true) ** 2)))
    sd_reps .append(np.sqrt(np.mean((mu_sd  - mu_true) ** 2)))
    kal_reps.append(np.sqrt(np.mean((mu_kal - mu_true) ** 2)))

rmse_smc = float(np.mean(smc_reps))
rmse_sd  = float(np.mean(sd_reps))
rmse_kal = float(np.mean(kal_reps))

import os
os.makedirs("figs/heatmap", exist_ok=True)

np.savez(
    f"figs/heatmap/heatmap_{GRID_IDX}.npz",
    grid_idx = GRID_IDX,
    nu_eta   = nu_eta,
    nu_eps   = nu_eps,
    rmse_smc = rmse_smc,
    rmse_sd  = rmse_sd,
    rmse_kal = rmse_kal,
    rel_gap_sd  = (rmse_sd  - rmse_smc) / rmse_smc,
    rel_gap_kal = (rmse_kal - rmse_smc) / rmse_smc,
)

print(f"[{GRID_IDX:>2}] nu_eta={nu_eta:>5}  nu_eps={nu_eps:>5}  "
      f"SMC={rmse_smc:.4f}  SD={rmse_sd:.4f}  Kal={rmse_kal:.4f}  "
      f"gap_kal={100*(rmse_kal-rmse_smc)/rmse_smc:.1f}%")
