"""
aggregate_heatmap.py 
collects HPC heatmap array results into a dict

to run locally after HPC jobs have finished use:
    python aggregate_heatmap.py

constructs the dict in the same format as run_gaussian_approx_2d_sweep() so it can
be passed directly to plot_gaussian_approx_2d_heatmap()
"""

import numpy as np
import os
import robust_ssm as ss

NU_GRID = [3, 5, 8, 12, 20, 50, 100, 300]
N_GRID  = len(NU_GRID) # length of each heatmap grid side
N_TOTAL = N_GRID ** 2 # n parameter pairs to set up

rmse_smc = np.full((N_GRID, N_GRID), np.nan)
rmse_sd  = np.full((N_GRID, N_GRID), np.nan)
rmse_kal = np.full((N_GRID, N_GRID), np.nan)

missing = [] # to collect names of missing files
for idx in range(N_TOTAL):
    path = f"figs/heatmap/heatmap_{idx}.npz"
    if not os.path.exists(path):
        missing.append(idx)
        continue
    d = np.load(path)
    i = idx // N_GRID
    j = idx  % N_GRID
    rmse_smc[i, j] = float(d["rmse_smc"])
    rmse_sd [i, j] = float(d["rmse_sd"])
    rmse_kal[i, j] = float(d["rmse_kal"])

if missing:
    print(f"WARNING: missing grid indices {missing}")

rel_gap_sd  = (rmse_sd  - rmse_smc) / rmse_smc
rel_gap_kal = (rmse_kal - rmse_smc) / rmse_smc

sweep = dict(
    nu_eta_list = NU_GRID,
    nu_eps_list = NU_GRID,
    rmse_smc    = rmse_smc,
    rmse_sd     = rmse_sd,
    rmse_kalman = rmse_kal,
    rel_gap_sd  = rel_gap_sd,
    rel_gap_kal = rel_gap_kal,
)

np.savez("figs/heatmap/heatmap_aggregated.npz", **{k: v for k, v in sweep.items()
                                                        if not isinstance(v, list)},
         nu_eta_list=NU_GRID, nu_eps_list=NU_GRID)
print("saved to figs/heatmap/heatmap_aggregated.npz")

print("\ngap summary (Kalman vs SMC):")
for i, nu_eta in enumerate(NU_GRID):
    row = " ".join(f"{rel_gap_kal[i,j]*100:5.1f}%" for j in range(N_GRID))
    print(f"  nu_eta={nu_eta:>4}:  {row}")
print(f"          nu_eps:  " + "  ".join(f"{v:>5}" for v in NU_GRID))

# Optionally plot immediately
import matplotlib.pyplot as plt
figs = ss.plot_gaussian_approx_2d_heatmap(sweep, threshold=0.05)
for fig, name in zip(figs, ["kal_gap", "sd_gap", "safezones", "contours"]):
    fig.savefig(f"figs/gaussian_approx_2d_heatmap_{name}.png", dpi=150, bbox_inches="tight")
print("saved updated heatmap figures.")