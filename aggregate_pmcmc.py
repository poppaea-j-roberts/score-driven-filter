"""
aggregate_pmcmc.py
collects HPC array job results into summary arrays

to run locally after all PBS jobs have finished:
    python aggregate_pmcmc.py
"""

import numpy as np
import os

N_REPS = 100
phi_means, sigma_means, nu_means = [], [], []
joint_phi_means, joint_sigma_means, joint_nu_means = [], [], []
phi_rates, sigma_rates, nu_rates, joint_rates = [], [], [], []
missing = [] # collect names of missing runs

for i in range(N_REPS):
    path = f'figs/hpc_runs/pmcmc_rep_{i}.npz'
    if not os.path.exists(path):
        missing.append(i)
        continue
    d = np.load(path)
    phi_means.append(float(d['phi_mean']))
    sigma_means.append(float(d['sigma_mean']))
    nu_means.append(float(d['nu_mean']))
    joint_phi_means.append(float(d['joint_phi_mean']))
    joint_sigma_means.append(float(d['joint_sigma_mean']))
    joint_nu_means.append(float(d['joint_nu_mean']))
    phi_rates.append(float(d['phi_accept_rate']))
    sigma_rates.append(float(d['sigma_accept_rate']))
    nu_rates.append(float(d['nu_accept_rate']))
    joint_rates.append(float(d['joint_accept_rate']))

if missing:
    print(f"WARNING: missing reps {missing}")

phi_means       = np.array(phi_means)
sigma_means     = np.array(sigma_means)
nu_means        = np.array(nu_means)
joint_phi_means = np.array(joint_phi_means)
joint_sigma_means = np.array(joint_sigma_means)
joint_nu_means  = np.array(joint_nu_means)

TRUE = dict(phi=0.8, sigma=1.0, nu=8.0)

print("marginal chains:")
for name, means, true_val in [
    ('φ', phi_means,   TRUE['phi']),
    ('σ', sigma_means, TRUE['sigma']),
    ('ν', nu_means,    TRUE['nu']),
]:
    print(f"  {name}   mean={means.mean():.4f}  std={means.std():.4f}  "
          f"bias={means.mean()-true_val:.4f}  (true={true_val})")

print("\njoint chain (φ, σ, ν):")
for name, means, true_val in [
    ('φ', joint_phi_means,   TRUE['phi']),
    ('σ', joint_sigma_means, TRUE['sigma']),
    ('ν', joint_nu_means,    TRUE['nu']),
]:
    print(f"  {name}   mean={means.mean():.4f}  std={means.std():.4f}  "
          f"bias={means.mean()-true_val:.4f}  (true={true_val})")

print(f"\nmean acceptance rates:")
print(f"  φ (marginal)={np.mean(phi_rates):.3f}  "
      f"σ (marginal)={np.mean(sigma_rates):.3f}  "
      f"ν (marginal)={np.mean(nu_rates):.3f}  "
      f"joint={np.mean(joint_rates):.3f}")

np.savez('figs/hpc_runs/pmcmc_aggregated.npz',
         phi_means=phi_means, sigma_means=sigma_means, nu_means=nu_means,
         joint_phi_means=joint_phi_means, joint_sigma_means=joint_sigma_means,
         joint_nu_means=joint_nu_means,
         phi_rates=phi_rates, sigma_rates=sigma_rates,
         nu_rates=nu_rates, joint_rates=joint_rates)
print("\nsaved to figs/hpc_runs/pmcmc_aggregated.npz")