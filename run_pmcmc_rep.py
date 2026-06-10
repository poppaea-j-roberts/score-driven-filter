"""
run_pmcmc_rep.py  —  single PMCMC MC rep for HPC array job

Usage:
    python run_pmcmc_rep.py <rep_idx>

This script is to be called via PBS array job with $PBS_ARRAY_INDEX as rep_idx.
Results saved to figs/pmcmc_rep_<rep_idx>.npz for aggregation afterwards.

Four chains per rep:
    1. φ  free  (marginal)
    2. σ  free  (marginal)
    3. ν  free  (marginal)
    4. (φ, σ, ν) free  (joint blocked MWG — 3× filter calls per iteration)

Tuned settings (reparameterise=False, resample_threshold=1.0, N=1000):
    phi_sd   = 0.06523   (tuned from notebook)
    sigma_sd = 0.08047   (tuned from notebook)
    nu_sd    = 1.75625  (tuned from notebook)
"""

import sys
import numpy as np
from dataclasses import replace
import robust_ssm as ss

# settings
REP_IDX  = int(sys.argv[1])
SEED     = 42
N_ITER   = 2000
N_PART   = 1000
BURN     = N_ITER // 2
PHI_SD   = 0.06523   # tuned with N=1000, resample_threshold=1.0, reparameterise=False
SIGMA_SD = 0.08047
NU_SD    = 1.75625   # tuned with N=1000, resample_threshold=1.0, reparameterise=False

TRUE = dict(phi=0.8, sigma=1.0, nu=8.0, w=1.0)

# simulate one dataset
p_base = ss.ModelParams(
    phi=TRUE['phi'], sigma=TRUE['sigma'],
    nu=TRUE['nu'], w=TRUE['w'],
    T=300, N=N_PART, seed=SEED + REP_IDX,
)
rng_i  = np.random.default_rng(SEED + REP_IDX)
p_i    = replace(p_base, seed=SEED + REP_IDX)
_, y_i = ss.simulate(p_i, rng_i)

# chain 1: φ free
res_phi = ss.pmcmc_inference(
    y_i, p_i,
    n_iter=N_ITER,
    free=('phi',),
    proposal_sd={'phi': PHI_SD},
    n_particles=N_PART,
    seed=SEED + REP_IDX,
    reparameterise=False,
)

# chain 2: σ free
res_sigma = ss.pmcmc_inference(
    y_i, p_i,
    n_iter=N_ITER,
    free=('sigma',),
    proposal_sd={'sigma': SIGMA_SD},
    n_particles=N_PART,
    seed=SEED + REP_IDX,
    reparameterise=False,
)

# chain 3: ν free
res_nu = ss.pmcmc_inference(
    y_i, p_i,
    n_iter=N_ITER,
    free=('nu',),
    proposal_sd={'nu': NU_SD},
    n_particles=N_PART,
    seed=SEED + REP_IDX,
    reparameterise=False,
)

# chain 4: joint (φ, σ, ν) — blocked MWG, one MH step per param per iteration
res_joint = ss.pmcmc_inference(
    y_i, p_i,
    n_iter=N_ITER,
    free=('phi', 'sigma', 'nu'),
    proposal_sd={'phi': PHI_SD, 'sigma': SIGMA_SD, 'nu': NU_SD},
    n_particles=N_PART,
    seed=SEED + REP_IDX,
    reparameterise=False,
)

# post-burn-in samples
phi_samples      = res_phi['samples']['phi'][BURN:]
sigma_samples    = res_sigma['samples']['sigma'][BURN:]
nu_samples       = res_nu['samples']['nu'][BURN:]
joint_phi        = res_joint['samples']['phi'][BURN:]
joint_sigma      = res_joint['samples']['sigma'][BURN:]
joint_nu         = res_joint['samples']['nu'][BURN:]

np.savez(
    f'figs/pmcmc_rep_{REP_IDX}.npz',
    phi_samples        = phi_samples,
    sigma_samples      = sigma_samples,
    nu_samples         = nu_samples,
    joint_phi          = joint_phi,
    joint_sigma        = joint_sigma,
    joint_nu           = joint_nu,
    phi_mean           = phi_samples.mean(),
    sigma_mean         = sigma_samples.mean(),
    nu_mean            = nu_samples.mean(),
    joint_phi_mean     = joint_phi.mean(),
    joint_sigma_mean   = joint_sigma.mean(),
    joint_nu_mean      = joint_nu.mean(),
    phi_accept_rate    = res_phi['accept_rate'],
    sigma_accept_rate  = res_sigma['accept_rate'],
    nu_accept_rate     = res_nu['accept_rate'],
    joint_accept_rate  = res_joint['accept_rate'],
)

print(f"Rep {REP_IDX} done.")
print(f"  φ̄={phi_samples.mean():.3f} (accept={res_phi['accept_rate']:.3f})  "
      f"σ̄={sigma_samples.mean():.3f} (accept={res_sigma['accept_rate']:.3f})  "
      f"ν̄={nu_samples.mean():.3f} (accept={res_nu['accept_rate']:.3f})")
print(f"  joint: φ̄={joint_phi.mean():.3f}  σ̄={joint_sigma.mean():.3f}  ν̄={joint_nu.mean():.3f}  "
      f"(accept={res_joint['accept_rate']:.3f})")
