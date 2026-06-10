# A Score-Driven Approximation for a Robust Dynamical Model
Simulation study relating to StatML MP2. Evaluating a score-driven filter as an approximation to the optimal filter for a robust state-space model in which state innovations are Sub-Gaussian and observation noise is Student-t distributed. We compare the score-driven filter against Sequential Monte Carlo (SMC) as a theoretical best, perform parameter inference under both likelihoods, and analyze how the model converges to the standard linear-Gaussian SSM as the degrees-of-freedom parameters grow.

---

## Repository Structure

```
.
├── robust_ssm.py                   # shared functions: model, filters, likelihood, PMCMC
├── aggregate_pmcmc.py              # collects HPC PMCMC outputs in figs/hpc_runs/
├── aggregate_heatmap.py            # collects HPC heatmap outputs in figs/heatmap/
│
├── run_pmcmc_rep.py                # HPC run: one PMCMC replication
├── run_pmcmc_array.pbs             # HPC PBS array job (100 replications)
├── run_heatmap_rep.py              # HPC run: one (nu_eps, nu_eta) grid point
├── run_heatmap_array.pbs           # HPC PBS array job (64 grid points × 100 reps)
│
├── subgaussian_verification.ipynb  # Sub-Gaussian distribution, score function, shape equivalence
├── filtering.ipynb                 # SD vs SMC filtering paths and MC RMSE sweep
├── inference.ipynb                 # profile likelihoods, SD MLE study, PMCMC results
├── gaussian_approx_analysis.ipynb  # Gaussian-approximation studies including heatmaps
└── diagnostics.ipynb               # additional filter and chain diagnostics
```

---

## Reproducing Results

### Notebooks

All notebooks share `robust_ssm.py` as the core library. Figures are written to `figs/`.

### HPC jobs

The PMCMC inference study and the 2-D Gaussian-approximation heatmap were run as PBS array jobs on the Imperial HPC cluster.

**PMCMC inference (100 replications):**
```bash
qsub run_pmcmc_array.pbs
python aggregate_pmcmc.py     # collects figs/hpc_runs/pmcmc_rep_*.npz
```

**Gaussian-approximation heatmap (8×8 grid, 100 reps per point):**
```bash
qsub run_heatmap_array.pbs
python aggregate_heatmap.py   # collects figs/heatmap/heatmap_*.npz
```

---

## Dependencies

Python 3.11+:

```
numpy
scipy
matplotlib
pandas
jupyter
```
