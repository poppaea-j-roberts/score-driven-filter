"""
robust_ssm.py
======================
Filtering of Robust Dynamical State Space Models

  State:       mu_t = phi * mu_{t-1} + eta_t,   eta_t ~ SubGaussian(0, Var(eta_t); nu)
  Observation: y_t  = mu_t + e_t,              e_t  ~ Student-t(0, sigma; nu)

  nu is the degrees of freedom which controls tail heaviness for both distributions
  eta_t is the state innovation 
  e_t is the observation noise

  Given x ~ t(0, w; nu), the sub-Gaussian innovation is:
      eta_t = x / (1 + x^2 / (nu * w^2))

Score-driven updating:
  The score of an observation y_t under the Student-t log-likelihood with location mu is proportional to
      x_t = (y_t - mu_hat_t) / (1 + (y_t - mu_hat_t)^2 / (nu * sigma^2))    *(later we apply rescaling to sigma)
  The filtered estimate is then:
      mu_hat_t = phi * mu_hat_{t-1} + kappa * x_t

  kappa is an unknown scaling parameter which is always estimated jointly
  with other parameters in a single optimisation and has no true value

Bootstrap particle filter (SMC) is implemented according to:
  Gordon, N.J., Salmond, D.J. & Smith, A.F.M. 
  Novel approach to nonlinear/non-Gaussian Bayesian state estimation
  https://www.ece.iastate.edu/~namrata/EE520/Gordonnovelapproach.pdf
  
  Likelihood for weighting: p(y_t | mu_t^(i)) = t(y_t; mu_t^(i), sigma; nu)
      = Gamma((nu+1)/2) / [Gamma(nu/2) * sqrt(nu*pi) * sigma]
        * (1 + (y_t - mu_t^(i))^2 / (nu * sigma^2))^(-(nu+1)/2) [https://www.mathworks.com/help/stats/t-location-scale-distribution.html]
   that is, the probability of the observation given the latent state is given by the probability of said
   observation under the student t-distirbution centred on the latent state and with proposed parameters.
"""
from __future__ import annotations

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize
from dataclasses import dataclass, replace
from typing import Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
#  Parameters
# ============================================================================
@dataclass
class ModelParams:
    """
    Model Parameters

    phi   : AR(1) coefficient (|phi| < 1 for stationarity)
    mu_0  : true initial state / intercept
    sigma : noise of the Student-t distributed observation noise
    w     : scale parameter for the t-distribution which underlies the sub-Gaussian innovations
    nu    : degrees of freedom — currently equal for state innovations and observations
    kappa : estimated by ML. Scales the score driven update
    T     : number of time steps
    N     : number of SMC particles
    seed  : random seed
    """
    phi:   float = 0.8
    mu_0:  float = 0.0
    sigma: float = 1.0
    w:     float = 1.0
    nu:    float = 8.0
    kappa: float | None = None
    T:     int   = 300
    N:     int   = 1000
    seed:  int   = 123


# ============================================================================
#  Sub-Gaussian Distribution Sampling Function
# ============================================================================
def sub_gaussian_rvs(
    size: int,
    scale: float,
    nu: float,
    rng: np.random.Generator,
    loc: float = 0.0,
) -> np.ndarray:
    """
    Draw samples from a sub-Gaussian distribution via the Student-t
    score transformation:
        x ~ t(0, w; nu)
        z = x / (1 + x^2 / (v * scale^2))

    Parameters
    ----------
    size  : number of sub-Gaussian samples
    scale : scale of the underlying Student-t (use w for state, sigma for obs)
    nu    : degrees of freedom
    rng   : numpy random Generator
    loc   : location shift added after the transform (default 0).
            Used when initialising the SMC prior centred at mu_0 rather
            than zero. See the prior draw in bootstrap_filter().
    """
    x = rng.standard_t(df=nu, size=size) * scale
    z = x / (1.0 + x**2 / (nu * scale**2))
    return loc + z


# ============================================================================
#  Simulation of Latent States and Observations
# ============================================================================
def simulate(p: ModelParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the sub-Gaussian AR(1) state process and Student-t observations.

    Model (1-indexed):
        State:       mu_t = phi * mu_{t-1} + eta_t,   eta_t ~ SubGaussian(0, w; nu)
        Observation: y_t  = mu_t + e_t,              e_t  ~ t(0, sigma; nu)

    The state array mu has shape (T+1,) to include the initial condition:
        mu[0] = mu_0  (initial state, set to p.mu_0; not observed)
        mu[t] = mu_t  for t = 1,...,T
    """
    innov = sub_gaussian_rvs(p.T, scale=p.w, nu=p.nu, rng=rng)
    mu = np.empty(p.T + 1)
    mu[0] = p.mu_0
    for t in range(1, p.T + 1):
        mu[t] = p.phi * mu[t - 1] + innov[t - 1]

    # sigma rescaling for identifiability
    e = rng.standard_t(df=p.nu, size=p.T) * p.sigma * np.sqrt((p.nu - 2.0) / p.nu)
    y = mu[1:] + e
    return mu, y


# ============================================================================
# Variance of sub-Gaussian
# ============================================================================
def sub_gaussian_variance(w: float, nu: float) -> float:
    """
    eta = w*Z / (1 + Z²/nu),  Z ~ t(0,1;nu).
    """
    return w**2 * nu**2 / ((nu + 1) * (nu + 3))


# ============================================================================
# Stationary scale of the AR(1) state process
# ============================================================================
def stationary_scale(p: ModelParams) -> float:
    """
    Stationary standard deviation of an AR(1) with sub-Gaussian innovations:

        sigma_stat = sqrt( Var(eta) / (1 - phi²) )
                   = w * nu / sqrt( (nu+1)(nu+3)(1-phi²) )

    Var(mu) = phi²*Var(mu) + Var(eta)
    (innovations are i.i.d. and independent of the past state)
    """
    sg_var = sub_gaussian_variance(p.w, p.nu)
    return float(np.sqrt(sg_var / max(1.0 - p.phi**2, 1e-8)))

# ============================================================================
#  Score-Driven Filter Implementation
# ============================================================================
def score_driven_filter(y: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    Score-driven filter for the sub-Gaussian SSM

    The score of log t(y_t; mu, sigma; nu) w.r.t. mu is proportional to the
    scaled-score innovation:
        x_t = (y_t - mu_pred) / (1 + (y_t - mu_pred)^2 / (nu * sigma^2))

    The predict-update recursion is performed:
        mu_pred  = phi * mu_hat_{t-1}
        x_t      = score innovation
        mu_hat_t = mu_pred + kappa * x_t
    """
    T     = p.T
    kappa = p.kappa
    mu_hat    = np.empty(T + 1)
    mu_hat[0] = p.mu_0

    for t in range(1, T + 1):
        mu_pred   = p.phi * mu_hat[t - 1] # propagation
        r         = y[t - 1] - mu_pred # prediction error
        x_t = r / (1.0 + r**2 / (p.sigma**2 * (p.nu - 2.0))) # calculate the score of for the observation, with sigma rescaled
        mu_hat[t] = mu_pred + kappa * x_t # update given by score scaled by estimated kappa

    return mu_hat[1:]


# ============================================================================
#  Bootstrap Particle Filtering as in "Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation" (Gordon et al. 1993)
# ============================================================================
def bootstrap_filter(
    y: np.ndarray,
    p: ModelParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SMC using the bootstrap filter

    We aim to approximate p(mu_t | y_{1:t}) for t = 1,...,T.

    Initialization:
      Draw mu_0^(i) ~ SubGaussian prior centred at mu_0, with scale equal to
      the correct sub-Gaussian stationary std from stationary_scale(p).

    For t = 1,...,T:
      Propagate:
              mu_t^(i) = phi * mu_{t-1}^(i) + eta_t^(i),  eta_t^(i) ~ SubGaussian(0, w; nu)

      Compute log-weights:
              log w_t^(i) = log p(y_t | mu_t^(i))
                           = log t(y_t; mu_t^(i), sigma; nu)
          LSE trick: Subtract max(log w) before exponentiating for numerical stability

      Normalise:  w_t^(i) <- w_t^(i) / sum_j w_t^(j)

      Compute filtered estimates:
              E[mu_t | y_{1:t}]   = sum_i w_t^(i) * mu_t^(i)
              Var[mu_t | y_{1:t}] = sum_i w_t^(i) * (mu_t^(i) - mean)^2

      Multinomial resampling:
          Draw u^j ~ U(0,1) independently for j = 1,...,N.
          Set new particle j to mu_t^{i*} where
              i* = min{i : sum_{k=1}^{i} w_t^(k) >= u^j}.
          After resampling, all particles carry equal weight 1/N.
    """
    T, N, nu = p.T, p.N, p.nu

    w_stat = stationary_scale(p)
    particles = sub_gaussian_rvs(N, scale=w_stat, nu=nu, rng=rng, loc=p.mu_0)

    mu_filt  = np.empty(T)
    var_filt = np.empty(T)
    ess_arr  = np.empty(T)

    for t in range(T):
        innov     = sub_gaussian_rvs(N, scale=p.w, nu=nu, rng=rng)
        particles = p.phi * particles + innov

        # Compute log likelihood of each observation, under a t-distribution centred at the particle location
        log_w = stats.t.logpdf(y[t], df=nu, loc=particles, scale=p.sigma * np.sqrt((nu - 2.0) / nu)) # sigma rescaled
        log_w -= log_w.max() # LSE
        w = np.exp(log_w) # exp for unnormalized weights
        w /= w.sum() # normalize weights

        mu_filt[t]  = w @ particles # dot product for weighted mean for particle approx / mean estimated by filtering
        var_filt[t] = w @ (particles - mu_filt[t])**2 # weighted sample variance
        ess_arr[t]  = 1.0 / (w**2).sum() # ESS = 1 / sum_i w_i^2

        # multinomial resampling
        u    = rng.uniform(0.0, 1.0, size=N)
        cumw = np.cumsum(w)
        idx  = np.searchsorted(cumw, u).clip(0, N - 1)
        particles = particles[idx]
    return mu_filt, var_filt, ess_arr


# ============================================================================
#  Kalman Filter implementation
# ============================================================================
def kalman_filter(y: np.ndarray, p: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard linear Kalman filter, optimally applied for filtering Gaussian state-space models

    When nu is large, our model is approximately Gaussian

    In this case, the Kalman filter is the optimal filter
    It therefore serves as a lower bound on achievable RMSE at large nu.
    At high enough nu, we expect RMSE_Kalman <= RMSE_SMC <= RMSE_ScoreDriven:
      - SMC has residual Monte Carlo error from finite N particles; it converges
        to Kalman as N -> infinity.
      - Score-driven uses the Student-t score r/(1 + r^2/(nu*sigma^2)) which
        converges to the linear residual r only as nu -> infinity. At high enough nu, it
        is close but not exact, and kappa is estimated from data rather than
        set to the analytic Kalman gain, adding further approximation error.
    A large gap between Kalman and SMC suggests N is too small. A large gap
    between Kalman and score-driven reflects the cost of the score-driven
    approximation in the near-Gaussian regime.

    The algorithm produces two quantities at each t:
        mu_{t|1:t} : the filtered mean  E[mu_t | y_{1:t}]
        P_{t|1:t}  : the filtered variance  Var[mu_t | y_{1:t}]

    These are propagated by alternating Predict and Update steps.

    Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
    Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space Methods (2nd ed.), Ch 2 
    Shumway, R.H. & Stoffer, D.S. (2017). Time Series Analysis and Its Applications Ch 6
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb
    """
    T   = p.T
    Q   = p.w**2      # state noise variance: sub-Gaussian -> N(0, w^2) as nu -> \inf as score term -> x
    R   = p.sigma**2  # observation noise variance: t(0, sigma, nu) -> N(0, sigma^2) as nu -> \inf
    phi = p.phi

    # -------------------------------------------------------------------------
    # Initialisation from stationary AR(1)
    #
    # The AR(1) mu_t = phi * mu_{t-1} + eta_t has stationary variance:
    #   Var_stationary = Q / (1 - phi^2) as previously calculated
    # -------------------------------------------------------------------------
    mu_t = float(p.mu_0)
    P_t  = Q / max(1.0 - phi**2, 1e-8)  # 1e-8 prevents division by zero near phi=1

    mu_filt  = np.empty(T)
    var_filt = np.empty(T)

    for t in range(T):
        # Predict
        # Project the current posterior forward one step through the state equation
        mu_pred = phi * mu_t         # predicted mean
        P_pred  = phi**2 * P_t + Q  # predicted variance

        # Update 
        # Condition on the new observation y[t].
        v_t = y[t] - mu_pred   # innovation based on how far off the prediction was
        S_t = P_pred + R       # innovation variance
        K_t = P_pred / S_t     # Kalman gain

        mu_t = mu_pred + K_t * v_t       # posterior mean
        P_t  = (1.0 - K_t) * P_pred     # posterior variance

        mu_filt[t]  = mu_t
        var_filt[t] = P_t

    return mu_filt, var_filt


# ============================================================================
#  Joint MLE for Score-Driven Parameters
# ============================================================================
# the score-driven filter depends on:
_SD_PARAM_NAMES: list[str] = ["phi", "kappa", "sigma", "nu"]

# Valid (lo, hi) bounds for each parameter
_SD_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "phi":       (-0.999, 0.999),
    "kappa":     (1e-3,   50.0),
    "sigma":     (1e-3,   20.0),      # natural sigma; used only by profile-likelihood sweeps, not MLE
    "log_sigma": (-10.0,  4.0),       # reparameterize sigma: lambda = log(sigma) for MLE
                                      # Matlab fmincon can take ±Inf, but scipy L-BFGS-B requires finite bounds
                                      # bounds are e^{-10} ≈ 0 to e^4 ≈ 55 in sigma-space —> big enough to be effectively unconstrained
    "nu":        (2.01,   300.0),
}

# reasonable visual ranges for profile likelihood plots (tighter than optimisation bounds)
_PROFILE_GRID_DEFAULTS: dict[str, tuple[float, float]] = {
    "phi":   (0.5,  0.99),
    "sigma": (0.4,  2.5),
    "nu":    (2.0,  30.0),
    "kappa": (0.0,  3.0),
}


# ============================================================================
#  Accumulate log likelihood for parameter(s) values in the score-driven filter
# ============================================================================
def _sd_neg_loglik_joint(
    theta: np.ndarray,
    y: np.ndarray,
    p_base: ModelParams,
    free_names: list[str],
) -> float:
    """
    Compute the Negative PED log-likelihood as a function of a potentially joint parameter vector.
    This function is called at each itreration.

    Can take any subset of {phi, sigma, nu} and always including kappa, while holding the remaining
    parameters fixed at their values specified in p_base.

    The score-driven filtered state mu_hat_t is a deterministic function of
    the data y_{1:t} and the current parameter estimates. The PED log-likelihood

        log L(theta) = sum_{t=1}^T  log t(y_t; mu_{t|t-1}(theta), sigma, nu)

    is therefore a smooth, deterministic scalar that scipy.optimize.minimize
    can evaluate and minimise by the quasi-Newton method L-BFGS-B since we supply bounds.
    """
    # unpack values of parameters we are optimizing into named dict
    params: dict[str, float] = {n: float(v) for n, v in zip(free_names, theta)}

    # sigma reparameterization
    if "log_sigma" in params:
        params["sigma"] = float(np.exp(params.pop("log_sigma")))

    for n in _SD_PARAM_NAMES:
        if n not in params:
            params[n] = float(getattr(p_base, n))

    phi, kappa, sigma, nu = params["phi"], params["kappa"], params["sigma"], params["nu"]

    # if the optimizer returns a value outside our reasonable range, penalize heavily (should be unlikely to hit this)
    if not (-1.0 < phi < 1.0) or kappa <= 0.0 or sigma <= 0.0 or nu <= 2.0:
        return 1e10

    mu_hat = float(p_base.mu_0)
    nll    = 0.0
    nu_s2  = sigma**2 * (nu - 2.0) # denominator of score update

    for y_t in y: # through all observations
        mu_pred = phi * mu_hat
        nll -= stats.t.logpdf(y_t, df=nu, loc=mu_pred, scale=sigma * np.sqrt((nu - 2.0) / nu))
        r      = y_t - mu_pred
        mu_hat = mu_pred + kappa * r / (1.0 + r**2 / nu_s2)

    return nll


def estimate_sd_params_ml(
    y: np.ndarray, # observations
    p: ModelParams, # values for fixed parameters and starting points for those which are being estimated
    free: tuple[str, ...] = ("phi", "sigma", "nu"), # defines which parameters we will estimate
) -> dict[str, float]:
    """
    Estimate parameters of the score-driven model by minimizing the
    negative prediction-error-decomposition (PED) likelihood.

    Structural parameters: {phi, sigma, nu}:
        Part of the data-generating process. We have set values we are trying
        to recover from the observed series and choose whether we want to infer them.

    kappa: the score-driven filter gain:
        Scales the score innovation at each update step. Has no counterpart in
        the DGP. Always include in `free`.

    OPTIMIZATION
    ------------
    L-BFGS-B: a quasi-Newton method, constraints are placed on parameter values.
    Because the score-driven state is a deterministic function of past data and
    parameters, the PED log-likelihood is smooth and L-BFGS-B should be able to approximate
    its gradient reliably via finite differences.

    sigma is reparameterized as lambda = log(sigma) to reduce skewness in the likelihood surface
    and hopefully make the quadratic approximation used by L-BFGS-B more realistic.
    """
    _all_free = set(_SD_PARAM_NAMES)
    if any(n not in _all_free for n in free):
        raise ValueError(f"'free' must be a subset of {_all_free}.")

    free_names = [n for n in _SD_PARAM_NAMES if n in free]

    # reparameterize sigma: rename "sigma" -> "log_sigma" in the parameter vector passed
    # to the optimiser.  The optimiser therefore steps in lambda = log(sigma) space rather
    # than sigma space.  The caller still passes "sigma" in `free` and receives sigma back.
    free_names_opt = ["log_sigma" if n == "sigma" else n for n in free_names]

    def _start(name: str) -> float:
        if name == "log_sigma":
            return float(np.log(p.sigma))
        if name == "kappa":
            return float(p.kappa) if p.kappa is not None else 1.0
        return float(getattr(p, name))

    x0  = np.array([_start(n) for n in free_names_opt])
    bds = [_SD_DEFAULT_BOUNDS[n] for n in free_names_opt]  # reparameterize sigma: picks log_sigma bounds

    # run the optimization, repeatedly calling the negative loglik maximiser with different values of the parameter
    # vector until NLL stops improving
    # no jac specified in the function call means gradients will be estimated by finite differences as default,
    # which should converge quickly enough with only 4 parameters at most
    result = minimize(
        _sd_neg_loglik_joint,
        x0,
        args=(y, p, free_names_opt),               # reparameterize sigma: optimiser passes log_sigma to loglik
        method="L-BFGS-B",
        bounds=bds,
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8},
    )

    # reparameterize sigma: convert lambda_hat back to sigma = exp(lambda_hat) before returning
    out: dict[str, float] = {}
    for name, val in zip(free_names_opt, result.x):
        if name == "log_sigma":                    # reparameterize sigma
            out["sigma"] = float(np.exp(val))      # reparameterize sigma
        else:
            out[name] = float(val)
    out["neg_loglik"] = float(result.fun)
    out["success"]    = bool(result.success)
    return out


def estimate_sigma_oracle(
    y: np.ndarray, # observations
    mu: np.ndarray, # true latent states
    p: ModelParams,
    sigma_range: tuple[float, float] = (0.4, 2.5),
    grid_points: int = 200,
) -> tuple[float, plt.Figure]:
    """
    We are interested in whether the upward bias in estimate of sigma is due to the
    error in the score driven filter itself. Therefore, we apply the same PED likelihood
    but to the true latent states rather than the filter's estimates. 

    The PED likelihood used in estimate_params_ml() evaluates
        log t(y_t; mu_{t|t-1}, sigma, nu)
    where mu_{t|t-1} is the score-driven filter's one-step-ahead prediction of the latent state.
    The residual y_t - mu_{t|t-1} therefore contains both
    genuine observation noise and state estimation error. Our theory is that the optimizer 
    assigns both of these to sigma, causing upward bias.

    In this oracle function we bypass the filter. The
    oracle uses the true state mu[t] as the location, so the
    residual y_t - mu_t = e_t is pure observation noise. This isolates sigma
    from both filter estimation error and state innovation variance, and should
    recover sigma ≈ 1. Of course this is not possible in practice but it is useful for comparison.
    """
    grid = np.linspace(sigma_range[0], sigma_range[1], grid_points)

    # Oracle log-likelihood: use true states mu[1..T] as location of the t-likelihood
    # Residual y_t - mu_t = e_t is pure observation noise, enabling isolation of sigma.
    mu_pred_oracle = mu[1:] # extract true latent states after the initial

    ll_oracle = np.array([
        np.sum(stats.t.logpdf(
            y,
            df=p.nu,
            loc=mu_pred_oracle,
            scale=sig * np.sqrt((p.nu - 2.0) / p.nu), # standard deviation parameterization as observation noise
        ))
        for sig in grid
    ])
    ll_oracle -= ll_oracle.max() # normalize log likelihood relative to its max for visual

    # PED (filter-based) likelihood for comparison: profile over kappa
    ll_ped = np.empty(grid_points)
    for i, sig in enumerate(grid):
        p_sig = replace(p, sigma=sig, kappa=None)
        res = estimate_sd_params_ml(y, p_sig, free=("kappa",))
        ll_ped[i] = -res["neg_loglik"]
    ll_ped -= ll_ped.max()

    sigma_hat = float(grid[np.argmax(ll_oracle)])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid, ll_oracle, "g-",  lw=1.8, label="Oracle LL (true states)")
    ax.plot(grid, ll_ped,    "b--", lw=1.5, label="PED LL (filter predictions, κ profiled)")
    ax.axvline(p.sigma, c="k",    lw=1.2, ls="--", label=f"True σ={p.sigma}")
    ax.axvline(sigma_hat, c="g",  lw=1.0, ls=":",  label=f"Oracle MLE σ={sigma_hat:.3f}")
    ax.axhline(-1.92,    c="gray", lw=0.8, ls=":",  label="−1.92 (≈95% CI boundary)")
    ax.set_xlabel("σ")
    ax.set_ylabel("Log-likelihood (relative to max)")
    ax.set_title("Oracle vs PED profile log-likelihood for σ", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()

    return sigma_hat, fig


def _t_logpdf_numpy(x: np.ndarray, df: float, loc, scale: float) -> np.ndarray:
    # Student-t log-pdf in pure numpy — avoids scipy's ~6-layer Python dispatch stack.
    # Mathematically identical to stats.t.logpdf(x, df=df, loc=loc, scale=scale).
    # Formula: lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi) - log(scale)
    #          - (df+1)/2 * log1p(z^2/df),  where z = (x - loc) / scale.
    log_norm = (math.lgamma((df + 1.0) / 2.0)
                - math.lgamma(df / 2.0)
                - 0.5 * math.log(df * math.pi)
                - math.log(scale))
    z = (x - loc) / scale
    return log_norm - (df + 1.0) / 2.0 * np.log1p(z * z / df)


def _bootstrap_filter_loglik(
    y: np.ndarray, # take only the observed data
    p: ModelParams, # and a proposed set of parameters
    rng: np.random.Generator,
    resample_threshold: float = 1.0,
) -> float:
    """
    Bootstrap particle filter that returns the log marginal likelihood estimate.

    Approximates log L(theta) = log p(y_{1:T} | theta) by the product of
    one-step predictive likelihoods:

        p(y_t | y_{1:t-1}) ≈  (1/N) * sum_{i=1}^N  w_t^(i)

    where w_t^(i) = p(y_t | mu_t^(i)) is the Student-t likelihood of the
    observation given particle i's state value.
    """
    T, N, nu = p.T, p.N, p.nu

    w_stat    = stationary_scale(p) # MC sampling for the variance of a SG (replacing naive AR(1) w/1-phi^2)
    particles = sub_gaussian_rvs(N, scale=w_stat, nu=nu, rng=rng, loc=p.mu_0) # SG samples for t=0

    log_lik_estimate = 0.0

    for t in range(T):
       
        innov     = sub_gaussian_rvs(N, scale=p.w, nu=nu, rng=rng)
        particles = p.phi * particles + innov # Propagate particles through the state equation

        # Compute unnormalized log-weights
        # w_t^(i) = p(y_t | mu_t^(i)), i.e. the Student-t likelihood of the
        # observation given this particle's state value.
        # numpy-inlined version (~10x faster than stats.t.logpdf)
        log_w = _t_logpdf_numpy(y[t], df=nu, loc=particles, scale=p.sigma * np.sqrt((nu - 2.0) / nu))

        # Accumulate the log predictive likelihood for this time step
        # log[ (1/N) * sum_i exp(log_w_i) ]  using the log-sum-exp trick
        log_w_max         = log_w.max() # to be factored out for LSE

        # https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf
        log_lik_estimate += (
            log_w_max
            + np.log(np.sum(np.exp(log_w - log_w_max))) # average weight = likelihood
            - np.log(N)
        )

        # Normalize weights
        w  = np.exp(log_w - log_w_max)
        w /= w.sum()

        # resampling when ESS falls below threshold
        # Always resampling (threshold=1.0) prevents weight degeneracy but may cause
        # path degeneracy. Resampling only when ESS < threshold*N reduces
        # the frequency of resampling, lowering the variance of the log-likelihood
        # estimate and improving PMMH mixing — at the cost of occasional weight
        # concentration between resample steps.
        ess = 1.0 / (w**2).sum()
        if ess < resample_threshold * N:
            u         = rng.uniform(0.0, 1.0, size=N)
            cumw      = np.cumsum(w) # CDF of weight dist, sum of weights up to the particle
            idx       = np.searchsorted(cumw, u).clip(0, N - 1) # invert the CDF
            particles = particles[idx] # select elements

    return log_lik_estimate


_SMC_PARAM_NAMES: list[str] = ["phi", "sigma", "nu", "w"]

_SMC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "phi":   (-0.999, 0.999),
    "sigma": (1e-3,   20.0),
    "nu":    (2.01,   300.0),
    "w":     (1e-3,   20.0),
}


def pmcmc_inference(
    y: np.ndarray, # parameters are inferred based on the observed data
    p_init: ModelParams, # parameter initialization values
    n_iter: int = 2000, # number of chain steps
    proposal_sd: dict[str, float] | None = None, # each parameter has a proposed s.d. in the gaussian random walk
    free: tuple[str, ...] = ("phi", "sigma", "nu"), # define the parameters we want to infer, the rest held at p_init values
    log_prior_fn=None, # can add informative prior here, otherwise will be uniform over defined bounds
    n_particles: int | None = None,
    seed: int = 0,
    resample_threshold: float = 1.0,
    reparameterise: bool = True,    # if False, propose directly in constrained space (no logit/log transform)
) -> dict:
    """
    Particle Marginal Metropolis-Hastings (PMMH) for inference of
    the parameters in the SMC bootstrap filter.

    At each iteration:
      1. PROPOSE: Draw candidate theta* via Gaussian random walk on
         (optionally) unconstrained (logit phi, log sigma) space (based on reparameterise setting).
      2. RUN FILTER: Compute log L̂(y | theta*) via _bootstrap_filter_loglik.
      3. MH ACCEPT/REJECT: Accept with probability
             alpha = [L̂_new * p(theta*) * J(theta*)] / [L̂^{i-1} * p(theta^{i-1}) * J(theta^{i-1})]
         where J(theta) = |d theta / d psi| is the Jacobian of the inverse
         reparameterisation (J = 1 when reparameterise=False).   The last accepted likelihood is carried forward on
         rejection (PMMH).

    Because E[L̂] = L exactly, the chain's theta-marginal targets p(theta | y)
    regardless of particle count N. Larger N improves mixing (reduces noise in L̂).
    https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf
    """
    import functools
    rng = np.random.default_rng(seed)

    _filter = functools.partial(_bootstrap_filter_loglik, resample_threshold=resample_threshold) # bake resampling into the filter function for looping
    free_names = [n for n in _SMC_PARAM_NAMES if n in free] # params to estimate
    if not free_names:
        raise ValueError("No valid free parameters specified.")

    # Proposed standard deviations for the random walk. Can be tuned to adjust acceptance rate.
    # When reparameterise=True (default): proposals are in unconstrained space
    #   (logit for phi, log for sigma), so step sizes are in that transformed space.
    #   At phi~0.7 the logit Jacobian dψ/dphi ≈ 3.5, so 0.05 in ψ was only ~0.014 in phi.
    #   Rule of thumb: proposal_sd ≈ 2.4 * posterior_std_unconstrained / sqrt(d).
    #   https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoap%2F1034625254
    # When reparameterise=False: proposals are made directly in constrained space
    #   (phi units, sigma units), so step sizes are directly interpretable.
    _default_proposal_sd: dict[str, float] = {
        "phi":   0.3,    # reparameterise=True: logit space (~0.05-0.08 in phi near phi=0.7); False: phi units directly
        "sigma": 0.2,    # reparameterise=True: log space (~0.2 in sigma near sigma=1);       False: sigma units directly
        "nu":    1.0,    # nu has a wide range; larger steps to explore
        "w":     0.05,
    }
    if proposal_sd is None:
        proposal_sd = {}
        # setting step sizes:
    prop_sd = {n: proposal_sd.get(n, _default_proposal_sd[n]) for n in free_names}

    # possibility to override particle number here in the future
    p_current = p_init
    if n_particles is not None:
        from dataclasses import replace as _replace
        p_current = _replace(p_current, N=n_particles)
    N_particles = p_current.N

    # Default log-prior: flat (0.0) within bounds, -inf outside.
    # A flat prior means the posterior is proportional to the likelihood,
    # so we are effectively doing maximum likelihood via MCMC. This is a
    def _flat_log_prior(theta_dict: dict[str, float]) -> float:
        for name, val in theta_dict.items():
            lo, hi = _SMC_PARAM_BOUNDS[name]
            if not (lo < val < hi):
                return -np.inf   # zero probability outside the valid range
        return 0.0               # log(1) = 0: flat prior within bounds

    _log_prior = log_prior_fn if log_prior_fn is not None else _flat_log_prior

    # -------------------------------------------------------------------------
    # build a ModelParams from the current theta dict + p_current fixed values.
    # -------------------------------------------------------------------------
    from dataclasses import replace as _replace
    def _make_params(theta_dict: dict[str, float]) -> ModelParams:
        return _replace(p_current, **theta_dict)

    # -------------------------------------------------------------------------
    # Option to reparameterise to unconstrained space for better geometry
    # -------------------------------------------------------------------------
    # phi  in (-1, 1)  ->  psi    = log((1+phi)/(1-phi))   [transform to logit-like, unconstrained]
    # sigma > 0        ->  log_s  = log(sigma)              [log transform, unconstrained]
    # https://sites.stat.columbia.edu/gelman/book/BDA3.pdf
    #
    # Proposing on (psi, log_s) makes the posterior geometry closer to elliptical.
    # Because we change variables, the MH ratio picks up log-Jacobian corrections:
    #   phi:   log|d phi / d psi|   = log((1 - phi²) / 2)  [actually = log(2e^psi/(e^psi+1)²)]
    #   sigma: log|d sigma / d log_s| = log(sigma)
    # These are added for the proposed point and subtracted for the current point.
    # -------------------------------------------------------------------------
    def _to_unconstrained(name: str, val: float) -> float:
        if not reparameterise:
            return val
        if name == "phi":
            return float(np.log((1.0 + val) / (1.0 - val)))   # logit((phi+1)/2): maps (-1,1) -> (-inf,inf)
        if name == "sigma":
            return float(np.log(val))
        return val   # nu — no transform

    def _to_constrained(name: str, u: float) -> float:
        if not reparameterise:
            return u
        if name == "phi":
            e = np.exp(u)
            return float((e - 1.0) / (e + 1.0))
        if name == "sigma":
            return float(np.exp(u))
        return u

    def _log_jacobian(name: str, constrained_val: float) -> float:
        """
        log|d constrained / d unconstrained| at constrained_val.
        any change of variables tranforms a parameter and then samples it
        this requires a jacobian adjustment
        https://mc-stan.org/docs/2_21/stan-users-guide/changes-of-variables.html
        """
        if not reparameterise:
            return 0.0   # no transform, no Jacobian correction
        if name == "phi":
            # d phi / d psi = (1 - phi²) / 2
            return float(np.log((1.0 - constrained_val**2) / 2.0))
        if name == "sigma":
            # d sigma / d log_sigma = sigma
            return float(np.log(constrained_val))
        return 0.0   # nu — no transform, Jacobian = 1

    # -------------------------------------------------------------------------
    # Initialise the chain
    # -------------------------------------------------------------------------
    theta_current: dict[str, float] = {n: float(getattr(p_current, n)) for n in free_names} # extract starting param values

    log_lik_current   = _filter(y, _make_params(theta_current), rng) # compute log lik at initialization value
    log_prior_current = _log_prior(theta_current) # compute prior as 0 in bounds and -inf outside

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------
    samples: dict[str, np.ndarray] = {n: np.empty(n_iter) for n in free_names}
    log_liks  = np.empty(n_iter)
    n_accepted_per_param: dict[str, int] = {n: 0 for n in free_names}

    # -------------------------------------------------------------------------
    # Blocked Metropolis-within-Gibbs
    # -------------------------------------------------------------------------
    # Each iteration cycles through the free parameters one at a time.
    # Updating phi and sigma separately with individually-tuned step sizes
    # prevents the joint proposal from being dominated by the less-informed
    # direction and allows different mixing rates per parameter.
    # Each block runs one MH step: propose -> filter -> accept/reject.
    # The likelihood carried forward is always the one at the current theta.
    # -------------------------------------------------------------------------
    for i in range(n_iter):

        for name in free_names:   # one block per parameter

            # transform current proposal to unconstrained space 
            u_current = _to_unconstrained(name, theta_current[name]) # (does nothing if reparameterise=False)
            u_star    = u_current + rng.normal(0.0, prop_sd[name]) # apply random walk pertubation to proposal
            val_star  = _to_constrained(name, u_star) # (does nothing if reparameterise=False)

            theta_star = {**theta_current, name: val_star} # new dict with new proposal parameter values

            log_prior_star = _log_prior(theta_star) # new prior calculated based on this new proposal

            if np.isfinite(log_prior_star): # is the proposal in bounds?
                log_lik_star = _filter(y, _make_params(theta_star), rng) # then bother running the filter
            else:
                log_lik_star = -np.inf # otherwise don't bother, force a rejection

            # Jacobian for this parameter only (others unchanged). Does nothing if were not reparameterizing. 
            log_jac_star_param    = _log_jacobian(name, val_star)
            log_jac_current_param = _log_jacobian(name, theta_current[name])

            # MH ratio
            log_alpha = (
                (log_lik_star  + log_prior_star  + log_jac_star_param) # log_jac_star_param 0 when reparameterise=False
              - (log_lik_current + log_prior_current + log_jac_current_param) # log_jac_current_param 0 when reparameterise=False
            )

            if np.log(rng.uniform()) <= log_alpha:
                theta_current     = theta_star
                log_lik_current   = log_lik_star
                log_prior_current = log_prior_star
                n_accepted_per_param[name] += 1

        for n in free_names:
            samples[n][i] = theta_current[n]
        log_liks[i] = log_lik_current # save whatever theta* we finish with, either the last accepted or new proposal

    # Report per-parameter acceptance rates
    n_total = n_iter
    for name in free_names:
        rate = n_accepted_per_param[name] / n_total
        status = 'ok' if 0.15 < rate < 0.45 else 'TUNE proposal_sd'
        print(f"  [PMCMC] {name:6s} acceptance rate: {rate:.3f}  ({status})")
    accept_rate = sum(n_accepted_per_param.values()) / (n_iter * len(free_names))

    return {
        "samples":      samples,
        "log_liks":     log_liks,
        "accept_rate":  accept_rate,
        "free":         free_names,
        "n_iter":       n_iter,
        "n_particles":  N_particles,
    }


def plot_concavity(p: ModelParams, nu_list=(3, 4, 5, 6, 7, 8, 300)) -> plt.Figure:
    """
    Side-by-side panels:
      Left: Student-t log-density
      Right: Score function

    The dotted vertical lines mark |r| = sigma*sqrt(nu-2): the inflection point
    of the log-density and the maximum of the score function, under the
    sigma-rescaled parameterisation (scale = sigma*sqrt((nu-2)/nu)).
    """
    r      = np.linspace(-6 * p.sigma, 6 * p.sigma, 1200)
    cmap   = plt.cm.plasma
    colors = [cmap(i / (len(nu_list) - 1)) for i in range(len(nu_list))]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

    for nu_val, col in zip(nu_list, colors):
        lbl = f"ν={int(nu_val)}"

        log_f = stats.t.logpdf(r, df=nu_val, loc=0.0, scale=p.sigma * np.sqrt((nu_val - 2.0) / nu_val))
        score = r / (1.0 + r**2 / (p.sigma**2 * (nu_val - 2.0)))

        ax0.plot(r, log_f, color=col, lw=1.6, label=lbl)
        ax1.plot(r, score, color=col, lw=1.6, label=lbl)

        r_infl = p.sigma * np.sqrt(nu_val - 2.0)
        if r_infl <= r.max() * 0.95:
            ax0.axvline( r_infl, color=col, lw=0.7, ls=":")
            ax0.axvline(-r_infl, color=col, lw=0.7, ls=":")
            ax1.axvline( r_infl, color=col, lw=0.7, ls=":")
            ax1.axvline(-r_infl, color=col, lw=0.7, ls=":")

    ax0.set_xlabel(r"Residual $y_t - \mu_t$")
    ax0.set_ylabel(r"$\log f(y_t - \mu_t)$")
    ax0.set_title(
        r"Student-$t$ Log-Density  ($\sigma=$" + f"{p.sigma}" + r")" + "\n"
        r"Dotted lines: inflection points $|r| = \sigma\sqrt{\nu-2}$"
    )
    ax0.legend(fontsize=8, loc="lower center")
    ax0.grid(True, lw=0.4, alpha=0.6)

    ax1.set_xlabel(r"Residual $y_t - \mu_t$")
    ax1.set_ylabel(r"Score $s_t = r\,/\,(1 + r^2/\sigma^2(\nu-2))$")
    ax1.set_title(
        r"Score Function  ($\sigma=$" + f"{p.sigma}" + r")" + "\n"
        r"Dotted lines: inflection points $|r| = \sigma\sqrt{\nu-2}$"
    )
    ax1.axhline(0, c="k", lw=0.6)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, lw=0.4, alpha=0.6)

    fig.tight_layout()
    return fig


# ============================================================================
#  Metrics
# ============================================================================
def compute_metrics(true: np.ndarray, est: np.ndarray, label: str) -> dict:
    err = est - true
    return {
        "label": label,
        "RMSE":  float(np.sqrt(np.mean(err**2))),
        "MAE":   float(np.mean(np.abs(err))),
        "Bias":  float(np.mean(err)),
        "Corr":  float(np.corrcoef(true, est)[0, 1]),
        "MaxAE": float(np.max(np.abs(err))),
    }


def print_metrics_table(metrics_list: list[dict], header: str = ""):
    if header:
        print(f"\n  {header}")
    row_fmt = "  {:<22}  {:>7}  {:>7}  {:>8}  {:>7}  {:>7}"
    print(row_fmt.format("Filter", "RMSE", "MAE", "Bias", "Corr", "MaxAE"))
    print("  " + "─" * 64)
    for m in metrics_list:
        print(row_fmt.format(
            m["label"],
            f"{m['RMSE']:.4f}",
            f"{m['MAE']:.4f}",
            f"{m['Bias']:+.4f}",
            f"{m['Corr']:.4f}",
            f"{m['MaxAE']:.3f}",
        ))


def plot_paths(res: dict) -> plt.Figure:
    """
    Three-panel figure: (1) latent paths + filtered estimates, (2) estimate errors,
    (3) ESS: effective sample size.
    """
    p        = res["params"]
    mu       = res["mu"]
    y        = res["y"]
    t        = np.arange(1, p.T + 1)
    kappa_ml = res["kappa_ml"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"SMC and Score-Driven Filter Estimates of Latent States with Student-t Noise\n"
        f"ν={p.nu},  φ={p.phi},  σ={p.sigma},  w={p.w},  "
        f"κ={kappa_ml:.3f},  T={p.T},  N={p.N}",
        fontsize=12,
    )

    # Panel 1: state trajectories
    ax = axes[0]
    ax.scatter(t, y, s=3, c="#B8B8B8", zorder=1, label="Obs $y_t$")
    ax.plot(t, mu[1:],          "-",  color="#222222", lw=1.8, zorder=4, label=r"True $\mu_t$")
    ax.plot(t, res["mu_smc"],   "-",  color="#6A00A8", lw=1.2, alpha=0.9, label="SMC filtered mean")
    ax.plot(t, res["mu_sd"],    "--", color="#FCA636", lw=1.2, alpha=0.9,
            label=f"Score-driven  κ={kappa_ml:.3f}")
    sd_smc = np.sqrt(np.maximum(res["var_smc"], 0))
    ax.fill_between(
        t,
        res["mu_smc"] - 1.645 * sd_smc,
        res["mu_smc"] + 1.645 * sd_smc,
        alpha=0.15, color="#6A00A8", label="SMC 90% CI",
    )
    ax.set_ylabel("State / Observation")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, lw=0.4, alpha=0.6)

    # Panel 2: estimation errors
    ax = axes[1]
    ax.plot(t, res["mu_smc"] - mu[1:], "-",  color="#6A00A8", lw=0.9, alpha=0.85, label="SMC error")
    ax.plot(t, res["mu_sd"]  - mu[1:], "--", color="#FCA636", lw=0.9, alpha=0.85, label="Score-driven error")
    ax.axhline(0, c="#222222", lw=0.8)
    ax.set_ylabel("Filtering Error")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.6)

    # Panel 3: effective sample size
    ax = axes[2]
    ax.plot(t, res["ess"], "-", color="#6A00A8", lw=1.0, alpha=0.85, label="ESS")
    ax.axhline(p.N * 0.5,  c="orange", lw=0.9, ls="--", label=f"50% ESS = {p.N//2}")
    ax.axhline(p.N * 0.25, c="red",    lw=0.7, ls=":",  label=f"25% ESS = {p.N//4}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time  $t$")
    ax.set_ylabel("ESS")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.6)

    plt.tight_layout()
    return fig

def plot_rolling_rmse(res: dict, window: int = 30) -> plt.Figure:
    """Rolling RMSE over time for both filters."""
    p      = res["params"]
    sq_smc = (res["mu_smc"] - res["mu"][1:])**2
    sq_sd  = (res["mu_sd"]  - res["mu"][1:])**2
    kern   = np.ones(window) / window
    t      = np.arange(window, p.T + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, np.sqrt(np.convolve(sq_smc, kern, "valid")),
            "-",  color="#6A00A8", lw=1.4, label="SMC")
    ax.plot(t, np.sqrt(np.convolve(sq_sd,  kern, "valid")),
            "--", color="#FCA636", lw=1.4, label="Score-driven")
    ax.set_xlabel("Time  $t$")
    ax.set_ylabel("Rolling RMSE")
    ax.set_title(f"Rolling RMSE  (ν={p.nu},  φ={p.phi},  κ_ML={res['kappa_ml']:.3f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_dof_sweep(sweep_results: dict) -> plt.Figure:
    """
    Plot RMSE vs ν and ML-estimated κ vs ν side by side.
    """
    nu_vals    = sorted(sweep_results.keys())
    smc_rmse   = [sweep_results[v]["metrics_smc"]["RMSE"] for v in nu_vals]
    sd_rmse    = [sweep_results[v]["metrics_sd"]["RMSE"]  for v in nu_vals]
    kappa_vals = [sweep_results[v]["kappa_ml"]            for v in nu_vals]

    kf_rmse = [
        sweep_results[v]["metrics_kf"]["RMSE"]
        if "metrics_kf" in sweep_results[v] else None
        for v in nu_vals
    ]
    has_kf = any(v is not None for v in kf_rmse)

    fig, (ax_rmse, ax_kappa) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: RMSE vs ν ---
    ax_rmse.plot(nu_vals, smc_rmse, "o-",  color="#6A00A8", lw=1.5, ms=6, label="SMC")
    ax_rmse.plot(nu_vals, sd_rmse,  "s--", color="#FCA636", lw=1.5, ms=6, label="Score-driven (ML κ)")
    if has_kf:
        kf_nu   = [v for v, r in zip(nu_vals, kf_rmse) if r is not None]
        kf_vals = [r for r in kf_rmse if r is not None]
        ax_rmse.plot(kf_nu, kf_vals, "^-", color="#CC4678", lw=1.5, ms=6, label="Kalman")
    ax_rmse.set_xlabel(r"Degrees of Freedom  $\log(\nu)$")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE vs ν")
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(True, lw=0.4, alpha=0.6)
    ax_rmse.set_xscale("log")

    # --- Right: ML-estimated κ vs ν ---
    ax_kappa.plot(nu_vals, kappa_vals, "D-", color="#222222", lw=1.5, ms=6, label="κ_ML")
    ax_kappa.set_xlabel(r"Degrees of Freedom  $\log(\nu)$")
    ax_kappa.set_ylabel("κ_ML")
    ax_kappa.set_title("ML-estimated κ vs ν")
    ax_kappa.legend(fontsize=9)
    ax_kappa.grid(True, lw=0.4, alpha=0.6)
    ax_kappa.set_xscale("log")

    plt.tight_layout()
    return fig


def plot_dof_sweep_mc(mc_sweep_results: dict) -> plt.Figure:
    """
    Plot MC-averaged RMSE vs ν and ML-estimated κ vs ν side by side, with ±1 std error bars.
    Expects mc_sweep_results[nu_val] as returned by run_mc_experiment().
    """
    nu_vals = sorted(mc_sweep_results.keys())

    smc_mean  = [mc_sweep_results[v]["metrics_smc"]["RMSE"]["mean"] for v in nu_vals]
    smc_std   = [mc_sweep_results[v]["metrics_smc"]["RMSE"]["std"]  for v in nu_vals]
    sd_mean   = [mc_sweep_results[v]["metrics_sd"]["RMSE"]["mean"]  for v in nu_vals]
    sd_std    = [mc_sweep_results[v]["metrics_sd"]["RMSE"]["std"]   for v in nu_vals]
    kf_mean   = [mc_sweep_results[v]["metrics_kf"]["RMSE"]["mean"]  for v in nu_vals]
    kf_std    = [mc_sweep_results[v]["metrics_kf"]["RMSE"]["std"]   for v in nu_vals]
    kappa_mean = [mc_sweep_results[v]["kappa_ml"]["mean"] for v in nu_vals]
    kappa_std  = [mc_sweep_results[v]["kappa_ml"]["std"]  for v in nu_vals]

    fig, (ax_rmse, ax_kappa) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: RMSE vs ν ---
    ax_rmse.errorbar(nu_vals, smc_mean,  yerr=smc_std,   fmt="o-",  color="#6A00A8",
                     lw=1.5, ms=6, capsize=3, capthick=1, label="SMC")
    ax_rmse.errorbar(nu_vals, sd_mean,   yerr=sd_std,    fmt="s--", color="#FCA636",
                     lw=1.5, ms=6, capsize=3, capthick=1, label="Score-driven (ML κ)")
    ax_rmse.errorbar(nu_vals, kf_mean,   yerr=kf_std,    fmt="^-",  color="#CC4678",
                     lw=1.5, ms=6, capsize=3, capthick=1, label="Kalman")
    ax_rmse.set_xlabel(r"$\log(\nu)$")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE vs ν")
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(True, lw=0.4, alpha=0.6)
    ax_rmse.set_xscale("log")

    # --- Right: ML-estimated κ vs ν ---
    ax_kappa.errorbar(nu_vals, kappa_mean, yerr=kappa_std, fmt="D-", color="#222222",
                      lw=1.5, ms=6, capsize=3, capthick=1, label="κ_ML")
    ax_kappa.set_xlabel(r"$\log(\nu)$")
    ax_kappa.set_ylabel("κ_ML")
    ax_kappa.set_title("ML-estimated κ vs ν")
    ax_kappa.legend(fontsize=9)
    ax_kappa.grid(True, lw=0.4, alpha=0.6)
    ax_kappa.set_xscale("log")

    plt.tight_layout()
    return fig


def plot_scatter_comparison(res: dict) -> plt.Figure:
    """
    Scatter: SD estimate vs SMC estimate (coloured by true state),
    plus SMC vs true and SD vs true.
    """
    p      = res["params"]
    mu_t   = res["mu"][1:]
    smc    = res["mu_smc"]
    sd     = res["mu_sd"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"(ν={p.nu},  φ={p.phi},  κ_ML={res['kappa_ml']:.3f})",
        fontsize=12,
    )

    pairs = [
        (mu_t, smc, "True μ_t",  "SMC estimated state"),
        (mu_t, sd,  "True μ_t",  "Score-driven estimated state"),
        (smc,  sd,  "SMC estimated state",       "Score-driven estimated state"),
    ]
    for ax, (x, y_vals, xl, yl) in zip(axes, pairs):
        ax.scatter(x, y_vals, color="#6A00A8", s=4, alpha=0.4)
        lims = [min(x.min(), y_vals.min()), max(x.max(), y_vals.max())]
        ax.plot(lims, lims, "k--", lw=0.8, label="y = x")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        corr = np.corrcoef(x, y_vals)[0, 1]
        ax.set_title(f"{yl} vs {xl}\nCorr={corr:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    return fig


# ============================================================================
#  Experiment Runner
# ============================================================================

def run_experiment(p: ModelParams, verbose: bool = True) -> dict:
    """
    Full single experiment: simulate → SMC filter → SD filter (ML kappa) → metrics.

    If p.kappa is None, kappa is estimated by maximising the prediction-error-
    decomposition likelihood (Harvey & Luati 2014) before running the filter.
    If p.kappa is already set, that value is used directly.
    """
    rng    = np.random.default_rng(p.seed)
    mu, y  = simulate(p, rng)

    # Separate RNG for SMC so its draws are independent of the simulation draws.
    rng_smc = np.random.default_rng(p.seed + 1000)
    mu_smc, var_smc, ess = bootstrap_filter(y, p, rng_smc)

    kappa_ml = p.kappa if p.kappa is not None else estimate_sd_params_ml(y, p, free=('kappa',))['kappa']
    p_fit    = replace(p, kappa=kappa_ml)
    mu_sd    = score_driven_filter(y, p_fit)

    kf_mean, kf_var = kalman_filter(y, p)

    m_smc = compute_metrics(mu[1:], mu_smc, "SMC")
    m_sd  = compute_metrics(mu[1:], mu_sd,  f"Score-driven (κ={kappa_ml:.3f})")
    m_kf  = compute_metrics(mu[1:], kf_mean, "Kalman")

    if verbose:
        print_metrics_table(
            [m_kf, m_smc, m_sd],
            header=f"ν={p.nu:<6} φ={p.phi}  σ={p.sigma}  w={p.w}  κ_ML={kappa_ml:.3f}",
        )

    return {
        "params":      p,
        "mu":          mu,
        "y":           y,
        "mu_smc":      mu_smc,
        "var_smc":     var_smc,
        "ess":         ess,
        "mu_sd":       mu_sd,
        "kf_mean":     kf_mean,
        "kf_var":      kf_var,
        "kappa_ml":    kappa_ml,
        "metrics_smc": m_smc,
        "metrics_sd":  m_sd,
        "metrics_kf":  m_kf,
    }


# ============================================================================
#  Monte Carlo Experiment Runner
# ============================================================================
def run_mc_experiment(p: ModelParams, n_reps: int = 100) -> dict:
    """
    average of run_experiment over n_reps independent replications
    replication i uses seed p.seed + i, giving independent draws of (mu, y)
    """
    keys = ["RMSE", "MAE", "Bias", "Corr", "MaxAE"]
    smc_vals = {k: [] for k in keys}
    sd_vals  = {k: [] for k in keys}
    kf_vals  = {k: [] for k in keys}
    kappas   = []

    for i in range(n_reps):
        res = run_experiment(replace(p, seed=p.seed + i), verbose=False)
        for k in keys:
            smc_vals[k].append(res["metrics_smc"][k])
            sd_vals[k].append(res["metrics_sd"][k])
            kf_vals[k].append(res["metrics_kf"][k])
        kappas.append(res["kappa_ml"])

    def agg(vals):
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in vals.items()}

    return {
        "params":      p,
        "n_reps":      n_reps,
        "metrics_smc": agg(smc_vals),
        "metrics_sd":  agg(sd_vals),
        "metrics_kf":  agg(kf_vals),
        "kappa_ml":    {"mean": float(np.mean(kappas)), "std": float(np.std(kappas))},
    }


# ============================================================================
#  Dual-nu Model: Separate Degrees of Freedom for Innovations and Observations
# ============================================================================
#   nu_eta controls how bounded / light-tailed the state dynamics are.
#           For finite nu_eta, the sub-Gaussian has bounded support
#           [-sqrt(nu_eta)*w/2, +sqrt(nu_eta)*w/2] and is lighter-tailed
#           than Gaussian. As nu_eta -> inf it approaches the Gaussian
#
#   nu_eps controls how heavy-tailed the observation residuals are.
#           t(nu_eps) is always heavier-tailed than Gaussian.  As nu_eps -> inf it
#           approaches N(0, sigma^2).
# ============================================================================
@dataclass
class DualNuParams:
    phi:    float       = 0.8
    mu_0:   float       = 0.0
    sigma:  float       = 1.0
    w:      float       = 1.0
    nu:     float       = 8.0        # base df; used when nu_eta / nu_eps are None
    nu_eta: float | None = None      # innovation df; None -> resolves to nu
    nu_eps: float | None = None      # observation df; None -> resolves to nu
    T:      int         = 300
    N:      int         = 500
    seed:   int         = 123

    def __post_init__(self):
        # resolve Nones
        if self.nu_eta is None:
            self.nu_eta = self.nu
        if self.nu_eps is None:
            self.nu_eps = self.nu

    @classmethod
    def from_model_params(cls, p: ModelParams) -> "DualNuParams":
        """Create DualNuParams from a ModelParams, setting nu_eta = nu_eps = p.nu."""
        return cls(
            phi=p.phi, mu_0=p.mu_0, sigma=p.sigma, w=p.w,
            nu=p.nu, T=p.T, N=p.N, seed=p.seed,
            # nu_eta and nu_eps left as None -> resolve to p.nu in __post_init__
        )


def simulate_dual(p: DualNuParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    simulate from the dual-nu sub-Gaussian SSM
    when nu_eta = nu_eps = nu this is identical to simulate(ModelParams(nu=nu), rng)
    """
    innov = sub_gaussian_rvs(p.T, scale=p.w, nu=p.nu_eta, rng=rng)
    mu = np.empty(p.T + 1)
    mu[0] = p.mu_0
    for t in range(1, p.T + 1):
        mu[t] = p.phi * mu[t - 1] + innov[t - 1]
    e = rng.standard_t(df=p.nu_eps, size=p.T) * p.sigma * np.sqrt((p.nu_eps - 2.0) / p.nu_eps)
    y = mu[1:] + e
    return mu, y


def _stationary_scale_dual(p: DualNuParams) -> float:
    """
    stationary std of the AR(1) using the true sub-Gaussian innovation variance
    Uses nu_eta (not nu_eps) because innovations drive state spread.
    """
    sg_var = sub_gaussian_variance(p.w, p.nu_eta)
    return float(np.sqrt(sg_var / max(1.0 - p.phi**2, 1e-8)))


def bootstrap_filter_dual(
    y: np.ndarray,
    p: DualNuParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    bootstrap particle filter for the dual-nu model
    when nu_eta = nu_eps this is identical to bootstrap_filter()
    """
    T, N = p.T, p.N

    w_stat    = _stationary_scale_dual(p)
    particles = sub_gaussian_rvs(N, scale=w_stat, nu=p.nu_eta, rng=rng, loc=p.mu_0)

    mu_filt  = np.empty(T)
    var_filt = np.empty(T)
    ess_arr  = np.empty(T)

    for t in range(T):
        # propagate with sub-Gaussian innovations (nu_eta controls state dynamics)
        innov     = sub_gaussian_rvs(N, scale=p.w, nu=p.nu_eta, rng=rng)
        particles = p.phi * particles + innov

        # weight by t-likelihood (nu_eps controls how outliers are discounted)
        log_w = stats.t.logpdf(
            y[t], df=p.nu_eps, loc=particles,
            scale=p.sigma * np.sqrt((p.nu_eps - 2.0) / p.nu_eps),
        )
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()

        mu_filt[t]  = w @ particles
        var_filt[t] = w @ (particles - mu_filt[t])**2
        ess_arr[t]  = 1.0 / (w**2).sum()

        u    = rng.uniform(size=N)
        cumw = np.cumsum(w)
        idx  = np.searchsorted(cumw, u).clip(0, N - 1)
        particles = particles[idx]

    return mu_filt, var_filt, ess_arr


def kalman_filter_corrected(
    y: np.ndarray,
    p: DualNuParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman filter using the true innovation variance as Q

    Q = _sub_gaussian_variance(w, nu_eta)
    R = sigma^2                     

    this is the fairest Gaussian approximation: it uses the correct variances
    but assumes Gaussian *shapes* for both noise components. Any gap versus
    exact SMC is due purely to distributional non-Gaussianity (heavy tails /
    bounded support), not variance mismatch.

    as nu_eta -> inf and nu_eps -> inf, we expect it to converge to the optimal filter.
    """
    T   = p.T
    Q   = sub_gaussian_variance(p.w, p.nu_eta)
    R   = p.sigma**2
    phi = p.phi

    mu_t = float(p.mu_0)
    P_t  = Q / max(1.0 - phi**2, 1e-8)

    mu_filt  = np.empty(T)
    var_filt = np.empty(T)

    for t in range(T):
        mu_pred = phi * mu_t
        P_pred  = phi**2 * P_t + Q
        v_t     = y[t] - mu_pred
        S_t     = P_pred + R
        K_t     = P_pred / S_t
        mu_t    = mu_pred + K_t * v_t
        P_t     = (1.0 - K_t) * P_pred
        mu_filt[t]  = mu_t
        var_filt[t] = P_t

    return mu_filt, var_filt


# ============================================================================
#  Analytical Gaussian Approximation Metrics
# ============================================================================
def kl_t_vs_gaussian(nu: float) -> float:
    """
    exact closed-form KL(t(nu) || N) as a function of nu only.

    both distributions are matched to the same variance, so sigma cancels.
    derived from KL(P||Q) = -H(P) - E_P[log Q]:

        KL = -(nu+1)/2 * [psi((nu+1)/2) - psi(nu/2)]
             - log B(nu/2, 1/2)
             + 1/2 * log(2*pi / (nu-2))
             + 1/2

    where psi is the digamma function and B is the beta function.
    KL -> 0 as nu -> inf (t converges to Gaussian).
    """
    from scipy.special import digamma, betaln
    kl = (
        -(nu + 1.0) / 2.0 * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0))
        - betaln(nu / 2.0, 0.5)
        + 0.5 * np.log(2.0 * np.pi / (nu - 2.0))
        + 0.5
    )
    return float(kl)


def excess_kurtosis_t(nu: float) -> float:
    """excess kurtosis 6/(nu-4) for nu > 4, else inf."""
    if nu <= 4.0:
        return float("inf")
    return 6.0 / (nu - 4.0)


def sub_gaussian_moments_numerical(
    w: float,
    nu: float,
    n_mc: int = 300_000,
    seed: int = 42,
) -> dict:
    """Monte Carlo variance ratio and excess kurtosis of SubGaussian(0, var; nu)."""
    rng = np.random.default_rng(seed)
    x   = rng.standard_t(df=nu, size=n_mc) * w
    z   = x / (1.0 + x**2 / (nu * w**2))
    var  = float(np.var(z))
    # Fisher's excess kurtosis = E[z^4] / E[z^2]^2 - 3
    m2   = float(np.mean(z**2))
    m4   = float(np.mean(z**4))
    kurt = m4 / m2**2 - 3.0
    return {
        "variance":        var,
        "variance_ratio":  var / w**2,
        "excess_kurtosis": float(kurt),
    }


def score_deviation_from_linear(
    nu_eps: float,
    k_values: np.ndarray | None = None,
) -> np.ndarray:
    """
    fractional deviation of score innovation from the Kalman update.

    the Kalman-equivalent linear update uses r directly, and the fractional
    deviation at |r| = k * sigma is purely analytical:
        d(k) = 1 - x_t/r = k^2 / ((nu_eps - 2) + k^2)
    """
    if k_values is None:
        k_values = np.linspace(0.1, 5.0, 200)
    k_sq = k_values**2
    return k_sq / ((nu_eps - 2.0) + k_sq)


def mean_expected_score_deviation(
    nu_eps: float,
    sigma: float = 1.0,
    n_mc: int = 200_000,
) -> float:
    """
    expected fractional deviation of score from linear, averaged over the
    t(nu_eps) distribution of observation residuals:
        E_{r ~ t(nu_eps)}[ 1 - x_t/|r| ] = E_r[ r^2 / ((nu_eps-2)*sigma^2 + r^2) ]

    0 = perfectly linear (nu_eps -> inf), larger values
    indicate more robust (and more non-Gaussian) observation updates.
    """
    rng   = np.random.default_rng(42)
    scale = sigma * np.sqrt((nu_eps - 2.0) / nu_eps)
    r     = rng.standard_t(df=nu_eps, size=n_mc) * scale
    r     = r[np.abs(r) > 1e-6]
    k_sq  = (r / sigma)**2
    dev   = k_sq / ((nu_eps - 2.0) + k_sq)
    return float(np.mean(dev))


# ============================================================================
#  Gaussian Approximation Plots
# ============================================================================
def _gaussian_approx_precompute(
    w: float = 1.0,
    nu_list: np.ndarray | None = None,
) -> dict:
    """
    pre-compute all analytical metrics shared across the three
    Gaussian-approximation plot functions
    """
    if nu_list is None:
        nu_list = np.unique(np.concatenate([
            np.linspace(2.5, 10.0, 25),
            np.linspace(10.0, 50.0, 25),
            np.linspace(50.0, 300.0, 30),
        ]))

    kl_nu      = nu_list[nu_list > 2.0]
    kl_vals    = [kl_t_vs_gaussian(nu) for nu in kl_nu]
    kurt_eps   = np.array([excess_kurtosis_t(nu) if nu > 4.0 else np.nan for nu in nu_list])
    sg_data    = [sub_gaussian_moments_numerical(w, nu) for nu in nu_list]
    var_ratio  = np.array([d["variance_ratio"]  for d in sg_data])
    sg_kurt    = np.array([d["excess_kurtosis"] for d in sg_data])
    kl_sg_vals = [kl_subgaussian_vs_gaussian(w=w, nu=float(nu)) for nu in kl_nu]

    return dict(
        nu_list=nu_list, kl_nu=kl_nu, kl_vals=kl_vals,
        kurt_eps=kurt_eps, var_ratio=var_ratio,
        sg_kurt=sg_kurt, kl_sg_vals=kl_sg_vals,
    )


def plot_gaussian_approx_combined(
    w: float = 1.0,
    nu_list: np.ndarray | None = None,
) -> plt.Figure:
    """
    2x2 figure combining KL divergence and excess kurtosis for both noise components.

    Layout:
      (0,0)  KL( t(nu_eps) || N )            log-log axes
      (0,1)  Excess kurtosis of t(nu_eps)    log x-axis, positive (heavy tails)
      (1,0)  KL( SubGaussian(nu_eta) || N )  log-log axes
      (1,1)  Excess kurtosis of SG(nu_eta)   log x-axis, negative (light tails)

    Row 0: observation noise  t(nu_eps) — steelblue
    Row 1: state innovations  SubGaussian(nu_eta) — firebrick

    Both KL plots share a y-axis scale so the ~10x difference between the two
    components is immediately visible.
    """
    d          = _gaussian_approx_precompute(w, nu_list)
    nu_list    = d["nu_list"]
    kl_nu      = d["kl_nu"]
    kl_vals    = np.array(d["kl_vals"])
    kurt_eps   = d["kurt_eps"]
    kl_sg_vals = np.array(d["kl_sg_vals"])
    sg_kurt    = d["sg_kurt"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    kl_ymin = min(kl_vals.min(), kl_sg_vals.min()) * 0.8
    kl_ymax = max(kl_vals.max(), kl_sg_vals.max()) * 1.3

    # (0,0) — KL for observation noise t(nu_eps)
    ax = axes[0, 0]
    ax.plot(kl_nu, kl_vals, color="steelblue", lw=1.8)
    ax.set_xlabel(r"$\nu_\varepsilon$  (observation DoF)")
    ax.set_ylabel(r"KL divergence from $\mathcal{N}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(kl_ymin, kl_ymax)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # (0,1) — Excess kurtosis for t(nu_eps)
    ax = axes[0, 1]
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.plot(nu_list, kurt_eps, color="steelblue", lw=1.8)
    ax.set_xlabel(r"$\nu_\varepsilon$  (observation DoF)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_xscale("log")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # (1,0) — KL for state innovations SubGaussian(nu_eta)
    ax = axes[1, 0]
    ax.plot(kl_nu, kl_sg_vals, color="firebrick", lw=1.8)
    ax.set_xlabel(r"$\nu_\eta$  (innovation DoF)")
    ax.set_ylabel(r"KL divergence from $\mathcal{N}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(kl_ymin, kl_ymax)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # (1,1) — Excess kurtosis for SubGaussian(nu_eta)
    ax = axes[1, 1]
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.plot(nu_list, sg_kurt, color="firebrick", lw=1.8)
    ax.set_xlabel(r"$\nu_\eta$  (innovation DoF)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_xscale("log")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # row labels as bold left-aligned titles on the KL panels
    axes[0, 0].set_title(
        r"Observation noise  $t(\nu_\varepsilon)$",
        loc="left", fontweight="bold", fontsize=10,
    )
    axes[1, 0].set_title(
        r"State innovations  SubGaussian$(\nu_\eta)$",
        loc="left", fontweight="bold", fontsize=10,
    )

    plt.tight_layout()
    return fig

def plot_score_convergence(
    sigma: float = 1.0,
    nu_list: tuple = (4, 6, 8, 12, 20, 50, 100, 300),
) -> plt.Figure:
    """
    2 panels showing how the score innovation converges to the linear
    (Kalman-equivalent) update as nu_eps increases.

    1 — Deviation curves d(k) = k^2 / ((nu_eps-2) + k^2) vs |r|/sigma.
        Each curve shows how much the score shrinks the residual at each
        magnitude. Large nu_eps -> d -> 0 (score becomes linear in r).

    2 — Mean expected deviation E_r[1 - x_t/|r|] vs nu_eps (log x-axis).
        Single-number summary of score non-linearity per nu_eps value.
    """
    k_vals = np.linspace(0.1, 4.0, 300)
    cmap   = plt.cm.plasma
    colors = [cmap(i / (len(nu_list) - 1)) for i in range(len(nu_list))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Panel 1: deviation curves
    ax = axes[0]
    for nu_val, col in zip(nu_list, colors):
        if nu_val <= 2.0:
            continue
        dev = score_deviation_from_linear(nu_val, k_vals)
        ax.plot(k_vals, dev, color=col, lw=1.6,
                label=rf"$\nu_\varepsilon$={int(nu_val)}")
    ax.set_xlabel(r"Size of residual  $|r|\,/\,\sigma$")
    ax.set_ylabel("Proportion")
    ax.set_title(
        r"Proportion of residual discarded by score driven update vs linear"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.02)
    ax.grid(True, lw=0.4, alpha=0.5)

    # Panel 2: mean expected deviation vs nu_eps (log x-axis)
    ax = axes[1]
    nu_grid  = np.unique(np.concatenate([np.linspace(2.5, 20.0, 30),
                                          np.linspace(20.0, 300.0, 30)]))
    mean_dev = np.array([mean_expected_score_deviation_quad(nu, sigma)
                         for nu in nu_grid])
    ax.plot(nu_grid, mean_dev, "b-", lw=1.8)
    ax.set_xlabel(r"$\nu_\varepsilon$  (observation DoF)")
    ax.set_ylabel("Proportion")
    ax.set_title(
        r"Proportion of residual discarded averaged over the true distribution of residuals"
    )
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)

    plt.tight_layout()
    return fig


def run_gaussian_approx_1d_sweep(
    vary: str,
    nu_vary_list: list[float],
    nu_fixed: float = 8.0,
    p_base: DualNuParams | None = None,
    n_reps: int = 10,
) -> dict:
    """
    vary one nu while fixing the other, comparing SMC, score-driven, and Kalman.
    """
    if p_base is None:
        p_base = DualNuParams()

    n = len(nu_vary_list)
    rmse_smc  = np.empty((n, n_reps))
    rmse_sd   = np.empty((n, n_reps))
    rmse_kcor = np.empty((n, n_reps))

    for j, nu_val in enumerate(nu_vary_list):
        for rep in range(n_reps):
            seed = rep * 500 + j
            if vary == "nu_eps":
                p = replace(p_base, nu_eps=nu_val, nu_eta=nu_fixed, seed=seed)
            else:
                p = replace(p_base, nu_eta=nu_val, nu_eps=nu_fixed, seed=seed)

            rng          = np.random.default_rng(p.seed)
            mu, y        = simulate_dual(p, rng)
            rng_smc      = np.random.default_rng(p.seed + 100)
            mu_smc, _, _ = bootstrap_filter_dual(y, p, rng_smc)
            mu_kcor, _   = kalman_filter_corrected(y, p)

            mp_sd    = ModelParams(
                phi=p.phi, sigma=p.sigma, w=p.w, mu_0=p.mu_0,
                nu=p.nu_eps, T=p.T, N=p.N, seed=p.seed,
            )
            kappa_ml   = estimate_sd_params_ml(y, mp_sd, free=('kappa',))['kappa']
            mu_sd_filt = score_driven_filter(y, replace(mp_sd, kappa=kappa_ml))

            rmse_smc[j, rep]  = np.sqrt(np.mean((mu_smc     - mu[1:])**2))
            rmse_sd[j, rep]   = np.sqrt(np.mean((mu_sd_filt - mu[1:])**2))
            rmse_kcor[j, rep] = np.sqrt(np.mean((mu_kcor    - mu[1:])**2))

    return {
        "vary":                  vary,
        "nu_vary":               nu_vary_list,
        "nu_fixed":              nu_fixed,
        "rmse_smc":              rmse_smc,
        "rmse_sd":               rmse_sd,
        "rmse_kalman_corrected": rmse_kcor,
    }


def plot_gaussian_approx_1d_sweeps(
    results_eps: dict,
    results_eta: dict,
) -> plt.Figure:
    """
    Two-panel RMSE comparison from 1D sweeps.

    Each panel shows mean RMSE across replications for SMC,
    score-driven, and Kalman-corrected.

    The SMC–SD gap is the primary comparison (SD approximation cost).
    The SMC–Kalman gap shows the cost of the Gaussian approximation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res in zip(axes, [results_eps, results_eta]):
        nu_vary = np.array(res["nu_vary"])
        for key, lbl, col, ls in [
            ("rmse_smc",
             "SMC (reference)",
             "steelblue",  "-"),
            ("rmse_sd",
             r"Score-driven ($\hat{\kappa}_{\mathrm{ML}}$)",
             "seagreen",   "-"),
            ("rmse_kalman_corrected",
             r"Kalman  ($Q=\mathrm{Var}(\eta)$, $R=\sigma^2$)",
             "firebrick",  "--"),
        ]:
            rmse = res[key]
            mean = rmse.mean(axis=1)
            ax.plot(nu_vary, mean, color=col, lw=1.8, ls=ls, label=lbl)

        if res["vary"] == "nu_eps":
            varying_nu  = r"$\nu_\varepsilon$"
            fixed_nu    = r"$\nu_\eta$"
            varying_dist = r"$t$-distributed observation noise"
            fixed_dist   = r"SubGaussian innovations"
        else:
            varying_nu  = r"$\nu_\eta$"
            fixed_nu    = r"$\nu_\varepsilon$"
            varying_dist = r"SubGaussian innovations"
            fixed_dist   = r"$t$-distributed observation noise"
        ax.set_xlabel(varying_nu)
        ax.set_ylabel("RMSE")
        ax.set_title(
            f"Vary {varying_nu}  ({varying_dist})\n"
            f"{fixed_nu} = {res['nu_fixed']}  ({fixed_dist} fixed)"
        )
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", lw=0.4, alpha=0.5)

    plt.tight_layout()
    return fig


def run_gaussian_approx_2d_sweep(
    nu_eta_list: list[float],
    nu_eps_list: list[float],
    p_base: DualNuParams | None = None,
    n_reps: int = 5,
    verbose: bool = True,
) -> dict:
    """
    2D sweep over a grid of (nu_eta, nu_eps) values.

    At each grid point, compute three filters (SMC, score-driven, Kalman-corrected)
    and record their mean RMSE across n_reps replications.

    Positive gap: filter worse than SMC at this (nu_eta, nu_eps)
    """
    if p_base is None:
        p_base = DualNuParams()

    n_eta = len(nu_eta_list)
    n_eps = len(nu_eps_list)
    rmse_smc = np.zeros((n_eta, n_eps))
    rmse_sd  = np.zeros((n_eta, n_eps))
    rmse_kal = np.zeros((n_eta, n_eps))

    for i, nu_eta in enumerate(nu_eta_list):
        for j, nu_eps in enumerate(nu_eps_list):
            smc_reps, sd_reps, kal_reps = [], [], []
            for rep in range(n_reps):
                p       = replace(p_base, nu_eta=nu_eta, nu_eps=nu_eps,
                                  seed=rep * 1000 + i * 50 + j)
                rng     = np.random.default_rng(p.seed)
                mu, y   = simulate_dual(p, rng)
                rng_smc = np.random.default_rng(p.seed + 100)
                mu_smc, _, _ = bootstrap_filter_dual(y, p, rng_smc)
                mu_kal, _    = kalman_filter_corrected(y, p)

                mp_sd    = ModelParams(
                    phi=p.phi, sigma=p.sigma, w=p.w, mu_0=p.mu_0,
                    nu=p.nu_eps, T=p.T, N=p.N, seed=p.seed,
                )
                kappa_ml = estimate_sd_params_ml(y, mp_sd, free=('kappa',))['kappa']
                mu_sd_f  = score_driven_filter(y, replace(mp_sd, kappa=kappa_ml))

                smc_reps.append(np.sqrt(np.mean((mu_smc  - mu[1:])**2)))
                sd_reps.append( np.sqrt(np.mean((mu_sd_f - mu[1:])**2)))
                kal_reps.append(np.sqrt(np.mean((mu_kal  - mu[1:])**2)))

            rmse_smc[i, j] = float(np.mean(smc_reps))
            rmse_sd[i, j]  = float(np.mean(sd_reps))
            rmse_kal[i, j] = float(np.mean(kal_reps))

        if verbose:
            print(f"  nu_eta={nu_eta:.0f} done  ({i+1}/{n_eta})")

    rel_gap_sd  = (rmse_sd  - rmse_smc) / (rmse_smc + 1e-8)
    rel_gap_kal = (rmse_kal - rmse_smc) / (rmse_smc + 1e-8)
    return {
        "nu_eta_list":  nu_eta_list,
        "nu_eps_list":  nu_eps_list,
        "rmse_smc":     rmse_smc,
        "rmse_sd":      rmse_sd,
        "rmse_kalman":  rmse_kal,
        "rel_gap_sd":   rel_gap_sd,
        "rel_gap_kal":  rel_gap_kal,
    }


def plot_gaussian_approx_2d_heatmap(
    sweep: dict, threshold: float = 0.05
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Four separate figures from the 2D (nu_eta, nu_eps) sweep.

    1: Kalman − SMC) / SMC
        when is the Gaussian approximation adequate?
    2: (SD − SMC) / SMC
        SD approximation cost
    3: Binary safe-zones = Kalman and SD gaps vs threshold.
    4: Absolute RMSE contours for SMC, score-driven, and Kalman
    """
    from matplotlib.lines import Line2D

    nu_eta  = np.array(sweep["nu_eta_list"])
    nu_eps  = np.array(sweep["nu_eps_list"])
    gap_sd  = sweep["rel_gap_sd"]
    gap_kal = sweep["rel_gap_kal"]
    rmse_s  = sweep["rmse_smc"]
    rmse_sd = sweep["rmse_sd"]
    rmse_k  = sweep["rmse_kalman"]

    _ticks = [3, 5, 8, 12, 20, 50, 100, 300]

    def _ax_labels(ax):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(_ticks); ax.set_xticklabels([str(v) for v in _ticks])
        ax.set_yticks(_ticks); ax.set_yticklabels([str(v) for v in _ticks])
        ax.set_xlabel(r"$\nu_\varepsilon$  (observation df)  →  near-Gaussian")
        ax.set_ylabel(r"$\nu_\eta$  (innovation df)  →  near-Gaussian")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)

    def _gap_heatmap(gap, title):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(nu_eps, nu_eta, np.clip(gap * 100, 0, 50),
                           cmap="RdYlGn_r", shading="nearest", vmin=0, vmax=30)
        plt.colorbar(im, ax=ax, label="Relative RMSE gap (%)")
        cs = ax.contour(nu_eps, nu_eta, gap * 100,
                        levels=[5, 10, 20],
                        colors=["green", "orange", "red"],
                        linewidths=1.2, linestyles=["--", "-.", ":"])
        ax.clabel(cs, fmt="%.0f%%", fontsize=8)
        ax.set_title(title)
        _ax_labels(ax)
        plt.tight_layout()
        return fig

    # Kalman gap heatmap
    fig1 = _gap_heatmap(gap_kal, r"Kalman gap = (Kalman $-$ SMC) / SMC")

    # SD gap heatmap
    fig2 = _gap_heatmap(gap_sd, r"SD gap = (Score-driven $-$ SMC) / SMC")

    # binary safe-zones for both filters
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gap, lbl in [
        (axes[0], gap_kal, f"Kalman safe zone: gap < {threshold*100:.0f}%"),
        (axes[1], gap_sd,  f"SD safe zone: gap < {threshold*100:.0f}%"),
    ]:
        im = ax.pcolormesh(nu_eps, nu_eta, (gap < threshold).astype(float),
                           cmap="RdYlGn", shading="nearest", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label=f"1 = gap < {threshold*100:.0f}%")
        ax.set_title(lbl)
        _ax_labels(ax)
    plt.tight_layout()

    # absolute RMSE contours
    fig4, ax = plt.subplots(figsize=(6, 5))
    all_rmse = np.stack([rmse_s, rmse_sd, rmse_k])
    levels   = np.linspace(all_rmse.min(), all_rmse.max(), 7)
    cs_s  = ax.contour(nu_eps, nu_eta, rmse_s, levels=levels,
                       colors="steelblue", linewidths=1.2, linestyles="-")
    cs_sd = ax.contour(nu_eps, nu_eta, rmse_sd, levels=levels,
                       colors="seagreen",  linewidths=1.2, linestyles="-")
    cs_k  = ax.contour(nu_eps, nu_eta, rmse_k,  levels=levels,
                       colors="firebrick", linewidths=1.2, linestyles="--")
    ax.clabel(cs_s,  fmt="%.3f", fontsize=7, colors="steelblue")
    ax.clabel(cs_sd, fmt="%.3f", fontsize=7, colors="seagreen")
    ax.clabel(cs_k,  fmt="%.3f", fontsize=7, colors="firebrick")
    ax.set_title("RMSE contours: SMC / Score-driven / Kalman")
    ax.legend(handles=[
        Line2D([0], [0], color="steelblue", lw=1.2,         label="SMC"),
        Line2D([0], [0], color="seagreen",  lw=1.2,         label="Score-driven"),
        Line2D([0], [0], color="firebrick", lw=1.2, ls="--", label="Kalman"),
    ], fontsize=9)
    _ax_labels(ax)
    plt.tight_layout()

    return fig1, fig2, fig3, fig4


def mean_expected_score_deviation_quad(nu_eps: float, sigma: float = 1.0) -> float:
    """
    Expected fractional deviation of the score innovation from the linear
    (Kalman) update, computed by numerical quadrature.
    """
    # reparameterize the t-distribution so that Var(r) = sigma^2 exactly
    scale = sigma * np.sqrt((nu_eps - 2.0) / nu_eps)

    def integrand(r: float) -> float:
        k_sq = (r / sigma) ** 2                          # residual in sigma units
        dev  = k_sq / ((nu_eps - 2.0) + k_sq)           # fractional deviation d(r)
        pdf  = stats.t.pdf(r, df=nu_eps, loc=0.0, scale=scale)  # true density
        return dev * pdf

    # quad integrates over (-inf, inf); limit=200 subdivisions for accuracy
    # near the heavy tails
    result, _ = quad(integrand, -np.inf, np.inf, limit=200)
    return float(result)


def kl_subgaussian_vs_gaussian(
    w: float = 1.0,
    nu: float = 8.0,
    n_mc: int = 300_000,
    seed: int = 42,
) -> float:
    """
    KL( SubGaussian(0,Var(SG);nu) || N(0, Var(SG)) ) by MC log-density

    Derived fully in the report sec:kl_mc. The reference Gaussian is
    variance-matched so the KL reflects shape difference only.
    """
    rng = np.random.default_rng(seed)
    z   = sub_gaussian_rvs(n_mc, scale=w, nu=nu, rng=rng)

    var_sg = sub_gaussian_moments_numerical(w, nu)["variance"]
    std_sg = np.sqrt(var_sg)

    def _pdf_at(zi: float) -> float:
        z_max = np.sqrt(nu) * w / 2.0
        if abs(zi) >= z_max:
            return 0.0
        if abs(zi) < 1e-12:
            return float(stats.t.pdf(0.0, df=nu, scale=w))
        disc  = 1.0 - 4.0 * zi**2 / (nu * w**2)
        D     = np.sqrt(max(disc, 0.0))
        x_in  = (nu * w**2 / (2.0 * zi)) * (1.0 - D)
        x_out = (nu * w**2 / (2.0 * zi)) * (1.0 + D)
        def jac(x: float) -> float:
            u = x**2 / (nu * w**2)
            d = abs(1.0 - u)
            return (1.0 + u)**2 / d if d > 1e-14 else 1e14
        return float(stats.t.pdf(x_in,  df=nu, scale=w) * jac(x_in) +
                     stats.t.pdf(x_out, df=nu, scale=w) * jac(x_out))

    log_p = np.array([np.log(_pdf_at(float(zi)) + 1e-300) for zi in z])
    log_q = stats.norm.logpdf(z, loc=0.0, scale=std_sg)
    return float(np.mean(log_p - log_q))


def plot_kalman_vs_sd_heatmap(sweep: dict) -> plt.Figure:
    """
    heatmap of (RMSE_Kal - RMSE_SD) / RMSE_SD

    Positive = red = Kalman worse than SD: heavy-tailed observations
                      fit poorly with the Kalman update; SD's robust score helps it
    Negative = (blue) = Kalman better than SD: SD over-discounts genuine large
                      residuals, particularly when nu_eta is large (near-Gaussian
                      innovations allow large state jumps that SD under-corrects).
    White = the two filters are equivalent.
    """
    nu_eta  = np.array(sweep["nu_eta_list"])
    nu_eps  = np.array(sweep["nu_eps_list"])
    rmse_sd = sweep["rmse_sd"]
    rmse_k  = sweep["rmse_kalman"]

    gap = (rmse_k - rmse_sd) / rmse_sd  # positive 

    abs_max = np.nanmax(np.abs(gap)) * 100
    vlim    = np.ceil(abs_max / 5) * 5   # round up to nearest 5%

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(
        nu_eps, nu_eta, gap * 100,
        cmap="RdBu_r", shading="nearest",
        vmin=-vlim, vmax=vlim,
    )
    plt.colorbar(im, ax=ax, label=r"(RMSE$_\mathrm{Kal}$ $-$ RMSE$_\mathrm{SD}$) / RMSE$_\mathrm{SD}$  (%)")

    # zero contour
    ax.contour(nu_eps, nu_eta, gap * 100, levels=[0],
               colors="black", linewidths=1.4, linestyles="-")

    _ticks = [3, 5, 8, 12, 20, 50, 100, 300]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(_ticks); ax.set_xticklabels([str(v) for v in _ticks])
    ax.set_yticks(_ticks); ax.set_yticklabels([str(v) for v in _ticks])
    ax.set_xlabel(r"$\nu_\varepsilon$  (observation df)  →  near-Gaussian")
    ax.set_ylabel(r"$\nu_\eta$  (innovation df)  →  near-Gaussian")
    ax.set_title(r"Kalman vs SD: (RMSE$_\mathrm{Kal}$ $-$ RMSE$_\mathrm{SD}$) / RMSE$_\mathrm{SD}$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig