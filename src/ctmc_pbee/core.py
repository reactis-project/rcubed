"""
ctmc_pbee.core
==============
Three-state CTMC model with one absorbing (collapse) state: generator matrix,
state probabilities, and the parametric studies of the paper built on the
metrics (ctmc_pbee.metrics) and on the spectral analysis (ctmc_pbee.spectral).

Reference
---------
Nardin, Marelli, Sudret, Broccardo.
"A Generalized Framework for Performance-Based Earthquake Engineering:
Integrated Assessment of Structural Reliability and Resilience."
Structural Safety (under review, STRUCS-S-26-00168).

Modules
-------
core     : generator Q, transient block Q_T, pi(t); parametric studies (this module)
metrics  : reliability (F, beta) and resilience (Q0, R0, rho), Eq. (10)-(19)
spectral : eigenmode expansion, QSD, spectral approximation of R0, Eq. (20)-(27)

Equation index (numbering of the paper)
---------------------------------------
Eq. (8)        -> pi_t                   : state probabilities  pi(t) = pi(0) exp(Q t)
Eq. (9)        -> extract_Q_T            : transient sub-generator Q_T
(Sec. 3)       -> build_Q                : generator Q from the physical rates lambda_01, lambda_02, lambda_12, mu10
Eq. (28)-(29)  -> build_Q_adim           : dimensionless Q_tilde(alpha, gamma, epsilon), Q = mu10 * Q_tilde,
                                           to be used with the dimensionless time t_tilde = mu10 * t

Usage
-----
    import numpy as np
    from ctmc_pbee import core, metrics, spectral

    mu10 = 8.4
    Q   = core.build_Q(lam01=1/50, lam02=1/2475, lam12=1/100, mu10=mu10)   # rates [1/year]
    Q_T = core.extract_Q_T(Q)

    F, S  = metrics.F_tau(Q_T, t=50.0)
    beta  = metrics.beta_index(F)
    rho   = metrics.rho_index(metrics.R0_exact(Q_T, t=50.0))
    R0_sp = spectral.R0_spectral(spectral.spectral_analysis(Q_T))

    # dimensionless form of the paper (Eq. 28-29): Q_tilde with time t_tilde = mu10 * t
    Q_tilde = core.build_Q_adim(alpha=(1/50)/mu10, gamma=(1/2475)/mu10, epsilon=(1/100)/(1/50))
    F_adim, _ = metrics.F_tau(core.extract_Q_T(Q_tilde), t=mu10 * 50.0)   # == F
"""

import numpy as np
from scipy.linalg import expm

from ctmc_pbee import metrics, spectral


# ---------------------------------------------------------------------------
# 1.  Generator construction  (Eq. 28-29, Section 3)
# ---------------------------------------------------------------------------

def build_Q(lam01: float, lam02: float, lam12: float, mu10: float) -> np.ndarray:
    """
    Build the full 3×3 generator matrix of the three-state CTMC from the
    physical transition rates (Section 3).

    Parameters
    ----------
    lam01 : rate ds0 -> ds1, damaging earthquakes [year^{-1}]
    lam02 : rate ds0 -> ds2, direct collapse      [year^{-1}]
    lam12 : rate ds1 -> ds2, collapse of the damaged structure [year^{-1}]
    mu10  : rate ds1 -> ds0, recovery (repair)    [year^{-1}]

    Returns
    -------
    Q : (3, 3) ndarray  [year^{-1}]
        Full generator.  Rows sum to zero; state 2 (collapse) is absorbing.

    See also
    --------
    build_Q_adim : the same matrix from the dimensionless parameters (Eq. 28-29).
    """
    Q = np.array([
        [-(lam01 + lam02),  lam01,              lam02  ],
        [ mu10,            -(mu10 + lam12),      lam12  ],
        [ 0.0,              0.0,                 0.0    ],
    ])
    assert np.allclose(Q.sum(axis=1), 0.0), "Row-sum check failed."
    return Q


def build_Q_adim(alpha: float, gamma: float, epsilon: float) -> np.ndarray:
    """
    Dimensionless generator Q_tilde(alpha, gamma, epsilon) of Eq. (29).

    With the dimensionless parameters of Eq. (28)
        alpha = lambda_01 / mu10,  gamma = lambda_02 / mu10,  epsilon = lambda_12 / lambda_01
    the dimensional generator factors as  Q = mu10 * Q_tilde.  Q_tilde must be used
    with the dimensionless time  t_tilde = mu10 * t  (Section 3): the Kolmogorov
    equations become  d pi / d t_tilde = pi Q_tilde,  so the whole evolution depends only
    on (alpha, gamma, epsilon).  Hence, for any time-dependent quantity,

        F_tau(extract_Q_T(build_Q_adim(a, g, e)), mu10 * t)
            == F_tau(extract_Q_T(build_Q(lam01, lam02, lam12, mu10)), t)

    and the same holds for pi_t, R0_exact, pi_T_eigenmode, ...  Only when mu10 = 1
    [year^-1] does t_tilde coincide numerically with t in years.

    Parameters
    ----------
    alpha   : lambda_01 / mu10   (damage-to-recovery ratio)
    gamma   : lambda_02 / mu10   (collapse-to-recovery ratio)
    epsilon : lambda_12 / lambda_01  (damage-escalation ratio)

    Returns
    -------
    Q_tilde : (3, 3) ndarray, dimensionless.  Rows sum to zero; state 2 is absorbing.

    See also
    --------
    build_Q : dimensional generator from the physical rates [year^-1].
    """
    return build_Q(alpha, gamma, alpha * epsilon, 1.0)


def extract_Q_T(Q: np.ndarray) -> np.ndarray:
    """
    Extract the transient sub-generator Q_T (Eq. 9).

    For the three-state model the transient states are {0, 1};
    state 2 is the absorbing collapse state.

    Parameters
    ----------
    Q : (C+1, C+1) full generator

    Returns
    -------
    Q_T : (C, C) transient sub-generator
    """
    C = Q.shape[0] - 1          # number of transient states
    return Q[:C, :C].copy()


# ---------------------------------------------------------------------------
# 2.  Kolmogorov solution  (Eq. 8)
# ---------------------------------------------------------------------------

def pi_t(Q: np.ndarray, t: float,
         pi0: np.ndarray | None = None) -> np.ndarray:
    """
    State probability vector at time t  (Eq. 8).

        pi(t) = pi(0) · exp(Q t)

    Parameters
    ----------
    Q   : full generator (C+1, C+1)
    t   : time [year]
    pi0 : initial distribution (row vector, length C+1); default e_0

    Returns
    -------
    pi : (C+1,) row probability vector
    """
    if pi0 is None:
        pi0 = np.zeros(Q.shape[0])
        pi0[0] = 1.0
    return pi0 @ expm(Q * t)


# ---------------------------------------------------------------------------
# 3.  Parametric studies  (case study, Section 4, and toy example, Section 3)
# ---------------------------------------------------------------------------

def set_recovery_rate(Q_base: np.ndarray, mu_new: float) -> np.ndarray:
    """
    Return a copy of the full generator with the recovery rate mu10 replaced.

    The seismic rates (lambda_01, lambda_02, lambda_12) are kept fixed and the
    diagonal of row 1 is updated so that the row still sums to zero.

    Parameters
    ----------
    Q_base : (3, 3) full generator  (ds0, ds1, ds2 absorbing)
    mu_new : new recovery rate mu10 [year^-1]

    Returns
    -------
    Q : (3, 3) ndarray
    """
    Q = Q_base.copy()
    Q[1, 0] = mu_new
    Q[1, 1] = -(mu_new + Q_base[1, 2])   # -(mu10 + lam12)
    return Q


def reliability_resilience_summary(Q_full: np.ndarray, T_hor: float, n_quad: int = 600) -> dict:
    """
    Reliability + resilience pipeline for one full generator and one horizon.

    Used for Table 5 of the paper.

    Parameters
    ----------
    Q_full : (3, 3) full generator  (ds0, ds1, ds2 absorbing)
    T_hor  : time horizon [year]
    n_quad : Gauss-Legendre points for the M0 integral

    Returns
    -------
    dict with keys
        mu10, lam01, lam02, lam12 : rates read from Q_full [year^-1]
        F, S, beta                : failure / survival probability, Eq. (10)-(11)
        R0, rho                   : exact resilience metric and index, Eq. (18)-(19)
        R0_sp, rho_sp, dR0        : spectral approximation, Eq. (27), and |R0 - R0_sp|
        sigma1, sigma2, c1, c2    : eigenvalues and modal coefficients of Q_T
        q0, w0, ratio             : QSD component, weight, |sigma2 / sigma1|
    """
    Q_T = Q_full[:2, :2].copy()       # transient 2x2 block

    F, S   = metrics.F_tau(Q_T, T_hor)
    beta   = metrics.beta_index(F)
    R0_ex  = metrics.R0_exact(Q_T, T_hor, n_quad=n_quad)
    rho_ex = metrics.rho_index(R0_ex)
    spec   = spectral.spectral_analysis(Q_T)
    R0_sp  = spectral.R0_spectral(spec)
    rho_sp = metrics.rho_index(R0_sp)

    return dict(
        mu10=Q_full[1, 0], lam01=Q_full[0, 1],
        lam02=Q_full[0, 2], lam12=Q_full[1, 2],
        F=F, S=S, beta=beta,
        R0=R0_ex, rho=rho_ex,
        R0_sp=R0_sp, rho_sp=rho_sp, dR0=abs(R0_ex - R0_sp),
        sigma1=spec['sigma1'], sigma2=spec['sigma2'],
        c1=spec['c1'], c2=float(spec['c'][1]),
        q0=spec['q0'], w0=spec['w0'],
        ratio=abs(spec['sigma2'] / spec['sigma1']),
    )


def beta_rho_vs_mu10(Q_base: np.ndarray, mus, T_hor: float,
                         n_quad: int = 300):
    """
    (beta, rho) for a sweep of recovery rates, keeping the seismic rates fixed.

    Used for Figure 13 of the paper.

    Parameters
    ----------
    Q_base : (3, 3) full generator
    mus    : iterable of recovery rates mu10 [year^-1]
    T_hor  : time horizon [year]
    n_quad : Gauss-Legendre points for the M0 integral

    Returns
    -------
    betas, rhos : ndarrays, one value per mu
    """
    betas, rhos = [], []
    for mu in mus:
        r = reliability_resilience_summary(set_recovery_rate(Q_base, mu), T_hor, n_quad=n_quad)
        betas.append(r['beta'])
        rhos.append(r['rho'])
    return np.array(betas), np.array(rhos)


def beta_from_dimensionless(alpha: float, gamma: float, epsilon: float,
                     mu10: float, T_hor: float) -> float:
    """
    Reliability index beta(T_hor) of the three-state model, from the
    dimensionless parameters of Eq. (28)-(29).  Toy example, Section 3.
    Q_tilde is evaluated at the dimensionless time  t_tilde = mu10 * T_hor.
    """
    Q_T = extract_Q_T(build_Q_adim(alpha, gamma, epsilon))
    F, _ = metrics.F_tau(Q_T, mu10 * T_hor)
    return metrics.beta_index(F)


def beta_rho_vs_mu10_from_rates(lam01: float, lam02: float, lam12: float, mu10_arr,
                     T_hor: float, n_quad: int = 50):
    """
    (beta, rho) of the three-state model for fixed seismic rates and a sweep of
    recovery rates mu10.  Toy example, Figure 7.

    Parameters
    ----------
    lam01, lam02, lam12 : seismic rates [year^-1]
    mu10_arr            : iterable of recovery rates [year^-1]
    T_hor               : time horizon [year]
    n_quad              : Gauss-Legendre points for the M0 integral

    Returns
    -------
    betas, rhos : ndarrays, one value per mu10
    """
    betas, rhos = [], []
    for m in mu10_arr:
        Q_T = extract_Q_T(build_Q(lam01, lam02, lam12, m))
        F, _ = metrics.F_tau(Q_T, T_hor)
        betas.append(metrics.beta_index(F))
        rhos.append(metrics.rho_index(metrics.R0_exact(Q_T, T_hor, n_quad=n_quad)))
    return np.array(betas), np.array(rhos)


def Q0_rho(Q_full: np.ndarray, T_hor: float, n_quad: int = 300):
    """
    Compute Q0(T_hor) = 1 - R0(T_hor)  (Eq. 17-18)
    and rho = -Phi^{-1}(R0)            (Eq. 19).

    Parameters
    ----------
    Q_full : (3, 3) full generator  — ds0, ds1, ds2(absorbing)
    T_hor  : time horizon [year]
    n_quad : Gauss-Legendre quadrature points for M0 integral

    Returns
    -------
    Q0  : float in [0, 1]   — conditional fraction of time in ds0
    rho : float              — standardized resilience index
    """
    Q_T = Q_full[:2, :2].copy()
    R0  = metrics.R0_exact(Q_T, T_hor, n_quad=n_quad)
    Q0  = 1.0 - R0
    rho = metrics.rho_index(R0)
    return float(Q0), float(rho)
