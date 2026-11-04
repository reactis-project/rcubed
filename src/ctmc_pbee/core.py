"""
ctmc_pbee.core
==============
Reliability and resilience of a CTMC with one absorbing (collapse) state:
the equations of the paper (Eq. 8-27) and the parametric studies built on them.

Reference
---------
Nardin, Marelli, Sudret, Broccardo (2025).
"A Generalized Framework for Performance-Based Earthquake Engineering:
Integrated Assessment of Structural Reliability and Resilience."

Equation index (numbering of the paper)
---------------------------------------
Eq. (8)        -> pi_t                   : state probabilities  pi(t) = pi(0) exp(Q t)
Eq. (9)        -> extract_Q_T            : transient sub-generator Q_T
Eq. (10)       -> F_tau                  : failure (absorption) probability and survival function
Eq. (11)       -> beta_index             : reliability index
Eq. (13), (16) -> cond_occ_time          : conditional expected occupation time  E[A0(t) | T>t]
Eq. (14)       -> occ_numerator          : numerator  E[A0(t) 1_{T>t}]  (integral form)
Eq. (15)       -> M0_kernel              : matrix-valued kernel  M0(t)
Eq. (17)       -> R0_complement_exact    : Q0(t) = 1 - R0(t), conditional occupation fraction of ds0
Eq. (18)       -> R0_exact               : resilience metric  R0(t)  (exact)
Eq. (19)       -> rho_index              : resilience index  rho
Eq. (20)       -> eigenmode_expansion    : eigenmode expansion of  exp(Q_T t)
Eq. (21)       -> pi_T_eigenmode         : transient distribution via eigenmode expansion
Eq. (22)       -> spectral_analysis      : left/right eigenvectors of Q_T (dominant mode)
Eq. (23)       -> pi_tilde               : normalized transient distribution
Eq. (24)       -> spectral_analysis q0   : QSD limit of the undamaged-state probability
Eq. (25)       -> S_T_asymptotic         : asymptotic survival function
Eq. (26)       -> R0_asymptotic_integrand: asymptotic result for 1 - R0(t)
Eq. (27)       -> R0_spectral            : spectral approximation  R0 ~ 1 - w0 * q0
Eq. (28)-(29)  -> build_Q                : dimensionless parameters and Q = mu10 * Q_tilde (Section 3)

Usage
-----
    from ctmc_pbee import core as ct
    import numpy as np

    Q = ct.build_Q(alpha=0.5, gamma=1e-3, epsilon=2.0, mu10=1.0)
    Q_T = ct.extract_Q_T(Q)

    F, S   = ct.F_tau(Q_T, t=50.0)
    beta   = ct.beta_index(F)

    R0     = ct.R0_exact(Q_T, t=50.0)
    rho    = ct.rho_index(R0)

    spec   = ct.spectral_analysis(Q_T)
    R0_sp  = ct.R0_spectral(spec)
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats  import norm

# ---------------------------------------------------------------------------
# 1.  Generator construction  (Eq. 28-29, Section 3)
# ---------------------------------------------------------------------------

def build_Q(alpha: float, gamma: float, epsilon: float,
            mu10: float = 1.0) -> np.ndarray:
    """
    Build the full 3×3 generator matrix for the three-state CTMC.

    Factorisation (Eq. 29):
        Q = mu10 * Q_tilde(alpha, gamma, epsilon)

    Parameters
    ----------
    alpha   : lambda_01 / mu10   (damage-to-recovery ratio)
    gamma   : lambda_02 / mu10   (collapse-to-recovery ratio)
    epsilon : lambda_12 / lambda_01  (damage-escalation ratio)
    mu10    : recovery rate [year^{-1}], default 1.0

    Returns
    -------
    Q : (3, 3) ndarray  [year^{-1}]
        Full generator.  Rows sum to zero; state 2 is absorbing.
    """
    lam01 = alpha   * mu10
    lam02 = gamma   * mu10
    lam12 = epsilon * lam01          # = epsilon * alpha * mu10

    Q = np.array([
        [-(lam01 + lam02),  lam01,              lam02  ],
        [ mu10,            -(mu10 + lam12),      lam12  ],
        [ 0.0,              0.0,                 0.0    ],
    ])
    assert np.allclose(Q.sum(axis=1), 0.0), "Row-sum check failed."
    return Q


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
# 3.  Reliability  –  Eq. (10) and Eq. (11)
# ---------------------------------------------------------------------------

def F_tau(Q_T: np.ndarray, t: float,
          pi0_T: np.ndarray | None = None) -> tuple[float, float]:
    """
    Failure (absorption) probability and survival function  (Eq. 10).

        F_T(t) = 1 - S_T(t)
        S_T(t) = pi_T(0) · exp(Q_T t) · 1_T

    Parameters
    ----------
    Q_T   : (C, C) transient sub-generator
    t     : time horizon [year]
    pi0_T : initial transient distribution (length C); default e_0

    Returns
    -------
    F : float  – failure probability in [0, 1]
    S : float  – survival probability in [0, 1]
    """
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C)
        pi0_T[0] = 1.0

    ones_T = np.ones(C)
    S = float(pi0_T @ expm(Q_T * t) @ ones_T)
    S = float(np.clip(S, 0.0, 1.0))
    F = 1.0 - S
    return F, S


def beta_index(F: float) -> float:
    """
    Reliability index  (Eq. 11).

        beta = -Phi^{-1}(F_T(t))

    Parameters
    ----------
    F : failure probability in (0, 1)

    Returns
    -------
    beta : float
    """
    eps = 1e-12
    F   = float(np.clip(F, eps, 1.0 - eps))
    return float(-norm.ppf(F))


# ---------------------------------------------------------------------------
# 4.  Resilience – exact formulation  –  Eq. (12)–(19)
# ---------------------------------------------------------------------------

def M0_kernel(Q_T: np.ndarray, t: float,
              n_quad: int = 500) -> np.ndarray:
    """
    Matrix-valued kernel M0(t)  (Eq. 15).

        M0(t) = int_0^t  exp(Q_T s) D0 exp(Q_T (t-s)) ds

    where D0 = e0 e0^T selects the undamaged state.

    Computed via Gauss–Legendre quadrature on [0, t].

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : upper integration limit [year]
    n_quad : number of quadrature points

    Returns
    -------
    M0 : (C, C) ndarray
    """
    C  = Q_T.shape[0]
    D0 = np.zeros((C, C))
    D0[0, 0] = 1.0                        # D0 = e0 e0^T

    s_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # map from [-1,1] to [0,t]
    s_nodes  = 0.5 * t * (s_nodes + 1.0)
    weights  = 0.5 * t * weights

    M0 = np.zeros((C, C))
    for s, w in zip(s_nodes, weights):
        M0 += w * (expm(Q_T * s) @ D0 @ expm(Q_T * (t - s)))
    return M0


def occ_numerator(Q_T: np.ndarray, t: float,
                  pi0_T: np.ndarray | None = None,
                  n_quad: int = 500) -> float:
    """
    Numerator of the conditional occupation time  (Eq. 14).

        E[A0(t) 1_{T>t}] = e0^T M0(t) 1_T
                         = int_0^t  e0^T exp(Q_T s) D0 exp(Q_T(t-s)) 1_T ds

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : time [year]
    pi0_T  : initial distribution; default e_0
    n_quad : quadrature points

    Returns
    -------
    num : float
    """
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C)
        pi0_T[0] = 1.0

    ones_T = np.ones(C)
    M0     = M0_kernel(Q_T, t, n_quad=n_quad)
    return float(pi0_T @ M0 @ ones_T)


def cond_occ_time(Q_T: np.ndarray, t: float,
                  pi0_T: np.ndarray | None = None,
                  n_quad: int = 500) -> float:
    """
    Conditional expected occupation time of ds0  (Eq. 13 / Eq. 16).

        E[A0(t) | T > t] = e0^T M0(t) 1_T  /  S_T(t)

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : time [year]
    pi0_T  : initial distribution; default e_0
    n_quad : quadrature points

    Returns
    -------
    E_cond : float  [year]
    """
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C)
        pi0_T[0] = 1.0

    _, S   = F_tau(Q_T, t, pi0_T)
    num    = occ_numerator(Q_T, t, pi0_T, n_quad=n_quad)
    return num / S if S > 1e-15 else 0.0


def R0_exact(Q_T: np.ndarray, t: float,
             pi0_T: np.ndarray | None = None,
             n_quad: int = 500) -> float:
    """
    Exact resilience metric R0(t)  (Eq. 18).

        R0(t) = 1 - E[A0(t) | T>t] / t
              = 1 - e0^T M0(t) 1_T  /  (t S_T(t))

    R0 in [0,1]:  0 = always undamaged (perfect resilience),
                  1 = always in damaged states prior to collapse.

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : time horizon [year]
    pi0_T  : initial distribution; default e_0
    n_quad : quadrature points for M0

    Returns
    -------
    R0 : float in [0, 1]
    """
    E_cond = cond_occ_time(Q_T, t, pi0_T, n_quad=n_quad)
    R0     = 1.0 - E_cond / t
    return float(np.clip(R0, 0.0, 1.0))


def rho_index(R0: float) -> float:
    """
    Resilience index  (Eq. 19).

        rho = -Phi^{-1}(R0)

    Parameters
    ----------
    R0 : resilience metric in (0, 1)

    Returns
    -------
    rho : float
    """
    eps = 1e-12
    R0  = float(np.clip(R0, eps, 1.0 - eps))
    return float(-norm.ppf(R0))


# ---------------------------------------------------------------------------
# 5.  Spectral analysis  –  Eq. (20)–(27)
# ---------------------------------------------------------------------------

def spectral_analysis(Q_T: np.ndarray,
                      initial_state: int = 0) -> dict:
    """
    Full spectral decomposition of the transient sub-generator Q_T.

    Computes eigenvalues, left/right eigenvectors, modal coefficients,
    and the quasi-stationary distribution (QSD).

    Normalisation convention (Eq. 22):
        nu @ Q_T = sigma1 * nu
        Q_T @ w  = sigma1 * w
        nu @ 1_T = 1          (nu is a probability vector)
        nu @ w   = 1          (bi-orthonormal pair)

    Parameters
    ----------
    Q_T           : (C, C) transient sub-generator
    initial_state : index of the starting state (default 0 = ds0)

    Returns
    -------
    dict with keys
    --------------
    sigma   : (C,) eigenvalues sorted by decreasing real part (all < 0)
    W       : (C, C) right eigenvectors as columns, sorted consistently
    Nu      : (C, C) left eigenvectors as rows, sorted consistently,
                     bi-orthonormal to W  (Nu @ W = I)
    nu      : (C,) dominant left eigenvector = QSD  (Eq. 22–24)
    w       : (C,) dominant right eigenvector       (Eq. 22)
    sigma1  : float  dominant (least negative) eigenvalue  (Eq. 22)
    sigma2  : float  second eigenvalue
    q0      : float  QSD probability of undamaged state  (Eq. 24)
    w0      : float  w[initial_state]  (Eq. 25)
    c       : (C,) modal coefficients c_k = pi0 @ w_k   (Eq. 21)
    c1      : float  modal coefficient of dominant mode
    spectral_gap : float  |sigma2 - sigma1|
    """
    C = Q_T.shape[0]

    # --- right eigenproblem:  Q_T w_k = sigma_k w_k
    raw_vals, W_raw = np.linalg.eig(Q_T)

    # sort by decreasing real part (dominant = index 0)
    order  = np.argsort(np.real(raw_vals))[::-1]
    sigma  = np.real(raw_vals[order])
    W      = np.real(W_raw[:, order])     # columns are right eigenvectors

    # --- left eigenvectors via inverse (bi-orthonormal):  Nu = inv(W)^T
    Nu = np.real(np.linalg.inv(W).T)     # rows are left eigenvectors

    # Dominant pair
    nu_raw = Nu[:, 0]                    # dominant left eigenvector (un-normalised row)
    w_raw  = W[:, 0]                     # dominant right eigenvector

    # Apply normalisation  nu 1_T = 1  and  nu w = 1  (Eq. 22)
    ones_T = np.ones(C)
    nu     = nu_raw / (nu_raw @ ones_T)  # nu 1_T = 1
    # Re-scale w so that nu @ w = 1
    w      = w_raw  / (nu @ w_raw)       # nu @ w = 1

    # QSD  (Eq. 23–24)
    q0     = float(nu[0])                # conditional probability of ds0
    w0     = float(w[initial_state])     # initial excitation of dominant mode (Eq. 25)
    sigma1 = float(sigma[0])
    sigma2 = float(sigma[1]) if C > 1 else np.nan

    # --- modal coefficients  c_k = pi0_T @ w_k  (Eq. 21)
    pi0_T = np.zeros(C)
    pi0_T[initial_state] = 1.0

    # Build full right-eigenvector matrix with proper normalisation
    # Columns of W_norm are the w_k normalised so nu_k @ w_k = 1
    W_norm  = W.copy()
    Nu_norm = Nu.copy()
    for k in range(C):
        nu_k  = Nu_norm[:, k] / (Nu_norm[:, k] @ ones_T)   # nu_k 1_T = 1
        scale = nu_k @ W_norm[:, k]
        W_norm[:, k]  /= scale                              # nu_k @ w_k = 1
        Nu_norm[:, k]  = nu_k

    c  = np.array([float(pi0_T @ W_norm[:, k]) for k in range(C)])
    c1 = c[0]

    return {
        "sigma"        : sigma,
        "W"            : W_norm,
        "Nu"           : Nu_norm,
        "nu"           : nu,           # QSD = dominant left eigenvector
        "w"            : w,            # dominant right eigenvector
        "sigma1"       : sigma1,
        "sigma2"       : sigma2,
        "q0"           : q0,           # Eq. 24
        "w0"           : w0,           # Eq. 25
        "c"            : c,            # Eq. 21
        "c1"           : c1,
        "spectral_gap" : abs(sigma2 - sigma1) if C > 1 else np.nan,
    }


def eigenmode_expansion(Q_T: np.ndarray, t: float,
                        spec: dict | None = None) -> np.ndarray:
    """
    Eigenmode expansion of exp(Q_T t)  (Eq. 20).

        exp(Q_T t) = sum_k  exp(sigma_k t)  w_k nu_k

    Parameters
    ----------
    Q_T  : (C, C) transient sub-generator
    t    : time [year]
    spec : pre-computed spectral_analysis dict (optional, for efficiency)

    Returns
    -------
    expQt : (C, C) matrix exponential (reconstructed from eigenmodes)
    """
    if spec is None:
        spec = spectral_analysis(Q_T)

    C      = Q_T.shape[0]
    expQt  = np.zeros((C, C))
    W, Nu  = spec["W"], spec["Nu"]
    sigma  = spec["sigma"]

    for k in range(C):
        # Rank-1 projector:  w_k  (column)  ⊗  nu_k  (row)
        projector = np.outer(W[:, k], Nu[:, k])
        expQt    += np.exp(sigma[k] * t) * projector
    return expQt


def pi_T_eigenmode(Q_T: np.ndarray, t: float,
                   spec: dict | None = None,
                   pi0_T: np.ndarray | None = None) -> np.ndarray:
    """
    Transient distribution via eigenmode expansion  (Eq. 21).

        pi_T(t) = pi_T(0) exp(Q_T t) = sum_k  c_k exp(sigma_k t) nu_k

    Parameters
    ----------
    Q_T   : (C, C) transient sub-generator
    t     : time [year]
    spec  : pre-computed spectral dict (optional)
    pi0_T : initial distribution; default e_0

    Returns
    -------
    pi_T : (C,) row vector
    """
    if spec is None:
        spec = spectral_analysis(Q_T)

    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C); pi0_T[0] = 1.0

    # Modal coefficients for this specific pi0_T
    c_k  = np.array([float(pi0_T @ spec["W"][:, k]) for k in range(C)])
    pi_T = np.zeros(C)
    for k in range(C):
        pi_T += c_k[k] * np.exp(spec["sigma"][k] * t) * spec["Nu"][:, k]
    return pi_T


def pi_tilde(Q_T: np.ndarray, t: float,
             pi0_T: np.ndarray | None = None) -> np.ndarray:
    """
    Normalised transient distribution (conditional on survival)  (Eq. 23).

        pi~_T(t) = pi_T(t) / (pi_T(t) @ 1_T)

    Converges to the QSD nu as t -> inf  (Eq. 24).

    Parameters
    ----------
    Q_T   : (C, C) transient sub-generator
    t     : time [year]
    pi0_T : initial distribution; default e_0

    Returns
    -------
    pi_tilde : (C,) normalised row vector
    """
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C); pi0_T[0] = 1.0

    pi_T = pi0_T @ expm(Q_T * t)
    norm = pi_T @ np.ones(C)
    if norm < 1e-15:
        return np.zeros(C)
    return pi_T / norm


def S_T_asymptotic(t: float, spec: dict) -> float:
    """
    Dominant-mode asymptotic approximation of the survival function  (Eq. 25).

        S_T(t) ~ w0 * exp(sigma1 * t),   t -> inf

    Parameters
    ----------
    t    : time [year]
    spec : spectral_analysis dict

    Returns
    -------
    S_asymp : float
    """
    return float(spec["w0"] * np.exp(spec["sigma1"] * t))


def R0_complement_exact(Q_T: np.ndarray, t: float,
                        pi0_T: np.ndarray | None = None,
                        n_quad: int = 500) -> float:
    """
    Exact value of 1 - R0(t) = Q0(t)  (Eq. 17).

        1 - R0(t) = E[A0(t)|T>t] / t
                  = e0^T M0(t) 1_T / (t S_T(t))

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : time [year]
    pi0_T  : initial distribution; default e_0
    n_quad : quadrature points

    Returns
    -------
    comp : float  in [0, 1]
    """
    return 1.0 - R0_exact(Q_T, t, pi0_T, n_quad=n_quad)


def R0_asymptotic_integrand(Q_T: np.ndarray, t: float,
                            spec: dict | None = None,
                            pi0_T: np.ndarray | None = None,
                            n_quad: int = 500) -> float:
    """
    Asymptotic approximation of 1 - R0(t)  (Eq. 26).

        1 - R0(t) ~ (w0 / t) * int_0^t  pi~_0(s) ds

    Evaluated numerically via Gauss–Legendre quadrature.

    Parameters
    ----------
    Q_T    : (C, C) transient sub-generator
    t      : time [year]
    spec   : spectral_analysis dict (optional)
    pi0_T  : initial distribution; default e_0
    n_quad : quadrature points

    Returns
    -------
    one_minus_R0 : float
    """
    if spec is None:
        spec = spectral_analysis(Q_T)
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C); pi0_T[0] = 1.0

    s_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    s_nodes = 0.5 * t * (s_nodes + 1.0)
    weights = 0.5 * t * weights

    integral = 0.0
    for s, w in zip(s_nodes, weights):
        pi_tilde_s = pi_tilde(Q_T, s, pi0_T)
        integral  += w * pi_tilde_s[0]           # pi~_0(s)

    return float(spec["w0"] / t * integral)


def R0_spectral(spec: dict) -> float:
    """
    Spectral (asymptotic) approximation of R0  (Eq. 27).

        R0 ≈ 1 - w0 * q0

    Valid when t >> 1/|sigma1| and the spectral gap is large.

    Parameters
    ----------
    spec : spectral_analysis dict

    Returns
    -------
    R0_sp : float in [0, 1]
    """
    R0_sp = 1.0 - spec["w0"] * spec["q0"]
    return float(np.clip(R0_sp, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 7.  Parametric studies  (case study, Section 4, and toy example, Section 3)
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

    F, S   = F_tau(Q_T, T_hor)
    beta   = beta_index(F)
    R0_ex  = R0_exact(Q_T, T_hor, n_quad=n_quad)
    rho_ex = rho_index(R0_ex)
    spec   = spectral_analysis(Q_T)
    R0_sp  = R0_spectral(spec)
    rho_sp = rho_index(R0_sp)

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
    """
    Q_T = extract_Q_T(build_Q(alpha, gamma, epsilon, mu10))
    F, _ = F_tau(Q_T, T_hor)
    return beta_index(F)


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
        Q_T = extract_Q_T(build_Q(lam01 / m, lam02 / m, lam12 / lam01, m))
        F, _ = F_tau(Q_T, T_hor)
        betas.append(beta_index(F))
        rhos.append(rho_index(R0_exact(Q_T, T_hor, n_quad=n_quad)))
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
    R0  = R0_exact(Q_T, T_hor, n_quad=n_quad)
    Q0  = 1.0 - R0
    rho = rho_index(R0)
    return float(Q0), float(rho)
