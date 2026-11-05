"""
ctmc_pbee.spectral
==================
Spectral analysis of the transient sub-generator Q_T: eigenmode expansion,
quasi-stationary distribution (QSD) and spectral approximation of the
resilience metric.

Reference
---------
Nardin, Marelli, Sudret, Broccardo.
"A Generalized Framework for Performance-Based Earthquake Engineering:
Integrated Assessment of Structural Reliability and Resilience."
Structural Safety (under review, STRUCS-S-26-00168).

Equation index (numbering of the paper)
---------------------------------------
Eq. (20)       -> eigenmode_expansion    : eigenmode expansion of  exp(Q_T t)
Eq. (21)       -> pi_T_eigenmode         : transient distribution via eigenmode expansion
Eq. (22)       -> spectral_analysis      : left/right eigenvectors of Q_T (dominant mode)
Eq. (23)       -> pi_tilde               : normalized transient distribution
Eq. (24)       -> spectral_analysis q0   : QSD limit of the undamaged-state probability
Eq. (25)       -> S_T_asymptotic         : asymptotic survival function
Eq. (26)       -> R0_asymptotic_integrand: asymptotic result for 1 - R0(t)
Eq. (27)       -> R0_spectral            : spectral approximation  R0 ~ 1 - w0 * q0

Usage
-----
    from ctmc_pbee import core, metrics, spectral

    Q_T   = core.extract_Q_T(core.build_Q(lam01=1/50, lam02=1/2475, lam12=1/100, mu10=8.4))
    spec  = spectral.spectral_analysis(Q_T)
    R0_sp = spectral.R0_spectral(spec)               # compare with metrics.R0_exact(Q_T, t)
"""

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Spectral analysis  –  Eq. (20)–(27)
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
