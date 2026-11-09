"""
rcubed.metrics
=================
Reliability and resilience metrics of a CTMC with one absorbing (collapse) state.

Reference
---------
Nardin, Marelli, Sudret, Broccardo.
"A Generalized Framework for Performance-Based Earthquake Engineering:
Integrated Assessment of Structural Reliability and Resilience."
Structural Safety (accepted for publication).

Equation index (numbering of the paper)
---------------------------------------
Eq. (10)       -> F_tau                  : failure (absorption) probability and survival function
Eq. (11)       -> beta_index             : reliability index
Eq. (13), (16) -> cond_occ_time          : conditional expected occupation time  E[A0(t) | T>t]
Eq. (14)       -> occ_numerator          : numerator  E[A0(t) 1_{T>t}]  (integral form)
Eq. (15)       -> M0_kernel              : matrix-valued kernel  M0(t)
Eq. (17)       -> R0_complement_exact    : Q0(t) = 1 - R0(t), conditional occupation fraction of ds0
Eq. (18)       -> R0_exact               : resilience metric  R0(t)  (exact)
Eq. (19)       -> rho_index              : resilience index  rho

Unconditional formulation of the REC2026 conference paper (Section 3 below; superseded by
the conditional one above, kept to reproduce that paper):
occ_time_truncated     : expected time in each transient state up to t, truncated fundamental matrix
R0_unconditional       : R0 = n00(t) / t, unconditional fraction of time in ds0
rho_unconditional      : rho = -Phi^{-1}(1 - R0)

All functions take the transient sub-generator Q_T (see core.extract_Q_T).

Usage
-----
    from rcubed import core, metrics

    Q_T  = core.extract_Q_T(core.build_Q(lam01=1/50, lam02=1/2475, lam12=1/100, mu10=8.4))
    F, S = metrics.F_tau(Q_T, t=50.0)
    beta = metrics.beta_index(F)
    rho  = metrics.rho_index(metrics.R0_exact(Q_T, t=50.0))
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats  import norm


# ---------------------------------------------------------------------------
# 1.  Reliability  –  Eq. (10) and Eq. (11)
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
# 2.  Resilience – exact formulation  –  Eq. (12)–(19)
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
# 3.  Unconditional formulation  –  REC2026 conference paper
# ---------------------------------------------------------------------------
# Earlier, unconditional definition of the resilience metric, used in
#   Nardin et al., REC2026 conference paper (toy example, Figs. 2-5).
# The time spent in ds0 is averaged over all trajectories, including those that
# collapse before t (after collapse the structure is no longer in ds0).
# The Structural Safety paper replaces it with the conditional metric of
# Section 2 (fraction of time in ds0 given survival, Eq. 17-19).

def occ_time_truncated(Q_T: np.ndarray, t: float,
                       pi0_T: np.ndarray | None = None) -> np.ndarray:
    """
    Expected time spent in each transient state up to t (truncated fundamental matrix).

        N(t)   = int_0^t exp(Q_T s) ds = (-Q_T)^{-1} (I - exp(Q_T t))
        n_j(t) = pi_T(0) N(t) e_j

    Parameters
    ----------
    Q_T   : (C, C) transient sub-generator
    t     : time horizon [year]
    pi0_T : initial transient distribution; default e_0

    Returns
    -------
    n : (C,) ndarray  expected occupation times [year]  (n[0] = n00(t) for pi0_T = e_0)
    """
    C = Q_T.shape[0]
    if pi0_T is None:
        pi0_T = np.zeros(C)
        pi0_T[0] = 1.0
    N_t = np.linalg.solve(-Q_T, np.eye(C) - expm(Q_T * t))
    return pi0_T @ N_t


def R0_unconditional(Q_T: np.ndarray, t: float,
                     pi0_T: np.ndarray | None = None) -> float:
    """
    Unconditional resilience metric of the REC2026 paper.

        R0(t) = n00(t) / t

    Expected fraction of [0, t] spent in the undamaged state ds0 (1 = always undamaged).

    Returns
    -------
    R0 : float in [0, 1]
    """
    n = occ_time_truncated(Q_T, t, pi0_T)
    return float(np.clip(n[0] / t, 0.0, 1.0))


def rho_unconditional(R0: float) -> float:
    """
    Resilience index of the REC2026 paper.

        rho = -Phi^{-1}(1 - R0)

    Parameters
    ----------
    R0 : unconditional resilience metric (R0_unconditional), in (0, 1)
    """
    eps = 1e-12
    R0  = float(np.clip(R0, eps, 1.0 - eps))
    return float(-norm.ppf(1.0 - R0))
