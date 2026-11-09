"""
rcubed.montecarlo
====================
Monte Carlo simulation of the CTMC trajectories, used to verify the closed-form
reliability and resilience metrics (REC2026 conference paper, Figs. 3-4).

Each trajectory starts in ds0 and is simulated with the standard
(Gillespie) algorithm up to the time horizon: exponential holding time in the
current state, then a jump to state j with probability q_ij / (-q_ii).  The
absorbing (collapse) state is the last one.

Estimators (one sample per trajectory)
--------------------------------------
F(t)                : indicator of collapse before t                    -> metrics.F_tau
R0, unconditional   : (time in ds0) / t, all trajectories               -> metrics.R0_unconditional
Q0, conditional     : (time in ds0) / t, surviving trajectories only     -> metrics.R0_complement_exact

Usage
-----
    from rcubed import core, montecarlo

    Q  = core.build_Q(lam01=0.02, lam02=1e-4, lam12=5e-3, mu10=0.1)
    mc = montecarlo.run(Q, T_hor=50.0, n_sim=10_000, seed=0)
    mc["F"], mc["R0_unconditional"], mc["Q0_conditional"]
"""

import numpy as np


def simulate_trajectory(Q: np.ndarray, T_hor: float, rng: np.random.Generator,
                        x0: int = 0) -> np.ndarray:
    """
    Simulate one trajectory of the CTMC on [0, T_hor].

    Parameters
    ----------
    Q     : (n, n) full generator [year^-1]
    T_hor : time horizon [year]
    rng   : numpy random Generator
    x0    : initial state (default 0, undamaged)

    Returns
    -------
    time_in_state : (n,) ndarray, time spent in each state [year]; sums to T_hor
    """
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    t, x = 0.0, x0
    time_in_state = np.zeros(n)
    while t < T_hor:
        rate = -Q[x, x]
        if rate <= 0.0:                         # absorbing state
            time_in_state[x] += T_hor - t
            break
        dt = rng.exponential(1.0 / rate)
        if t + dt >= T_hor:
            time_in_state[x] += T_hor - t
            break
        time_in_state[x] += dt
        p = Q[x].copy()
        p[x] = 0.0
        x = rng.choice(n, p=p / rate)
        t += dt
    return time_in_state


def run(Q: np.ndarray, T_hor: float, n_sim: int = 10_000, seed: int | None = 0,
        x0: int = 0) -> dict:
    """
    Monte Carlo estimate of F, R0 (unconditional) and Q0 (conditional) at T_hor.

    Parameters
    ----------
    Q     : (n, n) full generator [year^-1]; the last state is absorbing
    T_hor : time horizon [year]
    n_sim : number of trajectories
    seed  : seed of the random generator (None: not reproducible)
    x0    : initial state

    Returns
    -------
    dict with
        time_in_state    : (n_sim, n) time spent in each state by each trajectory
        collapsed        : (n_sim,) bool, collapse before T_hor
        F                : estimate of F(T_hor)
        R0_unconditional : estimate of n00(T_hor) / T_hor (all trajectories)
        Q0_conditional   : estimate of E[A0 | T > T_hor] / T_hor (surviving trajectories)
        F_samples, R0_samples : per-trajectory samples (for convergence plots)
    """
    rng = np.random.default_rng(seed)
    tis = np.array([simulate_trajectory(Q, T_hor, rng, x0) for _ in range(n_sim)])
    collapsed = tis[:, -1] > 0.0
    frac0 = tis[:, 0] / T_hor
    survivors = ~collapsed
    return {
        "time_in_state": tis,
        "collapsed": collapsed,
        "F": float(collapsed.mean()),
        "R0_unconditional": float(frac0.mean()),
        "Q0_conditional": float(frac0[survivors].mean()) if survivors.any() else np.nan,
        "F_samples": collapsed.astype(float),
        "R0_samples": frac0,
    }


def running_mean_ci(samples: np.ndarray, z: float = 1.96):
    """
    Running mean of the samples and its confidence band (normal approximation).

    Returns
    -------
    n, mean, lower, upper : ndarrays of length len(samples)
    """
    samples = np.asarray(samples, float)
    n = np.arange(1, len(samples) + 1)
    mean = np.cumsum(samples) / n
    var = np.maximum(np.cumsum(samples ** 2) / n - mean ** 2, 0.0)
    se = np.sqrt(var / n)
    return n, mean, mean - z * se, mean + z * se
