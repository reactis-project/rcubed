"""
ctmc_pbee.case_study
====================
Case study of the paper (Section 4): input loading, generator matrices and the
functions that reproduce Figures 9-13 and Tables 4-5.

Typical use (see ``notebooks/02_case_study.ipynb``)::

    from ctmc_pbee import case_study as cs

    inputs = cs.load_inputs()          # hazard + state-dependent fragilities (pkl)
    Qsim   = cs.build_Q(inputs)        # generator matrices of the case study
    cs.fig09_median_fragilities(inputs)
    table4  = cs.fig10_state_probabilities(Qsim)
    ...

The state-dependent fragilities stored in the pkl file were derived as in
Nardin et al. (2025), https://www.sciencedirect.com/science/article/pii/S0951832025002686.

Figures are saved as SVG in ``output/case_study/``.
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from scipy.linalg import expm

from ctmc_pbee import core, generator, metrics, plotting, utils
from ctmc_pbee.paths import DATA, output_dir, require_data

INPUT_PKL = "SS-input-hazard-fragilities-20260512.pkl"

# Colors of the two structures: braced frame (BF, 'fixed') and base-isolated (BI, 'isolated')
STRUCT_COLORS = {
    'isolated': {'color': '#77355B', 'label': 'isolated', 'linestyle': '-'},
    'fixed':    {'color': '#08407F', 'label': 'fixed',    'linestyle': '-'},
}
# Median recovery times [days] of Table 3: 'low' = short repair, 'high' = long repair
RECOVERY_DAYS = {
    'fixed':    {'low': 30, 'high': 365},
    'isolated': {'low': 1,  'high': 50},
}
HAZARD_QTILES = ['lambda_0.05', 'lambda_0.5', 'lambda_0.95']
FRAGILITY_QTILES = ['p01', 'p05', 'p16', 'p50', 'p84', 'p95', 'p99']
SCENARIO_MAP = {'short': 'low', 'long': 'high'}   # paper scenario -> recovery level


def _figdir():
    return output_dir('case_study')


def _save(fname, fig=None, savefig=True):
    if not savefig:
        return
    target = _figdir() / fname
    (fig or plt.gcf()).savefig(target, transparent=True, format='svg', bbox_inches='tight')
    print('Saved:', target)


# ─────────────────────────────────────────────────────────────────────────────
# Inputs and generator matrices
# ─────────────────────────────────────────────────────────────────────────────
def load_inputs(pkl=INPUT_PKL, recovery='recovery.json'):
    """Load hazard, state-dependent fragilities and recovery rates from ``data/``.

    Returns a dict with keys ``hazard``, ``fixed``, ``isolated``, ``recovery``.
    """
    require_data(pkl, recovery)
    with open(DATA / pkl, 'rb') as f:
        data = pickle.load(f)
    with open(DATA / recovery) as f:
        rec = json.load(f)
    return {
        'hazard':   data['hazard'],
        'fixed':    data['pb_transitions_exc_fixed'],
        'isolated': data['pb_transitions_exc_isolated'],
        'recovery': rec,
    }


def build_Q(inputs, recovery_days=RECOVERY_DAYS, verbose=False):
    """Generator matrices of the case study for every hazard and fragility quantile.

    Returns ``Qsim[site][hazard_qtile][structure][fragility_qtile][recovery_level]``
    (site ``'ita'``; recovery levels ``'low'`` and ``'high'``, see ``RECOVERY_DAYS``).
    """
    return generator.build_Q_matrices(
        hazard               = inputs['hazard'],
        hazard_locations     = ['ita'],
        hazard_qtiles        = HAZARD_QTILES,
        struct_dicts         = {'fixed': inputs['fixed'], 'isolated': inputs['isolated']},
        pb_transition_qtiles = FRAGILITY_QTILES,
        recovery_dict        = recovery_days,
        verbose              = verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: median state-dependent fragilities
# ─────────────────────────────────────────────────────────────────────────────
def fig09_median_fragilities(inputs, savefig=True):
    """Figure 9: median state-dependent fragilities, BF vs BI."""
    fixed, isolated = inputs['fixed'], inputs['isolated']
    fig, axes = plotting.plot_median_fragilities(
        pga            = fixed['pb_transitions_exc']['DS0|DS0']['pga'],
        ds_probs_bf    = fixed['pb_transitions_exc'],
        ds_probs_iso   = isolated['pb_transitions_exc'],
        struct_colors2 = STRUCT_COLORS,
        general        = {'muteFig': False},
    )
    _save('03-BF-vs-ISO-median_comparison.svg', fig, savefig)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 / Table 4: long-term state probabilities
# ─────────────────────────────────────────────────────────────────────────────
def fig10_state_probabilities(Qsim, hqtile='lambda_0.05', stqtile='p84',
                              T_table=50, max_time=475, savefig=True):
    """Figure 10 and Table 4: state probabilities pi(t) for short and long recovery.

    One figure per structure (top: short recovery, bottom: long recovery).
    Returns Table 4 (pi at ``T_table`` years) as a DataFrame.
    """
    rate_levels = ['low', 'high']
    struct_lines = {   # line colors per recovery level
        'low':  {'color': '#08407F', 'label': 'low',  'linestyle': '-'},
        'high': {'color': '#816EF6', 'label': 'high', 'linestyle': '-'},
    }
    t_vals = np.linspace(0, max_time, max_time + 1)
    P0 = np.array([1.0, 0.0, 0.0])
    table = {}

    for st in ['fixed', 'isolated']:
        table[st] = {}
        fig, axes = plt.subplots(2, figsize=(3.5, 3), sharex=True)
        for rec_level, ax in zip(rate_levels, axes):
            Q = Qsim['ita'][hqtile][st][stqtile][rec_level]
            P_t = np.array([P0 @ expm(Q * t) for t in t_vals])
            table[st][rec_level] = P_t[T_table].round(2).tolist()

            t_cross, p_cross = plotting.plt_intersection_p50(P_t, t_vals)
            style = struct_lines[rec_level]
            for idx in range(3):
                ax.plot(t_vals, P_t[:, idx], linestyle=style['linestyle'],
                        color=style['color'], label=style['label'])
            if t_cross is not None:
                ax.axvline(t_cross, color="darkgrey", linestyle="--", alpha=0.7)
                ax.plot(t_cross, p_cross, 'x', color="black", markersize=8, mew=2)
            ax.grid(True)
            ax.set_xlim([0, max_time])

        axes[-1].set_xlabel("Time [t]")
        plt.tight_layout()
        for ax in axes:
            plt.setp(ax.get_yticklabels(), visible=False)
        _save(f"03m-state-probabilities-ita-lambda_095-{st}-mean.svg", fig, savefig)
        plt.show()

    return pd.DataFrame(
        [[table['fixed'][l] for l in rate_levels] + [table['isolated'][l] for l in rate_levels]],
        index=['ita'],
        columns=pd.MultiIndex.from_product([['fixed', 'isolated'], ['mu_low', 'mu_high']]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: reliability vs mu10
# ─────────────────────────────────────────────────────────────────────────────
def fig11_reliability(Qsim, hqtile='lambda_0.05', stqtile='p50',
                      mus=None, thors=(1.0, 10, 50.0),
                      ref_mus_fix=(8.43, 2.10, 0.69), ref_mus_iso=(253, 8.43, 5.05),
                      mute=False, savefig=True):
    """Figure 11: failure probability F and reliability index beta vs mu10.

    Seismic rates fixed (long-recovery Q), one curve per time horizon.
    ``mute=True`` removes tick labels (version used for the paper layout).
    """
    mus = np.logspace(-1, 3, 50) if mus is None else mus
    thors = np.asarray(thors)
    Q_fix = Qsim['ita'][hqtile]['fixed'][stqtile]['high']
    Q_iso = Qsim['ita'][hqtile]['isolated'][stqtile]['high']

    def fading(hex_color):
        r, g, b, _ = to_rgba(hex_color)
        return LinearSegmentedColormap.from_list(hex_color, [(r, g, b, 1.0), (r, g, b, 0.4)], N=256)

    normCol = LogNorm(vmin=thors.min(), vmax=thors.max())
    cmap_iso = fading(STRUCT_COLORS['isolated']['color'])
    cmap_fix = fading(STRUCT_COLORS['fixed']['color'])

    def F_beta(Q, mu, T):
        F, _ = metrics.F_tau(core.extract_Q_T(core.set_recovery_rate(Q, mu)), T)
        return F, metrics.beta_index(F)

    fig, (axF, axB) = plt.subplots(1, 2, figsize=(9, 2.5))
    for T_hor in thors:
        F_iso, b_iso = zip(*[F_beta(Q_iso, mu, T_hor) for mu in mus])
        F_fix, b_fix = zip(*[F_beta(Q_fix, mu, T_hor) for mu in mus])
        c_iso, c_fix = cmap_iso(normCol(T_hor)), cmap_fix(normCol(T_hor))

        axB.semilogx(mus, b_iso, color=c_iso, linestyle='-')
        axB.semilogx(mus, b_fix, color=c_fix, linestyle='-')
        axF.semilogx(mus, F_iso, color=c_iso, linestyle='-')
        axF.semilogx(mus, F_fix, color=c_fix, linestyle='-')

        for mu in ref_mus_iso:
            F, b = F_beta(Q_iso, mu, T_hor)
            axB.plot(mu, b, 'o', color=c_iso, markersize=4)
            axF.plot(mu, F, 'o', color=c_iso, markersize=4)
        for mu in ref_mus_fix:
            F, b = F_beta(Q_fix, mu, T_hor)
            axB.plot(mu, b, 'o', color=c_fix, markersize=4)
            axF.plot(mu, F, 'o', color=c_fix, markersize=4)

    axF.grid(True, which="both", ls="--", alpha=0.5)
    axB.grid(True, which="both", ls="--", alpha=0.5)
    axB.set_ylim([0, 3.5])
    if mute:
        for ax in (axF, axB):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.tight_layout()
    plt.box(True)
    plt.tight_layout()
    _save(("mIC" if mute else "IC") + "-ftau-beta-function-mu10.svg", fig, savefig)
    plt.show()
    return fig, (axF, axB)


# ─────────────────────────────────────────────────────────────────────────────
# Table 5: reliability and resilience metrics
# ─────────────────────────────────────────────────────────────────────────────
def table5(Qsim, hqtile='lambda_0.05', stqtile='p50', T_hors=(1.0, 50.0), show=True):
    """Table 5: beta, R0 (exact and spectral approximation, Eq. 27) and rho.

    Returns ``results[structure][(scenario, T_hor)]`` (dict from
    ``core.reliability_resilience_summary``); scenarios ``'short'``/``'long'``.
    """
    results = {'fixed': {}, 'isolated': {}}
    for st in results:
        for sc_label, lvl in SCENARIO_MAP.items():
            Q = Qsim['ita'][hqtile][st][stqtile][lvl]
            for T_hor in T_hors:
                results[st][(sc_label, T_hor)] = core.reliability_resilience_summary(Q, T_hor)
    if show:
        utils.print_metrics_table(results['fixed'],    'BRACED FRAME  (Table 5)', SCENARIO_MAP, list(T_hors))
        utils.print_metrics_table(results['isolated'], 'BASE-ISOLATED (Table 5)', SCENARIO_MAP, list(T_hors))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Q0 and rho vs mu10
# ─────────────────────────────────────────────────────────────────────────────
def fig12_Q0_rho(Qsim, hqtile='lambda_0.05', stqtile='p84', savefig=True):
    """Figure 12: mean Q0 over the mu10 range (left) and rho vs mu10 (right).

    Saves the labelled figure and a muted version (no labels/ticks/legends).
    """
    with plt.rc_context():
        Q_fix = Qsim['ita'][hqtile]['fixed'][stqtile]['high']
        Q_iso = Qsim['ita'][hqtile]['isolated'][stqtile]['high']
        fig, axes = plotting.plot_Q0_rho(Q_fix, Q_iso, savefig=False, mutefig=True)
        _save('fig_Q0_rho_aggregate.svg', fig, savefig)
        for ax in axes:
            ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        _save('mfig_Q0_rho_aggregate.svg', fig, savefig)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: (beta, rho) plane
# ─────────────────────────────────────────────────────────────────────────────
def fig13_beta_rho_plane(Qsim, results, hqtile='lambda_0.05', stqtile='p50',
                         T_hor=50.0, N_mu=80, n_quad=300, savefig=True):
    """Figure 13: parametric curves (beta(mu10), rho(mu10)) of the two structures.

    ``results`` is the output of :func:`table5` (reference points at ``T_hor``).
    """
    rc = {"font.size": 13, "axes.titlesize": 12, "axes.labelsize": 13,
          "xtick.labelsize": 11, "ytick.labelsize": 11, "mathtext.fontset": "stix"}
    with plt.rc_context(rc):
        mus_sweep = np.logspace(-1, 3, N_mu)
        Q_bf = Qsim['ita'][hqtile]['fixed'][stqtile]['high'].copy()
        Q_bi = Qsim['ita'][hqtile]['isolated'][stqtile]['high'].copy()
        betas_bf, rhos_bf = core.beta_rho_vs_mu10(Q_bf, mus_sweep, T_hor, n_quad)
        betas_bi, rhos_bi = core.beta_rho_vs_mu10(Q_bi, mus_sweep, T_hor, n_quad)

        col_bf, col_bi = STRUCT_COLORS['fixed']['color'], STRUCT_COLORS['isolated']['color']
        fig, ax = plt.subplots(figsize=(7, 6.5))
        fig.subplots_adjust(right=0.74)

        ax.plot(betas_bf, rhos_bf, color=col_bf, lw=2.0, alpha=0.9, zorder=3,
                label='Braced Frame (BF)')
        ax.plot(betas_bi, rhos_bi, color=col_bi, lw=2.0, alpha=0.9, zorder=3,
                ls='--', label='Base Isolated (BI)')

        # start (triangle: low mu10) / end (square: high mu10) of the sweep
        for bv, rv, col in [(betas_bf, rhos_bf, col_bf), (betas_bi, rhos_bi, col_bi)]:
            ax.scatter(bv[0],  rv[0],  s=180, marker='^', color=col,
                       edgecolor='white', linewidth=0.8, zorder=7)
            ax.scatter(bv[-1], rv[-1], s=180, marker='s', color=col,
                       edgecolor='white', linewidth=0.8, zorder=7)

        # reference points (short / long recovery of Table 5)
        for sc_label, mk in {'short': 's', 'long': '^'}.items():
            for st, col, offset in [('fixed', col_bf, (+6, -14)), ('isolated', col_bi, (+6, +6))]:
                r = results[st][(sc_label, T_hor)]
                b, rho, mu = r['beta'], r['rho'], r['mu10']
                ax.scatter(b, rho, s=140, marker=mk, color=col,
                           edgecolor='white', linewidth=1.0, zorder=9)
                ax.annotate(f'{sc_label}\nμ={mu:.1f} yr⁻¹', (b, rho), textcoords='offset points',
                            xytext=offset, fontsize=7, color=col, ha='left')

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xlabel(r'Reliability index $\beta_{50\mathrm{yr}}$')
        ax.set_ylabel(r'Resilience index $\rho_{50\mathrm{yr}}$')
        ax.set_title(
            r'$(\beta,\,\rho)$ plane — varying $\mu_{10}$' '\n'
            fr'$T_{{\mathrm{{hor}}}}={T_hor:.0f}$ yr  |  '
            r'$\blacktriangle$ = low $\mu_{10}$,  $\blacksquare$ = high $\mu_{10}$',
            fontsize=10,
        )
        ax.grid(True, ls='--', alpha=0.3)
        leg = [
            Line2D([], [], color=col_bf, lw=2, label='Braced Frame (BF)'),
            Line2D([], [], color=col_bi, lw=2, ls='--', label='Base Isolated (BI)'),
            Line2D([], [], marker='^', color='gray', ms=8, lw=0,
                   markeredgecolor='white', label=r'low $\mu_{10}$'),
            Line2D([], [], marker='s', color='gray', ms=8, lw=0,
                   markeredgecolor='white', label=r'high $\mu_{10}$'),
        ]
        ax.legend(handles=leg, fontsize=8, loc='lower right', framealpha=0.85)
        plt.tight_layout()
        _save('06-beta-vs-rho-parametric.svg', fig, savefig)
        plt.show()
    return fig, ax
