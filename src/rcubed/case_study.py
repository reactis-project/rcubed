"""
rcubed.case_study
====================
Case study of the paper (Section 4): input loading, generator matrices and the
functions that reproduce Figures 9-13 and Tables 4-5.

Typical use (see ``notebooks/02_case_study.ipynb``)::

    from rcubed import case_study as cs

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
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.linalg import expm

from rcubed import core, generator, metrics, plotting, utils
from rcubed.paths import DATA, output_dir, require_data

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
# Recovery-time axis under the left panel: t = 1/mu10 at mu10 = 10^-1 ... 10^3 (rounded labels)
FIG11_TIME_TICKS = {0.1: '10\n[years]', 1.0: '1\n[year]', 10.0: '1\n[month]',
                    100.0: '3\n[days]', 1000.0: '8\n[hours]'}


def fig11_reliability(Qsim, hqtile='lambda_0.05', stqtile='p50',
                      mus=None, thors=(1.0, 10.0, 50.0), savefig=True):
    """Figure 11: failure probability F_T (left) and reliability index beta (right) vs mu10.

    Seismic rates fixed (long-recovery Q), one curve per structure and time horizon
    (darker = shorter horizon). Filled markers: median recovery rates of Table 3
    (Eq. 30, same repair times as Fig. 13).
    """
    mus = np.logspace(-1, 3, 50) if mus is None else np.asarray(mus)
    thors = np.asarray(thors, float)
    mu_of = lambda days: np.log(2) / (np.asarray(days, float) / 365.0)   # Eq. 30
    ref_mus = {st: mu_of([d['short'], *d['long']]) for st, d in FIG13_REPAIR_DAYS.items()}
    Q = {st: Qsim['ita'][hqtile][st][stqtile]['high'] for st in ('fixed', 'isolated')}

    def F_beta(Qs, mu, T):
        F, _ = metrics.F_tau(core.extract_Q_T(core.set_recovery_rate(Qs, mu)), T)
        return F, metrics.beta_index(F)

    def shade(st, T):
        # alpha from 1 (shortest horizon) to 0.4 (longest), log-spaced, blended on white
        a = 1.0 - 0.6 * LogNorm(vmin=thors.min(), vmax=thors.max())(T)
        r, g, b, _ = to_rgba(STRUCT_COLORS[st]['color'])
        return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)

    rc = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
          "axes.titlesize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}
    with plt.rc_context(rc):
        fig, (axF, axB) = plt.subplots(1, 2, figsize=(10, 2.6))
        fig.subplots_adjust(wspace=0.18)
        legend_handles = []
        for T_hor in thors:
            pair = []
            for st in ('fixed', 'isolated'):
                c = shade(st, T_hor)
                F, b = np.array([F_beta(Q[st], mu, T_hor) for mu in mus]).T
                axF.semilogx(mus, F, color=c, lw=1.6)
                axB.semilogx(mus, b, color=c, lw=1.6)
                Fr, br = np.array([F_beta(Q[st], mu, T_hor) for mu in ref_mus[st]]).T
                axF.plot(ref_mus[st], Fr, 'o', color=c, ms=4.5, zorder=5)
                axB.plot(ref_mus[st], br, 'o', color=c, ms=4.5, zorder=5)
                pair.append(Line2D([], [], color=c, lw=4, marker='o', ms=8))
            legend_handles.append(tuple(pair))

        span = r'with $\mathrm{T_{hor}} \in [%s]$ [years]' % ', '.join(f'{t:g}' for t in thors)
        axF.set_title(r'$\mathcal{F}_{\mathcal{T}}\,(\mu_{10})$  ' + span)
        axB.set_title(r'$\beta\,(\mu_{10})$  ' + span)
        axF.set_ylabel(r'$\mathcal{F}_{\mathcal{T}}(\cdot)$ [-]')
        axB.set_ylabel(r'$\beta = -\Phi^{-1}(\mathcal{F}_{\mathcal{T}})$')
        axF.set_ylim(bottom=-0.01)
        axB.set_ylim(0, 3.5)
        axB.set_yticks([1, 2, 3])
        for ax in (axF, axB):
            ax.set_xlim(mus.min() / 1.3, mus.max() * 1.3)
            ax.grid(True, which='both', ls='--', alpha=0.5)
            # mu axis: arrow with label below the tick labels
            ax.annotate('', xy=(1.0, -0.2), xytext=(0.0, -0.2), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=0.9), annotation_clip=False)
            ax.text(0.5, -0.23, r'$\mu$ - recovery rates [year$^{-1}$]', transform=ax.transAxes,
                    ha='center', va='top')

        # recovery-time axis (reversed arrow) under the left panel
        axT = axF.secondary_xaxis(-0.52)
        axT.set_xticks(list(FIG11_TIME_TICKS), list(FIG11_TIME_TICKS.values()))
        axT.minorticks_off()
        axT.tick_params(direction='in', length=4)
        axT.spines['bottom'].set_visible(False)
        axF.annotate('', xy=(0.0, -0.52), xytext=(1.0, -0.52), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', lw=0.9), annotation_clip=False)
        axF.text(0.5, -0.44, r'$t$ - recovery times', transform=axF.transAxes,
                 ha='center', va='center')

        # legend under the right panel: one BF/BI pair per time horizon
        axB.legend(legend_handles, [rf'$\mathrm{{T_{{hor}}}}$ = {t:g} yr'
                                    for t in thors],
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                   handlelength=2.6, ncol=len(thors), loc='upper center',
                   bbox_to_anchor=(0.5, -0.38), frameon=False, fontsize=10, columnspacing=1.2,
                   handletextpad=0.5)

        _save("IC-ftau-beta-function-mu10.svg", fig, savefig)
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
# Recovery scenarios of Table 3: median repair time [days]
FIG13_REPAIR_DAYS = {
    'fixed':    {'short': 30, 'long': (120, 365)},
    'isolated': {'short': 1,  'long': (30, 50)},
}


def fig13_beta_rho_plane(Qsim, results=None, hqtile='lambda_0.05', stqtile='p50',
                         T_hor=50.0, n_quad=300, savefig=True):
    """Figure 13: (beta, rho) points of the two structures for the recovery scenarios
    of Table 3 (square: short repair; triangles + band: range of the long repair).

    ``results`` (output of :func:`table5`) is no longer needed; kept for compatibility.
    """
    rc = {"font.size": 13, "axes.titlesize": 12, "axes.labelsize": 13,
          "xtick.labelsize": 11, "ytick.labelsize": 11, "mathtext.fontset": "stix"}
    mu_of = lambda days: np.log(2) / (np.asarray(days, float) / 365.0)   # Eq. 30
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7, 6.5))
        # performance quadrants (as in Figure 7)
        ax.axvspan(0, 2, ymax=2 / 4.5, color='#f2dcdc', alpha=0.6, zorder=0)
        ax.axvspan(2, 4, ymin=2 / 4.5, color='#e6e6e6', alpha=0.6, zorder=0)
        for st in ['fixed', 'isolated']:
            col = STRUCT_COLORS[st]['color']
            Q_b = Qsim['ita'][hqtile][st][stqtile]['high'].copy()
            # long-term recovery: range of median repair times (band + triangles at the ends)
            d_long = np.linspace(*FIG13_REPAIR_DAYS[st]['long'], 30)
            b_l, r_l = core.beta_rho_vs_mu10(Q_b, mu_of(d_long), T_hor, n_quad)
            ax.plot(b_l, r_l, color=col, lw=7, alpha=0.35, solid_capstyle='round', zorder=3)
            ax.scatter(b_l[[0, -1]], r_l[[0, -1]], s=150, marker='^', facecolor='white',
                       edgecolor=col, linewidth=1.6, zorder=8)
            # short-term recovery: one point (square)
            b_s, r_s = core.beta_rho_vs_mu10(Q_b, mu_of([FIG13_REPAIR_DAYS[st]['short']]),
                                             T_hor, n_quad)
            ax.scatter(b_s, r_s, s=150, marker='s', facecolor='white',
                       edgecolor=col, linewidth=1.6, zorder=8)

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4.5)
        ax.axvline(2, color='gray', lw=0.6)
        ax.axhline(2, color='gray', lw=0.6)
        ax.set_xlabel(r'Reliability index $\beta_{50\mathrm{yr}}$')
        ax.set_ylabel(r'Resilience index $\rho_{50\mathrm{yr}}$')
        ax.set_title(r'$\beta_{50y}$-$\rho$ performance space', fontsize=12)
        ax.grid(True, ls='--', alpha=0.3)
        cbf, cbi = STRUCT_COLORS['fixed']['color'], STRUCT_COLORS['isolated']['color']
        leg = [
            Line2D([], [], marker='s', color=cbi, mfc='white', lw=0, ms=9, label=r'BI, $t_{med}$ = 1 day'),
            Line2D([], [], marker='^', color=cbi, mfc='white', lw=0, ms=9, label=r'BI, $t_{med}$ = 30-50 days'),
            Line2D([], [], marker='s', color=cbf, mfc='white', lw=0, ms=9, label=r'BF, $t_{med}$ = 30 days'),
            Line2D([], [], marker='^', color=cbf, mfc='white', lw=0, ms=9, label=r'BF, $t_{med}$ = 120-365 days')
        ]
        ax.legend(handles=leg, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
        plt.tight_layout()
        _save('06-beta-vs-rho-parametric.svg', fig, savefig)
        plt.show()
    return fig, ax