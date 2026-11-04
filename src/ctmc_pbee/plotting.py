"""
ctmc_pbee.plotting
==================
Figures of the paper (case study) and plotting helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch

from ctmc_pbee import core



def save_svg(general, fname, moveTo=None):
    """Save the current figure as a transparent SVG in general['figFolder']
    (or in the full path ``moveTo``, if given). Existing files are overwritten."""
    from pathlib import Path
    fname = fname.replace('..', '.')  # just in case
    target = Path(moveTo) if moveTo is not None else Path(general['figFolder']) / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, transparent=True, format='svg', bbox_inches='tight')
    print("Saved:", target)
    return

def plt_intersection_p50(P_t, t_vals):
    """
    Find the first time t where P_t[:,0] or P_t[:,2] crosses 0.50.

    Parameters
    ----------
    P_t : ndarray, shape (N, n_states)
        Probability trajectories over time.
    t_vals : ndarray, shape (N,)
        Time grid corresponding to rows of P_t.

    Returns
    -------
    t_cross : float or None
        Time of crossing (linear interpolation).
    p_cross : float or None
        Probability at crossing (should be ~0.50).
    """

    # Check state 0
    diff0 = P_t[:,0] - 0.50
    sign0 = np.sign(diff0)
    idx0 = np.where(sign0[:-1] * sign0[1:] < 0)[0]

    # Check state 2
    diff2 = P_t[:,2] - 0.50
    sign2 = np.sign(diff2)
    idx2 = np.where(sign2[:-1] * sign2[1:] < 0)[0]

    # pick the first crossing found
    candidates = []
    if idx0.size > 0:
        i = idx0[0]
        # linear interpolation
        t_cross = t_vals[i] + (0.50 - P_t[i,0]) * (t_vals[i+1]-t_vals[i]) / (P_t[i+1,0]-P_t[i,0])
        candidates.append((t_cross, 0.50))
    if idx2.size > 0:
        i = idx2[0]
        t_cross = t_vals[i] + (0.50 - P_t[i,2]) * (t_vals[i+1]-t_vals[i]) / (P_t[i+1,2]-P_t[i,2])
        candidates.append((t_cross, 0.50))

    if candidates:
        # return the earliest crossing
        t_cross, p_cross = min(candidates, key=lambda x: x[0])
        return t_cross, p_cross
    else:
        print("No crossing found for states 0 or 2 at 0.50.")
        return None, None

def fixed_aspect_ratio(ax, ratio):
    xvals, yvals = ax.get_xlim(), ax.get_ylim()
    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    ax.set_aspect(ratio * (xrange / yrange), adjustable='box')

def plot_fragilities(ds_probs, col_frag, fragil_ls, general, ax_ratio=0.5, labels=False):
    """
    Simulated state-dependent fragility curves of one structure on a 2x3 grid:
    every simulated curve (thin), mean (solid) and 5th/95th percentiles.

    Parameters
    ----------
    ds_probs  : output of cnn_fragilities.simulate_damage_state_curves + compute_residual
    col_frag  : line colour
    fragil_ls : {'mean': linestyle, 'perc': linestyle}
    general   : dict; general['muteFig'] = True hides titles, labels, tick labels and legend
    ax_ratio  : aspect ratio of each panel
    labels    : if True, label the mean and percentile curves in the legend

    Returns
    -------
    fig, axes
    """
    lab_mean = 'Mean' if labels else None
    lab_perc = '05th/95th percentile' if labels else None
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.005, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    layout_indices = {
        "DS0|DS0": 0,
        "DS1|DS0": 1,
        "DS2|DS0": 2,
        "SKIP"   : 3,
        "DS1|DS1": 4,
        "DS2|DS1": 5
    }

    for label, idx in layout_indices.items():
        if label == "SKIP":
            axes[idx].axis('off')
            continue

        ax = axes[idx]
        curves = ds_probs[label]['curves']
        for curve in curves:
            ax.plot(ds_probs[label]['pga'], curve, color=col_frag, alpha=0.3, lw=1.0)

        # Plot summary curves
        ax.plot(ds_probs[label]['pga'], ds_probs[label]['mean'], color=col_frag, ls=fragil_ls['mean'], lw=2.3, alpha=0.9, label=lab_mean)
        ax.plot(ds_probs[label]['pga'], ds_probs[label]['p05'], color=col_frag, ls=fragil_ls['perc'], lw=1.7, alpha=0.9, label=lab_perc)
        ax.plot(ds_probs[label]['pga'], ds_probs[label]['p95'], color=col_frag, ls=fragil_ls['perc'], lw=1.7, alpha=0.9, label=lab_perc)

        if general.get('muteFig', False):
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_title(label)
            ax.set_xlabel('PGA [m/s²]')
            ax.set_ylabel('Probability')
            ax.legend()

        ax.set_ylim([0, 1])
        ax.grid(True)
        fixed_aspect_ratio(ax, ax_ratio)

    return fig, axes

def plot_fragilities_v2(ds_probs, col_frag, fragil_ls, general, struct_color, ax_ratio=0.5):
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.005, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    layout_indices = {
        "DS0|DS0": 0,
        "DS1|DS0": 1,
        "DS2|DS0": 2,
        "SKIP"   : 3,
        "DS1|DS1": 4,
        "DS2|DS1": 5
    }

    line_color = struct_color['color']

    for label, idx in layout_indices.items():
        if label == "SKIP":
            axes[idx].axis('off')
            continue

        ax = axes[idx]
        pga    = ds_probs[label]['pga']
        p05    = ds_probs[label]['p05']
        p95    = ds_probs[label]['p95']
        median = ds_probs[label]['mean']

        # Shaded band between p05 and p95
        ax.fill_between(pga, p05, p95, color=line_color, alpha=0.2, linewidth=0)

        # 5th and 95th percentile dashed lines
        ax.plot(pga, p05, color=line_color, ls='--', lw=1.7, alpha=0.9,
                label=r'$05^{th}$-$95^{th}$ p.tile')
        ax.plot(pga, p95, color=line_color, ls='--', lw=1.7, alpha=0.9)

        # Median solid line
        ax.plot(pga, median, color=line_color, ls='-', lw=2.3, alpha=0.9,
                label=r'$50^{th}$ p.tile')

        if general.get('muteFig', False):
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_title(label)
            ax.set_xlabel('PGA [m/s²]')
            ax.set_ylabel('Probability')
            ax.legend()

        ax.set_ylim([0, 1])
        ax.grid(True)
        fixed_aspect_ratio(ax, ax_ratio)

    return fig, axes

def plot_hazard(hazard, hazard_col, im_grid_original_samples = None):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Italy
    plt.plot( hazard['im_axis'], hazard['ita']['lambda_0.05'],  color=hazard_col[0]['color'], linestyle = '--', label='0.05th percentile')
    plt.plot( hazard['im_axis'], hazard['ita']['lambda_0.5' ],  color=hazard_col[0]['color'], linestyle ='-',   label='0.50th percentile (median)')
    plt.plot( hazard['im_axis'], hazard['ita']['lambda_0.95'],  color=hazard_col[0]['color'], linestyle ='--',  label='0.95th percentile')
    ax.fill_between(hazard['im_axis'],hazard['ita']['lambda_0.05'],hazard['ita']['lambda_0.95'],color=hazard_col[0]['color'],alpha=0.2)
        
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xscale('log')  # Log scale is standard for hazard curves
    plt.yscale('log')  # Log scale is standard for hazard curves
    plt.xlabel('PGA [m/s2]')
    plt.ylabel('lambda [1/year]')
    
    # plt.ticklabel_format(style='sci')
    # plt.xticks([ 0.1, 0.5 ,1 ,2,3,4,5,10])
    
    if im_grid_original_samples is not None:
        # === ADD ORIGINAL DATA POINTS ===
        ax.scatter(im_grid_original_samples, hazard['ita']['lambda_0.5'],  color=hazard_col[0]['color'], marker='s', s=10, edgecolor='black', label='Original 0.50')
    
    plt.tight_layout()
    # # plt.show()
    
    # plt.setp( ax.get_xticklabels(), visible=False)
    # plt.setp( ax.get_yticklabels(), visible=False)
    return fig, ax

def plot_median_fragilities(pga, ds_probs_bf, ds_probs_iso, struct_colors2, general, ax_ratio=0.5,
                           labels=('Fixed', 'Isolated')):
    """
    Plot only the median curves for BF (fixed) and ISO (isolated) configurations.
    Replicates the 2x3 layout with the skipped cell [1,0].

    struct_colors2 : {'fixed': {'color': ...}, 'isolated': {'color': ...}}
    labels         : legend labels for (BF, ISO)
    """
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.005, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    layout_indices = {
        "DS0|DS0": 0,
        "DS1|DS0": 1,
        "DS2|DS0": 2,
        "SKIP"   : 3,
        "DS1|DS1": 4,
        "DS2|DS1": 5
    }

    col_bf  = struct_colors2['fixed']['color']
    col_iso = struct_colors2['isolated']['color']

    for label, idx in layout_indices.items():
        ax = axes[idx]

        if label == "SKIP":
            ax.axis('off')
            continue

        pga_arr = ds_probs_bf[label]['pga']

        # BF median
        ax.plot(pga_arr, ds_probs_bf[label]['mean'],
                color=col_bf,  ls='-', lw=2.3, alpha=0.9, label=labels[0])

        # ISO median
        ax.plot(pga_arr, ds_probs_iso[label]['mean'],
                color=col_iso, ls='-', lw=2.3, alpha=0.9, label=labels[1])

        if general.get('muteFig', False):
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_title(label)
            ax.set_xlabel('PGA [m/s²]')
            ax.set_ylabel('Probability')
            ax.legend()

        ax.set_ylim([0, 1])
        ax.grid(True)
        fixed_aspect_ratio(ax, ax_ratio)

    return fig, axes

def plot_fragilities_bands(ds_probs_dict, general, struct_colors, ax_ratio=0.5):
    """
    State-dependent fragility curves of several structures on one 2x3 grid:
    median (solid) and 5th-95th percentile band (shaded, dashed limits).

    Parameters
    ----------
    ds_probs_dict : {structure: {'DS1|DS0': {'pga', 'p05', 'p95', 'mean'}, ...}}
    general       : dict; general['muteFig'] = True hides titles, labels and tick labels
    struct_colors : {structure: {'color': ..., 'label': ...}}
    ax_ratio      : aspect ratio of each panel

    Returns
    -------
    fig, axes
    """
    fig = plt.figure(figsize=(18, 9), dpi=600)
    gs = gridspec.GridSpec(2, 3, hspace=0.005, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    layout_indices = {
        "DS0|DS0": 0,
        "DS1|DS0": 1,
        "DS2|DS0": 2,
        "SKIP"   : 3,
        "DS1|DS1": 4,
        "DS2|DS1": 5
    }

    for struct_type, ds_probs in ds_probs_dict.items():
        line_color = struct_colors[struct_type]['color']
        struct_label = struct_colors[struct_type]['label']

        for label, idx in layout_indices.items():
            if label == "SKIP":
                continue

            ax = axes[idx]
            pga    = ds_probs[label]['pga']
            p05    = ds_probs[label]['p05']
            p95    = ds_probs[label]['p95']
            median = ds_probs[label]['mean']

            ax.fill_between(pga, p05, p95, color=line_color, alpha=0.2, linewidth=0)
            ax.plot(pga, p05, color=line_color, ls='--', lw=1.7, alpha=0.9)
            ax.plot(pga, p95, color=line_color, ls='--', lw=1.7, alpha=0.9)
            ax.plot(pga, median, color=line_color, ls='-', lw=2.3, alpha=0.9,
                    label=struct_label)

    for label, idx in layout_indices.items():
        if label == "SKIP":
            axes[idx].axis('off')
            continue

        ax = axes[idx]
        ax.set_ylim([0, 1])
        ax.grid(True)
        fixed_aspect_ratio(ax, ax_ratio)

        if general.get('muteFig', False):
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_title(label)
            ax.set_xlabel('PGA [m/s²]')
            ax.set_ylabel('Probability')

    return fig, axes

def add_discrete_colorbar(fig, ax, cmap, values_log, label=r"$\alpha$"):
    """Discrete colorbar with one tick per swept value (values_log on a log10 scale)."""
    from matplotlib.colors import BoundaryNorm
    bounds = np.concatenate([
        [values_log[0] - (values_log[1] - values_log[0]) / 2],
        (values_log[:-1] + values_log[1:]) / 2,
        [values_log[-1] + (values_log[-1] - values_log[-2]) / 2],
    ])
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    ticks = np.arange(int(np.ceil(values_log.min())), int(np.floor(values_log.max())) + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([fr"$10^{{{int(t)}}}$" for t in ticks])
    cbar.set_label(label)
    return cbar


def plot_Q0_rho(
    Qtest1: np.ndarray,
    Qtest2: np.ndarray,
    mus: np.ndarray       = None,
    thors: np.ndarray     = None,
    ref_mus_fix: list     = None,
    ref_mus_iso: list     = None,
    color_fixed: str      = '#08407F',
    color_isolated: str   = '#77355B',
    n_quad: int           = 300,
    figsize: tuple        = (8, 2.8),
    savefig: bool         = False,
    fname: str            = 'fig_Q0_rho_aggregate.svg',
    mutefig: bool         = False,
):
    """
    Produce the two-panel figure:
      Left  — bar chart of mean Q0 aggregated over mu10, with min-max error bars
      Right — rho vs mu10 (semilog), one curve per (system, T_hor)

    Parameters
    ----------
    Qtest1      : (3,3) Q matrix for the braced frame    (seismic rates fixed)
    Qtest2      : (3,3) Q matrix for the base-isolated   (seismic rates fixed)
    mus         : array of mu10 values for the sweep
    thors       : array of time horizons [year]
    ref_mus_fix : reference mu10 values for the BF (plotted as dots)
    ref_mus_iso : reference mu10 values for the BI (plotted as squares)
    color_fixed / color_isolated : base colors for the two systems
    n_quad      : quadrature points for R0_exact
    savefig     : whether to save the figure
    fname       : output filename (svg recommended)
    mutefig     : if True, strip all labels/ticks (for journal-ready muted version)
    """

    # ── Defaults ──────────────────────────────────────────────────────────────
    if mus        is None: mus        = np.logspace(-1, 3, 50)
    if thors      is None: thors      = np.array([1.0, 10.0, 50.0])
    if ref_mus_fix is None: ref_mus_fix = [8.43, 0.69]
    if ref_mus_iso is None: ref_mus_iso = [252.99, 5.05]

    # ── Colormaps: same hue, fading alpha across T_hor ────────────────────────
    def make_cmap(hex_color):
        r, g, b, _ = to_rgba(hex_color)
        return LinearSegmentedColormap.from_list(
            hex_color, [(r, g, b, 1.0), (r, g, b, 0.35)], N=256
        )

    cmap_fix = make_cmap(color_fixed)
    cmap_iso = make_cmap(color_isolated)
    normCol  = LogNorm(vmin=thors.min(), vmax=thors.max())

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'mathtext.fontset': 'stix',
    })

    fig, (axQ, axR) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.38)

    labels = ['Braced\nframe', 'Base-\nisolated']
    x_base = np.array([0, 1])
    bar_w  = 0.55 / len(thors)

    for i, T_hor in enumerate(thors):

        # ── Sweep mu10 ────────────────────────────────────────────────────────
        Q0_fix, Q0_iso   = [], []
        rho_fix, rho_iso = [], []

        for mu in mus:
            q0f, rhof = core.Q0_rho(core.set_recovery_rate(Qtest1, mu), T_hor, n_quad)
            q0i, rhoi = core.Q0_rho(core.set_recovery_rate(Qtest2, mu), T_hor, n_quad)
            Q0_fix.append(q0f);  rho_fix.append(rhof)
            Q0_iso.append(q0i);  rho_iso.append(rhoi)

        Q0_fix  = np.array(Q0_fix);  rho_fix = np.array(rho_fix)
        Q0_iso  = np.array(Q0_iso);  rho_iso = np.array(rho_iso)

        col_fix = cmap_fix(normCol(T_hor))
        col_iso = cmap_iso(normCol(T_hor))
        offset  = (i - (len(thors) - 1) / 2) * bar_w

        # ── Left panel: Q0 bar chart ──────────────────────────────────────────
        for col, x_pos, Q0_arr in [(col_fix, x_base[0], Q0_fix),
                                    (col_iso, x_base[1], Q0_iso)]:
            mn, mx, me = Q0_arr.min(), Q0_arr.max(), Q0_arr.mean()
            axQ.bar(x_pos + offset, me, width=bar_w,
                    color=col, edgecolor='k', linewidth=0.5)
            axQ.errorbar(x_pos + offset, me,
                         yerr=[[me - mn], [mx - me]],
                         fmt='none', ecolor='k', capsize=3, lw=0.8)

        # ── Right panel: rho vs mu ────────────────────────────────────────────
        axR.semilogx(mus, rho_fix, color=col_fix, lw=1.6, ls='-',
                     label=f'BF  $T$={T_hor:.0f}yr')
        axR.semilogx(mus, rho_iso, color=col_iso, lw=1.6, ls='--',
                     label=f'BI  $T$={T_hor:.0f}yr')

        # Reference mu markers
        for mu_r in ref_mus_fix:
            q0r, rhor = core.Q0_rho(core.set_recovery_rate(Qtest1, mu_r), T_hor, n_quad)
            axR.plot(mu_r, rhor, 'o', color=col_fix, ms=4, zorder=5)
            axQ.plot(x_base[0] + offset, q0r, 'o',
                     color=col_fix, ms=3.5, zorder=6)

        for mu_r in ref_mus_iso:
            q0r, rhor = core.Q0_rho(core.set_recovery_rate(Qtest2, mu_r), T_hor, n_quad)
            axR.plot(mu_r, rhor, 's', color=col_iso, ms=4, zorder=5)
            axQ.plot(x_base[1] + offset, q0r, 's',
                     color=col_iso, ms=3.5, zorder=6)

    # ── Left panel formatting ─────────────────────────────────────────────────
    axQ.set_xticks(x_base)
    axQ.set_xticklabels(labels)
    axQ.set_ylim([0.55, 1.05])
    axQ.set_ylabel(r'$\mathcal{Q}_0(T_{\mathrm{hor}})$')
    axQ.set_title(r'Mean $\mathcal{Q}_0$ over $\mu_{10}$ range', fontsize=8)
    axQ.yaxis.grid(True, ls='--', alpha=0.4)
    axQ.set_axisbelow(True)

    # T_hor legend for left panel
    legend_bars = [
        Patch(facecolor=cmap_fix(normCol(T)), edgecolor='k',
              linewidth=0.5, label=f'$T$={T:.0f}yr')
        for T in thors
    ]
    axQ.legend(handles=legend_bars, fontsize=7, loc='lower right',
               framealpha=0.85)

    # ── Right panel formatting ────────────────────────────────────────────────
    axR.set_xlabel(r'$\mu_{10}\;[\mathrm{year}^{-1}]$')
    axR.set_ylabel(r'$\rho = -\Phi^{-1}(\mathcal{R}_0)$')
    axR.set_title(r'Resilience index $\rho$ vs $\mu_{10}$', fontsize=8)
    axR.set_ylim([0, 4.5])
    axR.grid(True, which='both', ls='--', alpha=0.35)
    axR.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.85)

    # Secondary x-axis: recovery time in days/years
    ax2 = axR.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(axR.get_xlim())
    tmed_days = np.array([1e3, 1e2, 1e1, 1.0, 0.1])   # days
    mu_ticks  = np.log(2) / tmed_days * 365
    ax2.set_xticks(mu_ticks)
    ax2.set_xticklabels(['1d', '10d', '100d', '1yr', '10yr'], fontsize=7)
    ax2.set_xlabel(r'$t_{\mathrm{med}}$ (median recovery time)',
                   fontsize=7, labelpad=2)

    # ── Muted version (no labels/ticks) ──────────────────────────────────────
    if mutefig:
        for ax in [axQ, axR, ax2]:
            ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
            ax.set_xticklabels([]); ax.set_yticklabels([])
        try: axQ.get_legend().remove()
        except AttributeError: pass
        try: axR.get_legend().remove()
        except AttributeError: pass

    plt.tight_layout()

    if savefig:
        plt.savefig(fname, bbox_inches='tight', transparent=True,
                    format=fname.split('.')[-1])
        print(f'Saved: {fname}')

    plt.show()
    return fig, (axQ, axR)
