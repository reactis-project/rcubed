"""
ctmc_pbee.utils
===============
Small helpers: nested dicts, figure saving, printing of result tables, comparisons.
"""

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def get_keys_by_level(d, level=0, keys_by_level=None):
    """
    Collect the keys of a nested dictionary, grouped by nesting level.
    
    Parameters
    ----------
    d : dict
        Dictionary to explore.
    level : int
        Current level (used internally).
    keys_by_level : dict
        Accumulator of the results (used internally).
    
    Returns
    -------
    dict
        Keys grouped by level: {0: [...], 1: [...], 2: [...], ...}

    Example
    -------        
    levels = get_keys_by_level(Q_matrices_with_recovery)

    for lvl, keys in levels.items():
        print(f"key{lvl+1} =", keys)
    """
    if keys_by_level is None:
        keys_by_level = defaultdict(set)

    for k, v in d.items():
        # Skip metadata entries
        if k == "_level_names":
            continue

        keys_by_level[level].add(k)
        if isinstance(v, dict):
            get_keys_by_level(v, level + 1, keys_by_level)

    return {lvl: sorted(list(keys), key=str) for lvl, keys in keys_by_level.items()}


def saveplot(fname, outdir_or_general, general=None):
    """
    Save the current figure as a transparent SVG, then show it.

    Two calling conventions are supported:
      saveplot(fname, outdir)     # legacy: outdir_or_general is a path/str
      saveplot(fname, general)    # new: outdir_or_general is a dict with
                                   #   'saveFig'       -> bool, skip saving if False
                                   #   'folderFig'     -> exploratory plots/ folder
                                   #   'folderFig4Art' -> article-ready figure4article/ folder
                                   # (figure is saved into both folders when both are set)
    """
    if isinstance(outdir_or_general, dict):
        general = outdir_or_general
        if not general.get('saveFig', True):
            plt.show()
            return
        folders = [general.get('folderFig'), general.get('folderFig4Art')]
    else:
        folders = [outdir_or_general]

    fname = f"{fname}.svg".replace("..", ".")
    for folder in filter(None, folders):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        fullpath = folder / fname
        plt.savefig(fullpath, transparent=True, format='svg', bbox_inches='tight')
        print("Saved:", fullpath)

    plt.show()


def recursive_defaultdict():
    """Nested defaultdict of arbitrary depth: d['a']['b']['c'] = 1 works directly."""
    return defaultdict(recursive_defaultdict)


def order_of_magnitude(x):
    """Order of magnitude of x as a short string, e.g. 3.2e-5 -> '~1e-5'."""
    if x == 0:
        return '0'
    return f'~1e{int(math.floor(math.log10(abs(x))))}'


def print_metrics_table(res, label, scenario_map, t_hors):
    """
    Print the reliability/resilience metrics of one structure as a text table
    (layout of Table 5 of the paper).

    Parameters
    ----------
    res          : dict {(scenario, T_hor): output of run_analysis}
    label        : table title
    scenario_map : dict {scenario label: recovery level}, e.g. {'short': 'low', 'long': 'high'}
    t_hors       : list of time horizons [year]
    """
    print(f'\n{"="*72}')
    print(f'  {label}')
    print(f'{"="*72}')
    keys = [(sc, T) for T in t_hors for sc in scenario_map]
    R    = [res[k] for k in keys]
    header = f'{"":22s}' + ''.join(f' | {sc:>5s} T={T:2.0f}yr' for (sc, T) in keys)
    print(header); print('-' * len(header))
    rows = [
        ('mu10 [yr^-1]', [f'{r["mu10"]:.2f}'   for r in R]),
        ('sigma1',       [f'{r["sigma1"]:.4f}'  for r in R]),
        ('sigma2',       [f'{r["sigma2"]:.4f}'  for r in R]),
        ('|s2/s1|',      [order_of_magnitude(r['ratio'])      for r in R]),
        ('beta',         [f'{r["beta"]:.3f}'    for r in R]),
        ('R0 Eq(18)',    [f'{r["R0"]:.4f}'      for r in R]),
        ('R0_sp Eq(27)', [f'{r["R0_sp"]:.4f}'   for r in R]),
        ('Delta R0',     [order_of_magnitude(r['dR0'])        for r in R]),
        ('rho',          [f'{r["rho"]:.2f}'     for r in R]),
    ]
    for name, vals in rows:
        print(f'  {name:22s}' + ''.join(f' | {v:>12s}' for v in vals))


def compare_nested(a, b, path='', rtol=0.0, atol=0.0):
    """
    Compare two nested structures (dicts, lists, arrays, numbers) value by value.

    Returns the list of differences as strings (empty list = identical).
    With rtol = atol = 0 arrays must be exactly equal.
    """
    import numpy as np
    diffs = []
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{path}: dict vs {type(b).__name__}"]
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing in {'first' if k not in a else 'second'}")
            else:
                diffs += compare_nested(a[k], b[k], f"{path}/{k}", rtol, atol)
        return diffs
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if x.shape != y.shape:
        return [f"{path}: shape {x.shape} vs {y.shape}"]
    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
        return [f"{path}: max abs diff {np.nanmax(np.abs(x - y)):.3e}"]
    return []
