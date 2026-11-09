"""
rcubed.generator
===================
Generator matrices Q of the case-study CTMC from seismic hazard, state-dependent
fragilities and recovery times (Section 4 of the paper).

    seismic_rates     : seismic transition rates lambda_ij = int P(DS_j | IM, DS_i) |d lambda(IM)|
    build_Q_matrices  : full 3x3 generators for every site / hazard quantile / structure /
                        fragility quantile / recovery scenario
    resample_hazard   : hazard curves on the IM axis of the fragilities
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools


# ====================================================== #
# ## Function to compute the jump matrix Qseism
# ====================================================== #
'''
# Vectorized computation of the seismic jump matrix Qseism.
# Given: 
    Parameters
    ----------
    lambda_im : dict
        Dictionary of hazard curves: keys are 'lambda_0.05', 'lambda_0.5', 'lambda_0.95', values are arrays of λ(IM)
    im_axis : np.ndarray
        IM values corresponding to the hazard curves
    pb_transitions : dict
        Dictionary of DS bins. Each entry has 'curves': np.ndarray of shape (n_realizations, n_IM_points)
    
    Returns
    -------
    Qseism : dict
        Nested dict: Qseism[im_qtile][pb_qtile] = np.array of λ values over all DS bins
    valid_masks : dict
        valid_masks[im_qtile] = boolean array indicating non-NaN IM points
    had_nan_dict : dict
        had_nan_dict[im_qtile] = True if any NaNs present in hazard curve
    df_Qseism : pd.DataFrame
        Tidy dataframe with columns: im_qtile, pb_qtile, ds, lambda_seis

# Computes Qseism as:
#    λ_{ij}​= λ_}{0}​ ∫P(DS_j​∣IM)f(IM)dIM
'''
def seismic_rates(lambda_im, im_axis, pb_transitions):

    desired_im_qtiles = ['lambda_0.05', 'lambda_0.5', 'lambda_0.95']
    pb_qtiles = ['p01','p05','p16','p50','p84','p95','p99']
    pb_qtile_vals = [float(q[1:])/100 for q in pb_qtiles]

    Qseism = {}
    valid_masks = {}
    had_nan_dict = {}
    tidy_records = []

    for im_qtile, lambda_sel in lambda_im.items():
        if im_qtile not in desired_im_qtiles:
            continue

        # Mask valid IM points
        valid_mask = ~np.isnan(lambda_sel)
        lambda_vals = lambda_sel[valid_mask]
        im_vals = im_axis[valid_mask]
        had_nan = np.any(~valid_mask)

        valid_masks[im_qtile] = valid_mask
        had_nan_dict[im_qtile] = had_nan

        if len(im_vals) < 2:
            continue

        # --- Vectorized derivative & PDF -------------------------------------
        lambda_0     = lambda_vals[0]
        dIM          = np.diff(im_vals)
        dlambda_dIM  = -np.diff(lambda_vals) / dIM
        f_im         = dlambda_dIM / lambda_0        
        # 1---------------------------------------------------------------------

        # Initialize container for this IM quantile
        Qseism[im_qtile] = {}

        # Loop over DS bins and PB-quantiles
        # To compute λ for multiple quantiles (p01, p05, … p99)
        # from your (totNcurves, totLengthCurve) curves, you need to:
        # 1. Compute the quantiles along the 30 realizations (axis 0).
        # 2. Apply the same boolean mask along the IM axis (axis 1).
        # 3. Integrate each quantile individually.
        # Loop over DS bins enforcing this order
        DS_ORDER = ['DS0|DS0', 'DS1|DS0', 'DS2|DS0', 'DS1|DS1', 'DS2|DS1']

        for ds_key in DS_ORDER:

            # Skip DS if missing in dictionary
            if ds_key not in pb_transitions:
                continue

            bin_data = pb_transitions[ds_key]

            # Ensure "curves" exists
            if 'curves' not in bin_data:
                continue

            # curves: shape = (n_realizations, n_IM_points)
            curves = np.array(bin_data['curves'])

            # Apply IM mask (same mask used on hazard)
            curves_valid = curves[:, valid_mask]  # shape (n_realizations, n_valid_IM)

            # ---- Compute PB quantiles for this DS --------------------------------
            # pb_qtile_vals is array like [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]
            pb_quantiles = np.quantile(curves_valid, q=pb_qtile_vals, axis=0)
            # shape = (n_pb_qtiles, n_valid_IM)
            # ----------------------------------------------------------------------

            # ---- Midpoint integration across IM ---------------------------------
            pb_mid = 0.5 * (pb_quantiles[:, :-1] + pb_quantiles[:, 1:])
            # shape = (n_pb_qtiles, n_valid_IM-1)

            # vectorized seismic rates for all PB quantiles
            lambda_ij_array = lambda_0 *  np.sum(pb_mid * f_im, axis=1)
            # shape = (n_pb_qtiles,)
            # ----------------------------------------------------------------------

            # ---- Store results in Qseism under fixed DS → correct order ----------
            for idx, pb_qtile in enumerate(pb_qtiles):

                if pb_qtile not in Qseism[im_qtile]:
                    Qseism[im_qtile][pb_qtile] = []

                # Append λ for this DS in correct DS_ORDER position
                Qseism[im_qtile][pb_qtile].append(lambda_ij_array[idx])

                # add each result to tidy dataframe
                tidy_records.append({
                    'im_qtile': im_qtile,
                    'pb_qtile': pb_qtile,
                    'ds': ds_key,
                    'lambda_seis': lambda_ij_array[idx]
                })
        # -------------------------------------------------------------------------

        # After filling, convert lists → np.array following DS_ORDER
        for pb_qtile in pb_qtiles:
            Qseism[im_qtile][pb_qtile] = np.array(Qseism[im_qtile][pb_qtile])

    # Create tidy pandas table
    df_Qseism = pd.DataFrame(tidy_records)

    return Qseism, valid_masks, had_nan_dict, df_Qseism


def build_Q_matrices(
    hazard: dict,
    hazard_locations: list,
    hazard_qtiles: list,
    struct_dicts: dict,
    pb_transition_qtiles: list,
    recovery_dict: dict,
    verbose: bool = True
) -> dict:
    Qfull = {}

    # Flatten combinations
    all_combinations = list(itertools.product(
        hazard_locations,
        hazard_qtiles,
        struct_dicts.keys(),
        pb_transition_qtiles
    ))
    total_steps = len(all_combinations)

    bar = tqdm(all_combinations, total=total_steps, unit="matrix", desc="Building Q matrices", disable=not verbose)

    for hl, hq, st, ptq in bar:
        if hl not in Qfull:
            Qfull[hl] = {}
        if hq not in Qfull[hl]:
            Qfull[hl][hq] = {}
        if st not in Qfull[hl][hq]:
            Qfull[hl][hq][st] = {}
        Qfull[hl][hq][st][ptq] = {}

        struct_data = struct_dicts[st]
        pb_transitions = struct_data['pb_transitions_exc']

        # Compute upper triangular rates
        temp_Qseism, _, _, _ = seismic_rates(
            lambda_im=hazard[hl],
            im_axis=hazard[hl]['PGA'],
            pb_transitions=pb_transitions
        )

        # Handle missing PB quantiles
        if ptq in temp_Qseism[hq]:
            rates = temp_Qseism[hq][ptq]
        elif 'p50' in temp_Qseism[hq]:
            rates = temp_Qseism[hq]['p50']
        else:
            raise KeyError(f"PB quantile {ptq} not found for {hl}, {hq}, {st}")

        # Recovery levels
        for level, t_med in recovery_dict[st].items():
            mu_ = (np.log(2) / t_med) * 365
            Q_mat = np.zeros((3,3))
            Q_mat[0,1] = rates[1]
            Q_mat[0,2] = rates[2]
            Q_mat[0,0] = -(rates[1] + rates[2])
            Q_mat[1,0] = mu_
            Q_mat[1,2] = rates[4]
            Q_mat[1,1] = -(mu_ + rates[4])
            Q_mat[2,:] = 0
            Qfull[hl][hq][st][ptq][level] = Q_mat

        # Update bar with current combination
        bar.set_postfix({"loc": hl, "hazard": hq, "struct": st, "pb": ptq}, refresh=False)

    return Qfull


def resample_hazard(hazard, im_axis):
    """
    Resample hazard curves on a new intensity-measure axis (linear interpolation,
    linear extrapolation outside the original range).

    The case-study fragilities are defined on PGA = linspace(0, 8, 100) [m/s^2],
    while the ESHM20 hazard curves are given at 25 PGA values: this brings the
    hazard on the fragility axis, so that both can be combined in the CTMC.

    Parameters
    ----------
    hazard  : {site: {'PGA': x, 'lambda_0.05': y, 'q_0.5': y, ...}, 'im_pga': x}
              (1-D curves, or 2-D arrays with one curve per column)
    im_axis : new intensity-measure axis

    Returns
    -------
    dict with the same sites and keys, every curve evaluated on im_axis;
    the key 'im_pga' (or 'im_axis') is replaced by 'im_axis'.
    """
    from scipy.interpolate import interp1d

    reference_im = im_axis
    hazard_resampled = {}
    for key, bin_data in hazard.items():
        if key in ('im_pga', 'im_axis'):
            hazard_resampled['im_axis'] = reference_im
            continue

        original_pga = np.array(bin_data['PGA'])
        resampled_bin_data = {'PGA': reference_im}
        for key_res in [k for k in bin_data.keys() if k != 'PGA']:
            curves_array = np.array(bin_data[key_res])
            if curves_array.shape[0] != len(original_pga):
                raise ValueError(f"{key}/{key_res} has shape {curves_array.shape}, "
                                 f"expected first dim = {len(original_pga)}")
            if curves_array.ndim == 1:
                f_interp = interp1d(original_pga, curves_array, kind='linear', fill_value='extrapolate')
                resampled_bin_data[key_res] = f_interp(reference_im)
            elif curves_array.ndim == 2:
                resampled_bin_data[key_res] = np.column_stack([
                    interp1d(original_pga, curves_array[:, j], kind='linear',
                             fill_value='extrapolate')(reference_im)
                    for j in range(curves_array.shape[1])
                ])
            else:
                raise ValueError(f"{key}/{key_res} has unexpected ndim = {curves_array.ndim}")
        hazard_resampled[key] = resampled_bin_data
    return hazard_resampled
