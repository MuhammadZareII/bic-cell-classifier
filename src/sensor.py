"""
COMSOL data loading and resonance-feature extraction for BIC sensor spectra.

The BIC (Bound State in the Continuum) resonator produces sharp dips in the
transmission spectrum whose position (λ₀) and quality factor (Q = λ₀/FWHM)
shift with the local refractive index of the surrounding medium.

All COMSOL files are simulations of the same sensor geometry in the same
seawater-like medium (n ≈ 1.35).  Each file represents a different parametric
sweep of the sensor design space:

Data formats exported by COMSOL
--------------------------------
bic_spectrum_single.csv  – 3 columns: lambda(m), lambda0(µm), T
                           Single reference BIC spectrum at fixed geometry.
bic_tsb_sweep.csv        – long format: TSB_nm, lambda_nm, T
                           Parametric sweep over translational symmetry break.
seawater_refined.csv     – long format: TSB_nm, lambda_nm, T, Tot
                           TSB sweep in the seawater reference medium.
bic_displacement_sweep.csv – 4 columns: lambda(m), d(nm), lambda0(µm), T
                           Parametric sweep over cube displacement d.
bic_nenv_sweep.csv       – columns: lambda(m), n_env_0, lambda0(µm), T1…T4
                           Sweep over environmental refractive index n_env.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _skip_comment_csv(filepath, ncols: int) -> pd.DataFrame:
    """Read a COMSOL CSV that uses '%' comment lines for its header."""
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split(",")
            if len(parts) >= ncols:
                rows.append([float(p) for p in parts[:ncols]])
    return pd.DataFrame(rows)


def load_single_spectrum(filepath) -> pd.DataFrame:
    """
    Load a single-geometry BIC spectrum (3-column COMSOL table).

    Returns DataFrame with columns: lambda_nm, T
    """
    df = _skip_comment_csv(filepath, 3)
    df.columns = ["lambda_m", "lambda0_um", "T"]
    df["lambda_nm"] = df["lambda_m"] * 1e9
    return df[["lambda_nm", "T"]].sort_values("lambda_nm").reset_index(drop=True)


def load_tsb_sweep(filepath) -> pd.DataFrame:
    """
    Load a long-format TSB-sweep CSV (bic_tsb_sweep.csv or seawater_refined.csv).

    Expected columns: TSB_nm, lambda_nm, T  (plus optional extras, ignored).

    Returns DataFrame with columns: TSB_nm, lambda_nm, T
    """
    df = pd.read_csv(filepath)
    required = {"TSB_nm", "lambda_nm", "T"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; got {set(df.columns)}")
    return df[["TSB_nm", "lambda_nm", "T"]].dropna()


def load_displacement_sweep(filepath) -> pd.DataFrame:
    """
    Load a cube-displacement parametric sweep (4-column COMSOL table).

    Returns DataFrame with columns: lambda_nm, d_nm, T
    """
    df = _skip_comment_csv(filepath, 4)
    df.columns = ["lambda_m", "d_nm", "lambda0_um", "T"]
    df["lambda_nm"] = df["lambda_m"] * 1e9
    return df[["lambda_nm", "d_nm", "T"]].dropna()


def load_nenv_sweep(filepath) -> pd.DataFrame:
    """
    Load a refractive-index-environment sweep (7-column COMSOL table).

    Returns DataFrame with columns: lambda_nm, n_env, T
    (uses the first transmittance column only).
    """
    df = _skip_comment_csv(filepath, 7)
    df.columns = ["lambda_m", "n_env", "lambda0_um", "T1", "T2", "T3", "T4"]
    df["lambda_nm"] = df["lambda_m"] * 1e9
    return df[["lambda_nm", "n_env", "T1"]].rename(columns={"T1": "T"}).dropna()


# ---------------------------------------------------------------------------
# Resonance analysis
# ---------------------------------------------------------------------------

def normalize_spectrum(lam_nm: np.ndarray, T: np.ndarray,
                       env_win_nm: float = 8.0,
                       top_frac: float = 0.0) -> np.ndarray:
    """
    Estimate and remove the slowly varying baseline.

    Parameters
    ----------
    lam_nm      : wavelength axis (nm)
    T           : transmittance
    env_win_nm  : moving-window width for upper-envelope mode (nm)
    top_frac    : if > 0, use median of the top ``top_frac`` fraction of
                  values as a flat baseline (more robust for broadband spectra)

    Returns
    -------
    T_norm : normalised transmittance, clipped to [0, …]
    """
    if top_frac > 0:
        n = max(3, round(top_frac * len(T)))
        baseline = float(np.median(np.sort(T)[-n:]))
        baseline = max(baseline, np.finfo(float).eps)
        return np.maximum(T / baseline, 0.0)

    step = float(np.median(np.diff(lam_nm)))
    win = max(3, round(env_win_nm / step))
    pad = win // 2
    T_padded = np.pad(T, pad, mode="edge")
    upper = np.array([T_padded[i: i + win].max()
                      for i in range(len(T_padded) - win + 1)])
    upper = upper[:len(T)]
    baseline = np.convolve(upper, np.ones(win) / win, mode="same")
    baseline = np.maximum(baseline, np.finfo(float).eps)
    return np.maximum(T / baseline, 0.0)


def find_dips(lam_nm: np.ndarray, T_norm: np.ndarray,
              min_prominence: float = 2e-3,
              min_distance_nm: float = 1.0) -> np.ndarray:
    """
    Locate resonance dips in a normalised transmission spectrum.

    Parameters
    ----------
    lam_nm        : wavelength axis (nm)
    T_norm        : normalised transmittance
    min_prominence: minimum peak prominence (fraction of normalised T)
    min_distance_nm: minimum dip separation (nm)

    Returns
    -------
    dip_lam : array of dip wavelengths (nm), sorted ascending
    """
    step = float(np.median(np.diff(lam_nm)))
    min_dist_pts = max(1, round(min_distance_nm / step))
    peaks, _ = find_peaks(-T_norm, prominence=min_prominence,
                          distance=min_dist_pts)
    return lam_nm[peaks]


def extract_q(lam_nm: np.ndarray, T: np.ndarray,
              lam_guess: float,
              half_win_nm: float = 4.0,
              fine_step_nm: float = 1e-3,
              edge_frac: float = 0.15):
    """
    Extract the quality factor Q = λ₀ / FWHM around one resonance dip.

    Uses a PCHIP-interpolated fine grid for sub-nm accuracy.

    Parameters
    ----------
    lam_nm      : wavelength axis (nm)
    T           : transmittance (normalised or raw)
    lam_guess   : approximate dip centre (nm)
    half_win_nm : half-width of the fitting window (nm)
    fine_step_nm: interpolation step for the fine grid (nm)
    edge_frac   : fraction of window edges used to estimate local baseline

    Returns
    -------
    Q, lambda0_nm, fwhm_nm  (all NaN if extraction fails)
    """
    mask = (lam_nm >= lam_guess - half_win_nm) & \
           (lam_nm <= lam_guess + half_win_nm)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan

    x = lam_nm[mask]
    y = T[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Fine interpolation
    xf = np.arange(x[0], x[-1], fine_step_nm)
    yf = PchipInterpolator(x, y)(xf)

    imin = int(np.argmin(yf))
    lam0 = float(xf[imin])
    tmin = float(yf[imin])

    # Asymmetric local baseline (median of left / right edge)
    m = max(2, round(edge_frac * len(yf)))
    base_l = float(np.median(yf[:m]))
    base_r = float(np.median(yf[-m:]))
    baseline = max(0.5 * (base_l + base_r), tmin + 1e-12)

    depth = baseline - tmin
    if depth <= 0:
        return np.nan, np.nan, np.nan

    y_hm = tmin + 0.5 * depth

    # Find half-maximum crossings
    left_side = yf[:imin]
    right_side = yf[imin:]

    li_arr = np.where(left_side >= y_hm)[0]
    ri_arr = np.where(right_side >= y_hm)[0]

    if len(li_arr) == 0 or len(ri_arr) == 0:
        return np.nan, np.nan, np.nan

    li = li_arr[-1]
    ri = ri_arr[0] + imin

    # Sub-pixel interpolation of crossings
    def _cross(ya, yb, xa, xb, level):
        return xa + (level - ya) * (xb - xa) / (yb - ya + 1e-30)

    if li + 1 < len(yf):
        lam_l = _cross(yf[li], yf[li + 1], xf[li], xf[li + 1], y_hm)
    else:
        lam_l = xf[li]

    if ri > 0:
        lam_r = _cross(yf[ri - 1], yf[ri], xf[ri - 1], xf[ri], y_hm)
    else:
        lam_r = xf[ri]

    fwhm = lam_r - lam_l
    if fwhm <= 0 or not np.isfinite(fwhm):
        return np.nan, np.nan, np.nan

    q = lam0 / fwhm
    return float(q), float(lam0), float(fwhm)


def sweep_q_vs_param(grouped: dict[str, tuple[np.ndarray, np.ndarray]],
                     n_dips: int = 2,
                     min_prominence: float = 2e-3,
                     min_distance_nm: float = 3.0,
                     half_win_nm: float = 6.0) -> pd.DataFrame:
    """
    Extract Q and λ₀ for each sweep value in a grouped spectrum dictionary.

    Parameters
    ----------
    grouped      : dict mapping param_value → (lam_nm, T_norm)
    n_dips       : expected number of resonance dips per spectrum.
                   Using n_dips=2 enables dual-dip analysis for simultaneous
                   temperature and analyte discrimination.
    min_prominence, min_distance_nm, half_win_nm : dip-finding / Q-extraction

    Returns
    -------
    DataFrame with columns: param, dip, lambda0_nm, Q, fwhm_nm
    """
    records = []
    for param_val, (lam_nm, T_norm) in sorted(grouped.items()):
        dip_lams = find_dips(lam_nm, T_norm,
                             min_prominence=min_prominence,
                             min_distance_nm=min_distance_nm)
        # Keep the n_dips deepest dips (by transmittance depth)
        if len(dip_lams) > n_dips:
            depths = [1.0 - T_norm[np.argmin(np.abs(lam_nm - dl))]
                      for dl in dip_lams]
            order = np.argsort(depths)[::-1]
            dip_lams = np.sort(dip_lams[order[:n_dips]])

        for dip_idx, dl in enumerate(dip_lams):
            q, lam0, fwhm = extract_q(lam_nm, T_norm, dl,
                                       half_win_nm=half_win_nm)
            records.append({
                "param": param_val,
                "dip": dip_idx + 1,
                "lambda0_nm": lam0,
                "Q": q,
                "fwhm_nm": fwhm,
            })

    return pd.DataFrame(records)
