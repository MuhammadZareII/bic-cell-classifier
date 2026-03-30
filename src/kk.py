"""
Kramers–Kronig conversion: absorbance spectra → refractive-index spectra.

Physical model
--------------
Absorbance A(ν̃) is first converted to the extinction coefficient k(ν̃):

    k(ν̃) = ln(10) · A(ν̃) / (4π · ν̃ · L)

where ν̃ is the wavenumber in cm⁻¹ and L is the optical path length in cm.

The refractive index n(ν̃) is then obtained via the Kramers–Kronig relation:

    n(ν̃) = n∞ + (2/π) P.V. ∫₀^∞ [ν̃′ k(ν̃′) / (ν̃′² − ν̃²)] dν̃′

implemented as a discrete principal-value trapezoidal integral.

Typical parameters for aqueous taste-compound spectra:
  path_length_cm = 0.2   (2 mm transmission cell)
  n_inf          = 1.35  (water-like high-frequency limit)
"""

import numpy as np
import pandas as pd

META_COLS = ["ID", "Class"]


def absorbance_to_k(a: np.ndarray, wn: np.ndarray,
                    path_length_cm: float = 1.0) -> np.ndarray:
    """
    Convert absorbance to extinction coefficient k(ν̃).

    Parameters
    ----------
    a             : absorbance array, shape (N,)
    wn            : wavenumbers (cm⁻¹), shape (N,)
    path_length_cm: optical path length L in cm

    Returns
    -------
    k : ndarray, shape (N,)
    """
    return (np.log(10.0) * a) / (4.0 * np.pi * wn * path_length_cm)


def kk_transform(wn: np.ndarray, k: np.ndarray,
                 n_inf: float = 1.0) -> np.ndarray:
    """
    Apply the discrete Kramers–Kronig relation to obtain n(ν̃) from k(ν̃).

    Parameters
    ----------
    wn    : wavenumbers (cm⁻¹), shape (N,)
    k     : extinction-coefficient array, shape (N,)
    n_inf : high-frequency refractive index limit n∞

    Returns
    -------
    n : ndarray, shape (N,)
    """
    n = np.empty_like(k)
    for i, nu in enumerate(wn):
        mask = np.ones(len(wn), dtype=bool)
        mask[i] = False                          # principal-value exclusion
        integrand = (wn[mask] * k[mask]) / (wn[mask] ** 2 - nu ** 2)
        n[i] = n_inf + (2.0 / np.pi) * np.trapezoid(integrand, wn[mask])
    return n


def convert_dataframe(df: pd.DataFrame,
                      path_length_cm: float = 1.0,
                      n_inf: float = 1.0) -> pd.DataFrame:
    """
    Convert an absorbance DataFrame to a refractive-index DataFrame.

    The input DataFrame must have columns ``ID``, ``Class``, and one column
    per wavenumber (column header = wavenumber in cm⁻¹).  Metadata columns
    are preserved; spectral columns are replaced with n(ν̃) values.

    Parameters
    ----------
    df            : input absorbance DataFrame
    path_length_cm: optical path length L in cm
    n_inf         : high-frequency refractive index limit n∞

    Returns
    -------
    DataFrame with the same shape, spectral columns replaced by n(ν̃).
    """
    spec_cols = [c for c in df.columns if c not in META_COLS]
    wn = np.asarray(spec_cols, dtype=float)

    n_rows = []
    for _, row in df.iterrows():
        a = row[spec_cols].to_numpy(dtype=float)
        k = absorbance_to_k(a, wn, path_length_cm)
        n = kk_transform(wn, k, n_inf)
        n_rows.append(n)

    n_df = pd.DataFrame(n_rows, columns=spec_cols, index=df.index)
    return pd.concat([df[META_COLS], n_df], axis=1)
