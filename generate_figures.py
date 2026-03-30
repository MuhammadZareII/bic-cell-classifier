#!/usr/bin/env python3
"""
Generate all publication figures for the BIC cell-classifier project.

Usage
-----
    python generate_figures.py

Figures written to results/figures/:
    fig1_kk_refractive_index.png     – K–K average n(ν̃) per cell type (BPE/MPE)
    fig2_discriminant_wavenumbers.png – between-class discriminant power
    fig3_transmission_spectrum.png   – BIC resonance spectrum (reference geometry)
    fig5_spectra_by_displacement.png – transmission vs cube displacement
    fig6_q_vs_displacement.png       – Q-factor and λ₀ vs displacement
    fig7_classification.png          – confusion matrix of KK-based classifier
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import (
    DATA_DIR, FIGURES_DIR,
    PATH_LENGTH_CM, N_INF,
    FONT_FAMILY, FONT_SIZE, FIGURE_DPI, FIG_W, FIG_H, FIG_W_WIDE, LINE_WIDTH,
)
from src.kk import convert_dataframe, META_COLS
from src.sensor import (
    load_single_spectrum, load_displacement_sweep,
    normalize_spectrum, find_dips, extract_q, sweep_q_vs_param,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "figure.dpi": FIGURE_DPI,
})

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (colour-blind-friendly)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  saved -> {path.relative_to(path.parents[3])}")
    plt.close(fig)


# ===========================================================================
# Figure 1 – K–K average refractive index per cell type
# ===========================================================================
def fig1_kk_refractive_index() -> None:
    print("[1/7] K–K refractive index by cell type …")

    df = pd.read_excel(DATA_DIR / "MOESM2_ESM-C.xlsx", sheet_name="Spectra_data")
    n_df = convert_dataframe(df, path_length_cm=PATH_LENGTH_CM, n_inf=N_INF)

    spec_cols = [c for c in n_df.columns if c not in META_COLS]
    wn = np.asarray(spec_cols, dtype=float)
    wl_nm = 1e7 / wn                          # wavenumber → wavelength (nm)

    classes = sorted(n_df["Class"].unique())
    grouped = n_df.groupby("Class")[spec_cols]
    means = grouped.mean()
    stds = grouped.std()

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, FIG_H))

    for i, cls in enumerate(classes):
        mu = means.loc[cls].values
        sd = stds.loc[cls].values
        ax.plot(wl_nm, mu, label=cls, color=PALETTE[i % len(PALETTE)],
                linewidth=LINE_WIDTH)
        ax.fill_between(wl_nm, mu - sd, mu + sd,
                        color=PALETTE[i % len(PALETTE)], alpha=0.15)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive index  $n(\\tilde{\\nu})$")
    ax.set_title("Kramers–Kronig refractive-index spectra by cell type\n"
                 "(mean ± 1 s.d.;  BPE = benign,  MPE = malignant)")
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    save(fig, "fig1_kk_refractive_index.png")


# ===========================================================================
# Figure 2 – Between-class discriminant power vs wavenumber
# ===========================================================================
def fig2_discriminant_wavenumbers() -> None:
    print("[2/7] Discriminant wavenumbers …")

    df = pd.read_excel(DATA_DIR / "MOESM2_ESM-C.xlsx", sheet_name="Spectra_data")
    n_df = convert_dataframe(df, path_length_cm=PATH_LENGTH_CM, n_inf=N_INF)

    spec_cols = [c for c in n_df.columns if c not in META_COLS]
    wn = np.asarray(spec_cols, dtype=float)
    wl_nm = 1e7 / wn

    # ANOVA F-statistic per wavenumber (between-class / within-class variance)
    classes = n_df["Class"].unique()
    grand_mean = n_df[spec_cols].mean().values
    n_total = len(n_df)
    n_classes = len(classes)

    ss_between = np.zeros(len(spec_cols))
    ss_within = np.zeros(len(spec_cols))

    for cls in classes:
        sub = n_df.loc[n_df["Class"] == cls, spec_cols].values
        nc = len(sub)
        cm = sub.mean(axis=0)
        ss_between += nc * (cm - grand_mean) ** 2
        ss_within += ((sub - cm) ** 2).sum(axis=0)

    df_between = n_classes - 1
    df_within = n_total - n_classes
    F = (ss_between / df_between) / (ss_within / df_within + 1e-30)

    # Top-10 discriminant wavenumbers (for annotation)
    top10 = np.argsort(F)[-10:]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_WIDE, FIG_H * 1.5),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax1, ax2 = axes

    # Panel 1: F-statistic
    ax1.plot(wl_nm, F, color=PALETTE[0], linewidth=LINE_WIDTH)
    for idx in top10:
        ax1.axvline(wl_nm[idx], color="gray", linestyle="--",
                    linewidth=0.6, alpha=0.7)
    ax1.set_ylabel("ANOVA $F$-statistic")
    ax1.set_title("Between-class discriminant power (BPE vs MPE) in the refractive-index domain")
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Panel 2: mean RI per cell type (for reference)
    grouped_means = n_df.groupby("Class")[spec_cols].mean()
    for i, cls in enumerate(sorted(classes)):
        ax2.plot(wl_nm, grouped_means.loc[cls].values,
                 label=cls, color=PALETTE[i % len(PALETTE)],
                 linewidth=LINE_WIDTH * 0.8)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Mean $n(\\tilde{\\nu})$")
    ax2.legend(fontsize=FONT_SIZE - 2, ncol=2)
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    save(fig, "fig2_discriminant_wavenumbers.png")


# ===========================================================================
# Figure 3 – BIC resonance spectrum at reference geometry
# ===========================================================================
def fig3_transmission_spectrum() -> None:
    print("[3/7] BIC resonance spectrum (reference geometry) …")

    df = load_single_spectrum(DATA_DIR / "bic_spectrum_single.csv")
    lam = df["lambda_nm"].values
    T = df["T"].values
    T_norm = normalize_spectrum(lam, T)

    dip_lams = find_dips(lam, T_norm, min_prominence=5e-3, min_distance_nm=1.0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(lam, T_norm, color=PALETTE[0], linewidth=LINE_WIDTH)

    for dl in dip_lams:
        q, lam0, fwhm = extract_q(lam, T_norm, dl)
        if np.isfinite(q):
            t_at_dip = float(T_norm[np.argmin(np.abs(lam - lam0))])
            ax.plot(lam0, t_at_dip, "o",
                    markerfacecolor="none", markeredgecolor=PALETTE[1],
                    markeredgewidth=1.5, markersize=7, zorder=5)
            ax.text(lam0 + 0.4, t_at_dip + 0.005,
                    f"Q={q:.0f}",
                    fontsize=FONT_SIZE - 1, va="bottom")

    ax.set_xlabel("$\\lambda$ (nm)")
    ax.set_ylabel("Normalized transmittance (a.u.)")
    ax.set_title("BIC resonance spectrum — reference geometry (normalized)")
    valid = (T_norm <= 1.005)
    ax.set_xlim(lam[valid][0], lam[valid][-1])
    ax.set_ylim(max(0, T_norm.min() - 0.02), 1.05)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    save(fig, "fig3_transmission_spectrum.png")


# ===========================================================================
# Figure 5 – Transmission spectra vs cube displacement
# ===========================================================================
def fig5_spectra_by_displacement() -> None:
    print("[5/7] Spectra by cube displacement …")

    df = load_displacement_sweep(DATA_DIR / "bic_displacement_sweep.csv")
    d_vals = sorted(df["d_nm"].unique())

    line_colors = ["#0072BD", "#D95319", "#EDB120",
                   "#7E2F8E", "#77AC30", "#4DBEEE"]

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, FIG_H))

    for i, d in enumerate(d_vals):
        sub = df[df["d_nm"] == d].sort_values("lambda_nm")
        lam = sub["lambda_nm"].values
        T = sub["T"].values
        if len(lam) < 10:
            continue
        T_norm = normalize_spectrum(lam, T, top_frac=0.20)
        ax.plot(lam, T_norm,
                color=line_colors[i % len(line_colors)],
                linewidth=LINE_WIDTH,
                label=f"$d$ = {d:.0f} nm")

    ax.set_xlabel("$\\lambda$ (nm)")
    ax.set_ylabel("Normalized transmittance (a.u.)")
    ax.set_title("BIC spectra vs cube displacement $d$ (normalized)")
    ax.set_xlim(450, 750)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="center right", fontsize=FONT_SIZE - 1)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    save(fig, "fig5_spectra_by_displacement.png")


# ===========================================================================
# Figure 6 – Q-factor and λ₀ vs cube displacement
# ===========================================================================
def fig6_q_vs_displacement() -> None:
    print("[6/7] Q vs displacement …")

    df = load_displacement_sweep(DATA_DIR / "bic_displacement_sweep.csv")

    grouped = {}
    for d in sorted(df["d_nm"].unique()):
        sub = df[df["d_nm"] == d].sort_values("lambda_nm")
        mask = (sub["lambda_nm"] >= 550) & (sub["lambda_nm"] <= 680)
        sub = sub[mask]
        lam = sub["lambda_nm"].values
        T = sub["T"].values
        if len(lam) < 10:
            continue
        T_norm = normalize_spectrum(lam, T, top_frac=0.20)
        grouped[d] = (lam, T_norm)

    metrics = sweep_q_vs_param(grouped, n_dips=1,
                               min_prominence=2e-3,
                               min_distance_nm=5.0,
                               half_win_nm=12.0)

    if metrics.empty or metrics["Q"].isna().all():
        print("    (no dips extracted – skipping fig 6)")
        return

    valid = metrics.dropna(subset=["Q", "lambda0_nm"])
    if valid.empty:
        print("    (dip metrics empty – skipping fig 6)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, FIG_H))
    ax_q, ax_l = axes

    d_vals_m = sorted(valid["param"].unique())
    q_vals = [valid.loc[valid["param"] == d, "Q"].mean() for d in d_vals_m]
    l_vals = [valid.loc[valid["param"] == d, "lambda0_nm"].mean() for d in d_vals_m]

    ax_q.plot(d_vals_m, q_vals, "-o", color=PALETTE[0],
              linewidth=LINE_WIDTH, markersize=5)
    ax_q.set_xlabel("Displacement $d$ (nm)")
    ax_q.set_ylabel("$Q = \\lambda_0 / \\Delta\\lambda$")
    ax_q.set_title("Quality factor vs displacement")
    ax_q.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax_l.plot(d_vals_m, l_vals, "-o", color=PALETTE[2],
              linewidth=LINE_WIDTH, markersize=5)
    ax_l.set_xlabel("Displacement $d$ (nm)")
    ax_l.set_ylabel("$\\lambda_0$ (nm)")
    ax_l.set_title("Resonance wavelength vs displacement")
    ax_l.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if len(l_vals) >= 2:
        sens = (l_vals[-1] - l_vals[0]) / (d_vals_m[-1] - d_vals_m[0])
        ax_l.text(0.05, 0.95,
                  f"$d\\lambda_0/dd \\approx {sens:.2f}$ nm/nm",
                  transform=ax_l.transAxes, va="top",
                  fontsize=FONT_SIZE - 1,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))

    fig.suptitle("BIC sensor — resonance sensitivity to cube displacement",
                 fontsize=FONT_SIZE + 1)
    fig.tight_layout()
    save(fig, "fig6_q_vs_displacement.png")


# ===========================================================================
# Figure 7 – Cell-type classification from K–K features
# ===========================================================================
def fig7_classification() -> None:
    print("[7/7] Cell-type classification (K–K features) …")

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        print("    scikit-learn not found – skipping fig 7 (pip install scikit-learn)")
        return

    df = pd.read_excel(DATA_DIR / "MOESM2_ESM-C.xlsx", sheet_name="Spectra_data")
    n_df = convert_dataframe(df, path_length_cm=PATH_LENGTH_CM, n_inf=N_INF)

    spec_cols = [c for c in n_df.columns if c not in META_COLS]
    X = n_df[spec_cols].values
    y = n_df["Class"].values
    classes = sorted(np.unique(y))

    # Pipeline: standardise → PCA (keep 95 % variance) → SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    overall_acc = float((y_pred == y).mean())

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 1.1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f",
              im_kw={"vmin": 0, "vmax": 1})
    ax.set_title(
        f"Cell-type classifier (BPE vs MPE) — confusion matrix\n"
        f"(5-fold CV, PCA + SVM,  accuracy = {overall_acc:.1%})"
    )
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("True cell type")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save(fig, "fig7_classification.png")

    print(f"    overall accuracy: {overall_acc:.1%}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("Generating figures ...\n")
    fig1_kk_refractive_index()
    fig2_discriminant_wavenumbers()
    fig3_transmission_spectrum()
    fig5_spectra_by_displacement()
    fig6_q_vs_displacement()
    fig7_classification()
    print("\nDone. All figures saved to results/figures/")
