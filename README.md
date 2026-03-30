# BIC Taste Sensor

Photonic **Bound-State-in-the-Continuum (BIC)** resonator for label-free taste-compound detection, with Kramers–Kronig spectral conversion and automated classification.

## Overview

BIC resonators exhibit ultra-sharp transmission dips (Q > 1 000) whose resonance wavelength and quality factor shift with the local refractive index of the surrounding medium.  Because standard spectral databases report *absorbance*, not refractive index, we apply the **Kramers–Kronig (K–K) relations** to convert absorbance spectra into refractive-index proxies — the quantity the sensor physically responds to.

The full pipeline is:

```
Absorbance spectra (FTIR database)
        │
        ▼  Kramers–Kronig conversion (src/kk.py)
Refractive-index spectra  n(ν̃)
        │
        ├──▶  Feature extraction → PCA + SVM classifier  (generate_figures.py)
        │
        └──▶  COMSOL-simulated BIC transmission spectra
                │  (sensor response to each compound)
                ▼
         Q-factor & λ₀ extraction  (src/sensor.py)
```

## Sensor design

| Parameter | Value |
|---|---|
| Structure | Si nano-cubes on SiO₂ substrate |
| Cube width | 130 nm (at 20 °C) |
| Periodicity | 400 × 200 nm |
| Substrate thickness | 150 nm |
| Environment | Seawater (n = 1.35, salinity 0.35 %) |
| Working range | 655 – 725 nm |
| Best Q-factor | > 1 000 |

The **translational symmetry break (TSB)** — a controlled gap between adjacent cubes — tunes the coupling from a dark BIC mode into the radiation continuum, enabling observation of the resonance.

## Results

| Figure | Description |
|---|---|
| fig1_kk_refractive_index | Mean K–K n(ν̃) ± 1 s.d. per taste class |
| fig2_discriminant_wavenumbers | ANOVA F-statistic — wavenumbers that best separate classes |
| fig3_transmission_spectrum | Sharp BIC resonances in the sweet-compound medium (Q ≈ 1 070) |
| fig5_spectra_by_displacement | Spectral evolution with cube displacement (umami medium) |
| fig6_q_vs_displacement | Q-factor and λ₀ vs displacement for each resonance mode |
| fig7_classification | Confusion matrix — 5-fold CV PCA+SVM on K–K features |

## Project structure

```
bic-taste-sensor/
├── data/
│   ├── MOESM2_ESM-C.xlsx        absorbance spectra (FTIR, taste compounds)
│   ├── sweet_table.csv          COMSOL: single-compound BIC spectrum
│   ├── bitter_table.csv         COMSOL: TSB sweep, bitter medium
│   ├── seawater_refined.csv     COMSOL: TSB sweep, seawater baseline
│   ├── umami_table.csv          COMSOL: displacement sweep, umami medium
│   └── nenv_table.csv           COMSOL: refractive-index sensitivity sweep
├── src/
│   ├── config.py                paths, sensor constants, plot style
│   ├── kk.py                    Kramers–Kronig conversion
│   └── sensor.py                COMSOL data loading, dip detection, Q extraction
├── results/
│   └── figures/                 generated PNG figures (300 DPI)
├── generate_figures.py          reproduce all publication figures
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce figures

```bash
python generate_figures.py
```

All seven figures are written to `results/figures/`.  The K–K step processes the full absorbance dataset (≈ 10 min on a laptop); the remaining steps are fast.

## Data sources

- **Absorbance spectra**: MOESM2_ESM-C supplementary dataset (ATR-FTIR, 2 mm path length, aqueous taste compounds).
- **Transmission spectra**: COMSOL Multiphysics 6.3 FDTD simulations of the Si/SiO₂ BIC resonator geometry.

## Dependencies

Python ≥ 3.11, plus the packages listed in `requirements.txt`.
