# BIC Cell Classifier

Photonic **Bound-State-in-the-Continuum (BIC)** resonator for label-free discrimination of benign and malignant prostate epithelial cells, with Kramers–Kronig spectral conversion and SVM classification.

## Concept

BIC resonators exhibit ultra-sharp transmission dips (Q > 1 000) whose resonance wavelength and quality factor shift when the refractive index (RI) of the surrounding medium changes.  Two resonance dips appear at distinct wavelengths; because each dip has a different RI and temperature sensitivity, tracking both simultaneously allows the two coupled effects — analyte-induced RI change and temperature-induced RI change — to be separated into two independent equations.  This dual-dip approach provides temperature-compensated detection without an external temperature sensor.

The biological signal of interest is the RI difference between benign prostate epithelium (**BPE**) and malignant prostate epithelium (**MPE**).  Standard spectral databases report *absorbance*, not refractive index, so the **Kramers–Kronig (K–K) relations** are used to convert ATR-FTIR absorbance spectra into refractive-index spectra — the physical quantity the sensor responds to.

## Pipeline

```
ATR-FTIR absorbance spectra (MOESM2_ESM-C.xlsx)
  82 samples: 35 BPE (benign), 47 MPE (malignant)
        │
        ▼  Kramers–Kronig conversion (src/kk.py)
Refractive-index spectra  n(ν̃)
        │
        ├──▶  Feature extraction → PCA + SVM classifier  (generate_figures.py)
        │       discriminates BPE from MPE on K–K RI features
        │
        └──▶  COMSOL-simulated BIC transmission spectra
                │  sensor characterisation sweeps (src/sensor.py)
                ▼
         Q-factor & λ₀ extraction per resonance dip
         dual-dip analysis for temperature compensation
```

## Sensor design

| Parameter | Value |
|---|---|
| Structure | Si nano-cubes on SiO₂ substrate |
| Cube width | 130 nm (at 20 °C) |
| Periodicity | 400 × 200 nm |
| Substrate thickness | 150 nm |
| Simulation medium | Seawater-like (n = 1.35, salinity 0.35 %) |
| Working range | 655 – 725 nm |
| Best Q-factor | > 1 000 |

The **translational symmetry break (TSB)** — a controlled gap between adjacent cubes — couples a dark BIC mode into the radiation continuum, making the resonance observable.  The TSB magnitude controls Q: a larger break increases radiation loss and broadens the dip.

## Dataset

`MOESM2_ESM-C.xlsx` — ATR-FTIR absorbance spectra from the supplementary material of the source paper.  Each row is one sample; the `Class` column labels it as `BPE` (benign prostate epithelium, n = 35) or `MPE` (malignant prostate epithelium, n = 47).  Spectral columns are wavenumbers in cm⁻¹.

## COMSOL simulation files

All COMSOL files are parametric sweeps of the same BIC sensor in the same seawater-like medium.  The sweep parameter differs between files; the medium does not.

| File | Sweep | Description |
|---|---|---|
| `bic_spectrum_single.csv` | — | Single reference BIC transmission spectrum at fixed geometry |
| `bic_tsb_sweep.csv` | TSB (nm) | Transmission vs translational symmetry break magnitude |
| `seawater_refined.csv` | TSB (nm) | TSB sweep in the seawater reference medium |
| `bic_displacement_sweep.csv` | d (nm) | Transmission vs cube displacement |
| `bic_nenv_sweep.csv` | n_env | Transmission vs environmental refractive index (sensitivity characterisation) |

## Results

| Figure | Description |
|---|---|
| fig1_kk_refractive_index | Mean K–K n(ν̃) ± 1 s.d. per cell type (BPE / MPE) |
| fig2_discriminant_wavenumbers | ANOVA F-statistic — wavenumbers that best separate BPE from MPE |
| fig3_transmission_spectrum | BIC resonance spectrum at reference geometry (Q ≈ 1 070) |
| fig5_spectra_by_displacement | Spectral evolution with cube displacement |
| fig6_q_vs_displacement | Q-factor and λ₀ vs cube displacement |
| fig7_classification | Confusion matrix — 5-fold CV PCA+SVM on K–K features |

## Project structure

```
bic-cell-classifier/
├── data/
│   ├── MOESM2_ESM-C.xlsx            ATR-FTIR spectra (BPE/MPE, 82 samples)
│   ├── bic_spectrum_single.csv      COMSOL: reference BIC spectrum
│   ├── bic_tsb_sweep.csv            COMSOL: TSB parametric sweep
│   ├── bic_displacement_sweep.csv   COMSOL: cube displacement sweep
│   ├── seawater_refined.csv         COMSOL: seawater medium TSB sweep
│   └── bic_nenv_sweep.csv           COMSOL: RI environment sweep
├── src/
│   ├── config.py                    paths, sensor constants, plot style
│   ├── kk.py                        Kramers–Kronig conversion
│   └── sensor.py                    COMSOL data loading, dip detection, Q extraction
├── results/
│   └── figures/                     generated PNG figures (300 DPI)
├── generate_figures.py              reproduce all publication figures
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

All figures are written to `results/figures/`.  The K–K step processes the full absorbance dataset (≈ 10 min on a laptop); the remaining steps are fast.

## Data sources

- **Cell spectra**: MOESM2_ESM-C supplementary dataset — ATR-FTIR absorbance spectra of BPE and MPE cell lines.
- **Transmission spectra**: COMSOL Multiphysics FDTD simulations of the Si/SiO₂ BIC resonator.

## Dependencies

Python ≥ 3.11, plus the packages listed in `requirements.txt`.
