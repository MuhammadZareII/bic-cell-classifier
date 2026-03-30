"""
Project-wide paths and constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Kramers–Kronig conversion
# ---------------------------------------------------------------------------
PATH_LENGTH_CM = 0.2   # 2 mm ATR/transmission cuvette used in source spectra
N_INF = 1.35           # high-frequency RI limit (water-like baseline)

# ---------------------------------------------------------------------------
# COMSOL sensor geometry (from Parameters.txt)
# ---------------------------------------------------------------------------
PERIOD_X_NM = 400      # periodicity in x (nm)
PERIOD_Y_NM = 200      # periodicity in y (nm)
CUBE_WIDTH_NM = 130    # Si cube width at 20 °C (nm)
SUBSTRATE_NM = 150     # SiO₂ substrate thickness (nm)
N_SI = 3.47            # Si refractive index at 20 °C
N_SIO2 = 1.46          # SiO₂ refractive index at 20 °C
N_ENV = 1.35           # seawater RI at 20 °C, 0.35 % salinity

# Sensor wavelength sweep
LAMBDA_S_NM = 655.0    # sweep start (nm)
LAMBDA_E_NM = 725.0    # sweep end   (nm)

# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------
FONT_FAMILY = "serif"
FONT_SIZE = 11
FIGURE_DPI = 300
FIG_W = 6.5            # single-column width (inches)
FIG_H = 4.0
FIG_W_WIDE = 8.5       # two-column / landscape width (inches)
LINE_WIDTH = 1.5
