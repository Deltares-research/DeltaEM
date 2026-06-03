# DeltaEM

[![License: MIT](https://img.shields.io/pypi/l/imod)](https://choosealicense.com/licenses/mit)
[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![Formatting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<img src="deltaem_logo.svg" alt="DeltaEM Logo" width="100" align="left">

The Deltares Electromagnetics (DeltaEM) package provides 1D and laterally
constrained (LCI) SimPEG inversion for [GEM-2](https://geophex.com/gem-2/)
frequency-domain electromagnetic data, with both a Streamlit GUI and
scriptable CLI workflows.

It reads GEM-2 CSV/XYZ input, stacks soundings by line, runs per-sounding or
full-line inversion, and returns conductivity layers, channel misfit,
Christiansen & Auken depth-of-investigation (DOI), and section/sounding/map
outputs.

<br clear="left"/>

## Quick start

```bash
git clone https://github.com/jkinggeo/gem2-inversion.git
cd gem2-inversion

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run gem2_simpeg_app.py
```

The app opens at <http://localhost:8501>. Pick **Upload (.csv or .xyz)** in the
sidebar and drop in a GEM-2 file to try it without touching paths.

### Conda / mamba

If you prefer conda:

```bash
conda create -n gem2 python=3.12 -c conda-forge
conda activate gem2
pip install -r requirements.txt
streamlit run gem2_simpeg_app.py
```

### Optional: pre-fill the path inputs

The two file-path inputs in the sidebar can be pre-filled from environment
variables, useful if you always work with the same campaign folder:

```bash
# Windows PowerShell
$env:GEM2_DEFAULT_XYZ = "C:\path\to\017_7015_gem.xyz"
$env:GEM2_DEFAULT_CSV = "C:\path\to\056_7015_gem.csv"

# macOS / Linux
export GEM2_DEFAULT_XYZ=/path/to/017_7015_gem.xyz
export GEM2_DEFAULT_CSV=/path/to/056_7015_gem.csv

streamlit run gem2_simpeg_app.py
```

## Requirements

* Python 3.10+ (developed and tested on 3.12)
* The packages pinned in [`requirements.txt`](requirements.txt):
  numpy, pandas, scipy, matplotlib, streamlit, simpeg, discretize, and
  (optional) rasterio for DEM sampling.

`simpeg` pulls in `discretize`, `pymatsolver`, `geoana`, `empymod`,
`SimPEG`-internal solvers, etc. On Windows, installing from PyPI with the
bundled MUMPS/Pardiso-free solvers is sufficient for the small 1D problems
used here.

`rasterio` is only needed when sampling sensor elevation from a GeoTIFF DEM
(`gem2_simpeg_invert.py --dem-path` or the DEM section of the Streamlit
sidebar). Comment it out in `requirements.txt` if its native wheels cause
install trouble on your platform.

## Repository layout

```
gem2-inversion/
├── gem2_simpeg_app.py          # Streamlit GUI (entry point)
├── gem2_simpeg_invert.py       # Per-sounding 1D SimPEG inversion (CLI + lib)
├── gem2_simpeg_lci.py          # Laterally-constrained per-line inversion (CLI + lib)
├── gem2_doi.py                 # Christiansen & Auken DOI on the inversion mesh
├── gem_csv_to_xyz.py           # GEM-2 CSV → whitespace-column XYZ
├── gem_inversion_xyz_export.py # Wide inversion CSV → long-format XYZ point cloud
├── requirements.txt
├── LICENSE
└── README.md
```

## Command-line usage

Every module is importable as a library *and* runnable as a script:

```bash
# Per-sounding inversion
python gem2_simpeg_invert.py input.xyz \
    --output-csv inverted.csv \
    --stack-out-spacing 1.0 --stack-window 2.0 \
    --n-layers 6 --max-depth 5.0 --first-thickness 0.4 \
    --use-channels q --n-workers 4

# Laterally-constrained 1D inversion (one section per line)
python gem2_simpeg_lci.py input.xyz \
    --output-csv inverted_lci.csv \
    --alpha-x-constraint 1.2 --alpha-y-vertical 3.0 \
    --n-layers 6 --max-depth 5.0

# Raw GEM-2 CSV → XYZ
python gem_csv_to_xyz.py raw.csv raw.xyz --sensor-height 0.1

# Inversion wide CSV → long-format point cloud (CloudCompare / ParaView / GIS)
python gem_inversion_xyz_export.py inverted.csv -o inverted_xyz_long.csv
```

`python <script>.py --help` lists every flag with its default.

## What the inversion does

* **Forward model**: `Simulation1DLayered` per sounding, GEM-2 geometry
  (`CoilSeparation = 1.66 m`, HCP / Z–Z), one quadrature (and optionally
  in-phase) channel per frequency in ppm.
* **Mesh**: `n_layers` cells with a geometric thickness schedule
  (`--first-thickness`, `--max-depth`, `--n-layers`).
* **Noise model**: relative error (default 5 %) with an absolute ppm floor
  (default 10 ppm), passed via `Data(standard_deviation=...)`.
* **Warm start**: apparent conductivity from the lowest-frequency Q channel
  per sounding instead of a global 0.5 S/m starting model.
* **Regularization**: smooth L2 on `log σ` with a small reference-model
  penalty (`--alpha-s`), `--alpha-x` for smoothness in depth (per-sounding)
  or laterally (LCI).
* **Directives**: `UpdateSensitivityWeights → UpdatePreconditioner →
  BetaEstimate_ByEig → BetaSchedule → TargetMisfit`, cooling factor 1.5.
* **DOI**: Christiansen & Auken cumulative-sensitivity DOI computed on the
  inversion mesh (no rediscretization), in natural-log-σ space to match
  `ExpMap`.

## File formats

* **Input CSV**: GEM-2 exporter CSV with X/Y, sensor height, and paired I/Q
  ppm columns per frequency. Converted to XYZ by `gem_csv_to_xyz.py`.
* **Input XYZ**: whitespace-column text with `/Line /Sample /X /Y /GPSalt
  /SensorHeight ... /I_<f>Hz /Q_<f>Hz ...` headers. Rows with `*` for X/Y
  are skipped.
* **Output CSV**: one row per stacked sounding with
  `sigma_layer_*`, `depth_top_layer_*_m`, `depth_bottom_layer_*_m`, `chi2`,
  `n_channels_used`, `doi_m`, etc.
* **Long-format point cloud**: optional one-row-per-(sounding × layer)
  CSV via `gem_inversion_xyz_export.py`, for CloudCompare / ParaView / GIS.

## Acknowledgements

* [SimPEG](https://simpeg.xyz/) — the inversion / forward modelling stack.
* Christiansen, A.V. & Auken, E. (2012). *A global measure for depth of
  investigation*. Geophysics, 77(4): WB171–WB177 — DOI methodology.
* Auken, E. & Christiansen, A.V. (2004). *Layered and laterally constrained
  2D inversion of resistivity data*. Geophysics, 69(3): 752–761 — LCI scheme.

