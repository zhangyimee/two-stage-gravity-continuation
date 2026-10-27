# Two-stage equivalent-source gravity downward continuation

Research code for *A Two-Stage Equivalent-Source Method for Gravity Downward Continuation over Undulating Observation Surfaces*, by Yimi Zhang, Shanhao Li, Yucheng Sun, Feng Luo and Zhongping Xu. Correspondence: singalflash@163.com.

The solver first constructs an initial equivalent-source model with a mean-elevation BTTB–FFT operator, then refines it with a direct prism operator at the actual observation elevations. The observations are unchanged. The final manuscript examples use one equivalent-source layer. CUDA is the implementation backend for direct evaluation.

## Contents

| Path | Purpose |
|---|---|
| `src/twostage/operators.py` | BTTB–FFT forward/adjoint operations, analytic kernels and shared numerical utilities |
| `src/twostage/solver.py` | Matrix-free mixed-precision CUDA direct operators and CGLS |
| `src/twostage/model.py`, `prism_truth.py` | Eight-prism model and independent analytic reference-field functions |
| `run_examples.py` | Manuscript clean, noisy, source-depth and optional field examples |
| `data/synthetic/` | Observation/elevation grids, target planes, analytic reference grids and stored noise realizations |
| `reference_results/manuscript/` | Archived synthetic outputs and original study summary JSON |
| `reference_results/preliminary/` | Archived source-depth selection results |
| `USER_GUIDE.md`, `TUTORIAL.md` | Options, input/output definitions and example procedures |
| `provenance/` | Source-function mapping, input origins and release-file checksums |

Exploratory tests, obsolete experiment suites, intermediate figure designs, private manuscript files and editor configuration are excluded. Historical outputs are distinguished from new runs.

## Installation

Recorded environment: Python 3.12.7, NumPy 1.26.4, SciPy 1.17.1 and Numba 0.63.1. The study used an NVIDIA RTX 4080 Laptop GPU with 12 GB memory. Direct-operator examples require an NVIDIA CUDA-capable device, compatible driver and functioning Numba CUDA backend. No CPU fallback is substituted for the published implementation.

Create a Python 3.12 environment and install dependencies:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS shell: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The driver/CUDA setup is separate from these Python dependencies. The recorded dependency set has not been tested in a fresh environment during packaging.

## Examples

Run commands from this directory in the configured environment:

```sh
python run_examples.py clean
python run_examples.py noise --seed 100
python run_examples.py depth-scan
```

Each command creates a new folder under `outputs/`. Existing output folders are refused. See the tutorial for protocols and expected historical results. No experiments were run while preparing this public package; static checks do not establish numerical reproduction.

The optional `field` command uses archived processed Olympic Dam grids. These third-party field inputs and field-derived grids are held outside this initial upload folder while exact attribution and reuse terms are completed. See `DATA_AVAILABILITY.md`. Until those files are added, `field` and `all` stop with an explanatory error. This initial package therefore supports the synthetic examples, not the complete field-data reproduction.

## Scientific and timing conventions

Coordinates x/y are east/north in metres; z is positive downward. Elevation grids are positive upward and are negated on input. Gravity is mGal and density contrast is kg/m^3. Synthetic target z values are -800, -600, -400 and -200 m, all above the shallowest true body.

Approximate-stage RMS is evaluated with the mean-plane operator; direct-stage RMS uses the actual observation geometry. Target RMSE uses independent synthetic reference fields. Field amplitude increments are not continuation errors.

Archived clean results report approximately 19.70 s for two-stage and 53.66 s for single-stage CGLS (about 2.72 times faster at matched misfit), with a final true-surface RMS of approximately 8.5925e-4 mGal. These are historical measurements, not new results or deterministic performance requirements. The original CGLS timer excludes initial residual/adjoint setup, input loading, kernel construction, target prediction and plotting. A zero-iteration refinement has zero internal iteration time but still requires an initial direct residual evaluation.

Saved DSAA grids were rounded at export, so metrics recomputed from grids need not exactly equal full-precision JSON metrics. The clean entry uses the final selected source layer directly; the original complete study also evaluated other full-grid depths, so warm-up state and timings may differ. The generic low-level operator supports multiple layers, but the final examples use one.

## License and status

The research software is licensed under the MIT License selected by the author; see `LICENSE`. This license does not relicense third-party field data. Synthetic inputs and archived research outputs are supplied as research materials; see `DATA_AVAILABILITY.md`.

The associated article is a manuscript prepared for submission to Computers & Geosciences. Publication acceptance, a software DOI and an external numerical replication are not claimed.
