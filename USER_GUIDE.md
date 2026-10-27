# User guide

## Command line

`python run_examples.py CASE [--seed 100] [--output-name NAME]`

| Option | Meaning |
|---|---|
| `clean` | Full-grid two-stage solve (300 FFT + 20 direct iterations) and zero-start single-stage comparison stopped at the final two-stage RMS, at most 600 iterations |
| `noise` | Stored noisy observations, sigma=0.1 mGal, discrepancy multiplier tau=1.05; stopping is checked before direct refinement |
| `depth-scan` | 101 x 76 preliminary grid; four source centers; 150 approximate + 40 direct iterations; synthetic-truth-based selection |
| `field` | Optional continuation from archived processed field grids; requires separately documented field inputs |
| `all` | clean, noise(seed selected) and field in sequence; requires field inputs, does not include depth-scan |
| `--seed` | One of 100–104; default 100. Only seed100 is the final manuscript noisy example |
| `--output-name` | New single folder name under outputs; defaults to a timestamp; existing folders are refused |

## Inputs

DSAA Surfer ASCII grids start with `DSAA`, dimensions nx/ny, x and y ranges, value range, then ny rows of nx values. Flattening is row-major with x varying fastest. Synthetic grids have 201 x 151 nodes. x/y and elevation/target values are metres; anomalies are mGal. Files `z800.grd` through `z200.grd` contain target z coordinates, whereas `true_continuation_z*.grd` contain reference gravity. `terrain.grd` is positive-up elevation.

Source top/bottom are 1000/1750 m with z positive downward in the final synthetic examples. All supplied input files are addressed relative to the package, not an author's machine path.

## Outputs

Each invocation creates `outputs/NAME/environment.json` and a subdirectory per case. Each case writes `result.json` containing solver history, metrics and internal timings. Grid files contain fitted/continued fields and residuals. `models.npz` stores model arrays for clean/noise/field. Depth selection writes its rows and selected depth to result.json. Reference results are read-only historical materials and are not overwritten.

For clean/noise, continuation error is prediction minus analytic truth. Observation fitting residuals are observed minus fitted. Consult output names and the entry functions when comparing signed residuals; RMSE is sign-independent.

## Dependencies and failures

No editable installation is needed: run_examples.py prepends the adjacent src directory. Imports require the recorded numerical dependencies. A missing CUDA device stops numerical examples. Missing optional field inputs stop field/all before creating outputs. Output-name collisions stop instead of overwriting results. Numerical failures are not retried automatically.

Memory, hardware, software versions and load influence runtime. The reference timing convention is retained; it is not end-to-end wall time. A fresh installation and packaged numerical examples have not been executed during this preparation.
