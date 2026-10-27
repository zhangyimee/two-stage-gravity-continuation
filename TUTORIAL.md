# Manuscript example tutorial

## 1 Noise-free comparison

`python run_examples.py clean --output-name clean_example`

Uses 30,351 observations and the same number of sources. It writes two-stage and single-stage fits, continuation grids at four target planes, errors, models and histories. Compare result.json with reference_results/manuscript/single_layer_full.json, selected_case. Historical target RMSE values for two-stage are approximately 0.001144, 0.002050, 0.003707 and 0.006798 mGal for z=-800,-600,-400,-200 m. The historical single-stage iteration count is 221; stopping, floating-point results and timing may vary by platform. This case underlies manuscript Tables 2–3, Figs. 5–7 and the clean profiles in Fig. 10.

## 2 Noisy observations

`python run_examples.py noise --seed 100 --output-name noise_example`

Uses the stored seed100 grid and sigma=0.1 mGal. The historical discrepancy stop was 7 approximate iterations and 0 refinement iterations; a direct residual check is still performed. Outputs underlie Figs. 8–9 and noisy profiles in Fig. 10. Seeds101–104 are supplied stored alternatives; they are not additional final-manuscript single-layer results.

## 3 Preliminary source-depth selection

`python run_examples.py depth-scan --output-name depth_example`

Uses every second input node (101 x 76), centers 1375,2125,2875,3625 m and 150+40 iterations. Selection minimizes mean RMSE over four known synthetic target planes. This supports the preliminary choice described in Section 3.1; it is not an automatic source-depth rule for real data. Historical output is reference_results/preliminary/multilayer_depth.json; its filename is historical and does not imply a multilayer final example.

## 4 Optional Olympic Dam example

After documented field inputs are provided in data/field_local, run `python run_examples.py field --output-name field_example`. It requires od_residual.grd and od_terrain.grd, uses 300+20 iterations and continues to elevations60,40,20,0 m. The supplied processed residual field is the starting point; the package does not claim a verified raw-to-regional preprocessing reconstruction. No subsurface truth is available for target-error estimation. This case supports Section4.2 and Figs.14–15.

## Verification status

These are execution instructions, not evidence that the packaged examples have been run. Packaging checks cover syntax, grid structure, paths and file hashes only. Do not interpret archived outputs as a fresh benchmark. New runs write to outputs and do not alter archived results.
