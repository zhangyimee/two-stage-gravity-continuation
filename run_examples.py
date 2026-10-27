"""Run only the cases used by the final manuscript.

Examples (from the package directory):
    python run_examples.py clean
    python run_examples.py noise --seed 100
    python run_examples.py field
    python run_examples.py depth-scan
    python run_examples.py all

New output goes to outputs/<timestamp>/. No reference data are overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from twostage import solver as E
from numba import cuda


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def environment():
    packages = {}
    for name in ("numpy", "scipy", "numba", "numba-cuda", "llvmlite"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": packages, "cuda_device": str(cuda.get_current_device().name),
            "timing_note": "Unmodified original CGLS internal timer; see README."}


def problem(anomaly=None, surface=None, downsample=1, top=1000.0, bottom=1750.0):
    cfg = E.base_cfg(anomaly=anomaly, surface=surface, downsample=downsample,
                     n_z=1, z_top=top, z_bot=bottom)
    data, fft, direct, adjoint = E.load_problem(cfg)
    E.add_downsample_metadata(data, cfg)
    return cfg, data, fft, direct, adjoint


def grid(out, name, values, data):
    E.R.write_grd(str(out / name), np.asarray(values).reshape(data["ny"], data["nx"]),
                  data["xmin"], data["xmax"], data["ymin"], data["ymax"])


def two_stage(data, fft, direct, adjoint, stop=None, record1=10, record2=2):
    zero = np.zeros(data["n_src"], dtype=np.float64)
    m1, h1 = E.cgls(fft.J_fwd, fft.J_adj, data["d_obs"], zero, 300,
                    stop_rms=stop, tau=1.05 if stop is not None else 1.0,
                    record_every=record1)
    m2, h2 = E.cgls(direct, adjoint, data["d_obs"], m1, 20,
                    stop_rms=stop, tau=1.05 if stop is not None else 1.0,
                    record_every=record2)
    return m1, m2, h1, h2


def targets(out, model, data, prefix):
    for label in ("z800", "z600", "z400", "z200"):
        z = E.R.read_grd(str(E.DATA / (label + ".grd")))[6].ravel()
        if data["input_downsample"] > 1:
            z = z.reshape(151, 201)[::data["input_downsample"],
                                   ::data["input_downsample"]].ravel()
        pred = E.gpu_fwd_mixed(data["xo"], data["yo"], z,
                              data["xs"], data["ys"], data["zs"], model, E.G_MGAL)
        truth = E.R.read_grd(str(E.DATA / ("true_continuation_" + label + ".grd")))[6]
        truth = truth[::data["input_downsample"], ::data["input_downsample"]]
        grid(out, prefix + "_" + label + ".grd", pred, data)
        grid(out, prefix + "_error_" + label + ".grd", pred-truth.ravel(), data)


def clean(out):
    cfg, data, fft, direct, adjoint = problem()
    m1, m2, h1, h2 = two_stage(data, fft, direct, adjoint)
    single, hs = E.cgls(direct, adjoint, data["d_obs"],
                        np.zeros(data["n_src"], dtype=np.float64), 600,
                        stop_rms=h2["rms"], tau=1.0, record_every=5)
    result = {"config": cfg, "n_obs": int(data["d_obs"].size),
              "n_src": int(data["n_src"]), "phase1": h1, "phase2": h2,
              "single_stage": hs,
              "two_stage_total_s": h1["elapsed_s"] + h2["elapsed_s"],
              "target_metrics": E.plane_metrics(m2, data),
              "single_stage_target_metrics": E.plane_metrics(single, data),
              "true_surface_rms": float(np.sqrt(np.mean((direct(m2)-data["d_obs"])**2)))}
    result["equal_misfit_speedup"] = hs["elapsed_s"] / result["two_stage_total_s"]
    for label, model in (("two_stage", m2), ("single_stage", single)):
        fit = direct(model)
        grid(out, label + "_fit.grd", fit, data)
        grid(out, label + "_residual.grd", data["d_obs"]-fit, data)
        targets(out, model, data, label)
    np.savez(out / "models.npz", stage1=m1, two_stage=m2, single_stage=single)
    save_json(out / "result.json", result)


def noise(out, seed):
    cfg, data, fft, direct, adjoint = problem(E.DATA / ("obs_noise_seed%d.grd" % seed))
    m1, m2, h1, h2 = two_stage(data, fft, direct, adjoint, stop=0.1, record1=1, record2=1)
    fit = direct(m2)
    result = {"config": cfg, "seed": seed, "sigma_mgal": 0.1, "tau": 1.05,
              "phase1": h1, "phase2": h2, "target_metrics": E.plane_metrics(m2, data),
              "true_surface_rms": float(np.sqrt(np.mean((data["d_obs"]-fit)**2)))}
    grid(out, "observed.grd", data["d_obs"], data)
    grid(out, "two_stage_fit.grd", fit, data)
    grid(out, "two_stage_residual.grd", data["d_obs"]-fit, data)
    targets(out, m2, data, "two_stage")
    np.savez(out / "models.npz", stage1=m1, two_stage=m2)
    save_json(out / "result.json", result)


def field(out):
    folder = ROOT / "data" / "field_local"
    cfg, data, fft, direct, adjoint = problem(folder / "od_residual.grd",
                                             folder / "od_terrain.grd")
    m1, m2, h1, h2 = two_stage(data, fft, direct, adjoint, record1=25, record2=5)
    fit = direct(m2)
    grid(out, "fit.grd", fit, data)
    grid(out, "residual.grd", data["d_obs"]-fit, data)
    levels = []
    for elev in (60, 40, 20, 0):
        z = np.full(data["d_obs"].size, -float(elev))
        values = E.gpu_fwd_mixed(data["xo"], data["yo"], z, data["xs"],
                                data["ys"], data["zs"], m2, E.G_MGAL)
        grid(out, "elev_%02dm.grd" % elev, values, data)
        iy, ix = np.unravel_index(np.argmax(values), (data["ny"], data["nx"]))
        levels.append({"elevation_m": elev, "min_mgal": float(values.min()),
                       "max_mgal": float(values.max()),
                       "peak_x_m": float(data["x"][ix]), "peak_y_m": float(data["y"][iy])})
    save_json(out / "result.json",
              {"config": cfg, "phase1": h1, "phase2": h2,
               "n_obs": int(data["d_obs"].size), "n_src": int(data["n_src"]),
               "true_surface_fit_rms": float(np.sqrt(np.mean((data["d_obs"]-fit)**2))),
               "fit_mae": float(np.mean(np.abs(data["d_obs"]-fit))),
               "fit_max_abs": float(np.max(np.abs(data["d_obs"]-fit))), "levels": levels,
               "interpretation_limit": "Feasibility and internal consistency; no subsurface truth."})
    np.savez(out / "models.npz", stage1=m1, two_stage=m2)


def depth_scan(out):
    rows = []
    for top, bottom in ((1000., 1750.), (1750., 2500.), (2500., 3250.), (3250., 4000.)):
        cfg, data, fft, direct, adjoint = problem(downsample=2, top=top, bottom=bottom)
        zero = np.zeros(data["n_src"], dtype=np.float64)
        m1, h1 = E.cgls(fft.J_fwd, fft.J_adj, data["d_obs"], zero, 150, record_every=150)
        m2, h2 = E.cgls(direct, adjoint, data["d_obs"], m1, 40, record_every=40)
        rows.append({"config": cfg, "source_center_m": (top+bottom)/2,
                     "phase1": h1, "phase2": h2, "target_metrics": E.plane_metrics(m2, data)})
    selected = min(rows, key=lambda r: np.mean([v["rmse"] for v in r["target_metrics"].values()]))
    save_json(out / "result.json", {"grid": "101 x 76", "rows": rows,
               "selection_rule": "minimum mean RMSE over four prespecified target planes",
               "selected_center_m": selected["source_center_m"],
               "note": "Preliminary selection uses synthetic target truth, not a field-data rule."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("clean", "noise", "field", "depth-scan", "all"))
    parser.add_argument("--seed", type=int, choices=range(100, 105), default=100)
    parser.add_argument("--output-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        parser.error("--output-name must be a single new folder name")
    if args.case in ("field", "all"):
        required = ("od_residual.grd", "od_terrain.grd")
        missing = [name for name in required if not (ROOT / "data" / "field_local" / name).is_file()]
        if missing:
            parser.error("Field inputs are not included in the initial public code package. "
                         "See DATA_AVAILABILITY.md. Synthetic cases clean, noise and depth-scan are included.")
    out = ROOT / "outputs" / args.output_name
    if out.exists():
        parser.error("Output folder already exists; choose another --output-name")
    if not cuda.is_available():
        parser.error("A working NVIDIA CUDA device and Numba CUDA backend are required.")
    out.mkdir(parents=True)
    save_json(out / "environment.json", environment())
    cases = ("clean", "noise", "field") if args.case == "all" else (args.case,)
    for case in cases:
        target = out / case
        target.mkdir()
        if case == "noise":
            noise(target, args.seed)
        else:
            {"clean": clean, "field": field, "depth-scan": depth_scan}[case](target)
        print("Completed:", case, "->", target, flush=True)


if __name__ == "__main__":
    main()

