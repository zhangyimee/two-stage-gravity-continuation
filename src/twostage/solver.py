"""Mixed-precision direct operators and continuous CGLS from experiments_v2.py.
Only the default source geometry and data path are adapted to the final manuscript.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
from numba import cuda, float64
from . import operators as R

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "data" / "synthetic"
G_MGAL = 6.6743e-11 * 1e5

@cuda.jit
def _fwd_kernel_mixed(xo, yo, zo, xs, ys, zs, v, out, gc):
    """FP32 geometry/kernel evaluation with FP64 weighted accumulation."""
    i = cuda.grid(1)
    if i >= xo.size:
        return
    acc = 0.0
    x0, y0, z0 = xo[i], yo[i], zo[i]
    for j in range(v.size):
        s = 0.0
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    sign = 1.0 if (ii + jj + kk) % 2 == 0 else -1.0
                    s += sign * R._gravity_gz_prism(
                        xs[j, ii] - x0, ys[j, jj] - y0, zs[j, kk] - z0, gc)
        acc += float64(s) * v[j]
    out[i] = acc


@cuda.jit
def _adj_kernel_mixed(xo, yo, zo, xs, ys, zs, u, out, gc):
    """FP32 geometry/kernel evaluation with FP64 shared-memory reduction."""
    j = cuda.blockIdx.x
    if j >= xs.shape[0]:
        return
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    sm = cuda.shared.array(256, dtype=float64)
    partial = 0.0
    for i in range(tid, u.size, block_size):
        x0, y0, z0 = xo[i], yo[i], zo[i]
        s = 0.0
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    sign = 1.0 if (ii + jj + kk) % 2 == 0 else -1.0
                    s += sign * R._gravity_gz_prism(
                        xs[j, ii] - x0, ys[j, jj] - y0, zs[j, kk] - z0, gc)
        partial += float64(s) * u[i]
    sm[tid] = partial
    cuda.syncthreads()
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            sm[tid] += sm[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        out[j] = sm[0]


def gpu_fwd_mixed(xo, yo, zo, xs, ys, zs, v, gc):
    threads = 256
    blocks = (xo.size + threads - 1) // threads
    dxo = R._to_device_cached("xo", xo)
    dyo = R._to_device_cached("yo", yo)
    dzo = R._to_device_cached("zo", zo)
    dxs = R._to_device_cached("xs", xs)
    dys = R._to_device_cached("ys", ys)
    dzs = R._to_device_cached("zs", zs)
    dv = cuda.to_device(np.ascontiguousarray(v, dtype=np.float64))
    dout = cuda.device_array(xo.size, dtype=np.float64)
    _fwd_kernel_mixed[blocks, threads](dxo, dyo, dzo, dxs, dys, dzs, dv, dout, np.float32(gc))
    cuda.synchronize()
    return dout.copy_to_host()


def gpu_adj_mixed(xo, yo, zo, xs, ys, zs, u, gc):
    threads = 256
    dxo = R._to_device_cached("xo", xo)
    dyo = R._to_device_cached("yo", yo)
    dzo = R._to_device_cached("zo", zo)
    dxs = R._to_device_cached("xs", xs)
    dys = R._to_device_cached("ys", ys)
    dzs = R._to_device_cached("zs", zs)
    du = cuda.to_device(np.ascontiguousarray(u, dtype=np.float64))
    dout = cuda.device_array(xs.shape[0], dtype=np.float64)
    _adj_kernel_mixed[xs.shape[0], threads](dxo, dyo, dzo, dxs, dys, dzs, du, dout, np.float32(gc))
    cuda.synchronize()
    return dout.copy_to_host()


def base_cfg(anomaly: Path | None = None, surface: Path | None = None,
             downsample: int = 1, n_z: int = 1,
             z_top: float = 1000.0, z_bot: float = 1750.0) -> dict:
    return {
        "input": {
            "anomaly_file": str(anomaly or DATA / "observed_gravity_clean.grd"),
            "surface_file": str(surface or DATA / "terrain.grd"),
            "eigval": 1.70141e38,
            "downsample": downsample,
            "smooth_sigma": 0.0,
        },
        "physics": {"G": 6.6743e-11},
        "sources": {"z_top": float(z_top), "z_bot": float(z_bot), "n_z": int(n_z)},
        "inversion": {
            "alpha": 0.0, "alpha_L": 0.0,
            "noise_pct": 0.0, "noise_sigma": 0.0,
        },
    }


def load_problem(cfg: dict) -> tuple[dict, object, object, object]:
    data = R.setup_sources(R.load_data(cfg), cfg)
    xo, yo, zo = data["xo"], data["yo"], data["zo"]
    xs, ys, zs = data["xs"], data["ys"], data["zs"]

    def gpu_fwd(v):
        return gpu_fwd_mixed(xo, yo, zo, xs, ys, zs, v, G_MGAL)

    def gpu_adj(u):
        return gpu_adj_mixed(xo, yo, zo, xs, ys, zs, u, G_MGAL)

    fft = R.FFTOperator(
        xo, yo, xs, ys, zs, G_MGAL,
        grid_shape=(data["ny"], data["nx"]),
        dx=data["dx"], dy=data["dy"], z_mean=data["z_obs_mean"],
    )
    return data, fft, gpu_fwd, gpu_adj


def cgls(J, JT, d: np.ndarray, x0: np.ndarray, max_iter: int,
         alpha: float = 0.0, alpha_L: float = 0.0, L=None,
         stop_rms: float | None = None, tau: float = 1.0,
         record_every: int = 1) -> tuple[np.ndarray, dict]:
    """Continuous CGLS for the augmented regularized system.

    The observation residual is checked before the first iteration, which
    permits zero-iteration refinement if the transferred FFT solution already
    satisfies the discrepancy principle.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    d = np.asarray(d, dtype=np.float64)
    sa = math.sqrt(alpha) if alpha > 0 else 0.0
    sl = math.sqrt(alpha_L) if alpha_L > 0 else 0.0

    r_obs = d - J(x)
    r_x = -sa * x if alpha > 0 else None
    r_l = -sl * (L @ x) if alpha_L > 0 else None

    def adjoint_residual():
        s = JT(r_obs)
        if alpha > 0:
            s = s + sa * r_x
        if alpha_L > 0:
            s = s + sl * (L.T @ r_l)
        return s

    n_obs = d.size
    rms0 = float(np.linalg.norm(r_obs) / math.sqrt(n_obs))
    history = [{"iter": 0, "rms": rms0, "elapsed_s": 0.0}]
    if stop_rms is not None and rms0 <= tau * stop_rms:
        return x, {"iterations": 0, "rms": rms0, "elapsed_s": 0.0,
                   "stop_reason": "discrepancy_at_initialization", "history": history}

    s = adjoint_residual()
    p = s.copy()
    gamma = float(np.dot(s, s))
    start = time.perf_counter()
    reason = "maximum_iterations"
    completed = 0
    for k in range(1, max_iter + 1):
        q_obs = J(p)
        denom = float(np.dot(q_obs, q_obs))
        q_x = None
        q_l = None
        if alpha > 0:
            q_x = sa * p
            denom += float(np.dot(q_x, q_x))
        if alpha_L > 0:
            q_l = sl * (L @ p)
            denom += float(np.dot(q_l, q_l))
        if denom <= 1e-30 or gamma <= 1e-30:
            reason = "numerical_stagnation"
            break
        step = gamma / denom
        x += step * p
        r_obs -= step * q_obs
        if alpha > 0:
            r_x -= step * q_x
        if alpha_L > 0:
            r_l -= step * q_l
        s_new = adjoint_residual()
        gamma_new = float(np.dot(s_new, s_new))
        beta = gamma_new / gamma
        p = s_new + beta * p
        s = s_new
        gamma = gamma_new
        completed = k
        rms = float(np.linalg.norm(r_obs) / math.sqrt(n_obs))
        if k % record_every == 0 or k == max_iter:
            history.append({"iter": k, "rms": rms,
                            "elapsed_s": time.perf_counter() - start})
        if not np.isfinite(rms):
            reason = "non_finite"
            break
        if stop_rms is not None and rms <= tau * stop_rms:
            reason = "discrepancy"
            break
    elapsed = time.perf_counter() - start
    rms = float(np.linalg.norm(r_obs) / math.sqrt(n_obs))
    if not history or history[-1]["iter"] != completed:
        history.append({"iter": completed, "rms": rms, "elapsed_s": elapsed})
    return x, {"iterations": completed, "rms": rms, "elapsed_s": elapsed,
               "stop_reason": reason, "history": history}


def plane_metrics(model: np.ndarray, data: dict, labels=("z800", "z600", "z400", "z200")) -> dict:
    metrics = {}
    for label in labels:
        _, _, _, _, _, _, zfield = R.read_grd(str(DATA / f"{label}.grd"))
        z = zfield.ravel()
        pred = gpu_fwd_mixed(
            data["xo"], data["yo"], z,
            data["xs"], data["ys"], data["zs"], model, G_MGAL,
        ).reshape(data["ny"], data["nx"])
        _, _, _, _, _, _, truth = R.read_grd(str(DATA / f"true_continuation_{label}.grd"))
        if truth.shape != pred.shape:
            truth = truth[::data["input_downsample"], ::data["input_downsample"]]
        diff = pred - truth
        metrics[label] = {
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "corr": float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
        }
    return metrics


def add_downsample_metadata(data: dict, cfg: dict) -> None:
    data["input_downsample"] = int(cfg["input"].get("downsample", 1))

