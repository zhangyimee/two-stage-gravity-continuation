"""Numerical operators extracted from the research implementation; see provenance."""
import math
import time
import numpy as np
from numba import cuda

Z32 = np.float32(0.)
O32 = np.float32(1.)
_geo_cache = {}

@cuda.jit(device=True, inline=True)
def _gravity_gz_prism(xi, yj, zk, Gc):
    """Unit-density vertical gravity at a prism corner (CUDA device function)."""
    r = math.sqrt(xi*xi + yj*yj + zk*zk)
    if r == Z32: return Z32
    l1 = math.log(yj+r) if (yj+r) != Z32 else Z32
    l2 = math.log(xi+r) if (xi+r) != Z32 else Z32
    at = math.atan(xi*yj/(zk*r)) if zk*r != Z32 else Z32
    return Gc * (xi*l1 + yj*l2 - zk*at)


def _to_device_cached(name, arr):
    key = id(arr)
    entry = _geo_cache.get(key)
    if entry is not None and entry[0] is arr:
        return entry[1]
    dev = cuda.to_device(np.ascontiguousarray(arr.astype(np.float32)))
    _geo_cache[key] = (arr, dev)
    return dev


def prism_gz_corner(xi, yj, zk, Gc):
    """Prism-corner vertical-gravity kernel used to construct the FFT operator."""
    r = np.sqrt(xi*xi + yj*yj + zk*zk)
    if r == 0: return 0.0
    l1 = np.log(yj + r) if (yj + r) != 0 else 0.0
    l2 = np.log(xi + r) if (xi + r) != 0 else 0.0
    at = np.arctan2(xi*yj, zk*r) if zk*r != 0 else 0.0
    return Gc * (xi*l1 + yj*l2 - zk*at)


def build_bttb_first_column_single(xo_grid, yo_grid, z_obs_flat, xs_grid, ys_grid,
                             z_src_top, z_src_bot, dx, dy, Gc):
    """Construct the first BTTB column for one aligned prism layer."""
    ny_src, nx_src = len(ys_grid), len(xs_grid)
    xo0, yo0, zo0 = xo_grid[0], yo_grid[0], z_obs_flat
    first_col_2d = np.zeros((ny_src, nx_src), dtype=np.float64)
    src_dx = xs_grid[1] - xs_grid[0] if len(xs_grid) > 1 else dx
    src_dy = ys_grid[1] - ys_grid[0] if len(ys_grid) > 1 else dy
    for iy in range(ny_src):
        for ix in range(nx_src):
            xs_center = xs_grid[ix]; ys_center = ys_grid[iy]
            xs0 = xs_center - src_dx/2; xs1 = xs_center + src_dx/2
            ys0 = ys_center - src_dy/2; ys1 = ys_center + src_dy/2
            s = 0.0
            s += prism_gz_corner(xs0-xo0, ys0-yo0, z_src_top-zo0, Gc)
            s -= prism_gz_corner(xs1-xo0, ys0-yo0, z_src_top-zo0, Gc)
            s -= prism_gz_corner(xs0-xo0, ys1-yo0, z_src_top-zo0, Gc)
            s += prism_gz_corner(xs1-xo0, ys1-yo0, z_src_top-zo0, Gc)
            s -= prism_gz_corner(xs0-xo0, ys0-yo0, z_src_bot-zo0, Gc)
            s += prism_gz_corner(xs1-xo0, ys0-yo0, z_src_bot-zo0, Gc)
            s += prism_gz_corner(xs0-xo0, ys1-yo0, z_src_bot-zo0, Gc)
            s -= prism_gz_corner(xs1-xo0, ys1-yo0, z_src_bot-zo0, Gc)
            first_col_2d[iy, ix] = s
    return first_col_2d


class FFTOperator:
    """BTTB-FFT forward and adjoint operators at a constant observation elevation."""
    def __init__(self, xo, yo, xs, ys, zs, Gc, grid_shape, dx, dy, z_mean):
        self.Gc = Gc; self.ny, self.nx = grid_shape
        self.dx = dx; self.dy = dy; self.z_mean = z_mean
        self.n_obs = len(xo); self.n_src = len(xs)
        self.xo_grid = np.unique(xo); self.yo_grid = np.unique(yo)

        xs_center = (xs[:, 0] + xs[:, 1]) / 2
        ys_center = (ys[:, 0] + ys[:, 1]) / 2
        zs_center = (zs[:, 0] + zs[:, 1]) / 2
        self.xs_grid_unique = np.unique(xs_center)
        self.ys_grid_unique = np.unique(ys_center)

        n_per_layer = len(self.xs_grid_unique) * len(self.ys_grid_unique)
        self.n_z = self.n_src // n_per_layer
        print(f"    [FFT] 多层BTTB: {self.ny}×{self.nx}×{self.n_z}层, 共{self.n_src}源")

        zs_top_unique = np.unique(zs[:, 0])
        zs_bot_unique = np.unique(zs[:, 1])

        z_layer_info = sorted(set(zip(zs_top_unique, zs_bot_unique)))
        self.z_layers = z_layer_info  # [(z_top, z_bot), ...]
        print(f"    [FFT] 等效源层: {self.z_layers}")

        Ny, Nx = 2*self.ny, 2*self.nx
        self.fft_first_list = []
        t0 = time.perf_counter()
        for k, (z_top, z_bot) in enumerate(self.z_layers):
            fc = build_bttb_first_column_single(
                self.xo_grid, self.yo_grid, z_mean,
                self.xs_grid_unique, self.ys_grid_unique,
                z_top, z_bot, dx, dy, Gc)

            embed = np.zeros((Ny, Nx), dtype=np.float64)
            embed[:self.ny, :self.nx] = fc
            embed[self.ny+1:, :self.nx] = fc[:0:-1, :]
            embed[:self.ny, self.nx+1:] = fc[:, :0:-1]
            embed[self.ny+1:, self.nx+1:] = fc[:0:-1, :0:-1]
            self.fft_first_list.append(np.fft.fft2(embed))
        print(f"    [FFT] {self.n_z}层BTTB构建完成: {time.perf_counter()-t0:.2f}s")

    def J_fwd(self, v):
        """BTTB-FFT forward and adjoint operators at a constant observation elevation."""
        v_3d = v.reshape(self.n_z, self.ny, self.nx)
        Ny, Nx = 2*self.ny, 2*self.nx
        result = np.zeros((self.ny, self.nx), dtype=np.float64)
        for k in range(self.n_z):
            embed_v = np.zeros((Ny, Nx), dtype=np.float64)
            embed_v[:self.ny, :self.nx] = v_3d[k]
            fft_v = np.fft.fft2(embed_v)
            fft_result = self.fft_first_list[k] * fft_v
            result += np.real(np.fft.ifft2(fft_result))[:self.ny, :self.nx]
        return result.flatten()

    def J_adj(self, u):
        """BTTB-FFT forward and adjoint operators at a constant observation elevation."""
        Ny, Nx = 2*self.ny, 2*self.nx
        embed_u = np.zeros((Ny, Nx), dtype=np.float64)
        embed_u[:self.ny, :self.nx] = u.reshape(self.ny, self.nx)
        fft_u = np.fft.fft2(embed_u)
        out = np.zeros((self.n_z, self.ny, self.nx), dtype=np.float64)
        for k in range(self.n_z):
            fft_result = self.fft_first_list[k] * fft_u
            out[k] = np.real(np.fft.ifft2(fft_result))[:self.ny, :self.nx]
        return out.flatten()


def build_laplacian_2d(nx, ny, n_z=1):
    """Build the unscaled five-point lateral Laplacian without row wraparound."""
    from scipy.sparse import diags, block_diag, eye

    N = nx * ny
    main_diag = -4 * np.ones(N)

    off_x = np.ones(N)
    off_x[nx-1::nx] = 0

    off_y = np.ones(N)

    L_single = diags([main_diag, off_x, off_x, off_y, off_y],
                     [0, 1, -1, nx, -nx], shape=(N, N), format='csr')

    if n_z > 1:
        L = block_diag([L_single] * n_z, format='csr')
    else:
        L = L_single
    return L


def read_grd(fname, eigval=1.70141e+38):
    """Read a Surfer ASCII DSAA grid; return geometry and a (ny, nx) array."""
    with open(fname, 'r') as f:
        f.readline()  # DSAA
        nx, ny = [int(x) for x in f.readline().split()]
        xmin, xmax = [float(x) for x in f.readline().split()]
        ymin, ymax = [float(x) for x in f.readline().split()]
        f.readline()  # zmin, zmax
    with open(fname, 'r') as f:
        for _ in range(5): f.readline()
        vals = []
        while len(vals) < nx*ny:
            vals.extend([float(x) for x in f.readline().split()])
    field = np.array(vals[:nx*ny]).reshape(ny, nx)
    field[field > eigval*0.9] = np.nan
    return nx, ny, xmin, xmax, ymin, ymax, field


def write_grd(fname, field, xmin, xmax, ymin, ymax):
    """Write a Surfer ASCII DSAA grid with the original scientific-format precision."""
    ny, nx = field.shape
    zmin = np.nanmin(field); zmax = np.nanmax(field)
    with open(fname, 'w') as f:
        f.write('DSAA\n')
        f.write(f'{nx} {ny}\n')
        f.write(f'{xmin} {xmax}\n')
        f.write(f'{ymin} {ymax}\n')
        f.write(f'{zmin} {zmax}\n')
        for j in range(ny):
            line = ' '.join([f'{v:.6e}' for v in field[j]])
            f.write(line + '\n')


def load_data(cfg):
    """Load aligned gravity and elevation grids; convert elevation to downward-positive z."""
    eigval = cfg['input'].get('eigval', 1.70141e+38)
    ds_factor = cfg['input'].get('downsample', 1)


    anomaly_file = cfg['input']['anomaly_file']
    nx, ny, xmin, xmax, ymin, ymax, anomaly, = read_grd(anomaly_file, eigval)[:7]
    print(f"  重力异常: {anomaly_file}")
    print(f"    原始网格: {nx}×{ny}, 区域: [{xmin:.0f},{xmax:.0f}]×[{ymin:.0f},{ymax:.0f}]m")
    print(f"    异常范围: [{np.nanmin(anomaly):.4f}, {np.nanmax(anomaly):.4f}] mGal")

    orig_nx, orig_ny = nx, ny


    smooth_sigma = cfg['input'].get('smooth_sigma', 0.0)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        anomaly = gaussian_filter(anomaly, sigma=smooth_sigma)
        print(f"  高斯预平滑: sigma={smooth_sigma}, 平滑后范围=[{np.nanmin(anomaly):.4f}, {np.nanmax(anomaly):.4f}] mGal")



    noise_sigma_cfg = cfg['inversion'].get('noise_sigma', None)
    if noise_sigma_cfg is not None:
        noise_sigma = float(noise_sigma_cfg)
        print(f"  噪声水平(显式指定): sigma={noise_sigma:.6e} mGal")
    else:
        noise_pct = cfg['inversion'].get('noise_pct', 0.0)
        if noise_pct > 0:
            noise_sigma = noise_pct * np.std(anomaly) / 100.0
            print(f"  噪声估计: noise_pct={noise_pct}%, sigma={noise_sigma:.6e} mGal")
        else:
            noise_sigma = 0.0


    surface_file = cfg['input']['surface_file']
    nx2, ny2, _, _, _, _, surface = read_grd(surface_file, eigval)
    print(f"  地形高程: {surface_file}")
    print(f"    高程范围: [{np.nanmin(surface):.1f}, {np.nanmax(surface):.1f}]m")
    assert (nx, ny) == (nx2, ny2), "重力异常与地形网格尺寸不一致!"


    if ds_factor > 1:
        idx_x = np.arange(0, nx, ds_factor)
        idx_y = np.arange(0, ny, ds_factor)
        anomaly = anomaly[idx_y][:, idx_x]
        surface = surface[idx_y][:, idx_x]
        nx, ny = len(idx_x), len(idx_y)
        print(f"  降采样(factor={ds_factor}): {nx}×{ny}={nx*ny}点")


    z_obs = -surface


    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    dx = x[1] - x[0] if nx > 1 else 200.0
    dy = y[1] - y[0] if ny > 1 else 200.0


    XX, YY = np.meshgrid(x, y)
    xo = XX.flatten(); yo = YY.flatten()
    zo = z_obs.flatten(); d_obs = anomaly.flatten()


    nan_mask = np.isnan(d_obs)
    if np.any(nan_mask):
        print(f"  警告: {np.sum(nan_mask)}个NaN值用0填充")
        d_obs[nan_mask] = 0.0

    data = {
        'nx': nx, 'ny': ny, 'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax, 'dx': dx, 'dy': dy,
        'x': x, 'y': y, 'xo': xo, 'yo': yo, 'zo': zo,
        'd_obs': d_obs, 'anomaly_2d': anomaly, 'surface_2d': surface,
        'z_obs_mean': np.mean(zo), 'z_obs_min': np.min(zo),
        'z_obs_max': np.max(zo),
        'noise_sigma': noise_sigma,
        'orig_nx': orig_nx, 'orig_ny': orig_ny,
    }
    return data


def setup_sources(data, cfg):
    """Construct aligned rectangular-prism sources in layer-major order."""
    nx, ny = data['nx'], data['ny']
    x, y = data['x'], data['y']
    dx, dy = data['dx'], data['dy']
    z_top = cfg['sources']['z_top']
    z_bot = cfg['sources']['z_bot']
    n_z = cfg['sources'].get('n_z', 1)

    if n_z > 1:
        dz = (z_bot - z_top) / n_z
        z_lo_list = z_top + np.arange(n_z) * dz
        z_hi_list = z_lo_list + dz
        z_centers = (z_lo_list + z_hi_list) / 2.0
    else:
        dz = 0.0
        z_lo_list = np.array([z_top])
        z_hi_list = np.array([z_bot])
        z_centers = np.array([z_top])
    n_src = nx * ny * n_z
    xs = np.zeros((n_src, 2), dtype=np.float64)
    ys = np.zeros((n_src, 2), dtype=np.float64)
    zs = np.zeros((n_src, 2), dtype=np.float64)
    idx = 0
    for k in range(n_z):
        z_lo = z_lo_list[k]
        z_hi = z_hi_list[k]
        for iy in range(ny):
            for ix in range(nx):
                xs[idx, 0] = x[ix] - dx/2; xs[idx, 1] = x[ix] + dx/2
                ys[idx, 0] = y[iy] - dy/2; ys[idx, 1] = y[iy] + dy/2
                zs[idx, 0] = z_lo; zs[idx, 1] = z_hi
                idx += 1
    data['xs'] = xs; data['ys'] = ys; data['zs'] = zs
    data['n_src'] = n_src; data['n_z'] = n_z
    print(f"  等效源: {nx}×{ny}×{n_z}={n_src}个, z=[{z_top:.0f},{z_bot:.0f}]m, {n_z}层")
    if n_z > 1:
        print(f"  各层中心深度: {[f'{z:.0f}' for z in z_centers]}m, 层厚={dz:.0f}m")
    print(f"  网格间距: dx={dx:.0f}m, dy={dy:.0f}m (BTTB结构)")
    return data

