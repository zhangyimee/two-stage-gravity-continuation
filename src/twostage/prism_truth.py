"""Independent CPU analytic prism forward model for synthetic truth."""
import numpy as np
from numba import njit, prange
G = 6.67430e-11

@njit(parallel=True, fastmath=True)
def _prism_gz_kernel(x_obs, y_obs, z_obs, 
                      x1, x2, y1, y2, z1, z2, density):
    """Analytical prism-gravity kernel; coordinates are in metres."""
    n = len(x_obs)
    gz = np.zeros(n)
    

    # 1 mGal = 1e-5 m/s^2
    # gz(mGal) = G * rho(kg/m^3) * f(m^3) * 1e5
    rho_factor = G * density * 1e5
    
    for i in prange(n):
        xo = x_obs[i]
        yo = y_obs[i]
        zo = z_obs[i]
        
        total = 0.0
        for ix in range(2):
            if ix == 0:
                dx = x1 - xo
            else:
                dx = x2 - xo
            
            for iy in range(2):
                if iy == 0:
                    dy = y1 - yo
                else:
                    dy = y2 - yo
                
                for iz in range(2):
                    if iz == 0:
                        dz = z1 - zo
                    else:
                        dz = z2 - zo
                    

                    sign = (-1) ** (ix + iy + iz)
                    
                    # r = sqrt(dx^2 + dy^2 + dz^2)
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    
                    if r < 1e-10:
                        continue
                    

                    # z*ln(x+r) + x*ln(y+r) - z*atan(x*y/(z*r))

                    

                    term1 = dx * np.log(dy + r) if (dy + r) > 1e-30 else 0.0
                    

                    term2 = dy * np.log(dx + r) if (dx + r) > 1e-30 else 0.0
                    

                    arg = dx * dy / (dz * r) if abs(dz * r) > 1e-30 else 0.0
                    term3 = -dz * np.arctan(arg)
                    
                    total += sign * (term1 + term2 + term3)
        
        gz[i] = rho_factor * total
    
    return gz


@njit(parallel=True, fastmath=True)
def _prism_gx_kernel(x_obs, y_obs, z_obs,
                      x1, x2, y1, y2, z1, z2, density):
    """Analytical prism-gravity kernel; coordinates are in metres."""
    n = len(x_obs)
    gx = np.zeros(n)
    rho_factor = G * density * 1e5
    
    for i in prange(n):
        xo = x_obs[i]
        yo = y_obs[i]
        zo = z_obs[i]
        
        total = 0.0
        for ix in range(2):
            dx = (x1 - xo) if ix == 0 else (x2 - xo)
            for iy in range(2):
                dy = (y1 - yo) if iy == 0 else (y2 - yo)
                for iz in range(2):
                    dz = (z1 - zo) if iz == 0 else (z2 - zo)
                    sign = (-1) ** (ix + iy + iz)
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < 1e-10:
                        continue
                    
                    # gx: -dx*ln(dz+r) - dy*atan(dx*dy/(dz*r)) + dz*ln(dy+r)... 

                    # gx = sum mu_s * [ -z*ln(y+r) - y*ln(z+r) + x*atan(y*z/(x*r)) ]

                    # gx = sum mu_s * [ z*ln(y+r) + y*ln(z+r) - x*atan(y*z/(x*r)) ] * (-1)

                    

                    # gx = G*rho * sum mu_s * [ 
                    #   -y*ln(z+r) - z*ln(y+r) + x*atan(y*z/(x*r)) ]

                    

                    # gx = G*rho * sum mu_s * [ 
                    #   z*ln(y+r) + y*ln(z+r) - x*atan(y*z/(x*r)) ]

                    
                    term1 = dz * np.log(dy + r) if (dy + r) > 1e-30 else 0.0
                    term2 = dy * np.log(dz + r) if (dz + r) > 1e-30 else 0.0
                    arg = dy * dz / (dx * r) if abs(dx * r) > 1e-30 else 0.0
                    term3 = -dx * np.arctan(arg)
                    
                    total += sign * (term1 + term2 + term3)
        
        gx[i] = rho_factor * total
    
    return gx


@njit(parallel=True, fastmath=True)
def _prism_gy_kernel(x_obs, y_obs, z_obs,
                      x1, x2, y1, y2, z1, z2, density):
    """Analytical prism-gravity kernel; coordinates are in metres."""
    n = len(x_obs)
    gy = np.zeros(n)
    rho_factor = G * density * 1e5
    
    for i in prange(n):
        xo = x_obs[i]
        yo = y_obs[i]
        zo = z_obs[i]
        
        total = 0.0
        for ix in range(2):
            dx = (x1 - xo) if ix == 0 else (x2 - xo)
            for iy in range(2):
                dy = (y1 - yo) if iy == 0 else (y2 - yo)
                for iz in range(2):
                    dz = (z1 - zo) if iz == 0 else (z2 - zo)
                    sign = (-1) ** (ix + iy + iz)
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < 1e-10:
                        continue
                    

                    # gy = G*rho * sum mu_s * [
                    #   z*ln(x+r) + x*ln(z+r) - y*atan(x*z/(y*r)) ]
                    term1 = dz * np.log(dx + r) if (dx + r) > 1e-30 else 0.0
                    term2 = dx * np.log(dz + r) if (dz + r) > 1e-30 else 0.0
                    arg = dx * dz / (dy * r) if abs(dy * r) > 1e-30 else 0.0
                    term3 = -dy * np.arctan(arg)
                    
                    total += sign * (term1 + term2 + term3)
        
        gy[i] = rho_factor * total
    
    return gy


def prism_gravity_all(x_obs, y_obs, z_obs, prisms, surface_z=None,
                       components='gz'):
    """Evaluate prism gravity using downward-positive coordinates in metres."""
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    z_obs = np.asarray(z_obs, dtype=np.float64)
    
    if isinstance(components, str):
        components = [components]
        single = True
    else:
        single = False
    
    results = {}
    for comp in components:
        results[comp] = np.zeros(len(x_obs))
    
    for prism in prisms:
        x1, x2 = prism['x1'], prism['x2']
        y1, y2 = prism['y1'], prism['y2']
        z1, z2 = prism['z1'], prism['z2']
        rho = prism['density']
        
        for comp in components:
            if comp == 'gz':
                results[comp] += _prism_gz_kernel(
                    x_obs, y_obs, z_obs, x1, x2, y1, y2, z1, z2, rho)
            elif comp == 'gx':
                results[comp] += _prism_gx_kernel(
                    x_obs, y_obs, z_obs, x1, x2, y1, y2, z1, z2, rho)
            elif comp == 'gy':
                results[comp] += _prism_gy_kernel(
                    x_obs, y_obs, z_obs, x1, x2, y1, y2, z1, z2, rho)
    
    if single:
        return results[components[0]]
    return results

