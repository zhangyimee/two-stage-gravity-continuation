"""Eight-prism synthetic model used in Table 1 (not a geological inversion)."""
import numpy as np

def build_model_b():
    """
    Returns (prisms, region) for Model B.

    Prisms are defined with z positive DOWN (depth).
    Density contrast in kg/m³.
    """
    prisms = [
        # ============================================================
        # Intermediate depth stocks
        # ============================================================

        # I2: Stock A — northwest
        {'x1': 8e3, 'x2': 16e3, 'y1': 6e3, 'y2': 14e3,
         'z1': 5e3, 'z2': 8e3, 'density': 220.0,
         'label': 'I1: Stock A'},

        # I3: Stock B — southeast of center
        {'x1': -12e3, 'x2': -6e3, 'y1': -10e3, 'y2': -4e3,
         'z1': 3.5e3, 'z2': 6e3, 'density': 180.0,
         'label': 'I2: Stock B'},

        # I4: Stock C — small, east
        {'x1': 18e3, 'x2': 22e3, 'y1': -4e3, 'y2': 2e3,
         'z1': 2.5e3, 'z2': 4e3, 'density': 150.0,
         'label': 'I3: Stock C'},

        # ============================================================
        # Shallow dikes, sills, and plugs
        # ============================================================

        # I5: Dike swarm (E-W trending, long)
        {'x1': -18e3, 'x2': 25e3, 'y1': -2e3, 'y2': -0.5e3,
         'z1': 0.8e3, 'z2': 1.3e3, 'density': 120.0,
         'label': 'I4: Dike D1 (E-W)'},

        # I6: Cross-cutting dike (NE-SW trending)
        {'x1': -5e3, 'x2': 15e3, 'y1': -8e3, 'y2': -6.5e3,
         'z1': 0.6e3, 'z2': 1.0e3, 'density': 100.0,
         'label': 'I5: Dike D2 (NE)'},

        # I7: Sill — sub-horizontal, thin, shallow
        {'x1': -8e3, 'x2': 2e3, 'y1': 8e3, 'y2': 16e3,
         'z1': 0.4e3, 'z2': 0.7e3, 'density': 80.0,
         'label': 'I6: Sill S1'},

        # I8: Small plug — northwest
        {'x1': -22e3, 'x2': -19e3, 'y1': 10e3, 'y2': 13e3,
         'z1': 0.3e3, 'z2': 0.8e3, 'density': 250.0,
         'label': 'I7: Plug P1'},

        # I9: Small plug — south
        {'x1': 5e3, 'x2': 8e3, 'y1': -22e3, 'y2': -19e3,
         'z1': 0.5e3, 'z2': 1.0e3, 'density': 200.0,
         'label': 'I8: Plug P2'},

    ]

    # Region: 80 km (E-W) × 60 km (N-S)
    region = [-35e3, 45e3, -25e3, 35e3]

    return prisms, region

