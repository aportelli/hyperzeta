import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap


def make_icosphere(subdivisions=4, radius=1.0) -> pv.PolyData:
    """
    Make an icosphere PyVista mesh.
    """
    mesh = pv.Icosahedron()
    mesh = mesh.subdivide(subdivisions, subfilter="loop")
    assert isinstance(mesh, pv.PolyData)
    pts = mesh.points.copy()
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = radius * pts / norms
    mesh.points = pts

    return mesh


def make_rwb_cmap(min: float, max: float) -> LinearSegmentedColormap:
    """
    Make a red-white-blue colormap from min/max values
    """
    if min < 0 < max:
        zero_pos = -min / (max - min)
        cmap = LinearSegmentedColormap.from_list(
            "red_white_blue_zero",
            [(0.0, "red"), (zero_pos, "white"), (1.0, "blue")],
        )
    elif min >= 0:
        cmap = LinearSegmentedColormap.from_list(
            "white_blue",
            [(0.0, "white"), (1.0, "blue")],
        )
    else:
        cmap = LinearSegmentedColormap.from_list(
            "red_white",
            [(0.0, "red"), (1.0, "white")],
        )

    return cmap
