"""
Wall shear stress (WSS) from 4D flow velocity and a lumen segmentation.

Method
------
For each wall point with inward unit normal n, velocity is sampled at a set
of points along the inward normal (default 3 points to 3 voxels), the
component tangential to the wall is fitted with a quadratic through zero
at the wall (no-slip), and the wall shear rate is the derivative of that
fit at the wall. WSS = mu * shear rate. This is the method of Stalder et
al. restricted to the wall-normal line, with the smoothed-spline fit
replaced by a quadratic, and the inward normals taken from the
segmentation surface rather than a 2D contour. Time-averaged WSS (TAWSS)
and the oscillatory shear index (OSI) follow Ku et al.

Wall points and normals come from the lumen mask: the marching-cubes
surface of the binary mask (or of a smoothed mask) gives vertices and
outward normals; sampling moves inward. Only numpy and scipy are used
(scikit-image's marching cubes if available, otherwise a gradient-of-mask
surface); VTK is optional for export.

Resolution caveat: 4D flow voxels (1.5 to 3 mm) are comparable to the
aortic boundary layer, so WSS is systematically underestimated
(Petersson et al.); the values are reproducible and useful for
comparisons but not absolute, and the sampling distance matters. The
analytic Poiseuille and Womersley tests in this package quantify the
bias for a given voxel size.

References
----------
    Stalder AF, Russe MF, Frydrychowicz A, Bock J, Hennig J, Markl M.
    Quantitative 2D and 3D phase contrast MRI: optimized analysis of blood
    flow and vessel wall parameters. Magn Reson Med 2008;60(5):1218-1231.
    PMID 18956416. doi:10.1002/mrm.21778.
    Potters WV, van Ooij P, Marquering H, vanBavel E, Nederveen AJ.
    Volumetric arterial wall shear stress calculation based on cine phase
    contrast MRI. J Magn Reson Imaging 2015;41(2):505-516. PMID 24436246.
    doi:10.1002/jmri.24560.
    Petersson S, Dyverfeldt P, Ebbers T. Assessment of the accuracy of MRI
    wall shear stress estimation using numerical simulations. J Magn Reson
    Imaging 2012;36(1):128-138. PMID 22336966. doi:10.1002/jmri.23610.
    Ku DN, Giddens DP, Zarins CK, Glagov S. Pulsatile flow and
    atherosclerosis in the human carotid bifurcation: positive correlation
    between plaque location and low oscillating shear stress.
    Arteriosclerosis 1985;5(3):293-302. PMID 3994585.
    doi:10.1161/01.atv.5.3.293.
    Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface
    construction algorithm. Computer Graphics (SIGGRAPH) 1987;21(4):163-169.
    doi:10.1145/37402.37422.

Conventions as elsewhere: velocity (nt, nk, nj, ni, 3) mm/s world
components, mask (nk, nj, ni) bool, voxel_to_world 4x4. WSS in Pa.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world_points, world_to_array_coordinates, voxel_spacing

BLOOD_VISCOSITY_PA_S = 3.5e-3


# ---------------------------------------------------------------------------
# Wall surface from the mask
# ---------------------------------------------------------------------------

def wall_points(mask, voxel_to_world, smooth_sigma_vox=1.0, step=1):
    """
    Wall points (n, 3) in world mm and inward unit normals (n, 3).

    Uses scikit-image marching cubes on a Gaussian-smoothed mask at the 0.5
    level when scikit-image is available (every `step`-th vertex); otherwise
    boundary voxels of the mask with normals from the smoothed-mask gradient.
    """
    m = np.asarray(mask, float)
    ms = ndimage.gaussian_filter(m, smooth_sigma_vox) if smooth_sigma_vox > 0 else m
    try:
        from skimage import measure
        verts, faces, normals, _ = measure.marching_cubes(ms, level=0.5)   # verts in (k, j, i)
        verts, normals = verts[::step], normals[::step]
        ijk = verts[:, ::-1]
        # marching_cubes normals point toward decreasing values (outward from the lumen)
        g = np.stack([ndimage.map_coordinates(np.gradient(ms, axis=a), verts.T, order=1) for a in range(3)], -1)
        inward_kji = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)      # gradient points into the lumen
    except ImportError:
        # Fallback without scikit-image: boundary voxels of the mask, moved
        # outward along the inward normal to the 0.5 level of the smoothed
        # mask (a boundary voxel centre lies about half a voxel inside the
        # wall; leaving it there overestimates WSS by roughly 50%).
        boundary = m.astype(bool) & ~ndimage.binary_erosion(m.astype(bool))
        kji = np.argwhere(boundary).astype(float)[::step]
        g = np.stack([ndimage.map_coordinates(np.gradient(ms, axis=a), kji.T, order=1) for a in range(3)], -1)
        inward_kji = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        # one-step Newton move to the 0.5 isolevel: delta = (ms - 0.5) / |grad ms|
        val = ndimage.map_coordinates(ms, kji.T, order=1)
        gn = np.linalg.norm(g, axis=-1) + 1e-12
        kji = kji - ((val - 0.5) / gn)[:, None] * inward_kji
        ijk = kji[:, ::-1]
    pts = voxel_to_world_points(voxel_to_world, ijk)
    # map the voxel-space direction to world: direction (di, dj, dk) -> M @ d
    M = np.asarray(voxel_to_world, float)[:3, :3]
    inward_world = inward_kji[:, ::-1] @ M.T
    inward_world /= np.linalg.norm(inward_world, axis=-1, keepdims=True) + 1e-12
    return pts, inward_world


# ---------------------------------------------------------------------------
# WSS estimation
# ---------------------------------------------------------------------------

def _tangential(v, n):
    return v - (v * n).sum(-1, keepdims=True) * n


def wall_shear_stress(velocity_t, mask, voxel_to_world, points=None, normals=None,
                      sample_distances_mm=None, viscosity=BLOOD_VISCOSITY_PA_S, fit="quadratic"):
    """
    WSS vector (nt, n, 3) in Pa at each wall point, tangential to the wall.

    sample_distances_mm : inward sampling distances from the wall; default
                          (1, 2, 3) x smallest voxel size.
    fit : "quadratic" (v_t(d) = a d + b d^2, shear = a) or "linear"
          (v_t(d) = a d, least squares through zero).
    Returns dict with wss (nt, n, 3) Pa, wss_magnitude (nt, n), points,
    normals, tawss (n), osi (n).
    """
    v_t = np.asarray(velocity_t, float)
    if v_t.ndim == 4:
        v_t = v_t[None]
    if points is None:
        points, normals = wall_points(mask, voxel_to_world)
    points = np.asarray(points, float); normals = np.asarray(normals, float)
    h = float(voxel_spacing(voxel_to_world).min())
    d = np.asarray((1.0, 2.0, 3.0) if sample_distances_mm is None else sample_distances_mm, float) * (h if sample_distances_mm is None else 1.0)
    nt, n = v_t.shape[0], points.shape[0]
    mf = np.asarray(mask, float)

    # sample points: (n, nd, 3)
    samp = points[:, None, :] + d[None, :, None] * normals[:, None, :]
    coords = world_to_array_coordinates(voxel_to_world, samp.reshape(-1, 3))          # (3, n*nd)
    inside = ndimage.map_coordinates(mf, coords, order=1, mode="constant", cval=0.0).reshape(n, len(d))

    # design matrix for the fit through zero
    if fit == "quadratic":
        A = np.stack([d, d ** 2], 1)
    elif fit == "linear":
        A = d[:, None]
    else:
        raise ValueError(fit)
    wss = np.zeros((nt, n, 3))
    for t in range(nt):
        vs = np.stack([ndimage.map_coordinates(v_t[t, ..., c] * mf, coords, order=1, mode="constant", cval=0.0)
                       for c in range(3)], -1).reshape(n, len(d), 3)
        # normalized interpolation so partial-volume at the wall does not pull velocities toward zero
        with np.errstate(invalid="ignore", divide="ignore"):
            vs = np.where(inside[..., None] > 0.5, vs / inside[..., None], 0.0)
        vt = _tangential(vs, normals[:, None, :])                                    # (n, nd, 3)
        # weighted LS with weights = inside (drops samples that left the lumen)
        w = (inside > 0.5).astype(float)
        for i in range(n):
            wi = w[i]
            if wi.sum() < A.shape[1]:
                continue
            Aw = A * wi[:, None]
            coef, *_ = np.linalg.lstsq(Aw, vt[i] * wi[:, None], rcond=None)        # (p, 3)
            shear_rate = coef[0] * 1e-3 / 1e-3                                       # (mm/s)/mm = 1/s
            wss[t, i] = viscosity * shear_rate
    mag = np.linalg.norm(wss, axis=-1)
    mean_vec = wss.mean(0)
    tawss = mag.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = np.where(tawss > 0, 0.5 * (1.0 - np.linalg.norm(mean_vec, axis=-1) / tawss), 0.0)
    return {"wss": wss, "wss_magnitude": mag, "points": points, "normals": normals,
            "tawss": tawss, "osi": osi, "sample_distances_mm": d}


def to_vtk_surface(points, normals, scalars: dict):
    """vtkPolyData of wall points (vertices) with point-data arrays from
    `scalars` {name: (n,) or (n, 3)}. For a triangulated surface use the
    faces from skimage.measure.marching_cubes directly."""
    import vtk
    from vtk.util import numpy_support as ns
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    pd.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for i in range(points.shape[0]):
        verts.InsertNextCell(1, [i])
    pd.SetVerts(verts)
    nrm = ns.numpy_to_vtk(np.ascontiguousarray(normals), deep=True); nrm.SetName("inward_normal")
    pd.GetPointData().AddArray(nrm)
    for name, arr in scalars.items():
        a = ns.numpy_to_vtk(np.ascontiguousarray(np.asarray(arr, float)), deep=True); a.SetName(name)
        pd.GetPointData().AddArray(a)
    return pd
