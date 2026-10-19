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

Wall points and normals come from the lumen mask: the 0.5 isosurface of
the binary mask, extracted with VTK's flying edges (the same marching-cubes
surface 3D Slicer builds for a segmentation), gives the vertices; the
normals come from the gradient of a smoothed mask; sampling moves inward.
Without VTK the boundary voxels of the mask, moved onto the same surface,
are used instead (numpy and scipy only).

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
    Schroeder W, Maynard R, Geveci B. Flying edges: a high-performance
    scalable isocontouring algorithm. IEEE LDAV 2015:33-40.
    doi:10.1109/LDAV.2015.7348069.

Conventions as elsewhere: velocity (nt, nk, nj, ni, 3) mm/s world
components, mask (nk, nj, ni) bool, voxel_to_world 4x4. WSS in Pa.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import array_axis_spacing, voxel_to_world_points, world_to_array_coordinates, voxel_spacing

BLOOD_VISCOSITY_PA_S = 3.5e-3


# ---------------------------------------------------------------------------
# Wall surface from the mask
# ---------------------------------------------------------------------------

def _vtk_isosurface(m, level=0.5):
    """
    Vertices (n, 3) in array (k, j, i) index coordinates and triangles
    (f, 3) of the `level` isosurface of the volume `m` (nk, nj, ni), from
    vtkFlyingEdges3D (Schroeder 2015): the marching-cubes surface, computed
    edge by edge; it is what 3D Slicer uses for segmentation surfaces.
    Raises ImportError without VTK.
    """
    import vtk
    from vtk.util import numpy_support as ns
    nk, nj, ni = m.shape
    img = vtk.vtkImageData()
    img.SetDimensions(ni, nj, nk)                       # VTK x (fastest) = array axis i
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    arr = ns.numpy_to_vtk(np.ascontiguousarray(m, dtype=np.float32).ravel(), deep=True)
    arr.SetName("mask")
    img.GetPointData().SetScalars(arr)
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, level)
    fe.ComputeNormalsOff(); fe.ComputeGradientsOff(); fe.ComputeScalarsOff()
    fe.Update()
    surf = fe.GetOutput()
    if surf.GetNumberOfPoints() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    xyz = ns.vtk_to_numpy(surf.GetPoints().GetData()).astype(float)        # (n, 3) in (i, j, k)
    polys = ns.vtk_to_numpy(surf.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return xyz[:, ::-1], polys.astype(int)


def wall_surface(mask, voxel_to_world, smooth_sigma_vox=1.0):
    """
    Triangulated wall surface of the lumen mask: dict with points (n, 3) in
    world mm, inward unit normals (n, 3), faces (f, 3) int (empty without
    VTK) and ijk (n, 3) index coordinates.

    The wall location is the 0.5 isosurface of the binary mask, extracted
    with VTK's flying edges (the same marching-cubes surface 3D Slicer
    builds for a segmentation; it lies at the voxel boundaries, unbiased on
    average). The normals are the gradient of a Gaussian-smoothed mask at
    those points, because the gradient of the binary mask is a staircase.
    The 0.5 isosurface of the smoothed mask is NOT used for the location:
    smoothing a convex lumen moves it inward by ~ sigma^2 * curvature, which
    overestimates WSS (8% on a 12 mm cylinder at 1 voxel, 25% at 1.5 mm
    voxels smoothed over 2 voxels).
    smooth_sigma_vox: smoothing in voxels along the coarsest axis and the
    same distance in mm along the finer ones (at least smooth_sigma_vox
    voxels on every axis), so the normals are equally smooth in all
    directions. Normals are transformed to world as covectors (inverse
    transpose of the voxel-to-world matrix), which matters on anisotropic,
    oblique or sheared grids.
    Without VTK: the boundary voxels of the mask, moved half a voxel
    outward along the inward normal to where the 0.5 isosurface lies (a
    boundary voxel centre is about half a voxel inside the wall; leaving it
    there overestimates WSS by ~50%); no faces.
    """
    m = np.asarray(mask, float)
    sp = array_axis_spacing(voxel_to_world)
    sigma = np.maximum(smooth_sigma_vox, smooth_sigma_vox * sp.max() / sp) if smooth_sigma_vox > 0 else 0.0
    ms = ndimage.gaussian_filter(m, sigma) if smooth_sigma_vox > 0 else m
    grad = [np.gradient(ms, axis=a) for a in range(3)]

    def inward_at(kji):
        g = np.stack([ndimage.map_coordinates(grad[a], kji.T, order=1) for a in range(3)], -1)
        return g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)      # gradient points into the lumen

    try:
        kji, faces = _vtk_isosurface(m, 0.5)
        inward_kji = inward_at(kji)
    except ImportError:
        boundary = m.astype(bool) & ~ndimage.binary_erosion(m.astype(bool))
        kji = np.argwhere(boundary).astype(float)
        inward_kji = inward_at(kji)
        kji = kji - 0.5 * inward_kji
        faces = np.zeros((0, 3), int)
    ijk = kji[:, ::-1]
    pts = voxel_to_world_points(voxel_to_world, ijk)
    # The index-space gradient of the mask is a covector (a normal), not a
    # direction: n_world = M^-T n_ijk, i.e. rows n_ijk @ inv(M). Mapping it
    # with M itself (as a displacement) is only right for isotropic
    # orthogonal voxels; on a 1 x 1 x 2.5 mm grid it tilts the normals by
    # up to 45 degrees and puts a spurious normal component into the WSS.
    M = np.asarray(voxel_to_world, float)[:3, :3]
    inward_world = inward_kji[:, ::-1] @ np.linalg.inv(M)
    inward_world /= np.linalg.norm(inward_world, axis=-1, keepdims=True) + 1e-12
    return {"points": pts, "normals": inward_world, "faces": faces, "ijk": ijk}


def wall_points(mask, voxel_to_world, smooth_sigma_vox=1.0, step=1):
    """
    Wall points (n, 3) in world mm and inward unit normals (n, 3): every
    `step`-th vertex of wall_surface(). Use wall_surface() for the
    triangles as well.
    """
    s = wall_surface(mask, voxel_to_world, smooth_sigma_vox)
    return s["points"][::step], s["normals"][::step]


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


def to_vtk_surface(points, normals, scalars: dict, faces=None):
    """vtkPolyData of the wall with point-data arrays from `scalars`
    {name: (n,) or (n, 3)}: a triangulated surface when `faces` (f, 3) from
    wall_surface() is given (so the WSS map can be shown as a model in 3D
    Slicer), otherwise vertices."""
    import vtk
    from vtk.util import numpy_support as ns
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    pd.SetPoints(pts)
    if faces is not None and len(faces):
        f = np.asarray(faces, np.int64)
        cells = vtk.vtkCellArray()
        cells.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(f) + 1, 3), deep=True),
                      ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(f.ravel()), deep=True))
        pd.SetPolys(cells)
    else:
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
