"""
Time-resolved pathlines (particle traces) and centerline-based plane placement.

Pathlines
---------
Every commercial 4D flow package (Arterys, cvi42, CAAS, Medis, GTFlow,
iTFlow) presents flow as particles emitted from a plane and advected through
the time-varying velocity field over the cardiac cycle. Streamlines are
instantaneous; pathlines are the 4D flow visualization readers expect.

advect_pathlines() integrates dx/dt = v(x, t) with RK4, trilinear spatial
interpolation and linear, periodic temporal interpolation, stopping a
particle when it leaves the lumen mask or the FOV. Output is a list of
(n, 4) arrays [z, y, x, t_ms] in physical mm, ready for to_vtk_polydata()
which gives a vtkPolyData with a per-point time scalar for Slicer's
Models module (colour by time or speed, animate with the Sequences module).

Centerline planes
-----------------
centerline_planes() takes an ordered centerline polyline (e.g. from the
SlicerVMTK Extract Centerline module) and returns plane definitions every
`step_mm` perpendicular to the local tangent, in the form plane_flow() in
qa.py accepts. This is what cvi42 and the Siemens prototype do to
place measurement planes without user interaction.

Conventions: velocity_t (nt, nz, ny, nx, 3) in mm/s, components (vz, vy, vx),
spacing (dz, dy, dx) mm, rr_ms the cardiac cycle length. Only numpy and scipy
are required; VTK is used only in to_vtk_polydata.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Velocity sampling
# ---------------------------------------------------------------------------

class _Sampler:
    """Trilinear-in-space, linear-and-periodic-in-time velocity sampler."""

    def __init__(self, velocity_t, spacing, rr_ms, mask=None):
        self.v = np.asarray(velocity_t, float)
        self.nt = self.v.shape[0]
        self.spacing = np.asarray(spacing, float)
        self.rr = float(rr_ms)
        self.dt_frame = self.rr / self.nt
        self.mask = None if mask is None else np.asarray(mask, float)
        self.shape = np.array(self.v.shape[1:4])

    def _frame_pair(self, t_ms):
        f = (t_ms % self.rr) / self.dt_frame
        i0 = int(np.floor(f)) % self.nt
        i1 = (i0 + 1) % self.nt
        return i0, i1, f - np.floor(f)

    def __call__(self, pos_mm, t_ms):
        """pos_mm (n, 3) -> velocity (n, 3) mm/s, zeros outside mask/FOV."""
        idx = (pos_mm / self.spacing).T                     # (3, n) voxel coords
        i0, i1, w = self._frame_pair(t_ms)
        out = np.zeros((pos_mm.shape[0], 3))
        for c in range(3):
            a = ndimage.map_coordinates(self.v[i0, ..., c], idx, order=1, mode="constant", cval=0.0)
            b = ndimage.map_coordinates(self.v[i1, ..., c], idx, order=1, mode="constant", cval=0.0)
            out[:, c] = (1 - w) * a + w * b
        return out

    def inside(self, pos_mm):
        idx = (pos_mm / self.spacing).T
        in_fov = np.all((idx >= 0) & (idx <= (self.shape - 1)[:, None]), axis=0)
        if self.mask is None:
            return in_fov
        m = ndimage.map_coordinates(self.mask, idx, order=1, mode="constant", cval=0.0)
        return in_fov & (m > 0.5)


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------

def advect_pathlines(velocity_t, spacing, rr_ms, seeds_mm, t0_ms=0.0, duration_ms=None,
                     step_ms=None, mask=None, backward=False):
    """
    seeds_mm   : (n, 3) seed positions in mm (z, y, x).
    t0_ms      : emission time (scalar or (n,) per seed for staggered emission).
    duration_ms: integration length (default one RR).
    step_ms    : RK4 step, default a quarter of a frame (about 6 ms for a
                 25-frame 600 ms cycle, under one voxel of travel at 1.5 m/s).
    Returns list of (k, 4) arrays [z, y, x, t_ms], one per seed.
    """
    S = _Sampler(velocity_t, spacing, rr_ms, mask)
    seeds = np.asarray(seeds_mm, float)
    n = seeds.shape[0]
    t0 = np.broadcast_to(np.asarray(t0_ms, float), (n,)).copy()
    duration = rr_ms if duration_ms is None else duration_ms
    h = (S.dt_frame / 4.0) if step_ms is None else step_ms
    sign = -1.0 if backward else 1.0
    n_steps = int(np.ceil(duration / h))

    pos = seeds.copy()
    t = t0.copy()
    alive = S.inside(pos)
    traces = [[np.r_[pos[i], t[i]]] for i in range(n)]

    for _ in range(n_steps):
        if not alive.any():
            break
        p = pos[alive]; tt = t[alive]
        # RK4 with per-particle time; sampler is called once per stage with
        # the median time (particles emitted together share it; staggered
        # emission differs by at most a frame, which linear time
        # interpolation tolerates)
        tm = float(np.median(tt))
        k1 = S(p, tm)
        k2 = S(p + 0.5 * h * sign * k1 / 1000.0, tm + 0.5 * h * sign)
        k3 = S(p + 0.5 * h * sign * k2 / 1000.0, tm + 0.5 * h * sign)
        k4 = S(p + h * sign * k3 / 1000.0, tm + h * sign)
        p_new = p + (h * sign / 1000.0) * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0   # mm/s * ms -> mm
        pos[alive] = p_new
        t[alive] = tt + h * sign
        still = S.inside(p_new)
        idx_alive = np.flatnonzero(alive)
        for j, i in enumerate(idx_alive):
            if still[j]:
                traces[i].append(np.r_[p_new[j], t[i]])
        alive[idx_alive[~still]] = False

    return [np.array(tr) for tr in traces]


def seeds_from_plane(mask, spacing, centre_idx, normal, radius_mm, spacing_mm=1.5, lumen_only=True):
    """Seed spacing 1.5 mm, i.e. one seed per acquired voxel, giving a few
    hundred seeds for a great vessel; halve for dense visualizations.
    Regular grid of seeds on a plane (centre in voxel index, normal in
    index space), kept inside the lumen mask. Returns (n, 3) mm positions."""
    sp = np.asarray(spacing, float)
    c = np.asarray(centre_idx, float) * sp
    n = np.asarray(normal, float) / sp
    n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    r = np.arange(-radius_mm, radius_mm + 1e-9, spacing_mm)
    U, V = np.meshgrid(r, r, indexing="ij")
    keep = (U ** 2 + V ** 2) <= radius_mm ** 2
    pts = c + U[keep][:, None] * a + V[keep][:, None] * b
    if lumen_only:
        idx = (pts / sp).T
        m = ndimage.map_coordinates(np.asarray(mask, float), idx, order=1, mode="constant", cval=0.0)
        pts = pts[m > 0.5]
    return pts


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def to_vtk_polydata(traces, velocity_sampler=None):
    """vtkPolyData with one polyline per trace, point data 'time_ms' and
    'index' (particle id). In Slicer: slicer.modules.models.logic().AddModel(pd)."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    times, ids = [], []
    offset = 0
    for i, tr in enumerate(traces):
        if tr.shape[0] < 2:
            continue
        for row in tr:
            pts.InsertNextPoint(row[2], row[1], row[0])     # VTK wants x, y, z
        lines.InsertNextCell(tr.shape[0], list(range(offset, offset + tr.shape[0])))
        times.extend(tr[:, 3]); ids.extend([i] * tr.shape[0])
        offset += tr.shape[0]
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts); pd.SetLines(lines)
    for name, arr in (("time_ms", times), ("index", ids)):
        a = ns.numpy_to_vtk(np.asarray(arr, float), deep=True); a.SetName(name)
        pd.GetPointData().AddArray(a)
    pd.GetPointData().SetActiveScalars("time_ms")
    return pd


# ---------------------------------------------------------------------------
# Centerline-based plane placement
# ---------------------------------------------------------------------------

def centerline_planes(centerline_mm, step_mm=10.0, radius_mm=15.0, smooth_window=5, spacing=None):
    """
    centerline_mm : (n, 3) ordered points in mm (z, y, x), e.g. from VMTK.
    Returns list of dicts {centre, normal, radius, physical=True, arc_mm}
    usable as **kwargs to qa.plane_flow (set physical=True there).
    Tangents are from a smoothed finite difference so planes do not jitter.
    """
    c = np.asarray(centerline_mm, float)
    if smooth_window > 1:
        k = np.ones(smooth_window) / smooth_window
        c = np.stack([np.convolve(np.pad(c[:, i], (smooth_window // 2, smooth_window // 2), mode="edge"),
                                  k, mode="valid") for i in range(3)], 1)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    targets = np.arange(0.0, arc[-1] + 1e-9, step_mm)
    planes = []
    for s in targets:
        i = int(np.searchsorted(arc, s))
        i = min(max(i, 1), c.shape[0] - 2)
        tangent = c[i + 1] - c[i - 1]
        tangent /= np.linalg.norm(tangent) + 1e-12
        centre = c[i] + (s - arc[i]) * tangent
        planes.append(dict(centre=centre, normal=tangent, radius=radius_mm, physical=True, arc_mm=float(s)))
    return planes
