"""
Plane flow quantification and mass-conservation QA.

Mass-conservation QA
--------------------
Through-plane flow per frame at each measurement plane, from the velocity
field and a plane definition (centre, normal, radius, optional lumen mask),
then consistency checks:
  inflow vs outflow at a branch  (e.g. AAo vs BCA+LCCA+LSA+DAo)
  Qp:Qs                           (MPA vs AAo)
  systemic venous return          (SVC+IVC vs AAo)
Run before and after each correction step and the differences are the
acceptance test for the filters.

Conventions (see geometry.py): velocity_t (nt, nk, nj, ni, 3) with world
components (vx, vy, vz), voxel_to_world 4x4 voxel (i, j, k, 1) to world
(x, y, z, 1) matrix. Plane centre is a world point and normal a world vector,
in mm. Only numpy and scipy are required.

Sampling: the plane disc is sampled on a regular in-plane grid (default step
half the smallest voxel size), the lumen mask is interpolated trilinearly
at the sample points (samples with mask > 0.5 are in the lumen), the
through-plane velocity is interpolated trilinearly, normalized by the mask
(interp(v m) / interp(m)) so that velocities outside the lumen do not leak in
at the wall, and each sample represents step^2 of area. This is independent of the orientation of the plane relative to the
voxel grid. Centerline planes every 10 mm with a 15 mm in-plane radius, which
exceeds any pediatric or adult great-vessel radius and is clipped by the
lumen mask.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

from .geometry import voxel_spacing, world_to_array_coordinates


# ---------------------------------------------------------------------------
# Plane flow and mass-conservation QA
# ---------------------------------------------------------------------------

def plane_basis(normal, voxel_to_world=None):
    """
    Unit normal and two unit in-plane vectors (n, a, b), b = n x a.
    With voxel_to_world, `a` is the voxel axis that is most parallel to the
    plane, projected onto the plane, so that a sampling grid built from the
    basis is fixed relative to the image (results do not depend on the
    orientation of the world coordinate system).
    """
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    if voxel_to_world is None:
        axes = np.eye(3)
    else:
        axes = np.asarray(voxel_to_world, float)[:3, :3].T     # rows: world directions of i, j, k
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    # ties (two voxel axes exactly in the plane, the usual axis-aligned case)
    # go to the lowest axis index within a tolerance, so that rounding after a
    # world rotation cannot switch the axis and change the sampling grid
    along = np.abs(axes @ n)
    ref = axes[int(np.flatnonzero(along <= along.min() + 1e-9)[0])]
    a = ref - (ref @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return n, a, b


def plane_samples(shape, voxel_to_world, centre, normal, radius, sampling_mm=None):
    """
    Regular grid of sample points on the disc of `radius` mm around world
    point `centre`, perpendicular to world vector `normal`, kept where it is
    inside the image.
    shape       : image shape (nk, nj, ni).
    sampling_mm : grid step; default half the smallest voxel size.
    Returns (points_world (n, 3), array_coords (3, n), unit_normal, area_per_sample,
    fraction of the disc outside the image).
    """
    if sampling_mm is None:
        sampling_mm = 0.5 * float(voxel_spacing(voxel_to_world).min())
    n, a, b = plane_basis(normal, voxel_to_world)
    r = np.arange(-radius, radius + 1e-9, sampling_mm)
    r = r - r.mean()                                   # symmetric about the centre
    U, V = np.meshgrid(r, r, indexing="ij")
    keep = (U ** 2 + V ** 2) <= radius ** 2
    pts = np.asarray(centre, float) + U[keep][:, None] * a + V[keep][:, None] * b
    idx = world_to_array_coordinates(voxel_to_world, pts)
    inside = np.all((idx >= 0) & (idx <= (np.array(shape[:3]) - 1)[:, None]), axis=0)
    return pts[inside], idx[:, inside], n, sampling_mm ** 2, float(1.0 - inside.mean())


def plane_flow(velocity_t, voxel_to_world, centre, normal, radius, lumen_mask=None,
               sampling_mm=None, **_ignored):
    """
    Through-plane flow per frame (units: velocity * mm^2, e.g. mm^3/s if
    velocity in mm/s). Positive along `normal`. Also returns the number of
    plane samples in the lumen, the mean through-plane velocity, the lumen
    cross-section area (mm^2) and outside_fov_fraction, the part of the
    disc that lies outside the image (those samples carry no flow; a
    warning is issued when it is non-zero and the lumen reaches the image
    border, since the flow may then be underestimated).
    lumen_mask may also be one mask per frame, (nt, nk, nj, ni), for example
    the fraction of refine_vessel_boundary; n_samples and area are then
    arrays, per frame.
    """
    if lumen_mask is not None and np.ndim(lumen_mask) == 4:
        frames = [plane_flow(velocity_t[t:t + 1], voxel_to_world, centre, normal, radius, lumen_mask=lumen_mask[t],
                             sampling_mm=sampling_mm) for t in range(np.shape(velocity_t)[0])]
        return {"flow": np.concatenate([f["flow"] for f in frames]),
                "mean_velocity": np.concatenate([f["mean_velocity"] for f in frames]),
                "n_samples": np.array([f["n_samples"] for f in frames]), "area": np.array([f["area"] for f in frames]),
                "outside_fov_fraction": frames[0]["outside_fov_fraction"]}
    velocity_t = np.asarray(velocity_t, float)
    shape = velocity_t.shape[1:4]
    pts, idx, n, area_per_sample, outside = plane_samples(shape, voxel_to_world, centre, normal, radius,
                                                          sampling_mm)
    if outside > 0:
        # harmless when the lumen lies entirely inside the image (the part of
        # the disc outside is then outside the lumen as well)
        lm_touches_border = lumen_mask is None or any(
            np.asarray(lumen_mask, bool).take([0, -1], axis=ax).any() for ax in range(3))
        if lm_touches_border:
            warnings.warn(f"{100 * outside:.1f}% of the plane disc lies outside the image; "
                          "flow through that part is not counted", stacklevel=2)
    vn_t = velocity_t @ n                                         # (nt, nk, nj, ni)
    weight = None
    if lumen_mask is not None and idx.shape[1]:
        mf = np.asarray(lumen_mask, float)
        weight = ndimage.map_coordinates(mf, idx, order=1, mode="constant", cval=0.0)
        # at half-voxel sampling, samples on the mask boundary get a weight of
        # exactly 0.5; they are excluded with a tolerance so that rounding after
        # a world rotation cannot switch them in or out
        inside = weight > 0.5 + 1e-6
        idx, weight = idx[:, inside], weight[inside]
        # normalized interpolation: velocities outside the lumen do not leak in
        vn_t = vn_t * mf
    if idx.shape[1] == 0:
        raise ValueError("plane selects no voxels; check centre/normal/radius")
    vn = np.stack([ndimage.map_coordinates(vn_t[t], idx, order=1, mode="nearest")
                   for t in range(vn_t.shape[0])])               # (nt, n_samples)
    if weight is not None:
        vn = vn / weight
    q = vn.sum(axis=1) * area_per_sample
    return {"flow": q, "mean_velocity": vn.mean(axis=1), "n_samples": int(idx.shape[1]),
            "area": float(idx.shape[1] * area_per_sample), "outside_fov_fraction": outside}


def net_volume(flow, heart_rate_bpm=None, rr_ms=None):
    """Integrate flow over the cycle. With the cycle length (rr_ms, or
    heart_rate_bpm with RR = 60 / HR), returns the per-beat volume; otherwise
    assumes the frames evenly span one cycle and returns the mean flow
    (per-second units). rr_ms is the convention of pathlines and pressure;
    heart_rate_bpm is kept for the report API."""
    flow = np.asarray(flow, float)
    if rr_ms is not None:
        return float(flow.mean() * rr_ms / 1000.0)
    if heart_rate_bpm is None:
        return float(flow.mean())
    return float(flow.mean() * 60.0 / heart_rate_bpm)


def mass_conservation(flows: dict, checks: list, heart_rate_bpm=None, rr_ms=None):
    """
    flows  : {plane_name: flow_curve}. Sign convention: positive = forward
             in the direction of the circulation for every plane.
    checks : list of (label, inflow_names, outflow_names), e.g.
             ("arch", ["AAo"], ["BCA", "LCCA", "LSA", "DAo"])
             ("Qp:Qs", ["MPA"], ["AAo"])
    Returns list of dicts with in, out, difference, percent error, ratio.
    """
    out = []
    for label, ins, outs in checks:
        q_in = sum(net_volume(flows[n], heart_rate_bpm, rr_ms) for n in ins)
        q_out = sum(net_volume(flows[n], heart_rate_bpm, rr_ms) for n in outs)
        denom = 0.5 * (abs(q_in) + abs(q_out)) + 1e-12
        out.append({"check": label, "in": q_in, "out": q_out,
                    "difference": q_in - q_out,
                    "percent_error": 100.0 * (q_in - q_out) / denom,
                    "ratio": q_in / (q_out + 1e-12)})
    return out


def qa_report(stages: dict, checks: list, heart_rate_bpm=None):
    """
    stages: {"raw": flows_dict, "background": flows_dict, "unwrap": ...}
    Prints percent mass-conservation error per check at each stage so the
    effect of each correction is a number.
    """
    labels = [c[0] for c in checks]
    print(f"{'stage':16s}" + "".join(f"{l:>14s}" for l in labels))
    for stage, flows in stages.items():
        res = mass_conservation(flows, checks, heart_rate_bpm)
        print(f"{stage:16s}" + "".join(f"{r['percent_error']:13.1f}%" for r in res))
