"""
Plane flow quantification and mass-conservation QA.

Mass-conservation QA
--------------------
Through-plane flow per frame at each measurement plane, from the velocity
field and a plane definition (centre, normal, radius, optional lumen mask),
then consistency checks the consensus statement asks for:
  inflow vs outflow at a branch  (e.g. AAo vs BCA+LCCA+LSA+DAo)
  Qp:Qs                           (MPA vs AAo)
  systemic venous return          (SVC+IVC vs AAo)
Run before and after each correction step and the differences are the
acceptance test for the filters.

Conventions: velocity_t (nt, nz, ny, nx, 3) with components (vz, vy, vx),
spacing (dz, dy, dx). Plane centre and normal are in voxel-index space
(z, y, x) unless `physical=True`, in which case they are in mm and converted
with spacing. Only numpy and scipy are required.

Defaults: plane slab thickness 0.6 voxel (one voxel layer sampled, area per
voxel = voxel volume / slab); centerline planes every 10 mm with a 15 mm
in-plane radius, which exceeds any pediatric or adult great-vessel radius and
is clipped by the lumen mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Plane flow and mass-conservation QA
# ---------------------------------------------------------------------------

def plane_voxels(shape, spacing, centre, normal, radius, physical=False, thickness=0.6):
    """
    Boolean (nz, ny, nx) selection of voxels within `thickness` voxels of the
    plane through `centre` with `normal`, and within `radius` mm of the centre.
    physical=False: centre and normal in voxel-index space (a plane normal is
    covariant, so it maps to physical space as n / spacing).
    physical=True : centre and normal in mm.
    Returns (selection, unit_normal_mm).
    """
    spacing = np.asarray(spacing, float)
    centre = np.asarray(centre, float)
    normal = np.asarray(normal, float)
    if physical:
        centre_mm = centre
        n_mm = normal / np.linalg.norm(normal)
    else:
        centre_mm = centre * spacing
        n_mm = normal / spacing
        n_mm /= np.linalg.norm(n_mm)
    grids = np.meshgrid(*[np.arange(n) * h for n, h in zip(shape, spacing)], indexing="ij")
    Pmm = np.stack(grids, -1) - centre_mm
    dist = Pmm @ n_mm
    inplane = Pmm - dist[..., None] * n_mm
    sel = (np.abs(dist) <= thickness * spacing.min()) & (np.linalg.norm(inplane, axis=-1) <= radius)
    return sel, n_mm


def plane_flow(velocity_t, spacing, centre, normal, radius, lumen_mask=None,
               physical=False, thickness=0.6, **_ignored):
    """
    Through-plane flow per frame (units: velocity * mm^2, e.g. mm^3/s if
    velocity in mm/s). Positive along `normal`. Also returns the sampled
    voxel count and mean through-plane velocity.
    """
    shape = velocity_t.shape[1:4]
    sel, n_mm = plane_voxels(shape, spacing, centre, normal, radius, physical, thickness)
    if lumen_mask is not None:
        sel &= lumen_mask
    if sel.sum() == 0:
        raise ValueError("plane selects no voxels; check centre/normal/radius")
    # effective area per sampled voxel: voxel volume / slab thickness
    vox_vol = float(np.prod(spacing))
    slab = 2 * thickness * min(spacing)
    area_per_voxel = vox_vol / slab
    vn = velocity_t[:, sel] @ n_mm                               # (nt, n_sel)
    q = vn.sum(axis=1) * area_per_voxel
    return {"flow": q, "mean_velocity": vn.mean(axis=1), "n_voxels": int(sel.sum()),
            "area": float(sel.sum() * area_per_voxel)}


def net_volume(flow, heart_rate_bpm=None, nt=None):
    """Integrate flow over the cycle. With heart rate, returns per-beat volume
    using RR = 60 / HR; otherwise assumes frames evenly span one cycle and
    returns mean flow (per-second units)."""
    flow = np.asarray(flow, float)
    if heart_rate_bpm is None:
        return float(flow.mean())
    rr = 60.0 / heart_rate_bpm
    return float(flow.mean() * rr)


def mass_conservation(flows: dict, checks: list, heart_rate_bpm=None):
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
        q_in = sum(net_volume(flows[n], heart_rate_bpm) for n in ins)
        q_out = sum(net_volume(flows[n], heart_rate_bpm) for n in outs)
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
