"""
Hemodynamic quantities derived from the velocity gradient: vorticity,
helicity density, kinetic energy and viscous energy loss (dissipation).

Velocity gradient
-----------------
The gradient tensor G_ab = dv_a/dx_b (world components) is computed by
finite differences along the array axes inside the lumen mask, central
where both neighbours are in the mask and one-sided at the wall, then
mapped to world derivatives with the inverse of the linear part of
voxel_to_world. This is exact for any affine voxel-to-world mapping and is
the same discretisation as build_divergence_operator in divergence_free.py.
Velocity gradients from noisy 4D flow are noisy; apply divergence_free
filtering (dfw) first for visualisation-grade maps.

Quantities (per voxel, per frame)
---------------------------------
vorticity          omega = curl v (1/s), world components.
helicity density   h = v . omega (mm/s^2 for mm/s velocity).
kinetic energy     KE = 1/2 rho |v|^2 per unit volume (J/m^3 = Pa).
viscous energy loss rate (dissipation)  phi = 2 mu sum_ab S_ab S_ab with
                   S = 1/2 (G + G^T) the strain-rate tensor (W/m^3).
Integrating phi over the lumen gives the energy loss in W, and KE over the
lumen gives kinetic energy in J, both reported per frame.

References
----------
Vorticity and helicity from 4D flow:
    Lorenz R, Bock J, Barker AJ, et al. 4D flow magnetic resonance imaging
    in bicuspid aortic valve disease demonstrates altered distribution of
    aortic blood flow helicity. Magn Reson Med 2014;71(4):1542-1553.
    PMID 23716466. doi:10.1002/mrm.24802.
    Garcia J, Barker AJ, Markl M. The role of imaging of flow patterns by
    4D flow MRI in aortic stenosis. JACC Cardiovasc Imaging
    2019;12(2):252-266. PMID 30732721. doi:10.1016/j.jcmg.2018.10.034.
Viscous energy loss from 4D flow:
    Barker AJ, van Ooij P, Bandi K, et al. Viscous energy loss in the
    presence of abnormal aortic flow. Magn Reson Med 2014;72(3):620-628.
    PMID 24122967. doi:10.1002/mrm.24962.
Finite-element formulation of the same quantities (reference for
validation):
    Sotelo J, Urbina J, Valverde I, et al. 3D quantification of wall shear
    stress and oscillatory shear index using a finite-element method in
    3D CINE PC-MRI data of the thoracic aorta. IEEE Trans Med Imaging
    2016;35(6):1475-1487. PMID 26780787. doi:10.1109/TMI.2016.2517406.
    Sotelo J, Dux-Santoy L, Guala A, et al. 3D axial and circumferential
    wall shear stress from 4D flow MRI data using a finite element method
    and a laplacian approach. Magn Reson Med 2018;79(5):2816-2823.
    PMID 28980342. doi:10.1002/mrm.26927.
Kinetic energy from 4D flow:
    Carlsson M, Heiberg E, Toger J, Arheden H. Quantification of left and
    right ventricular kinetic energy using four-dimensional intracardiac
    magnetic resonance imaging flow measurements. Am J Physiol Heart Circ
    Physiol 2012;302(4):H893-H900. PMID 22180653.
    doi:10.1152/ajpheart.00942.2011.

Conventions: velocity (nk, nj, ni, 3) or (nt, nk, nj, ni, 3) in mm/s with
world components (vx, vy, vz); voxel_to_world 4x4 (see geometry.py);
mask (nk, nj, ni) bool. Blood constants: density 1060 kg/m^3, dynamic
viscosity 3.5 mPa s (Newtonian approximation; the 4D flow literature uses
3.2 to 4.0 mPa s).
"""

from __future__ import annotations

import numpy as np

from .geometry import as_voxel_to_world, voxel_spacing

BLOOD_DENSITY_KG_M3 = 1060.0
BLOOD_VISCOSITY_PA_S = 3.5e-3


# ---------------------------------------------------------------------------
# Masked finite-difference gradient
# ---------------------------------------------------------------------------

def _masked_axis_derivative(f, mask, axis):
    """d f / d(array axis) with unit step, central inside the mask, one-sided
    at the mask boundary, zero where no neighbour is in the mask.
    f: (..., nk, nj, ni) with trailing dims allowed after the spatial ones
    via the `axis` index referring to spatial axes 0..2 counted from the
    first of the three spatial dims."""
    f = np.asarray(f, float)
    m = np.asarray(mask, bool)
    sdim = f.ndim - 3 if f.shape[-3:] == m.shape else f.ndim - 4
    ax = sdim + axis
    fwd = np.roll(f, -1, axis=ax); bwd = np.roll(f, 1, axis=ax)
    mf = np.roll(m, -1, axis=axis); mb = np.roll(m, 1, axis=axis)
    # edges of the array have no neighbour
    s = [slice(None)] * 3
    s[axis] = -1; mf[tuple(s)] = False
    s[axis] = 0; mb[tuple(s)] = False
    mf = mf & m; mb = mb & m
    shape_b = (1,) * sdim + m.shape + (1,) * (f.ndim - sdim - 3)
    mfb, mbb = mf.reshape(shape_b), mb.reshape(shape_b)
    both = mfb & mbb
    out = np.where(both, 0.5 * (fwd - bwd),
          np.where(mfb & ~mbb, fwd - f,
          np.where(mbb & ~mfb, f - bwd, 0.0)))
    return out


def velocity_gradient(velocity, mask, voxel_to_world):
    """
    World velocity gradient G[..., a, b] = d v_a / d x_b (1/s for mm/s and
    mm), shape (..., nk, nj, ni, 3, 3). Zero outside the mask.
    """
    v = np.asarray(velocity, float)
    m = np.asarray(mask, bool)
    M = as_voxel_to_world(voxel_to_world)[:3, :3]        # world = M @ (i, j, k)
    Minv = np.linalg.inv(M)                              # (i,j,k) = Minv @ world
    # derivatives along array axes (k, j, i) -> order as (i, j, k) to match M
    d_k = _masked_axis_derivative(v, m, 0)
    d_j = _masked_axis_derivative(v, m, 1)
    d_i = _masked_axis_derivative(v, m, 2)
    d_ijk = np.stack([d_i, d_j, d_k], -1)                # (..., 3 comps, 3 voxel axes)
    # d v_a / d x_b = sum_c d v_a / d xi_c * d xi_c / d x_b = d_ijk @ Minv
    G = d_ijk @ Minv
    return G * m[..., None, None]


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def vorticity(velocity, mask, voxel_to_world, G=None):
    """curl v, world components, 1/s. Shape (..., 3)."""
    G = velocity_gradient(velocity, mask, voxel_to_world) if G is None else G
    return np.stack([G[..., 2, 1] - G[..., 1, 2],
                     G[..., 0, 2] - G[..., 2, 0],
                     G[..., 1, 0] - G[..., 0, 1]], -1)


def helicity_density(velocity, mask, voxel_to_world, G=None, normalized=False):
    """v . curl v (mm/s^2). With normalized=True returns the cosine of the
    angle between v and curl v (relative helicity, -1..1), the quantity
    used by Lorenz 2014 and Garcia 2019; zero where either vanishes."""
    v = np.asarray(velocity, float)
    w = vorticity(v, mask, voxel_to_world, G)
    h = (v * w).sum(-1)
    if not normalized:
        return h
    denom = np.linalg.norm(v, axis=-1) * np.linalg.norm(w, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, h / denom, 0.0)


def kinetic_energy_density(velocity, mask, density=BLOOD_DENSITY_KG_M3):
    """1/2 rho |v|^2 in J/m^3 for velocity in mm/s. Zero outside the mask."""
    v = np.asarray(velocity, float) * 1e-3
    return 0.5 * density * (v ** 2).sum(-1) * np.asarray(mask, float)


def strain_rate(velocity, mask, voxel_to_world, G=None):
    """Symmetric strain-rate tensor S = (G + G^T)/2, 1/s."""
    G = velocity_gradient(velocity, mask, voxel_to_world) if G is None else G
    return 0.5 * (G + np.swapaxes(G, -1, -2))


def viscous_dissipation(velocity, mask, voxel_to_world, G=None, viscosity=BLOOD_VISCOSITY_PA_S):
    """Viscous energy loss rate per unit volume, phi = 2 mu S:S, in W/m^3
    (Barker 2014 eq. 1). Velocity in mm/s, lengths in mm give 1/s rates."""
    S = strain_rate(velocity, mask, voxel_to_world, G)
    return 2.0 * viscosity * (S ** 2).sum((-1, -2))


def lumen_integrals(velocity, mask, voxel_to_world, density=BLOOD_DENSITY_KG_M3,
                    viscosity=BLOOD_VISCOSITY_PA_S):
    """
    Per-frame totals over the lumen: kinetic energy (mJ), viscous energy
    loss rate (mW), mean absolute vorticity (1/s), mean |v| (mm/s),
    lumen volume (mL). velocity (nt, ...) or (nk, nj, ni, 3).
    """
    v = np.asarray(velocity, float)
    if v.ndim == 4:
        v = v[None]
    m = np.asarray(mask, bool)
    # voxel volume is |det M| (prod of the spacings only for orthogonal axes)
    vox_m3 = abs(float(np.linalg.det(np.asarray(voxel_to_world, float)[:3, :3]))) * 1e-9
    n_vox = int(m.sum())
    out = {"kinetic_energy_mJ": [], "energy_loss_mW": [], "mean_abs_vorticity_1_s": [],
           "mean_speed_mm_s": []}
    for t in range(v.shape[0]):
        G = velocity_gradient(v[t], m, voxel_to_world)
        ke = kinetic_energy_density(v[t], m, density)[m].sum() * vox_m3
        el = viscous_dissipation(v[t], m, voxel_to_world, G, viscosity)[m].sum() * vox_m3
        w = np.linalg.norm(vorticity(v[t], m, voxel_to_world, G), axis=-1)[m].mean()
        out["kinetic_energy_mJ"].append(ke * 1e3)
        out["energy_loss_mW"].append(el * 1e3)
        out["mean_abs_vorticity_1_s"].append(float(w))
        out["mean_speed_mm_s"].append(float(np.linalg.norm(v[t], axis=-1)[m].mean()))
    out = {k: np.array(val) for k, val in out.items()}
    out["lumen_volume_mL"] = n_vox * vox_m3 * 1e6
    return out
