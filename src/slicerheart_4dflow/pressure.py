"""
Relative pressure field from 4D flow velocity by the pressure Poisson
equation (PPE).

Method
------
For an incompressible Newtonian fluid the Navier-Stokes momentum equation
gives the pressure gradient from the velocity field:

    grad p = -rho (dv/dt + (v . grad) v) + mu laplacian v        (1)

Taking the divergence and solving div(grad p) = div(rhs) inside the lumen
with Neumann boundary condition grad p . n = rhs . n on the wall yields p
up to a constant (the pressure Poisson equation, Ebbers and Farneback
2009; Ebbers 2001 integrated the Navier-Stokes pressure gradient along
lines, and Krittian 2012 computes relative pressure with a finite-element
formulation instead). Here the Poisson problem is
discretised on the lumen voxels with the same masked finite-difference
operators as the divergence filter (central inside, one-sided at the
wall), which imposes the Neumann condition through the one-sided
derivatives, and solved by conjugate gradient. The constant is fixed by
setting the mean pressure in a reference region (default the whole lumen,
or a user-supplied mask such as a plane neighbourhood) to zero.

The temporal derivative is a central difference over the cardiac cycle
(periodic). The convective term uses the masked velocity gradient from
derivatives.py. Velocity in mm/s and lengths in mm are converted to SI
internally; output pressure is in Pa (1 mmHg = 133.322 Pa).

Validation: for Poiseuille flow the exact gradient is dp/dz = -4 mu
v_max / R^2; for Womersley flow it is the prescribed harmonic series. Both
phantoms in phantom.py carry their analytic gradients.

Limitations: noise in the velocity gradient enters the convective term
quadratically and the viscous term through second derivatives, so the
field should be filtered first (dfw); the result is a relative pressure
only; and near-wall partial volume biases the viscous term, which is
small compared with the inertial terms in the aorta but not in small
vessels.

References
----------
    Ebbers T, Wigstrom L, Bolger AF, Engvall J, Karlsson M. Estimation of
    relative cardiovascular pressures using time-resolved three-dimensional
    phase contrast MRI. Magn Reson Med 2001;45(5):872-879. PMID 11323814.
    doi:10.1002/mrm.1116.
    Ebbers T, Farneback G. Improving computation of cardiovascular relative
    pressure fields from velocity MRI. J Magn Reson Imaging
    2009;30(1):54-61. PMID 19557846. doi:10.1002/jmri.21775.
    Krittian SBS, Lamata P, Michler C, et al. A finite-element approach to
    the direct computation of relative cardiovascular pressure from
    time-resolved MR velocity data. Med Image Anal 2012;16(5):1029-1037.
    PMID 22626833. doi:10.1016/j.media.2012.04.003.
    Bock J, Frydrychowicz A, Lorenz R, et al. In vivo noninvasive 4D
    pressure difference mapping in the human aorta: phantom comparison and
    application in healthy volunteers and patients. Magn Reson Med
    2011;66(4):1079-1088. PMID 21437978. doi:10.1002/mrm.22907.
    Tyszka JM, Laidlaw DH, Asa JW, Silverman JM. Three-dimensional,
    time-resolved (4D) relative pressure mapping using magnetic resonance
    imaging. J Magn Reson Imaging 2000;12(2):321-329. PMID 10931596.
    doi:10.1002/1522-2586(200008)12:2<321::aid-jmri15>3.0.co;2-2.

Conventions as elsewhere: velocity_t (nt, nk, nj, ni, 3) mm/s world
components, mask (nk, nj, ni) bool, voxel_to_world 4x4, rr_ms cycle length.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .derivatives import mask_box, velocity_gradient, BLOOD_DENSITY_KG_M3, BLOOD_VISCOSITY_PA_S
from .divergence_free import _stack
from .geometry import as_voxel_to_world

MMHG_PER_PA = 1.0 / 133.322


def pressure_gradient(velocity_t, mask, voxel_to_world, rr_ms, density=BLOOD_DENSITY_KG_M3,
                      viscosity=BLOOD_VISCOSITY_PA_S, include_viscous=True, include_temporal=True,
                      include_convective=True, frame_duration_ms=None, periodic=None):
    """
    Navier-Stokes pressure gradient (nt, nk, nj, ni, 3) in Pa/mm, world
    components, zero outside the mask. Terms can be switched off for
    analysis (e.g. the inertial-only approximation of Bock 2011).

    The temporal term is a central difference over the frames. By default
    the frames are taken to span the whole RR interval evenly (frame
    duration rr_ms / nt) and the difference wraps around the cycle. For
    prospectively triggered data that covers only part of the RR interval,
    pass frame_duration_ms (the acquired temporal resolution) and
    periodic=False (one-sided differences at the first and last frame).
    """
    v = np.asarray(velocity_t, float) * 1e-3                       # m/s
    if v.ndim == 4:
        v = v[None]
    m = np.asarray(mask, bool)
    box = mask_box(m)
    if box is not None and m[box].size < m.size:
        # on the bounding box of the mask (the same values; zero outside it)
        grad = np.zeros(v.shape)
        grad[(slice(None),) + box] = pressure_gradient(
            np.asarray(velocity_t, float).reshape(v.shape)[(slice(None),) + box], m[box], voxel_to_world, rr_ms,
            density=density, viscosity=viscosity, include_viscous=include_viscous,
            include_temporal=include_temporal, include_convective=include_convective,
            frame_duration_ms=frame_duration_ms, periodic=periodic)
        return grad
    nt = v.shape[0]
    if frame_duration_ms is None:
        frame_duration_ms = rr_ms / nt
        periodic = True if periodic is None else periodic
    elif periodic is None:
        periodic = abs(frame_duration_ms * nt - rr_ms) < 0.02 * rr_ms
    dt = frame_duration_ms * 1e-3
    M = as_voxel_to_world(voxel_to_world)
    # world derivatives are per mm from velocity_gradient; convert to per m
    grad = np.zeros_like(v)
    for t in range(nt):
        G = velocity_gradient(v[t], m, voxel_to_world) * 1e3           # 1/s, d(m/s)/d(m)
        rhs = np.zeros(v.shape[1:])
        if include_temporal and nt > 1:
            if periodic:
                dvdt = (v[(t + 1) % nt] - v[(t - 1) % nt]) / (2 * dt)
            elif t == 0:
                dvdt = (v[1] - v[0]) / dt
            elif t == nt - 1:
                dvdt = (v[nt - 1] - v[nt - 2]) / dt
            else:
                dvdt = (v[t + 1] - v[t - 1]) / (2 * dt)
            rhs -= density * dvdt
        if include_convective:
            rhs -= density * np.einsum("...ab,...b->...a", G, v[t])
        if include_viscous:
            # laplacian v = div(grad v): gradient of each gradient column
            lap = np.zeros_like(v[t])
            for a in range(3):
                Ga = velocity_gradient(G[..., a, :], m, voxel_to_world) * 1e3    # d/dx_b of (dv_a/dx_c)
                lap[..., a] = np.trace(Ga, axis1=-2, axis2=-1)
            rhs += viscosity * lap
        grad[t] = rhs * m[..., None]
    return grad * 1e-3                                                 # Pa/m -> Pa/mm


def build_gradient_operator(mask, voxel_to_world) -> sp.csr_matrix:
    """
    Sparse (3M, M) world-gradient operator on the mask voxels: for a scalar
    field p stacked over mask voxels, G @ p is the stacked gradient
    [dp/dx; dp/dy; dp/dz] in 1/mm units, with the same masked stencils as
    build_divergence_operator (central inside, one-sided at the wall).
    """
    m = np.asarray(mask, bool)
    Minv = np.linalg.inv(as_voxel_to_world(voxel_to_world)[:3, :3])     # d(i,j,k)/d(x,y,z)
    idx = -np.ones(m.shape, np.int64)
    n = int(m.sum())
    idx[m] = np.arange(n)
    # derivative along each array axis (k, j, i): rows (M, M) each
    axis_ops = []
    for axis in range(3):
        fwd = np.full(m.shape, -1, np.int64); bwd = np.full(m.shape, -1, np.int64)
        s1 = [slice(None)] * 3; s1[axis] = slice(1, None)
        s0 = [slice(None)] * 3; s0[axis] = slice(None, -1)
        fwd[tuple(s0)] = idx[tuple(s1)]; bwd[tuple(s1)] = idx[tuple(s0)]
        me, f, b = idx[m], fwd[m], bwd[m]
        hf, hb = f >= 0, b >= 0
        rows, cols, vals = [], [], []
        c = hf & hb
        rows += [me[c], me[c]]; cols += [f[c], b[c]]; vals += [np.full(c.sum(), 0.5), np.full(c.sum(), -0.5)]
        fo = hf & ~hb
        rows += [me[fo], me[fo]]; cols += [f[fo], me[fo]]; vals += [np.full(fo.sum(), 1.0), np.full(fo.sum(), -1.0)]
        bo = ~hf & hb
        rows += [me[bo], me[bo]]; cols += [me[bo], b[bo]]; vals += [np.full(bo.sum(), 1.0), np.full(bo.sum(), -1.0)]
        axis_ops.append(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(n, n)).tocsr())
    d_k, d_j, d_i = axis_ops
    # dp/dx_b = sum_c dp/dxi_c * dxi_c/dx_b, xi = (i, j, k); Minv[c, b] = d xi_c / d x_b
    blocks = []
    for b in range(3):
        blocks.append(Minv[0, b] * d_i + Minv[1, b] * d_j + Minv[2, b] * d_k)
    return sp.vstack(blocks, format="csr")


def relative_pressure(velocity_t, mask, voxel_to_world, rr_ms, reference_mask=None, grad=None,
                      wall_weight=1.0, tol=1e-10, maxiter=10000, **kw):
    """
    Relative pressure (nt, nk, nj, ni) in Pa inside the mask, zero mean in
    reference_mask (default the lumen). Weighted least-squares solution of
    G p = g over the lumen voxels, G the masked gradient operator and g the
    Navier-Stokes pressure gradient; the normal equations G^T W G p = G^T W g
    are the discrete Poisson problem with the Neumann condition built into
    the one-sided wall stencils (Ebbers 2001). Solved by conjugate gradient,
    the frames in threads (the matrix is the same for every frame; starting
    from the previous frame saved only about 3% of the iterations on the
    clinical cases). Returns (pressure_pa, info).

    wall_weight < 1 down-weights gradient samples in the one-voxel wall
    layer (Ebbers and Farneback 2009 weight samples by reliability), where
    the one-sided second derivative of the viscous term is biased by
    no-slip. On Poiseuille and Womersley flow 0.1 reduces the gradient
    error from 7% to 3%. It is not the default because in a
    convection-dominated jet (stenosis phantom) excluding the wall layer
    biases the drop toward the centreline value by 30 to 40%, and aortic
    4D flow is convection-dominated; uniform weights give the
    least-squares pressure of the measured gradient field with no tuning.
    """
    from scipy import ndimage
    m = np.asarray(mask, bool)
    ref = m if reference_mask is None else (np.asarray(reference_mask, bool) & m)
    if grad is None:
        grad = pressure_gradient(velocity_t, m, voxel_to_world, rr_ms, **kw)
    nt = grad.shape[0]
    G = build_gradient_operator(m, voxel_to_world)
    wall = (m & ~ndimage.binary_erosion(m))[m]
    w = np.where(wall, float(wall_weight), 1.0)
    W = sp.diags(np.tile(w, 3), format="csr")
    A = (G.T @ W @ G).tocsr()
    # G^T G is singular (constant null space); pin the mean with a tiny shift
    A = A + sp.identity(A.shape[0], format="csr") * (1e-10 * A.diagonal().max())
    p_out = np.zeros((nt,) + m.shape)

    def solve(t):
        g = _stack(grad[t], m)
        return spla.cg(A, G.T @ (W @ g), rtol=tol, maxiter=maxiter)

    workers = min(nt, os.cpu_count() or 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:      # the sparse products release the GIL
            solutions = list(ex.map(solve, range(nt)))
    else:
        solutions = [solve(t) for t in range(nt)]
    infos = []
    for t, (p, status) in enumerate(solutions):
        field = np.zeros(m.shape); field[m] = p
        field[m] -= field[ref].mean()
        p_out[t] = field
        infos.append({"cg_status": int(status)})
    return p_out, {"cg": infos, "n_voxels": int(m.sum())}


def pressure_drop(pressure_t, mask_a, mask_b):
    """Mean pressure in region a minus region b, per frame (Pa)."""
    p = np.asarray(pressure_t, float)
    return np.array([p[t][mask_a].mean() - p[t][mask_b].mean() for t in range(p.shape[0])])
