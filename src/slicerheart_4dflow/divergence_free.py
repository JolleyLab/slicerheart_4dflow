"""
Divergence-free filtering of 4D flow velocity fields, with a choice of method.

Methods (`method=` argument of divergence_free_filter)
------------------------------------------------------
"fdm"     Song 1993 / Loecher 2012. Masked finite-difference projection:
          minimum-norm correction c such that D (v + c) = 0 inside the lumen,
          with D a sparse masked divergence operator (central differences
          inside, one-sided at the wall). Hard constraint; solved by CG on
          D D^T. Fast, mask-aware, removes exactly the curl-free part of the
          noise inside the lumen (about one third of isotropic white noise).

"relaxed" Mura 2016. Same operator, penalised instead of constrained:
          minimise ||c||^2 + lam ||D (v + c)||^2. lam ~ spacing^2 leaves some
          divergence; lam >> spacing^2 approaches "fdm". Use when the mask is
          unreliable and a hard constraint over-corrects at the wall.

"leray"   Helmholtz-Hodge projection on the whole FOV via FFT (Leray
          projector as a Fourier multiplier). No mask in the projection; the
          field outside the lumen is zero, so the wall acts as a hard boundary
          with a divergence sheet. Iterated with re-masking (alternating
          projections). Included because it is the building block of "dfw"
          and the simplest possible implementation; not recommended alone.

"dfw"     Ong 2015 / Bostan 2015 spirit, dependency-free reimplementation.
          Split the field into divergence-free and curl-free parts with the
          Leray projector, take an undecimated (shift-invariant) wavelet
          transform of each, soft-threshold the detail coefficients with a
          low threshold on the divergence-free part (denoising) and a higher
          threshold on the curl-free part (soft divergence removal that keeps
          the large, structured divergence at vessel edges), recombine.
          This is the only method here that also removes the solenoidal
          component of the noise, so it is the only one that can beat the
          sqrt(2/3) RMSE floor of pure projection. Ong's construction uses
          Deriaz-Perrier divergence-free wavelet bases; the Leray-split
          variant (Bostan, Unser, Ward, IEEE SPL 2015) is equivalent in effect
          and far simpler to implement.

Not implemented: divergence-free radial basis functions (Busch 2012/2013).
Loecher 2012 found them 30x slower than FDM with no accuracy advantage;
Pirrello 2024 found them weaker than DFW near the wall.

Conventions (see geometry.py)
-----------------------------
velocity       : (nk, nj, ni, 3), world components (vx, vy, vz).
voxel_to_world : 4x4 matrix mapping voxel (i, j, k, 1) to world (x, y, z, 1)
                 in mm; any affine mapping (oblique, flipped, anisotropic).
mask           : (nk, nj, ni) bool, lumen = True.
Derivatives are taken along the voxel axes and mapped to world space with the
chain rule, so divergence and the Helmholtz-Hodge split are those of the
physical field. Only numpy and scipy are required.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import array_axis_spacing, as_voxel_to_world, world_to_array_axis_vectors


# ===========================================================================
# Masked finite-difference operator (fdm, relaxed)
# ===========================================================================

def build_divergence_operator(mask, voxel_to_world) -> sp.csr_matrix:
    """
    Sparse (M, 3M) divergence operator over mask voxels, M = mask.sum(),
    acting on the stacked world components [vx; vy; vz] of the mask voxels.

    div v = sum_a d(u_a)/d(xi_a), where u = P v are the contravariant
    components of v along the voxel axes xi (P = inverse of the linear part
    of voxel_to_world), which holds for any affine voxel-to-world mapping.
    """
    mask = np.asarray(mask, bool)
    P = world_to_array_axis_vectors(voxel_to_world)
    idx = -np.ones(mask.shape, np.int64)
    m = int(mask.sum())
    idx[mask] = np.arange(m)
    rows, cols, vals = [], [], []
    for axis in range(3):
        h = 1.0   # unit voxel step; the geometry is in P
        fwd = np.full(mask.shape, -1, np.int64)
        bwd = np.full(mask.shape, -1, np.int64)
        s1 = [slice(None)] * 3; s1[axis] = slice(1, None)
        s0 = [slice(None)] * 3; s0[axis] = slice(None, -1)
        fwd[tuple(s0)] = idx[tuple(s1)]
        bwd[tuple(s1)] = idx[tuple(s0)]
        me, f, b = idx[mask], fwd[mask], bwd[mask]
        hf, hb = f >= 0, b >= 0
        off = axis * m
        c = hf & hb
        rows += [me[c], me[c]]; cols += [off + f[c], off + b[c]]
        vals += [np.full(c.sum(), 0.5 / h), np.full(c.sum(), -0.5 / h)]
        fo = hf & ~hb
        rows += [me[fo], me[fo]]; cols += [off + f[fo], off + me[fo]]
        vals += [np.full(fo.sum(), 1.0 / h), np.full(fo.sum(), -1.0 / h)]
        bo = ~hf & hb
        rows += [me[bo], me[bo]]; cols += [off + me[bo], off + b[bo]]
        vals += [np.full(bo.sum(), 1.0 / h), np.full(bo.sum(), -1.0 / h)]
    D_axes = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(m, 3 * m)).tocsr()
    # world components -> contravariant components along the array axes
    return (D_axes @ sp.kron(sp.csr_matrix(P), sp.identity(m, format="csr"), format="csr")).tocsr()


def _stack(velocity, mask):
    return np.concatenate([velocity[..., k][mask] for k in range(3)])


def _unstack(v, velocity, mask):
    out = velocity.copy()
    m = mask.sum()
    for k in range(3):
        out[..., k][mask] = v[k * m:(k + 1) * m]
    return out


def project_fdm(velocity, mask, voxel_to_world, D=None, tol=1e-8, maxiter=2000):
    D = build_divergence_operator(mask, voxel_to_world) if D is None else D
    m = D.shape[0]
    v = _stack(velocity, mask)
    A = (D @ D.T).tocsr()
    A = A + sp.identity(m, format="csr") * (1e-10 * A.diagonal().max())
    y, status = spla.cg(A, D @ v, rtol=tol, maxiter=maxiter)
    return _unstack(v - D.T @ y, velocity, mask), {"cg_status": int(status)}


def project_relaxed(velocity, mask, voxel_to_world, lam=None, D=None, tol=1e-8, maxiter=2000):
    """lam defaults to 10 x min(spacing)^2: the divergence penalty then
    dominates at the voxel scale while leaving the wall-induced divergence
    partly in place (Mura 2016 report ~50% divergence reduction as the
    practical operating point; lam ~ spacing^2 gives that, 10x is stricter)."""
    if lam is None:
        lam = 10.0 * float(np.min(array_axis_spacing(voxel_to_world))) ** 2
    D = build_divergence_operator(mask, voxel_to_world) if D is None else D
    m = D.shape[0]
    v = _stack(velocity, mask)
    A = sp.identity(3 * m, format="csr") + lam * (D.T @ D)
    c, status = spla.cg(A, -lam * (D.T @ (D @ v)), rtol=tol, maxiter=maxiter)
    return _unstack(v + c, velocity, mask), {"cg_status": int(status)}


# ===========================================================================
# FFT Leray projector (leray, dfw)
# ===========================================================================

def _pad(arr, pad):
    return np.pad(arr, [(pad, pad)] * 3 + [(0, 0)] * (arr.ndim - 3))


def _unpad(arr, pad, shape):
    return arr[pad:pad + shape[0], pad:pad + shape[1], pad:pad + shape[2]]


def leray_split(velocity, voxel_to_world, pad=8):
    """
    Helmholtz-Hodge split on the (zero-padded) box.
    Returns (div_free, curl_free), each (nk, nj, ni, 3), world components.

    A plane wave exp(i kappa . xi) on the voxel grid has the world wave
    vector K = M^-T kappa (M: linear part of voxel_to_world), so the Leray
    projector K (K . V) / |K|^2 is applied with world wave vectors.
    """
    shape = velocity.shape[:3]
    v = _pad(velocity, pad)
    # angular frequency per voxel along the array axes (k, j, i)
    kappa = [2 * np.pi * np.fft.fftfreq(v.shape[a]) for a in range(3)]
    Kk, Kj, Ki = np.meshgrid(*kappa, indexing="ij")
    kappa_ijk = np.stack([Ki, Kj, Kk], -1)
    m_inv = np.linalg.inv(as_voxel_to_world(voxel_to_world)[:3, :3])
    K = kappa_ijk @ m_inv          # K_a = sum_b kappa_b m_inv[b, a]
    k2 = (K ** 2).sum(-1)
    k2[0, 0, 0] = 1.0
    V = np.fft.fftn(v, axes=(0, 1, 2))
    kdotv = (K * V).sum(-1)
    kdotv[0, 0, 0] = 0.0
    Vcf = K * (kdotv / k2)[..., None]
    cf = np.fft.ifftn(Vcf, axes=(0, 1, 2)).real
    df = v - cf
    return _unpad(df, pad, shape), _unpad(cf, pad, shape)


def project_leray(velocity, mask, voxel_to_world, n_iter=3, pad=8):
    """Alternating projections: Leray-project the masked field, re-mask."""
    v = velocity * mask[..., None]
    for _ in range(n_iter):
        df, _ = leray_split(v, voxel_to_world, pad)
        v = df * mask[..., None]
    out = velocity.copy()
    out[mask] = v[mask]
    return out, {}


# ===========================================================================
# Undecimated isotropic wavelet transform (starlet, B3-spline kernel)
# ===========================================================================

_B3 = np.array([1, 4, 6, 4, 1], float) / 16.0


def _atrous_smooth(x, step):
    """Separable B3 à-trous smoothing with dilation `step`, mirror edges."""
    out = x
    for axis in range(3):
        acc = np.zeros_like(out)
        for i, w in enumerate(_B3):
            shift = (i - 2) * step
            acc += w * np.roll(out, shift, axis=axis)
        out = acc
    return out


def starlet(x, levels, mask=None):
    """
    x -> (coarse, [w_1 ... w_J]) with x = coarse + sum(w_j).

    With `mask`, smoothing is a normalized convolution (Knutsson 1993):
    S(x m) / S(m), so the kernel never averages in voxels outside the lumen
    and no mass leaks across the wall. Values outside the mask are undefined.
    """
    details = []
    c = x if mask is None else x * mask
    mf = None if mask is None else mask.astype(float)
    for j in range(levels):
        if mask is None:
            c_next = _atrous_smooth(c, 2 ** j)
        else:
            w = _atrous_smooth(mf, 2 ** j)
            c_next = np.where(w > 1e-6, _atrous_smooth(c * mf, 2 ** j) / np.maximum(w, 1e-6), 0.0)
        details.append(c - c_next)
        c = c_next
    return c, details


def _starlet_noise_scale(levels, shape, rng=None):
    """Per-level std of unit white noise after the transform (computed once)."""
    rng = rng or np.random.default_rng(0)
    n = rng.normal(size=shape)
    _, d = starlet(n, levels)
    return np.array([w.std() for w in d])


def _shrink(x, t, kind="garrote"):
    """Coefficient shrinkage. soft: biased toward zero by t. hard: no bias,
    discontinuous. garrote (non-negative garrote, Gao 1998): between the two,
    nearly unbiased for large coefficients, continuous."""
    if t <= 0:
        return x
    a = np.abs(x)
    if kind == "soft":
        return np.sign(x) * np.maximum(a - t, 0.0)
    if kind == "hard":
        return np.where(a > t, x, 0.0)
    if kind == "garrote":
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(a > t, x * (1.0 - (t / a) ** 2), 0.0)
        return g
    raise ValueError(kind)


def denoise_dfw(velocity, mask, voxel_to_world, k_df=1.5, k_cf=3.0, levels=3, pad=8,
                sigma=None, keep_coarse_cf=True, shrink="soft", split=True):
    """
    Leray-split wavelet shrinkage.

    k_df  : threshold on divergence-free details, in units of noise std
            (1.5 sigma; Ong 2015 used SureShrink, which lands near this for
            4D flow SNR).
    k_cf  : threshold on curl-free details (3.0 sigma): Ong set the
            non-divergence-free threshold to roughly twice the divergence-free
            one. levels=3 covers structure up to 8 voxels, the scale of a
            pediatric great vessel; pad=8 voxels keeps the FFT split from
            wrapping across the box.
    sigma : noise std of a single velocity component (same units as
            velocity). Estimated from the finest-level details inside the
            mask by MAD if None.
    keep_coarse_cf : keep the coarse curl-free residual (wall/edge structure).
                     False makes the result closer to a hard projection.
    shrink : "soft" (default), "garrote" or "hard".
    split  : False skips the Leray split and thresholds the raw field with
             k_df (ablation: plain mask-aware wavelet denoising).
    """
    v = velocity * mask[..., None]
    if split:
        df, cf = leray_split(v, voxel_to_world, pad)
    else:
        df, cf = v, np.zeros_like(v)
    scale = _starlet_noise_scale(levels, (32, 32, 32))

    if sigma is None:
        # MAD of finest detail of the raw field inside the mask
        _, d = starlet(v[..., 0], 1, mask)
        sigma = 1.4826 * np.median(np.abs(d[0][mask])) / scale[0]

    out = np.zeros_like(v)
    for comp in range(3):
        c_df, d_df = starlet(df[..., comp], levels, mask)
        c_cf, d_cf = starlet(cf[..., comp], levels, mask)
        rec = c_df + (c_cf if keep_coarse_cf else 0.0)
        for j in range(levels):
            rec += _shrink(d_df[j], k_df * sigma * scale[j], shrink)
            rec += _shrink(d_cf[j], k_cf * sigma * scale[j], shrink)
        out[..., comp] = rec
    res = velocity.copy()
    res[mask] = out[mask]
    return res, {"sigma": float(sigma)}


# ===========================================================================
# Unified entry point
# ===========================================================================

def divergence_stats(velocity, mask, voxel_to_world, D=None):
    D = build_divergence_operator(mask, voxel_to_world) if D is None else D
    d = D @ _stack(velocity, mask)
    return {"div_rms": float(np.sqrt(np.mean(d ** 2))),
            "div_mean_abs": float(np.mean(np.abs(d)))}


def divergence_free_filter(velocity, mask, voxel_to_world, method="fdm", D=None, **kw):
    """
    velocity (nk, nj, ni, 3) world components, mask (nk, nj, ni) bool,
    voxel_to_world 4x4.
    method: "fdm" | "relaxed" | "leray" | "dfw". kw passed to the method.
    Returns (filtered, info) with divergence before/after in info.
    """
    velocity = np.asarray(velocity, float)
    mask = np.asarray(mask, bool)
    if method in ("fdm", "relaxed") and D is None:
        D = build_divergence_operator(mask, voxel_to_world)
    before = divergence_stats(velocity, mask, voxel_to_world, D)
    if method == "fdm":
        out, info = project_fdm(velocity, mask, voxel_to_world, D=D, **kw)
    elif method == "relaxed":
        out, info = project_relaxed(velocity, mask, voxel_to_world, D=D, **kw)
    elif method == "leray":
        out, info = project_leray(velocity, mask, voxel_to_world, **kw)
    elif method == "dfw":
        out, info = denoise_dfw(velocity, mask, voxel_to_world, **kw)
    else:
        raise ValueError(method)
    after = divergence_stats(out, mask, voxel_to_world, D)
    info.update(method=method, n_voxels=int(mask.sum()),
                div_rms_before=before["div_rms"], div_rms_after=after["div_rms"])
    return out, info


def divergence_free_sequence(velocity_t, mask_t, voxel_to_world, method="fdm", verbose=False, **kw):
    """velocity_t (nt, nk, nj, ni, 3); mask_t (nk, nj, ni) or (nt, nk, nj, ni)."""
    velocity_t = np.asarray(velocity_t)
    mask_t = np.asarray(mask_t, bool)
    static = mask_t.ndim == 3
    D = build_divergence_operator(mask_t, voxel_to_world) if static else None
    out = np.empty_like(velocity_t, dtype=float)
    infos = []
    for t in range(velocity_t.shape[0]):
        mk = mask_t if static else mask_t[t]
        out[t], info = divergence_free_filter(velocity_t[t], mk, voxel_to_world, method=method,
                                              D=D if static else None, **kw)
        infos.append(info)
        if verbose:
            print(f"frame {t:3d} {method}: div RMS {info['div_rms_before']:.4g} -> {info['div_rms_after']:.4g}")
    return out, infos


# ===========================================================================
# 3D Slicer glue
# ===========================================================================

def apply_in_slicer(flow_field_sequence_node, mask_segmentation_node, segment_id,
                    reference_volume_node, method="fdm", **kw):
    """
    Filter, in place, every frame of a SlicerHeart 4D flow sequence. Data nodes
    may be grid transforms or vector volumes; their voxels must be on the
    grid of reference_volume_node (e.g. the magnitude volume), which also
    provides voxel_to_world (IJKToRAS). Velocity arrays from Slicer are
    [k, j, i, 3] with RAS components, which is the convention of this module.
    """
    import slicer
    mask = slicer.util.arrayFromSegmentBinaryLabelmap(mask_segmentation_node, segment_id,
                                                      reference_volume_node).astype(bool)
    voxel_to_world = slicer.util.arrayFromVTKMatrix(reference_volume_node.GetIJKToRASMatrix())
    D = build_divergence_operator(mask, voxel_to_world) if method in ("fdm", "relaxed") else None
    seq = flow_field_sequence_node
    for i in range(seq.GetNumberOfDataNodes()):
        node = seq.GetNthDataNode(i)
        is_volume = node.IsA("vtkMRMLVolumeNode")
        arr = slicer.util.arrayFromVolume(node) if is_volume else slicer.util.arrayFromGridTransform(node)
        if arr.shape[:3] != mask.shape:
            raise ValueError(f"frame {i}: grid {arr.shape[:3]} does not match the reference volume {mask.shape}")
        v_f, info = divergence_free_filter(arr, mask, voxel_to_world, method=method, D=D, **kw)
        arr[...] = v_f
        if is_volume:
            slicer.util.arrayFromVolumeModified(node)
        else:
            slicer.util.arrayFromGridTransformModified(node)
        print(i, info["div_rms_before"], info["div_rms_after"])
