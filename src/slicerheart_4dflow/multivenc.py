"""
Multi-venc (dual- and multiple-velocity-encoding) unwrapping.

DRAFT (for review): written against the published equations and tested on
synthetic phantoms only; no dual-venc scanner data have been processed yet.

Why
---
A low venc gives low velocity noise (sigma_v = sqrt(2) venc / (pi SNR)) but
aliases; a high venc gives the range. With two or more acquisitions of the
same field at different vencs, the higher-venc image(s) tell which wrap of
the low-venc image is the right one, so the result keeps the low-venc noise
with the high-venc range (Lee 1995; Nett 2012; Schnell 2017). Single-venc
unwrapping fails once venc is below about half the peak velocity (double
wraps), where multi-venc estimation still works (Löcke 2023).

Target acquisition: interleaved high/low venc with a shared reference scan
(Schnell 2017; Ma 2019; the dual-venc 5D flow of Weiss, Roy et al.), which
gives two velocity series on the same grid and the same frames, typically
with venc_low / venc_high = 1/2. Two separate single-venc scans with
identical parameters (Franco 2022) also work.

Methods (`method=` argument of multivenc_unwrap)
-------------------------------------------------
"sdv"  Standard dual-venc (Lee 1995, as formulated by Franco 2022):
       k = round((v_ref - v_low) / (2 venc_low)), v = v_low + 2 venc_low k.
       The reference is the high-venc image, or (use_ivenc) the "iVENC"
       image built from the wrapped phase difference of the two images,
       venc_i = venc_H venc_L / (venc_H - venc_L), when venc_i > venc_H.
       Requires the reference to be alias-free.

"odv"  Optimal dual-venc / optimal multiple motion encoding (Carrillo 2019;
       Herthum 2022), in the reduced form used by Franco 2022 and Löcke
       2023: with N measurements u_j at vencs venc_j, the velocity is the
       candidate u = u_low + 2 venc_low k, |u| <= venc_eff, that minimises
           J(u) = sum_j [1 - cos(pi (u_j - u) / venc_j)],
       the smallest |u| breaking ties. venc_eff is the least common multiple
       of the vencs (the period of J is 2 venc_eff for rational venc
       ratios), so the high venc need not be alias-free and the range can
       exceed the high venc (e.g. venc_eff = 300 for vencs 150 and 100).
       Works for any number of encodings.
       With correction=True, Franco 2022's correction: where the sign of the
       estimate disagrees with the mean of its neighbourhood, take the
       candidate local minimum of J with the smallest |u| of the
       neighbourhood's sign. Franco applied it to 8-connected pixels of 2D
       through-plane images; the 26-connected 3D neighbourhood (optionally
       including the adjacent frames) and the extra requirement that the
       estimate be at least one low-venc wrap away from the neighbourhood
       mean (so that small in-plane components changing sign across the
       lumen are not touched) are this package's choices.

Not implemented: the bi- and tri-conditional threshold rules of Ma 2019
(to be added from the published tables once checked against the paper),
and PRoM (Zhao 2022), which needs multi-coil complex data.

Findings used for the defaults: venc_low / venc_high = 1/2 is the most
robust ratio for every method (Carrillo 2019; Herthum 2022; Franco 2022);
the corrected ODV is the most robust to noise (Franco 2022).

Conventions: velocity arrays (..., 3) in velocity units with the encoded
components last, so a frame, a time series (nt, nk, nj, ni, 3) or a
respiratory-resolved 5D series all work; venc scalar or (3,) per encoding.
All encodings must share shape; the caller checks geometry and frame
timing. Only numpy and scipy are required.

References
----------
    Lee AT, Pike GB, Pelc NJ. Three-point phase-contrast velocity
    measurements with increased velocity-to-noise ratio. Magn Reson Med
    1995;33(1):122-126. doi:10.1002/mrm.1910330119.
    Nett EJ, Johnson KM, Frydrychowicz A, et al. Four-dimensional phase
    contrast MRI with accelerated dual velocity encoding. J Magn Reson
    Imaging 2012;35(6):1462-1471. PMID 22282344. doi:10.1002/jmri.23588.
    Schnell S, Ansari SA, Wu C, et al. Accelerated dual-venc 4D flow MRI for
    neurovascular applications. J Magn Reson Imaging 2017;46(1):102-114.
    PMID 28152256. doi:10.1002/jmri.25595.
    Ma LE, Markl M, Chow K, et al. Efficient triple-VENC phase-contrast MRI
    for improved velocity dynamic range. Magn Reson Med 2020;83(2):505-520.
    PMID 31423646. doi:10.1002/mrm.27943.
    Carrillo H, Osses A, Uribe S, Bertoglio C. Optimal dual-VENC unwrapping
    in phase-contrast MRI. IEEE Trans Med Imaging 2019;38(5):1263-1270.
    PMID 30475716. doi:10.1109/TMI.2018.2882553.
    Herthum H, Carrillo H, Osses A, Uribe S, Sack I, Bertoglio C. Multiple
    motion encoding in phase-contrast MRI: a general theory and application
    to elastography imaging. Med Image Anal 2022;78:102416.
    doi:10.1016/j.media.2022.102416.
    Franco P, Ma L, Schnell S, et al. Comparison of improved unidirectional
    dual velocity-encoding MRI methods. J Magn Reson Imaging
    2023;57(3):763-773. PMID 35716109. doi:10.1002/jmri.28305.
    Löcke M, Garay Labra JE, Franco P, Uribe S, Bertoglio C. A comparison of
    phase unwrapping methods in velocity-encoded MRI for aortic flows. Magn
    Reson Med 2023;90(5):2102-2115. PMID 37345719. doi:10.1002/mrm.29767.
    Weiss EK, Baraboo JJ, Ma LE, Falcão MBL, Roy C, Stuber M, Markl M.
    Dual-venc 5D flow MRI for increased velocity dynamic range: an in vivo
    validation study. Proc ISMRM 2024. doi:10.58530/2024/4767.
    Zhao S, Ahmad R, Potter LC. Venc design and velocity estimation for
    phase contrast MRI. IEEE Trans Med Imaging 2022;41(12):3712-3724.
    PMID 35862337. doi:10.1109/TMI.2022.3193132.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import ndimage

METHODS = ("sdv", "odv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_velocity(v, venc):
    """Wrap into [-venc, venc), as the scanner does."""
    venc = np.asarray(venc, float)
    return (np.asarray(v, float) + venc) % (2 * venc) - venc


def _as_vencs(vencs, n):
    out = [np.broadcast_to(np.asarray(v, float), (3,)).copy() for v in vencs]
    if len(out) != n:
        raise ValueError(f"{n} velocity arrays but {len(out)} vencs")
    for v in out:
        if np.any(v <= 0):
            raise ValueError("vencs must be positive")
    return out


def _check_shapes(velocities):
    vs = [np.asarray(v, float) for v in velocities]
    if len(vs) < 2:
        raise ValueError("multi-venc unwrapping needs at least two encodings")
    shape = vs[0].shape
    if shape[-1] != 3:
        raise ValueError("velocity arrays must have the three components last, shape (..., 3)")
    for v in vs[1:]:
        if v.shape != shape:
            raise ValueError(f"all encodings must share one shape; got {shape} and {v.shape} "
                             "(resample or crop before unwrapping)")
    return vs


def effective_venc(vencs, max_denominator=100):
    """
    Unambiguous velocity range of a set of vencs: the least common multiple
    of the vencs, per component (the cost of "odv" has period 2 x this for
    rational venc ratios; Herthum 2022). Ratios are rounded to fractions
    with denominators up to max_denominator; an irrational-looking ratio
    falls back to the largest venc.
    """
    vencs = _as_vencs(vencs, len(vencs))
    out = np.empty(3)
    for c in range(3):
        base = max(v[c] for v in vencs)
        fracs = []
        for v in vencs:
            f = Fraction(v[c] / base).limit_denominator(max_denominator)
            if abs(float(f) - v[c] / base) > 1e-6:
                fracs = None
                break
            fracs.append(f)
        if not fracs:
            out[c] = base
            continue
        # lcm of base * p_j / q_j  =  base * lcm(p_j) / gcd(q_j)
        ps = [f.numerator for f in fracs]
        qs = [f.denominator for f in fracs]
        out[c] = base * np.lcm.reduce(ps) / np.gcd.reduce(qs)
    return out


def ivenc(venc_low, venc_high):
    """venc of the phase-difference image of two encodings (Franco 2022 eq. 5)."""
    vl = np.broadcast_to(np.asarray(venc_low, float), (3,))
    vh = np.broadcast_to(np.asarray(venc_high, float), (3,))
    if np.any(vh <= vl):
        raise ValueError("venc_high must exceed venc_low in every component")
    return vh * vl / (vh - vl)


def ivenc_image(v_low, venc_low, v_high, venc_high):
    """
    Velocity image at venc_i = venc_H venc_L / (venc_H - venc_L) from the
    wrapped phase difference of the low- and high-venc images (Franco 2022
    eq. 4): u_i = wrap(venc_i (u_L / venc_L - u_H / venc_H)).
    """
    vi = ivenc(venc_low, venc_high)
    vl = np.broadcast_to(np.asarray(venc_low, float), (3,))
    vh = np.broadcast_to(np.asarray(venc_high, float), (3,))
    return wrap_velocity(vi * (np.asarray(v_low, float) / vl - np.asarray(v_high, float) / vh), vi)


# ---------------------------------------------------------------------------
# Standard dual-venc
# ---------------------------------------------------------------------------

def sdv_unwrap(v_low, venc_low, v_high, venc_high, use_ivenc=True):
    """
    Standard dual-venc unwrapping (Lee 1995; Franco 2022 eq. 2 and 3).

    Returns (velocity, wrap_count, info): the low-venc image unwrapped by the
    nearest integer number of 2 venc_low wraps towards the reference, and
    the integer wraps applied. The reference is the high-venc image, or the
    iVENC image when use_ivenc and venc_i > venc_high (i.e. venc_low >
    venc_high / 2). The reference must be alias-free; its own aliasing
    propagates into the result.
    """
    vl_img, vh_img = _check_shapes([v_low, v_high])
    vl, vh = _as_vencs([venc_low, venc_high], 2)
    vi = ivenc(vl, vh)
    use_i = np.asarray(use_ivenc) & (vi > vh)                   # per component
    ref = np.where(use_i, ivenc_image(vl_img, vl, vh_img, vh), vh_img)
    ref_venc = np.where(use_i, vi, vh)
    k = np.rint((ref - vl_img) / (2 * vl)).astype(int)
    v = vl_img + 2 * vl * k
    info = {"method": "sdv", "reference": ["ivenc" if u else "high" for u in use_i],
            "reference_venc": ref_venc.tolist(), "effective_venc": ref_venc.tolist(),
            "venc_low": vl.tolist(), "venc_high": vh.tolist(),
            "fraction_changed": float((k != 0).mean())}
    return v, k, info


# ---------------------------------------------------------------------------
# Optimal dual-venc / optimal multiple motion encoding
# ---------------------------------------------------------------------------

def _odv_candidates(vl, veff):
    """Integer wraps k of the low-venc image whose candidates can lie in
    [-veff, veff]: |u_low + 2 vl k| <= veff with |u_low| < vl."""
    kmax = int(np.ceil((veff + vl) / (2 * vl)))
    return np.arange(-kmax, kmax + 1)


def odv_cost(u, velocities, vencs):
    """J(u) = sum_j [1 - cos(pi (u_j - u) / venc_j)] for u broadcastable to the
    velocity arrays (..., 3)."""
    J = np.zeros(np.broadcast(u, velocities[0]).shape)
    for uj, vj in zip(velocities, vencs):
        J += 1.0 - np.cos(np.pi * (uj - u) / vj)
    return J


def _neighbourhood_mean(v, mask, include_time):
    """Mean of v over the 26-connected spatial neighbourhood (and the adjacent
    frames if include_time), over masked voxels only, excluding the centre."""
    w = np.asarray(mask, float)
    if v.ndim == 4:                                  # (nk, nj, ni, 3): single frame
        size = (3, 3, 3, 1)
    elif v.ndim >= 5:
        size = (1,) * (v.ndim - 5) + ((3 if include_time else 1), 3, 3, 3, 1)
    else:
        raise ValueError("velocity must be (nk, nj, ni, 3) or (..., nt, nk, nj, ni, 3)")
    wsum = ndimage.uniform_filter(w, size=size[:-1], mode="constant") * np.prod(size[:-1])
    vsum = ndimage.uniform_filter(v * w[..., None], size=size, mode="constant") * np.prod(size)
    vsum = vsum - v * w[..., None]
    wsum = wsum - w
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum[..., None] > 0, vsum / wsum[..., None], 0.0)
    return mean, wsum


def odv_unwrap(velocities, vencs, mask=None, correction=True, include_time=True, return_cost=False):
    """
    Optimal dual-venc / multiple motion encoding unwrapping (Carrillo 2019;
    Herthum 2022; reduced form of Franco 2022 eq. 9).

    velocities : list of arrays (..., 3), the same field at different vencs
                 (any order; the lowest venc per component is the one that is
                 unwrapped).
    vencs      : list of scalar or (3,) vencs, one per array.
    mask       : optional bool spatial mask (nk, nj, ni), broadcast over the
                 leading axes; the correction uses neighbours inside the mask
                 only and leaves voxels outside it as the plain argmin.
    correction : Franco 2022's sign-consistency correction (see module doc).
    include_time : with (..., nt, nk, nj, ni, 3) input, let the neighbourhood
                 of the correction include the adjacent frames.

    Returns (velocity, wrap_count, info) with wrap_count the integer wraps
    applied to the lowest-venc image, and info holding effective_venc, the
    fraction of voxels changed, the fraction corrected and, if return_cost,
    the ambiguity map 'margin' (J of the runner-up minus J of the chosen
    candidate, normalised by 2 N; near zero means the data do not
    distinguish the candidates).
    """
    vs = _check_shapes(velocities)
    vencs = _as_vencs(vencs, len(vs))
    veff = effective_venc(vencs)
    venc_arr = np.stack(vencs)                                   # (N, 3)
    i_low = np.argmin(venc_arr, axis=0)                          # per component
    vl = venc_arr[i_low, np.arange(3)]
    u_low = np.stack([vs[i_low[c]][..., c] for c in range(3)], -1)
    n = len(vs)
    kmax = int(np.ceil((veff.max() + vl.max()) / (2 * vl.min())))
    ks = np.arange(-kmax, kmax + 1)

    # pass 1: best and runner-up cost on the candidate lattice, without
    # storing every candidate (ties within 1e-9 go to the smaller |u|);
    # frame by frame to bound the temporaries
    best_k = np.zeros(u_low.shape, int)
    best_J = np.full(u_low.shape, np.inf)
    second_J = np.full(u_low.shape, np.inf)
    frames = [slice(None)] if u_low.ndim == 4 else [(i,) for i in np.ndindex(u_low.shape[:-4])]
    for fr in frames:
        fr = fr if fr == slice(None) else fr[0] if len(fr) == 1 else fr
        u0 = u_low[fr]
        bJ, sJ, bk = best_J[fr], second_J[fr], best_k[fr]
        babs = np.full(u0.shape, np.inf)
        for k in ks:
            u = u0 + 2 * vl * k
            J = np.zeros(u0.shape)
            for uj, vj in zip(vs, vencs):
                J += 1.0 - np.cos(np.pi * (uj[fr] - u) / vj)
            J[np.abs(u) > veff + 1e-9] = np.inf                  # outside the unambiguous range
            with np.errstate(invalid="ignore"):
                better = (J < bJ - 1e-9) | ((np.abs(J - bJ) <= 1e-9) & (np.abs(u) < babs))
            sJ[:] = np.where(better, bJ, np.minimum(sJ, J))
            bJ[:] = np.where(better, J, bJ)
            babs = np.where(better, np.abs(u), babs)
            bk[:] = np.where(better, k, bk)
    v = u_low + 2 * vl * best_k
    info = {"method": "odv", "n_encodings": n, "vencs": [x.tolist() for x in vencs],
            "venc_low": vl.tolist(), "effective_venc": veff.tolist(),
            "fraction_changed": float((best_k != 0).mean()), "fraction_corrected": 0.0}

    # pass 2 (correction): only the voxels whose sign disagrees with their
    # neighbourhood are re-examined, on the full candidate lattice
    if correction:
        m = np.ones(u_low.shape[:-1], bool) if mask is None else \
            np.ascontiguousarray(np.broadcast_to(np.asarray(mask, bool), u_low.shape[:-1]))
        nb_mean, nb_count = _neighbourhood_mean(v, m, include_time)
        # Franco's trigger is the sign disagreement alone (2D through-plane
        # velocity in the aorta). On three-component data the small in-plane
        # components change sign across the lumen, so this package also
        # requires the estimate to be at least a wrap away from the
        # neighbourhood mean before it is re-examined.
        disagree = ((np.sign(nb_mean) * np.sign(v) < 0) & (np.abs(v - nb_mean) > vl)
                    & (nb_count[..., None] >= 1) & m[..., None])
        if disagree.any():
            comp_all = np.broadcast_to(np.arange(3), u_low.shape)[disagree]
            u0_all = u_low[disagree]
            target_all = np.sign(nb_mean[disagree])
            uj_all = [uj[disagree] for uj in vs]
            k_old = best_k[disagree]
            k_new = k_old.copy()
            chunk = 1 << 20
            for a in range(0, u0_all.size, chunk):
                sl = slice(a, a + chunk)
                comp, u0, target = comp_all[sl], u0_all[sl], target_all[sl]
                vl_s, veff_s = vl[comp], veff[comp]
                u_c = u0[None] + 2 * vl_s[None] * ks[:, None]                  # (K, M)
                J_c = np.zeros_like(u_c)
                for uj, vj in zip(uj_all, vencs):
                    J_c += 1.0 - np.cos(np.pi * (uj[sl][None] - u_c) / vj[comp][None])
                J_c[np.abs(u_c) > veff_s[None] + 1e-9] = np.inf
                pad = np.full((1, u_c.shape[1]), np.inf)
                J_pad = np.concatenate([pad, J_c, pad])
                local_min = (J_c <= J_pad[:-2]) & (J_c <= J_pad[2:]) & np.isfinite(J_c)
                ok = local_min & (np.sign(u_c) == target[None])
                cost = np.where(ok, np.abs(u_c), np.inf)
                pick = np.argmin(cost, axis=0)
                has = np.isfinite(cost[pick, np.arange(cost.shape[1])])
                k_new[sl] = np.where(has, ks[pick], k_old[sl])
            fixed = best_k.copy()
            fixed[disagree] = k_new
            info["fraction_corrected"] = float((fixed != best_k).mean())
            best_k = fixed
            v = u_low + 2 * vl * best_k
    if return_cost:
        info["margin"] = np.where(np.isfinite(second_J), (second_J - best_J) / (2.0 * n), 1.0)
    return v, best_k, info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def multivenc_unwrap(velocities, vencs, method="odv", mask=None, **kwargs):
    """
    Unwrap the lowest-venc image of a multi-venc acquisition.

    velocities : list of (..., 3) arrays (same field, different vencs); the
                 first is taken as the low-venc image for "sdv".
    vencs      : list of scalar or (3,) vencs, one per array.
    method     : "sdv" (two encodings, reference alias-free) or "odv"
                 (any number of encodings; default).
    mask       : optional bool spatial mask for the odv correction.
    kwargs     : use_ivenc for "sdv"; correction, include_time, return_cost
                 for "odv".

    Returns (velocity, wrap_count, info); velocity has the noise of the
    low-venc image and the range of info['effective_venc'].
    """
    if method == "sdv":
        if len(velocities) != 2:
            raise ValueError("sdv takes exactly two encodings (low, high)")
        vl, vh = _as_vencs(vencs, 2)
        if np.any(vl > vh):
            raise ValueError("sdv expects velocities[0] at the lower venc")
        return sdv_unwrap(velocities[0], vl, velocities[1], vh, **kwargs)
    if method == "odv":
        return odv_unwrap(velocities, vencs, mask=mask, **kwargs)
    raise ValueError(f"unknown multi-venc method {method!r}; choose from {METHODS}")


def noise_ratio(v_combined, v_reference, mask):
    """Ratio of the temporal velocity standard deviation of the combined image
    to that of a reference image inside `mask` (static tissue): the noise
    reduction obtained from the low venc. Arrays (nt, nk, nj, ni, 3)."""
    m = np.asarray(mask, bool)
    s1 = np.asarray(v_combined, float)[:, m].std(axis=0).mean()
    s0 = np.asarray(v_reference, float)[:, m].std(axis=0).mean()
    return float(s1 / s0) if s0 > 0 else float("nan")
