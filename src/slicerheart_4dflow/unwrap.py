"""
Velocity anti-aliasing (phase unwrapping) for 4D flow MRI, plus noise masking
and PC-MRA, and a fixed-order preprocessing pipeline.

Unwrapping methods (`method=` of unwrap_velocity)
-------------------------------------------------
"laplacian3d"  Loecher et al. JMRI 2016, single-step Laplacian unwrapping of
               each frame. The Laplacian of the true phase is estimated from
               the wrapped phase without unwrapping it (Schofield & Zhu 2003):
                   L = cos(p) lap(sin p) - sin(p) lap(cos p)
               then lap(psi) = L is solved in one shot with the DCT (Neumann
               boundaries). The integer wrap count is round((psi - p) / 2 pi)
               and the output is p + 2 pi k, so the measured noise is kept and
               only wraps are changed. Handles aliasing present in every frame
               (stenotic jets, baffles), which temporal unwrapping cannot.

"laplacian4d"  Same, with time as a fourth dimension (Loecher's 4D variant).
               Uses temporal continuity as well, so fewer residual wraps where
               the jet core is small. Time step is scaled so one frame counts
               as `time_scale` voxels.

"temporal"     Xiang 1995 as already in the module: detect jumps > pi between
               consecutive frames along each voxel's time course.

"laplacian_then_temporal"  Recommended default. Laplacian removes wraps that
               persist through the cycle; temporal then mops up isolated
               single-frame wraps the spatial solve leaves at the mask edge.

Noise masking and PC-MRA
------------------------
noise_mask(): Bock 2007. Voxels with low magnitude or high temporal velocity
std are noise. pcmra(): Bock 2010 velocity-weighted angiogram
sqrt(mean_t mag^2 |v|^2), the standard segmentation input.

Pipeline order (Lurie Children's 2023: background correction first cut
mass-conservation error at branch points from 28% to 7%):
    background phase -> unwrap -> noise mask -> divergence filter

Conventions match the other modules: velocity_t (nt, nz, ny, nx, 3),
magnitude_t (nt, nz, ny, nx), velocities in the same units as venc.
Only numpy and scipy are required.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage


# ---------------------------------------------------------------------------
# Laplacian unwrapping core
# ---------------------------------------------------------------------------

def _laplacian_neumann(f, spacing):
    """Discrete Laplacian with mirrored (Neumann) edges, N-dimensional."""
    out = np.zeros_like(f)
    for ax, h in enumerate(spacing):
        fp = np.roll(f, -1, axis=ax)
        fm = np.roll(f, 1, axis=ax)
        # half-sample mirror (f[-1] = f[0]), matching DCT-II eigenfunctions
        sl = [slice(None)] * f.ndim
        sl[ax] = -1; fp[tuple(sl)] = np.take(f, -1, axis=ax)
        sl[ax] = 0;  fm[tuple(sl)] = np.take(f, 0, axis=ax)
        out += (fp - 2 * f + fm) / h ** 2
    return out


def _poisson_dct(rhs, spacing):
    """Solve lap(psi) = rhs with Neumann boundaries via DCT-II."""
    nd = rhs.ndim
    R = sfft.dctn(rhs, type=2, norm="ortho")
    denom = np.zeros_like(R)
    for ax, h in enumerate(spacing):
        n = rhs.shape[ax]
        k = np.arange(n)
        lam = (2 * np.cos(np.pi * k / n) - 2) / h ** 2       # DCT eigenvalues of the 1D Laplacian
        shape = [1] * nd; shape[ax] = n
        denom = denom + lam.reshape(shape)
    denom[(0,) * nd] = 1.0
    P = R / denom
    P[(0,) * nd] = 0.0
    return sfft.idctn(P, type=2, norm="ortho")


def laplacian_unwrap_phase(phase, spacing):
    """
    phase : wrapped phase in radians, any dimensionality (3D or 4D).
    Returns unwrapped phase = phase + 2 pi k with integer k.
    """
    s, c = np.sin(phase), np.cos(phase)
    L = c * _laplacian_neumann(s, spacing) - s * _laplacian_neumann(c, spacing)
    psi = _poisson_dct(L, spacing)
    # psi is the unwrapped phase up to an additive constant; pin it to the
    # wrapped phase by the median offset before rounding
    psi += np.median(phase - psi)
    k = np.round((psi - phase) / (2 * np.pi))
    return phase + 2 * np.pi * k


# ---------------------------------------------------------------------------
# Velocity-level API
# ---------------------------------------------------------------------------

def temporal_unwrap(velocity_t, venc, max_wraps=3):
    """Xiang 1995 along time: cumulative jumps larger than venc are wraps."""
    v = velocity_t.copy()
    d = np.diff(v, axis=0)
    jumps = -np.round(d / (2 * venc))
    jumps = np.clip(jumps, -max_wraps, max_wraps)
    k = np.concatenate([np.zeros_like(v[:1]), np.cumsum(jumps, axis=0)], axis=0)
    return v + 2 * venc * k


def unwrap_velocity(velocity_t, venc, spacing, method="laplacian_then_temporal",
                    mask=None, time_scale=1.0, max_wraps=3):
    """
    velocity_t : (nt, nz, ny, nx, 3), aliased velocities in [-venc, venc).
    venc       : scalar or (3,) per component, same units as velocity.
    spacing    : (dz, dy, dx).
    mask       : optional (nz, ny, nx) bool; outside it the phase is set to
                 zero before unwrapping so noise cannot inject wraps, and the
                 output outside is left as input.
    time_scale : for laplacian4d, one frame counts as this many voxels.
    max_wraps  : temporal pass recovers up to +/- this many wraps (3, as in
                 the existing module; Loecher 2016 reports up to 2 venc of
                 aliasing recovered in vivo).
    Returns unwrapped velocity_t.
    """
    velocity_t = np.asarray(velocity_t, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    nt = velocity_t.shape[0]
    out = velocity_t.copy()
    m = None if mask is None else np.asarray(mask, bool)

    for comp in range(3):
        p = velocity_t[..., comp] / venc[comp] * np.pi       # (nt, nz, ny, nx)
        if m is not None:
            p = p * m[None]
        if method == "temporal":
            u = temporal_unwrap(velocity_t[..., comp], venc[comp], max_wraps)
            u = u / venc[comp] * np.pi
        elif method == "laplacian3d":
            u = np.stack([laplacian_unwrap_phase(p[t], spacing) for t in range(nt)])
        elif method == "laplacian4d":
            u = laplacian_unwrap_phase(p, (time_scale * min(spacing),) + tuple(spacing))
        elif method == "laplacian_then_temporal":
            u = np.stack([laplacian_unwrap_phase(p[t], spacing) for t in range(nt)])
            u = temporal_unwrap(u / np.pi * venc[comp], venc[comp], max_wraps) / venc[comp] * np.pi
        else:
            raise ValueError(method)
        u = u / np.pi * venc[comp]
        if m is not None:
            out[..., comp][:, m] = u[:, m]
        else:
            out[..., comp] = u
    return out


# ---------------------------------------------------------------------------
# Noise masking and PC-MRA
# ---------------------------------------------------------------------------

def noise_mask(magnitude_t, velocity_t, magnitude_percentile=30.0,
               velocity_std_percentile=97.0, min_component_voxels=50):
    """
    Bock 2007. Noise = low time-averaged magnitude OR very high temporal
    velocity std (random phase). Returns (nz, ny, nx) bool, True = noise.
    Small isolated signal islands inside noise are also flagged.

    Defaults: magnitude below the 30th percentile (air, lung; the body is
    normally 55 to 70% of a thoracic FOV) or velocity std in the top 3%
    (random phase in air exceeds any physiological variation); islands of
    fewer than 50 voxels are noise. Keller 2016 cautions against aggressive
    noise masking on low-venc data; loosen the percentiles rather than tighten.
    """
    mag = magnitude_t.mean(0)
    vstd = velocity_t.std(0).sum(-1)
    low_mag = mag < np.percentile(mag, magnitude_percentile)
    high_std = vstd > np.percentile(vstd, velocity_std_percentile)
    noise = low_mag | high_std
    signal = ~noise
    lab, n = ndimage.label(signal)
    if n:
        sizes = ndimage.sum(signal, lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.flatnonzero(sizes < min_component_voxels) + 1)
        noise |= small
    return noise


def pcmra(magnitude_t, velocity_t, gamma=1.0):
    """Bock 2010 velocity-weighted PC-MRA: sqrt(mean_t mag^2 |v|^2)^gamma."""
    speed2 = (velocity_t ** 2).sum(-1)
    return np.sqrt((magnitude_t ** 2 * speed2).mean(0)) ** gamma


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(velocity_t, magnitude_t, venc, spacing, lumen_mask=None,
               background_method="war+msac", unwrap_method="laplacian_then_temporal",
               divergence_method=None, divergence_kwargs=None, verbose=True):
    """
    Fixed-order preprocessing: background phase -> unwrap -> noise mask ->
    PC-MRA -> optional divergence filter (needs lumen_mask).
    Returns dict with 'velocity', 'noise', 'pcmra', 'report'.
    """
    try:
        from .background_phase import correct_background_phase
    except ImportError:
        from background_phase import correct_background_phase
    report = {}

    v, info = correct_background_phase(velocity_t, magnitude_t, method=background_method,
                                       lumen_mask=lumen_mask)
    report["background"] = info["report"]

    signal = ~noise_mask(magnitude_t, v)
    v = unwrap_velocity(v, venc, spacing, method=unwrap_method, mask=signal)
    report["unwrap"] = {"method": unwrap_method,
                        "voxels_changed": int((np.abs(v - velocity_t) > 1e-9).any(-1).sum())}

    noise = noise_mask(magnitude_t, v)
    v[:, noise] = 0.0
    report["noise_fraction"] = float(noise.mean())
    angio = pcmra(magnitude_t, v)

    if divergence_method is not None and lumen_mask is not None:
        try:
            from .divergence_free import divergence_free_sequence
        except ImportError:
            from divergence_free import divergence_free_sequence
        v, infos = divergence_free_sequence(v, lumen_mask, spacing, method=divergence_method,
                                            **(divergence_kwargs or {}))
        report["divergence"] = infos
    if verbose:
        print({k: (val if not isinstance(val, list) else f"{len(val)} entries") for k, val in report.items()})
    return {"velocity": v, "noise": noise, "pcmra": angio, "report": report}
