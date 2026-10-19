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

Conventions (see geometry.py): velocity_t (nt, nk, nj, ni, 3), magnitude_t
(nt, nk, nj, ni), voxel_to_world 4x4 voxel (i, j, k, 1) to world (x, y, z, 1)
matrix, velocities in the same units as venc. Aliasing happens independently
in each velocity-encoding direction, so unwrapping must be applied to the
velocity components as encoded (each with its own venc), before they are
rotated into world components; only the voxel size along each array axis is
taken from voxel_to_world. Only numpy and scipy are required.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .background_phase import air_threshold, otsu_threshold
from .geometry import array_axis_spacing


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


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _wrapped_laplacian(phase, spacing):
    """Laplacian of the true phase estimated from wrapped first differences:
    sum_axis [W(p[i+1] - p[i]) - W(p[i] - p[i-1])] / h^2, W wrapping to
    (-pi, pi], with mirrored (zero-difference) edges. Exact whenever the
    true phase changes by less than pi per voxel (Itoh's condition)."""
    out = np.zeros_like(phase)
    for ax, h in enumerate(spacing):
        d = _wrap(np.diff(phase, axis=ax))
        pad = [(0, 0)] * phase.ndim
        pad[ax] = (1, 1)
        d = np.pad(d, pad)                       # zero difference across the mirrored edges
        sl_p = [slice(None)] * phase.ndim; sl_p[ax] = slice(1, None)
        sl_m = [slice(None)] * phase.ndim; sl_m[ax] = slice(None, -1)
        out += (d[tuple(sl_p)] - d[tuple(sl_m)]) / h ** 2
    return out


def laplacian_unwrap_phase(phase, spacing, estimator="schofield_zhu"):
    """
    phase   : wrapped phase in radians, any dimensionality (3D or 4D).
    spacing : sample distance along each array axis of `phase`, e.g.
              geometry.array_axis_spacing(voxel_to_world) for a 3D volume.
    estimator : how the Laplacian of the true phase is estimated from the
              wrapped phase.
              "schofield_zhu" (default): cos(p) lap(sin p) - sin(p) lap(cos p).
              The identity is exact in the continuum; for the discrete
              Laplacian it reduces per axis to sin(d+) - sin(d-) and so
              attenuates a phase step d per voxel by sin(d)/d (0.84 at
              1 rad, 0.54 at 1.8 rad). A noise-free ramp is recovered up to
              1.0 rad/voxel only; this is the source of the single-frame
              wraps the spatial pass leaves in steep jets, which the temporal
              pass of "laplacian_then_temporal" removes.
              "wrapped_difference": second differences of the wrapped first
              differences (Ghiglia & Pritt 1998, ch. 5), exact for steps up
              to 2 rad/voxel on a noise-free ramp, but a noisy difference
              near +/- pi wraps the wrong way, and on the aortic CFD at 5%
              noise it leaves 457 single-frame wraps against 53 for
              Schofield-Zhu (both 0 after the temporal pass). Kept as an
              option for low-noise, very steep data.
    Returns unwrapped phase = phase + 2 pi k with integer k.

    The Laplacian is taken along the array axes, so the result does not
    depend on whether the voxel axes are orthogonal in world space.
    """
    if estimator == "wrapped_difference":
        L = _wrapped_laplacian(phase, spacing)
    elif estimator == "schofield_zhu":
        s, c = np.sin(phase), np.cos(phase)
        L = c * _laplacian_neumann(s, spacing) - s * _laplacian_neumann(c, spacing)
    else:
        raise ValueError(estimator)
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
    """Xiang 1995 along time: cumulative jumps larger than venc are wraps.
    max_wraps bounds the cumulative wrap count of a voxel's time course
    (|k| <= max_wraps); the first frame is taken as unwrapped."""
    v = velocity_t.copy()
    d = np.diff(v, axis=0)
    jumps = -np.round(d / (2 * venc))
    k = np.concatenate([np.zeros_like(v[:1]), np.cumsum(jumps, axis=0)], axis=0)
    k = np.clip(k, -max_wraps, max_wraps)
    return v + 2 * venc * k


def unwrap_velocity(velocity_t, venc, voxel_to_world, method="laplacian_then_temporal",
                    mask=None, time_scale=1.0, max_wraps=3, edge_extension_voxels=3):
    """
    velocity_t : (nt, nk, nj, ni, 3), aliased velocities in [-venc, venc),
                 components in the velocity-encoding directions (as acquired).
    venc       : scalar or (3,) per component, same units as velocity.
    voxel_to_world : 4x4 voxel (i, j, k, 1) to world (x, y, z, 1) matrix; the
                 voxel size along each axis weights the Laplacian.
    mask       : optional (nk, nj, ni) bool; outside it the phase is replaced
                 before unwrapping so noise cannot inject wraps, and the
                 output outside is left as input.
    edge_extension_voxels : with a mask, the wrapped phase is extended
                 outward by nearest-neighbour fill for this many voxels and
                 set to zero beyond. A velocity step from a wall voxel to
                 zero at the mask edge is a phase discontinuity the spatial
                 solve has to absorb; extending the phase first moves that
                 step away from the lumen. On the aortic CFD with a tight
                 (Otsu) noise mask the residual single-frame wraps of the
                 Laplacian pass fall from 493 (zero fill) to 53 (3 voxels).
    time_scale : for laplacian4d, one frame counts as this many voxels.
    max_wraps  : temporal pass recovers up to +/- this many wraps (3, as in
                 the existing module; Loecher 2016 reports up to 2 venc of
                 aliasing recovered in vivo).
    Returns unwrapped velocity_t.
    """
    velocity_t = np.asarray(velocity_t, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    spacing = tuple(array_axis_spacing(voxel_to_world))
    nt = velocity_t.shape[0]
    out = velocity_t.copy()
    m = None if mask is None else np.asarray(mask, bool)
    if m is not None and edge_extension_voxels > 0 and not m.all():
        dist, idx = ndimage.distance_transform_edt(~m, return_indices=True)
        keep = dist <= edge_extension_voxels
    else:
        idx = keep = None

    for comp in range(3):
        p = velocity_t[..., comp] / venc[comp] * np.pi       # (nt, nz, ny, nx)
        if m is not None:
            if idx is None:
                p = p * m[None]
            else:
                p = p[:, idx[0], idx[1], idx[2]] * keep[None]   # nearest lumen phase in the ring
        if method == "temporal":
            u = temporal_unwrap(velocity_t[..., comp], venc[comp], max_wraps)
            u = u / venc[comp] * np.pi
        elif method == "laplacian3d":
            u = np.stack([laplacian_unwrap_phase(p[t], spacing) for t in range(nt)])
        elif method == "laplacian4d":
            u = laplacian_unwrap_phase(p, (time_scale * min(spacing),) + tuple(spacing))
        elif method == "laplacian_then_temporal":
            # Spatial pass first, then temporal clean-up of what it left. The
            # temporal pass relies on the Xiang 1995 condition that the true
            # velocity changes by less than venc between frames; with too few
            # cardiac phases for the peak velocity it can misread a recovered
            # systolic rise as a wrap, so use laplacian3d alone in that case.
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

def noise_mask(magnitude_t, velocity_t, magnitude_threshold="otsu", velocity_std_percentile=97.0,
               min_component_voxels=50, magnitude_percentile=None):
    """
    Bock 2007. Noise = low time-averaged magnitude OR very high temporal
    velocity std (random phase). Returns (nk, nj, ni) bool, True = noise.
    Small isolated signal islands inside noise are also flagged.

    magnitude_threshold : "otsu" (default) separates air from body by Otsu's
        threshold of the time-averaged magnitude, which adapts to the air
        fraction of the field of view (30% of a thoracic FOV, 85% of a
        phantom or an isolated vessel); on a field of view without air (a
        cropped ROI) nothing is flagged by magnitude, with a warning (see
        background_phase.air_threshold). A float is used as a fixed threshold.
    magnitude_percentile : if given, overrides magnitude_threshold with the
        fixed-fraction rule of earlier versions (magnitude below this
        percentile is noise; 30 suits a thoracic FOV where the body is
        55 to 70% of the volume, but under-masks an air-dominated FOV).
    velocity_std_percentile : velocity std in the top 3% is noise (random
        phase in air exceeds any physiological variation); None disables.
    Islands of fewer than min_component_voxels signal voxels are noise.
    Keller 2016 cautions against aggressive noise masking on low-venc data;
    loosen the thresholds rather than tighten.

    Bock J, Kreher BW, Hennig J, Markl M. Optimized pre-processing of
    time-resolved 2D and 3D phase contrast MR data. Proc ISMRM 2007;15:3138.
    """
    mag = magnitude_t.mean(0)
    if magnitude_percentile is not None:
        thr = np.percentile(mag, magnitude_percentile)
    elif isinstance(magnitude_threshold, str):
        if magnitude_threshold != "otsu":
            raise ValueError("magnitude_threshold must be 'otsu' or a number")
        thr = air_threshold(mag)        # -inf (nothing is air) on a FOV without an air class
    else:
        thr = float(magnitude_threshold)
    noise = mag < thr
    if velocity_std_percentile is not None:
        vstd = velocity_t.std(0).sum(-1)
        noise |= vstd > np.percentile(vstd, velocity_std_percentile)
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

def preprocess(velocity_t, magnitude_t, venc, voxel_to_world, lumen_mask=None,
               background_method="war+msac", unwrap_method="laplacian_then_temporal",
               divergence_method=None, divergence_kwargs=None, encoding_to_world=None,
               verbose=True):
    """
    Fixed-order preprocessing: background phase -> unwrap -> noise mask ->
    PC-MRA -> optional divergence filter (needs lumen_mask).

    velocity_t        : (nt, nk, nj, ni, 3) velocity components as encoded,
                        each aliased at its venc.
    encoding_to_world : optional 3x3 matrix whose columns are the world
                        directions of the three encoded components (e.g. a
                        signed permutation for Siemens through-plane /
                        in-plane encodings). Identity if None, i.e. the
                        encoded components are already (vx, vy, vz).

    Returns dict with 'velocity' (world components), 'noise', 'pcmra',
    'report'. report['timing_s'] holds the wall-clock seconds of each step
    (background, unwrap, noise_mask, pcmra, divergence) and the total, so
    the cost of a configuration is visible next to its effect.
    """
    try:
        from .background_phase import correct_background_phase
    except ImportError:
        from background_phase import correct_background_phase
    report = {}
    timing = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    v, info = correct_background_phase(velocity_t, magnitude_t, method=background_method,
                                       lumen_mask=lumen_mask)
    report["background"] = info["report"]
    timing["background"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    signal = ~noise_mask(magnitude_t, v)
    v_corrected = v
    v = unwrap_velocity(v, venc, voxel_to_world, method=unwrap_method, mask=signal)
    report["unwrap"] = {"method": unwrap_method,
                        "voxel_frames_changed": int((np.abs(v - v_corrected) > 1e-9).any(-1).sum())}
    timing["unwrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    noise = noise_mask(magnitude_t, v)
    v[:, noise] = 0.0
    report["noise_fraction"] = float(noise.mean())
    timing["noise_mask"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    angio = pcmra(magnitude_t, v)
    timing["pcmra"] = time.perf_counter() - t0
    if encoding_to_world is not None:
        v = v @ np.asarray(encoding_to_world, float).T     # encoded -> world components

    if divergence_method is not None and lumen_mask is not None:
        try:
            from .divergence_free import divergence_free_sequence
        except ImportError:
            from divergence_free import divergence_free_sequence
        t0 = time.perf_counter()
        v, infos = divergence_free_sequence(v, lumen_mask, voxel_to_world, method=divergence_method,
                                            **(divergence_kwargs or {}))
        report["divergence"] = infos
        timing["divergence"] = time.perf_counter() - t0
    timing["total"] = time.perf_counter() - t_start
    report["timing_s"] = {k: round(val, 4) for k, val in timing.items()}
    if verbose:
        print({k: (val if not isinstance(val, list) else f"{len(val)} entries") for k, val in report.items()})
    return {"velocity": v, "noise": noise, "pcmra": angio, "report": report}
