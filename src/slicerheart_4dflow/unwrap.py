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
               as `time_scale` voxels. The time axis is Neumann by default
               (the existing behaviour); `time_boundary="periodic"` is
               Loecher's choice for a retrospectively gated cycle, where the
               last frame neighbours the first.

"temporal"     Xiang 1995 as already in the module: detect jumps > pi between
               consecutive frames along each voxel's time course.

"laplacian_then_temporal"  Recommended default. Laplacian removes wraps that
               persist through the cycle; temporal then mops up isolated
               single-frame wraps the spatial solve leaves at the mask edge.

Noise masking and PC-MRA
------------------------
noise_mask(): Bock 2007. Voxels with low magnitude or high temporal velocity
std are noise. On aliased data (before unwrapping), pass venc: the temporal
std of wrapped velocity is that of the wrap jumps (2 venc), which marks every
aliased voxel as noise and excludes it from unwrapping; with venc the rule
uses the frame-to-frame differences wrapped into [-venc, venc), which wraps
do not change, with a threshold relative to venc that does not depend on
the field of view. pcmra(): velocity-weighted angiogram (Bock 2010), computed
here as sqrt(mean_t mag^2 |v|^2), the standard segmentation input.

Pipeline order (background correction before any other filter, Keller 2016;
see also Keller 2017):
    background phase -> unwrap -> noise mask -> divergence filter

    Keller EJ, Collins JD, Rigsby CK, Carr JC, Markl M, Schnell S. Optimized
    4D flow MRI processing for evaluation of abdominal blood flow. Proc ISMRM
    2016;24:1939.
    Keller EJ, Collins JD, Rigsby CK, Carr JC, Markl M, Schnell S. Superior
    abdominal 4D flow MRI data consistency with adjusted preprocessing
    workflow and noncontrast acquisitions. Acad Radiol 2017;24(3):350-358.
    PMID 27940231. doi:10.1016/j.acra.2016.10.007.
    Bock J, Frydrychowicz A, Stalder AF, et al. 4D phase contrast MRI at 3 T:
    effect of standard and blood-pool contrast agents on SNR, PC-MRA, and
    blood flow visualization. Magn Reson Med 2010;63(2):330-338.
    PMID 20024953. doi:10.1002/mrm.22199.

Conventions (see geometry.py): velocity_t (nt, nk, nj, ni, 3), magnitude_t
(nt, nk, nj, ni), voxel_to_world 4x4 voxel (i, j, k, 1) to world (x, y, z, 1)
matrix, velocities in the same units as venc. Aliasing happens independently
in each velocity-encoding direction, so unwrapping must be applied to the
velocity components as encoded (each with its own venc), before they are
rotated into world components; only the voxel size along each array axis is
taken from voxel_to_world. Only numpy and scipy are required.
"""

from __future__ import annotations

import functools
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .background_phase import (STATISTICS_SLAB_VOXELS, air_threshold, body_mask, otsu_threshold,
                               temporal_statistics)
from .geometry import array_axis_spacing


# ---------------------------------------------------------------------------
# Laplacian unwrapping core
# ---------------------------------------------------------------------------

BOUNDARIES = ("neumann", "periodic")


def _boundaries(boundary, ndim):
    """Per-axis boundary conditions: None -> all Neumann; a string applies to
    every axis; a sequence gives one per axis."""
    if boundary is None:
        return ("neumann",) * ndim
    if isinstance(boundary, str):
        boundary = (boundary,) * ndim
    boundary = tuple(boundary)
    if len(boundary) != ndim or any(b not in BOUNDARIES for b in boundary):
        raise ValueError(f"boundary must be one of {BOUNDARIES} per axis, got {boundary}")
    return boundary


def _axis_slice(ndim, ax, a, b):
    s = [slice(None)] * ndim
    s[ax] = slice(a, b)
    return tuple(s)


def _laplacian(f, spacing, boundary=None):
    """Discrete Laplacian, N-dimensional: (f[i+1] - 2 f[i] + f[i-1]) / h^2
    with, per axis, mirrored (Neumann) edges (half-sample mirror f[-1] = f[0],
    f[n] = f[n-1], matching the DCT-II eigenfunctions) or periodic edges
    (f[-1] = f[n-1], matching the DFT). Computed on slices, without rolled
    copies of the array."""
    boundary = _boundaries(boundary, f.ndim)
    out = np.zeros_like(f)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        n = f.shape[ax]
        if n == 1:
            continue
        sl = lambda a, c: _axis_slice(f.ndim, ax, a, c)
        out[sl(1, n - 1)] += (f[sl(2, n)] - 2 * f[sl(1, n - 1)] + f[sl(0, n - 2)]) / h ** 2
        if b == "neumann":
            out[sl(0, 1)] += (f[sl(1, 2)] - 2 * f[sl(0, 1)] + f[sl(0, 1)]) / h ** 2
            out[sl(n - 1, n)] += (f[sl(n - 1, n)] - 2 * f[sl(n - 1, n)] + f[sl(n - 2, n - 1)]) / h ** 2
        else:
            out[sl(0, 1)] += (f[sl(1, 2)] - 2 * f[sl(0, 1)] + f[sl(n - 1, n)]) / h ** 2
            out[sl(n - 1, n)] += (f[sl(0, 1)] - 2 * f[sl(n - 1, n)] + f[sl(n - 2, n - 1)]) / h ** 2
    return out


def _laplacian_neumann(f, spacing):
    """Discrete Laplacian with mirrored (Neumann) edges, N-dimensional."""
    return _laplacian(f, spacing, None)


@functools.lru_cache(maxsize=8)
def _poisson_denominator(shape, spacing, boundary, half_axis=None):
    """Eigenvalues of the discrete Laplacian in the spectral basis of each
    axis: DCT-II on Neumann axes, DFT on periodic axes. `half_axis` is the
    periodic axis carried as a half spectrum by rfft. Cached: the same grid
    is solved for every frame and component."""
    nd = len(shape)
    spec_shape = list(shape)
    if half_axis is not None:
        spec_shape[half_axis] = shape[half_axis] // 2 + 1
    denom = np.zeros(spec_shape)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        n = shape[ax]
        k = np.arange(spec_shape[ax])
        if b == "neumann":
            lam = (2 * np.cos(np.pi * k / n) - 2) / h ** 2       # DCT eigenvalues of the 1D Laplacian
        else:
            lam = (2 * np.cos(2 * np.pi * k / n) - 2) / h ** 2   # DFT eigenvalues
        s = [1] * nd; s[ax] = len(k)
        denom = denom + lam.reshape(s)
    denom[(0,) * nd] = 1.0
    return denom


def _poisson_spectral(rhs, spacing, boundary=None, workers=-1):
    """Solve lap(psi) = rhs by a spectral transform per axis: DCT-II on
    Neumann axes, DFT on periodic axes (Loecher 2016 used the DFT on every
    axis; the DCT is the Neumann choice they call ideal in space). The DCTs
    run first, on the real array, and a single periodic axis is then
    transformed by rfft (half spectrum), so every transform has real input
    and the complex array is half size; with more than one periodic axis a
    complex FFT over those axes follows the DCTs. workers: threads for the
    transforms (-1 = all cores; 1 when the caller runs one solve per
    thread)."""
    boundary = _boundaries(boundary, rhs.ndim)
    nd = rhs.ndim
    neumann_axes = [ax for ax, b in enumerate(boundary) if b == "neumann"]
    periodic_axes = [ax for ax, b in enumerate(boundary) if b == "periodic"]
    R = rhs
    if neumann_axes:
        R = sfft.dctn(R, type=2, norm="ortho", axes=neumann_axes, workers=workers)
    half = periodic_axes[0] if len(periodic_axes) == 1 else None
    if half is not None:
        R = sfft.rfft(R, axis=half, norm="ortho", workers=workers)
    elif periodic_axes:
        R = sfft.fftn(R, axes=periodic_axes, norm="ortho", workers=workers)
    P = R / _poisson_denominator(rhs.shape, tuple(float(h) for h in spacing), boundary, half)
    P[(0,) * nd] = 0.0
    if half is not None:
        P = sfft.irfft(P, n=rhs.shape[half], axis=half, norm="ortho", workers=workers, overwrite_x=True)
    elif periodic_axes:
        P = sfft.ifftn(P, axes=periodic_axes, norm="ortho", workers=workers, overwrite_x=True).real
    if neumann_axes:
        P = sfft.idctn(P, type=2, norm="ortho", axes=neumann_axes, workers=workers, overwrite_x=True)
    return P


def _poisson_dct(rhs, spacing, workers=-1):
    """Solve lap(psi) = rhs with Neumann boundaries via DCT-II."""
    return _poisson_spectral(rhs, spacing, None, workers)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _schofield_zhu_laplacian(phase, spacing, boundary=None):
    """cos(p) lap(sin p) - sin(p) lap(cos p) for the discrete Laplacian,
    evaluated per axis as sin(p[i+1] - p[i]) - sin(p[i] - p[i-1]) (the
    identity cos a sin b - sin a cos b = sin(b - a) applied to each
    neighbour): one sine of the first difference per axis instead of sin,
    cos and two Laplacians of the whole array. Mirrored edges contribute
    sin 0 = 0, periodic edges the sine of the difference across the edge."""
    boundary = _boundaries(boundary, phase.ndim)
    out = np.zeros_like(phase)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        n = phase.shape[ax]
        if n == 1:
            continue
        sl = lambda a, c: _axis_slice(phase.ndim, ax, a, c)
        d = np.sin(np.diff(phase, axis=ax))                   # sin(p[i+1] - p[i]), length n-1
        d /= h ** 2
        out[sl(0, n - 1)] += d
        out[sl(1, n)] -= d
        if b == "periodic":
            dw = np.sin(phase[sl(0, 1)] - phase[sl(n - 1, n)])    # sin(p[0] - p[n-1]) across the edge
            dw /= h ** 2
            out[sl(n - 1, n)] += dw
            out[sl(0, 1)] -= dw
    return out


def _wrapped_laplacian(phase, spacing, boundary=None):
    """Laplacian of the true phase estimated from wrapped first differences:
    sum_axis [W(p[i+1] - p[i]) - W(p[i] - p[i-1])] / h^2, W wrapping to
    (-pi, pi], with mirrored (zero-difference) edges on Neumann axes and
    the wrapped difference across the edge on periodic axes. Exact whenever
    the true phase changes by less than pi per voxel (Itoh's condition)."""
    boundary = _boundaries(boundary, phase.ndim)
    out = np.zeros_like(phase)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        n = phase.shape[ax]
        if n == 1:
            continue
        sl = lambda a, c: _axis_slice(phase.ndim, ax, a, c)
        d = _wrap(np.diff(phase, axis=ax))
        d /= h ** 2
        out[sl(0, n - 1)] += d
        out[sl(1, n)] -= d
        if b == "periodic":
            dw = _wrap(phase[sl(0, 1)] - phase[sl(n - 1, n)])
            dw /= h ** 2
            out[sl(n - 1, n)] += dw
            out[sl(0, 1)] -= dw
    return out


def laplacian_unwrap_phase(phase, spacing, estimator="schofield_zhu", boundary=None, workers=-1):
    """
    phase   : wrapped phase in radians, any dimensionality (3D or 4D).
    spacing : sample distance along each array axis of `phase`, e.g.
              geometry.array_axis_spacing(voxel_to_world) for a 3D volume.
    boundary : boundary condition per axis, "neumann" (default, mirrored
              edges, DCT solve) or "periodic" (DFT solve). A retrospectively
              gated cardiac cycle is periodic in time, so the time axis of a
              4D solve should be periodic (Loecher 2016): with Neumann edges
              a systolic peak falling in the first or last frame has no
              temporal neighbour to pull it from and may stay wrapped.
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
              differences (Ghiglia & Pritt 1998, ch. 5, minimum-norm methods;
              ISBN 978-0-471-24935-1), exact for steps up
              to 2 rad/voxel on a noise-free ramp, but a noisy difference
              near +/- pi wraps the wrong way, and on the aortic CFD at 5%
              noise it leaves 457 single-frame wraps against 53 for
              Schofield-Zhu (both 0 after the temporal pass). Kept as an
              option for low-noise, very steep data.
    workers : threads for the spectral solve (-1 = all cores).
    Returns unwrapped phase = phase + 2 pi k with integer k.

    The Laplacian is taken along the array axes, so the result does not
    depend on whether the voxel axes are orthogonal in world space.
    """
    boundary = _boundaries(boundary, phase.ndim)
    if estimator == "wrapped_difference":
        L = _wrapped_laplacian(phase, spacing, boundary)
    elif estimator == "schofield_zhu":
        L = _schofield_zhu_laplacian(phase, spacing, boundary)
    else:
        raise ValueError(estimator)
    psi = _poisson_spectral(L, spacing, boundary, workers)
    # psi is the unwrapped phase up to an additive constant; pin it to the
    # wrapped phase by the median offset before rounding
    psi += np.median(phase - psi)
    k = np.round((psi - phase) / (2 * np.pi))
    return phase + 2 * np.pi * k


# ---------------------------------------------------------------------------
# Velocity-level API
# ---------------------------------------------------------------------------

def temporal_unwrap(velocity_t, venc, max_wraps=3, out=None):
    """Xiang 1995 along time: cumulative jumps larger than venc are wraps.
    max_wraps bounds the cumulative wrap count of a voxel's time course
    (|k| <= max_wraps); the first frame is taken as unwrapped.

    Evaluated frame by frame: the jump between frames t-1 and t is
    -round(diff / 2 venc), the wrap count its running sum, clipped, so a
    frame needs only its predecessor and the running count, without the
    sequence-sized temporaries of diff, round, cumsum and clip. `out`
    receives the result and may be velocity_t itself (in place); it must
    have the result dtype, velocity_t's promoted with venc's (float64 for
    float64 velocities)."""
    v = np.asarray(velocity_t)
    nt = v.shape[0]
    two_venc = 2 * venc
    dtype = (np.empty(1, v.dtype) / two_venc).dtype      # dtype of the wrap count, as numpy promotes it
    if out is None:
        out = np.empty(v.shape, dtype)
    elif out.shape != v.shape or out.dtype != dtype:
        raise ValueError(f"out must have shape {v.shape} and dtype {dtype}")
    in_place = np.may_share_memory(out, v)
    k = np.zeros(v.shape[1:], dtype)
    prev = None
    for t in range(nt):
        cur = v[t]
        if t:
            jumps = -np.round((cur - prev) / two_venc)
            k = jumps if t == 1 else np.add(k, jumps, out=k)    # cumsum: first term, then running sum
        prev = cur.copy() if in_place else cur                  # the input frame, before it is overwritten
        np.add(cur, two_venc * np.clip(k, -max_wraps, max_wraps), out=out[t])
    return out


def _map_frames(fn, frames, workers=None):
    """fn over a list of frames, one frame per thread (numpy and scipy.fft
    release the GIL on large arrays; frames are independent). Results in
    input order, identical to the sequential ones."""
    workers = workers or min(len(frames), os.cpu_count() or 1)
    if workers <= 1 or len(frames) <= 1:
        return [fn(f) for f in frames]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, frames))


def unwrap_velocity(velocity_t, venc, voxel_to_world, method="laplacian_then_temporal",
                    mask=None, time_scale=1.0, max_wraps=3, edge_extension_voxels=3,
                    time_boundary="neumann", workers=None):
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
    time_boundary : for laplacian4d, the boundary condition of the time
                 axis: "neumann" (default, the existing behaviour, mirrored
                 edges) or "periodic" (Loecher 2016: the cardiac cycle of a
                 retrospectively gated acquisition is periodic, so the last
                 frame neighbours the first; use it for retrospective gating,
                 keep Neumann for prospectively triggered data that stops
                 before end-diastole). The spatial axes are always Neumann.
    max_wraps  : temporal pass recovers up to +/- this many wraps (3, as in
                 the existing module). Loecher 2016 fully unwrapped in vivo
                 aortic data with peak velocities about twice venc (venc
                 80 cm/s, peaks about 160 cm/s).
    workers    : threads for the per-frame spatial solves (default: one per
                 core). The result does not depend on it.
    Returns unwrapped velocity_t.
    """
    if method not in ("temporal", "laplacian3d", "laplacian4d", "laplacian_then_temporal"):
        raise ValueError(method)
    if time_boundary not in BOUNDARIES:
        raise ValueError(f"time_boundary must be one of {BOUNDARIES}")
    boundary4d = (time_boundary,) + ("neumann",) * 3
    velocity_t = np.asarray(velocity_t)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    spacing = tuple(array_axis_spacing(voxel_to_world))
    nt, shape = velocity_t.shape[0], velocity_t.shape[1:-1]
    m = None if mask is None else np.asarray(mask, bool)
    # The work is done one component at a time and, where the method allows,
    # one frame at a time, in float64: the only sequence-sized arrays are the
    # output and one component. Outside the mask the output is the input.
    out = np.array(velocity_t, float) if m is not None else np.empty(velocity_t.shape, float)
    nearest = keep = None
    if m is not None and method != "temporal" and edge_extension_voxels > 0 and not m.all():
        dist, idx = ndimage.distance_transform_edt(~m, return_indices=True)
        keep = dist <= edge_extension_voxels
        nearest = np.ravel_multi_index(tuple(idx), shape)   # flat index of the nearest lumen voxel

    def wrapped_phase(t, comp):
        """Frame t of component comp as wrapped phase, masked or extended."""
        p = np.asarray(velocity_t[t, ..., comp], float) / venc[comp]
        p *= np.pi
        if m is not None:
            if nearest is None:
                p *= m
            else:
                p = np.take(p, nearest)                      # nearest lumen phase in the ring
                p *= keep
        return p

    for comp in range(3):
        u = np.empty((nt,) + shape)
        # scalings applied to each frame after the method, in the order of
        # the method's definition: the temporal pass works in velocity units,
        # the Laplacian passes in phase, and the result is in velocity units
        if method == "temporal":
            u[...] = velocity_t[..., comp]
            temporal_unwrap(u, venc[comp], max_wraps, out=u)
            scalings = [(np.divide, venc[comp]), (np.multiply, np.pi),
                        (np.divide, np.pi), (np.multiply, venc[comp])]
        elif method == "laplacian4d":
            p = np.empty((nt,) + shape)
            for t in range(nt):
                p[t] = wrapped_phase(t, comp)
            u = laplacian_unwrap_phase(p, (time_scale * min(spacing),) + tuple(spacing), boundary=boundary4d)
            del p
            scalings = [(np.divide, np.pi), (np.multiply, venc[comp])]
        else:
            def solve(t):
                r = laplacian_unwrap_phase(wrapped_phase(t, comp), spacing, workers=1)
                r /= np.pi
                r *= venc[comp]
                u[t] = r
            _map_frames(solve, list(range(nt)), workers)
            if method == "laplacian_then_temporal":
                # Spatial pass first, then temporal clean-up of what it left. The
                # temporal pass relies on the Xiang 1995 condition that the true
                # velocity changes by less than venc between frames; with too few
                # cardiac phases for the peak velocity it can misread a recovered
                # systolic rise as a wrap, so use laplacian3d alone in that case.
                temporal_unwrap(u, venc[comp], max_wraps, out=u)
                scalings = [(np.divide, venc[comp]), (np.multiply, np.pi),
                        (np.divide, np.pi), (np.multiply, venc[comp])]
            else:
                scalings = []
        for t in range(nt):
            f = u[t]
            for op, sc in scalings:
                op(f, sc, out=f)
            if m is None:
                out[t, ..., comp] = f
            else:
                np.copyto(out[t, ..., comp], f, where=m)
    return out


# ---------------------------------------------------------------------------
# Noise masking and PC-MRA
# ---------------------------------------------------------------------------

def noise_mask(magnitude_t, velocity_t, magnitude_threshold="gmm", velocity_std_percentile=97.0,
               min_component_voxels=50, magnitude_percentile=None, venc=None, wrapped_difference_threshold=0.3):
    """
    Bock 2007. Noise = background by signal to phase-noise score (low
    time-averaged magnitude relative to the temporal velocity std) OR very
    high temporal velocity std (random phase). Returns (nk, nj, ni) bool,
    True = noise. Small isolated signal islands inside noise are also flagged.

    magnitude_threshold : "gmm" (default) separates background from body
        with background_phase.body_mask, a Kittler-Illingworth minimum-error
        split of log(magnitude / velocity std), which adapts to the air
        fraction of the field of view (30% of a thoracic FOV, 85% of a
        phantom or an isolated vessel) and is not fooled by coil shading;
        "otsu" is Otsu's threshold of the time-averaged magnitude alone
        (background_phase.air_threshold). Either treats a field of view
        without air (a cropped ROI) as all body, with a warning. A float is
        used as a fixed magnitude threshold.
    magnitude_percentile : if given, overrides magnitude_threshold with the
        fixed-fraction rule of earlier versions (magnitude below this
        percentile is noise; 30 suits a thoracic FOV where the body is
        55 to 70% of the volume, but under-masks an air-dominated FOV).
    velocity_std_percentile : velocity std in the top 3% is noise (random
        phase in air exceeds any physiological variation); None disables.
        Skipped when the "gmm" body mask finds no background class, since
        the highest velocity std is then the pulsatile lumen.
        Used when venc is not given. The percentile is of the whole field
        of view, so where vessels fill more than about 3% of it, pulsatile
        flow is marked too; and on aliased data every wrapped voxel is (see
        venc).
    venc : scalar or (3,) per component; given for velocities that may be
        aliased (before unwrapping). The velocity-std rule then uses the RMS
        of the frame-to-frame differences wrapped into [-venc, venc), mean
        over the components: it does not change where a voxel wraps, is
        venc / sqrt(3) (0.58 venc) for random phase and a few percent of
        venc for tissue and flow. A voxel is noise where it exceeds
        wrapped_difference_threshold * venc; velocity_std_percentile is
        not used.
    Islands of fewer than min_component_voxels signal voxels are noise.
    Keller 2016 cautions against aggressive noise masking on low-venc data;
    loosen the thresholds rather than tighten.

    Bock J, Kreher BW, Hennig J, Markl M. Optimized pre-processing of
    time-resolved 2D and 3D phase contrast MRI data. Proc ISMRM 2007;15:3138.
    """
    magnitude_t = np.asarray(magnitude_t)
    velocity_t = np.asarray(velocity_t)
    vstd_rule = velocity_std_percentile is not None
    # The magnitude and velocity std rules reduce in the input dtype, as
    # magnitude_t.mean(0) and velocity_t.std(0).sum(-1) do (slab by slab,
    # see _reduce_frames). body_mask scores by the float64 statistics of
    # temporal_statistics, computed once here and passed to it; for float64
    # input they are the same numbers (bit for bit) and serve the velocity
    # std rule too, so the 4D array is reduced once.
    stats = None
    if magnitude_percentile is not None:
        mag = _reduce_frames(lambda a: a.mean(0), magnitude_t)
        noise = mag < np.percentile(mag, magnitude_percentile)
    elif isinstance(magnitude_threshold, str):
        if magnitude_threshold == "gmm":
            stats = temporal_statistics(magnitude_t, velocity_t, pcmra=False)
            body, _, fit = body_mask(magnitude_t, velocity_t, min_component_voxels=min_component_voxels,
                                     return_score=True, venc=venc, stats=stats)
            noise = ~body
            if not np.isfinite(fit["threshold"]):
                # no background class (cropped ROI): the highest velocity std
                # is then the pulsatile lumen, not random phase
                vstd_rule = False
        elif magnitude_threshold == "otsu":
            mag = _reduce_frames(lambda a: a.mean(0), magnitude_t)
            noise = mag < air_threshold(mag)    # -inf (nothing is air) on a FOV without an air class
        else:
            raise ValueError("magnitude_threshold must be 'gmm', 'otsu' or a number")
    else:
        noise = _reduce_frames(lambda a: a.mean(0), magnitude_t) < float(magnitude_threshold)
    if venc is not None and velocity_t.shape[0] > 1:
        venc = np.broadcast_to(np.asarray(venc, float), (3,))
        difference = np.diff(np.asarray(velocity_t, float), axis=0)
        difference = (difference + venc) % (2 * venc) - venc              # wrapped into [-venc, venc)
        rms = np.sqrt(np.mean(difference ** 2, axis=0)) / venc           # (nk, nj, ni, 3), relative to venc
        noise |= rms.mean(-1) > wrapped_difference_threshold
    elif vstd_rule:
        if stats is not None and velocity_t.dtype == np.float64:
            vstd = np.sqrt(stats["var"]).sum(-1)            # velocity_t.std(0).sum(-1), as body_mask scored
        else:
            vstd = _reduce_frames(lambda a: a.std(0), velocity_t).sum(-1)
        noise |= vstd > np.percentile(vstd, velocity_std_percentile)
    signal = ~noise
    lab, n = ndimage.label(signal)
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)   # voxels per label; sizes[0] is the noise
        small = sizes < min_component_voxels
        small[0] = False
        noise |= small[lab]
    return noise


def _reduce_frames(fn, array_t):
    """fn (a reduction over axis 0, e.g. lambda a: a.mean(0)) of the 4D
    array_t, applied to spatial slabs of STATISTICS_SLAB_VOXELS voxels and
    assembled: the same numbers as fn(array_t), since numpy reduces axis 0
    frame by frame in index order for a slab as for the whole array, with
    slab-sized temporaries instead of the sequence-sized ones of
    array_t.std(0)."""
    shape = array_t.shape[1:]
    step = max(1, STATISTICS_SLAB_VOXELS // int(np.prod(shape[1:3])))
    out = None
    for k0 in range(0, shape[0], step):
        r = fn(array_t[:, k0:k0 + step])
        if out is None:
            out = np.empty(shape, r.dtype)
        out[k0:k0 + step] = r
    return out


def pcmra(magnitude_t, velocity_t, gamma=1.0):
    """Velocity-weighted PC-MRA (Bock 2010), computed as
    sqrt(mean_t mag^2 |v|^2)^gamma."""
    speed2 = (velocity_t ** 2).sum(-1)
    return np.sqrt((magnitude_t ** 2 * speed2).mean(0)) ** gamma


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(velocity_t, magnitude_t, venc, voxel_to_world, lumen_mask=None,
               background_method="war+msac", unwrap_method="laplacian_then_temporal",
               divergence_method=None, divergence_kwargs=None, encoding_to_world=None,
               verbose=True, background_kwargs=None, unwrap_kwargs=None, extra_encodings=None,
               background_first=False):
    """
    Fixed-order preprocessing: [multi-venc unwrap ->] background phase ->
    unwrap -> noise mask -> PC-MRA -> optional divergence filter (needs
    lumen_mask).

    velocity_t        : (nt, nk, nj, ni, 3) velocity components as encoded,
                        each aliased at its venc. With extra_encodings this
                        is the low-venc series.
    extra_encodings   : DRAFT. Optional list of (velocity_t, venc) or
                        (velocity_t, venc, magnitude_t) of the same field
                        acquired at other (higher) vencs, on the same grid
                        and frames (dual- or triple-venc); the magnitude is
                        needed for the dephasing fallback ("fallback_ratio"
                        in unwrap_kwargs). The
                        low-venc series is then unwrapped first with
                        multivenc.multivenc_unwrap (method and options from
                        unwrap_kwargs: "method" "odv" (default), "sdv" or
                        "schnell", their options, and
                        "reference_unwrap", a single-venc method applied to
                        the higher-venc series first, as Schnell 2017 do),
                        before
                        background correction, as in the dual-venc
                        pipelines of Schnell 2017 and Ma 2019; `venc` for
                        the later steps becomes the effective venc, and
                        unwrap_method applies afterwards only if given as
                        "then" in unwrap_kwargs (default: no further
                        unwrapping). The report gains a "multivenc" entry
                        and the wrap counts are returned as 'wrap_count'.
    background_first  : with extra_encodings, correct the background phase
                        of every series separately before combining them
                        (the offsets of the two encodings need not be equal
                        in velocity units: eddy-current terms scale with the
                        gradient moment and cancel, concomitant-field terms
                        do not), instead of once on the combined series.
                        Which order is better on scanner data is open; both
                        are provided so it can be tested.
    encoding_to_world : optional 3x3 matrix whose columns are the world
                        directions of the three encoded components (e.g. a
                        signed permutation for Siemens through-plane /
                        in-plane encodings). Identity if None, i.e. the
                        encoded components are already (vx, vy, vz).
    background_kwargs, unwrap_kwargs, divergence_kwargs : extra keyword
                        arguments for correct_background_phase (e.g. l1,
                        arto_tau for "arto"), unwrap_velocity (e.g.
                        time_boundary="periodic" for laplacian4d) and the
                        divergence filter.

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
    unwrap_kwargs = dict(unwrap_kwargs or {})
    wrap_count = None

    if extra_encodings:
        try:
            from .multivenc import multivenc_unwrap
        except ImportError:
            from multivenc import multivenc_unwrap
        t0 = time.perf_counter()
        series = [np.asarray(velocity_t, float)] + [np.asarray(e[0], float) for e in extra_encodings]
        vencs = [venc] + [e[1] for e in extra_encodings]
        for sv in series[1:]:
            if sv.shape != series[0].shape:
                raise ValueError(f"extra encoding shape {sv.shape} differs from {series[0].shape}; "
                                 "the series must share grid and frames (see multivenc.resample_frames "
                                 "for frame times, and register separate scans in the host application)")
        # separate scans may have moved between acquisitions: report the
        # magnitude shift so the user knows whether registration is needed
        mv_check = {}
        for n_extra, e in enumerate(extra_encodings):
            if len(e) >= 3 and e[2] is not None:
                try:
                    from .multivenc import encoding_shift
                except ImportError:
                    from multivenc import encoding_shift
                shift, peak = encoding_shift(magnitude_t, e[2])
                mv_check[f"shift_voxels_encoding_{n_extra + 1}"] = [round(float(x), 3) for x in shift]
                if np.abs(shift).max() > 0.5:
                    warnings.warn(f"extra encoding {n_extra + 1} appears shifted by {shift} voxels relative to the "
                                  "low-venc series; register the scans before multi-venc unwrapping")
        mv_method = unwrap_kwargs.pop("method", "odv")
        then = unwrap_kwargs.pop("then", None)
        reference_unwrap = unwrap_kwargs.pop("reference_unwrap", None)
        mv_kwargs = {k: unwrap_kwargs.pop(k) for k in ("correction", "include_time", "use_ivenc",
                                                        "ambiguity_threshold", "max_velocity", "bands",
                                                        "fallback_ratio", "max_wraps") if k in unwrap_kwargs}
        if "fallback_ratio" in mv_kwargs:
            if any(len(e) < 3 for e in extra_encodings):
                raise ValueError("fallback_ratio needs (velocity_t, venc, magnitude_t) for every extra encoding")
            mv_kwargs["magnitudes"] = [np.asarray(magnitude_t, float)] + [np.asarray(e[2], float)
                                                                          for e in extra_encodings]
        if background_first:
            corrected = []
            for sv in series:
                sv_c, info_c = correct_background_phase(sv, magnitude_t, method=background_method,
                                                        lumen_mask=lumen_mask, **(background_kwargs or {}))
                corrected.append(sv_c)
            report["background_per_encoding"] = [info_c["report"]]
            series = corrected
        signal0 = ~noise_mask(magnitude_t, series[0])
        if reference_unwrap is not None:
            # Schnell 2017: the high-venc series is unwrapped with a
            # single-venc method before it serves as the reference
            series = [series[0]] + [unwrap_velocity(sv, vc, voxel_to_world, method=reference_unwrap, mask=signal0)
                                    for sv, vc in zip(series[1:], vencs[1:])]
        velocity_t, wrap_count, mv_info = multivenc_unwrap(series, vencs, method=mv_method, mask=signal0,
                                                           **mv_kwargs)
        if reference_unwrap is not None:
            mv_info["reference_unwrap"] = reference_unwrap
        venc = np.asarray(mv_info["effective_venc"], float)
        report["multivenc"] = {k: val for k, val in mv_info.items() if k not in ("margin", "fallback_mask")}
        report["multivenc"]["background_first"] = bool(background_first)
        report["multivenc"].update(mv_check)
        timing["multivenc"] = time.perf_counter() - t0
        unwrap_method = then
        if background_first:
            background_method = None

    t0 = time.perf_counter()
    if background_method is not None:
        v, info = correct_background_phase(velocity_t, magnitude_t, method=background_method,
                                           lumen_mask=lumen_mask, **(background_kwargs or {}))
        report["background"] = info["report"]
    else:
        v = np.array(velocity_t, float)
        report["background"] = report.get("background_per_encoding", [])
    timing["background"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    signal = ~noise_mask(magnitude_t, v, venc=venc)      # v is still aliased: wrap-invariant rule
    v_corrected = v
    if unwrap_method is not None:
        v = unwrap_velocity(v, venc, voxel_to_world, method=unwrap_method, mask=signal, **unwrap_kwargs)
    report["unwrap"] = {"method": unwrap_method,
                        "voxel_frames_changed": sum(int((np.abs(v[t] - v_corrected[t]) > 1e-9).any(-1).sum())
                                                    for t in range(v.shape[0]))}
    timing["unwrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    noise = noise_mask(magnitude_t, v, venc=venc)
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
    out = {"velocity": v, "noise": noise, "pcmra": angio, "report": report}
    if wrap_count is not None:
        out["wrap_count"] = wrap_count
    return out
