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

import time

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .background_phase import air_threshold, otsu_threshold
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


def _laplacian(f, spacing, boundary=None):
    """Discrete Laplacian, N-dimensional, with mirrored (Neumann) or periodic
    edges per axis."""
    boundary = _boundaries(boundary, f.ndim)
    out = np.zeros_like(f)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        fp = np.roll(f, -1, axis=ax)
        fm = np.roll(f, 1, axis=ax)
        if b == "neumann":
            # half-sample mirror (f[-1] = f[0]), matching DCT-II eigenfunctions
            sl = [slice(None)] * f.ndim
            sl[ax] = -1; fp[tuple(sl)] = np.take(f, -1, axis=ax)
            sl[ax] = 0;  fm[tuple(sl)] = np.take(f, 0, axis=ax)
        out += (fp - 2 * f + fm) / h ** 2
    return out


def _laplacian_neumann(f, spacing):
    """Discrete Laplacian with mirrored (Neumann) edges, N-dimensional."""
    return _laplacian(f, spacing, None)


def _poisson_spectral(rhs, spacing, boundary=None):
    """Solve lap(psi) = rhs by a spectral transform per axis: DCT-II on
    Neumann axes, DFT on periodic axes (Loecher 2016 used the DFT on every
    axis; the DCT is the Neumann choice they call ideal in space)."""
    boundary = _boundaries(boundary, rhs.ndim)
    nd = rhs.ndim
    neumann_axes = [ax for ax, b in enumerate(boundary) if b == "neumann"]
    periodic_axes = [ax for ax, b in enumerate(boundary) if b == "periodic"]
    R = rhs
    if neumann_axes:
        R = sfft.dctn(R, type=2, norm="ortho", axes=neumann_axes)
    if periodic_axes:
        R = sfft.fftn(R, axes=periodic_axes, norm="ortho")
    denom = np.zeros(rhs.shape)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        n = rhs.shape[ax]
        k = np.arange(n)
        if b == "neumann":
            lam = (2 * np.cos(np.pi * k / n) - 2) / h ** 2       # DCT eigenvalues of the 1D Laplacian
        else:
            lam = (2 * np.cos(2 * np.pi * k / n) - 2) / h ** 2   # DFT eigenvalues
        shape = [1] * nd; shape[ax] = n
        denom = denom + lam.reshape(shape)
    denom[(0,) * nd] = 1.0
    P = R / denom
    P[(0,) * nd] = 0.0
    if periodic_axes:
        P = sfft.ifftn(P, axes=periodic_axes, norm="ortho")
    if neumann_axes:
        P = sfft.idctn(P, type=2, norm="ortho", axes=neumann_axes)
    return np.real(P) if np.iscomplexobj(P) else P


def _poisson_dct(rhs, spacing):
    """Solve lap(psi) = rhs with Neumann boundaries via DCT-II."""
    return _poisson_spectral(rhs, spacing, None)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _wrapped_laplacian(phase, spacing, boundary=None):
    """Laplacian of the true phase estimated from wrapped first differences:
    sum_axis [W(p[i+1] - p[i]) - W(p[i] - p[i-1])] / h^2, W wrapping to
    (-pi, pi], with mirrored (zero-difference) edges on Neumann axes and
    the wrapped difference across the edge on periodic axes. Exact whenever
    the true phase changes by less than pi per voxel (Itoh's condition)."""
    boundary = _boundaries(boundary, phase.ndim)
    out = np.zeros_like(phase)
    for ax, (h, b) in enumerate(zip(spacing, boundary)):
        if b == "periodic":
            d_fwd = _wrap(np.roll(phase, -1, axis=ax) - phase)       # p[i+1] - p[i], wrapping round
            out += (d_fwd - np.roll(d_fwd, 1, axis=ax)) / h ** 2
            continue
        d = _wrap(np.diff(phase, axis=ax))
        pad = [(0, 0)] * phase.ndim
        pad[ax] = (1, 1)
        d = np.pad(d, pad)                       # zero difference across the mirrored edges
        sl_p = [slice(None)] * phase.ndim; sl_p[ax] = slice(1, None)
        sl_m = [slice(None)] * phase.ndim; sl_m[ax] = slice(None, -1)
        out += (d[tuple(sl_p)] - d[tuple(sl_m)]) / h ** 2
    return out


def laplacian_unwrap_phase(phase, spacing, estimator="schofield_zhu", boundary=None):
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
    Returns unwrapped phase = phase + 2 pi k with integer k.

    The Laplacian is taken along the array axes, so the result does not
    depend on whether the voxel axes are orthogonal in world space.
    """
    boundary = _boundaries(boundary, phase.ndim)
    if estimator == "wrapped_difference":
        L = _wrapped_laplacian(phase, spacing, boundary)
    elif estimator == "schofield_zhu":
        s, c = np.sin(phase), np.cos(phase)
        L = c * _laplacian(s, spacing, boundary) - s * _laplacian(c, spacing, boundary)
    else:
        raise ValueError(estimator)
    psi = _poisson_spectral(L, spacing, boundary)
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
                    mask=None, time_scale=1.0, max_wraps=3, edge_extension_voxels=3,
                    time_boundary="neumann"):
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
    Returns unwrapped velocity_t.
    """
    if time_boundary not in BOUNDARIES:
        raise ValueError(f"time_boundary must be one of {BOUNDARIES}")
    boundary4d = (time_boundary,) + ("neumann",) * 3
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
            u = laplacian_unwrap_phase(p, (time_scale * min(spacing),) + tuple(spacing),
                                       boundary=boundary4d)
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
               min_component_voxels=50, magnitude_percentile=None, venc=None, wrapped_difference_threshold=0.3):
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
    if venc is not None and velocity_t.shape[0] > 1:
        venc = np.broadcast_to(np.asarray(venc, float), (3,))
        difference = np.diff(np.asarray(velocity_t, float), axis=0)
        difference = (difference + venc) % (2 * venc) - venc              # wrapped into [-venc, venc)
        rms = np.sqrt(np.mean(difference ** 2, axis=0)) / venc           # (nk, nj, ni, 3), relative to venc
        noise |= rms.mean(-1) > wrapped_difference_threshold
    elif velocity_std_percentile is not None:
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
    extra_encodings   : DRAFT. Optional list of (velocity_t, venc) of the
                        same field acquired at other (higher) vencs, on the
                        same grid and frames (dual- or triple-venc). The
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
                                 "the series must share grid and frames")
        mv_method = unwrap_kwargs.pop("method", "odv")
        then = unwrap_kwargs.pop("then", None)
        reference_unwrap = unwrap_kwargs.pop("reference_unwrap", None)
        mv_kwargs = {k: unwrap_kwargs.pop(k) for k in ("correction", "include_time", "use_ivenc",
                                                        "ambiguity_threshold", "max_velocity", "bands")
                     if k in unwrap_kwargs}
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
        report["multivenc"] = {k: val for k, val in mv_info.items() if k != "margin"}
        report["multivenc"]["background_first"] = bool(background_first)
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
                        "voxel_frames_changed": int((np.abs(v - v_corrected) > 1e-9).any(-1).sum())}
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
