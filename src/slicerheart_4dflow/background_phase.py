"""
Background phase (eddy current) correction for 4D flow MRI, with a choice of
fitting strategy.

Methods (`method=` argument of correct_background_phase)
--------------------------------------------------------
"ols"   Ordinary least squares polynomial fit to static-tissue candidates:
        Walker 1993 (first-order surface), Lankhaar 2005 (polynomial
        surfaces of order 0 to 5 compared). No robustness to candidate
        errors.

"msac"  Fischer et al. MRM 2022 (PMID 35916368). M-estimate SAmple
        Consensus: many minimal-subset fits scored by a truncated squared
        residual, best model refitted on its inliers. Robust to vessels,
        lung and wrap that leak into the candidate set.

"war+msac"  MSAC to strip gross outliers (slow steady flow, leaked vessels),
        then wrap-around rejection (as in "wrls") on the survivors, then an MSAC
        refit. Recommended default: WaR handles folded-in tissue whose offset
        is smooth and close to the true surface (invisible to MSAC); MSAC
        first keeps gross outliers from confusing the WaR mixture model.

"wrls"  Adapted from Pruitt et al. MRM 2019 (doi 10.1002/mrm.27572); this
        package's variant, kept for exploration. Three stacked parts:
          WRLS  weighted regularized least squares, weights = inverse temporal
                variance of velocity (Pruitt: inverse standard deviation),
                ridge penalty on polynomial coefficients (Pruitt: l1);
          OR    outlier rejection for steady (venous) flow: z-score the
                weighted residuals, drop |z| > 1, remove isolated voxels,
                refit. This step is not in Pruitt, who catches steady flow
                with the same residual test as wrap-around;
          WaR   wrap-around rejection: fit the central FOV first, compute
                weighted residuals everywhere, fit a two-component Gaussian
                mixture to them (Pruitt: three components), keep voxels
                within 2 SD of the dominant component (Pruitt: 3 SD), refit
                once (Pruitt: two iterations, first order then second).

"arto"  Pruitt et al. MRM 2019 as published: WRLS + ARTO (automatic rejection
        of temporally invariant outliers). Weights 1/sigma (inverse temporal
        standard deviation of velocity), l1 penalty on the polynomial
        coefficients solved by FISTA, and the ARTO loop: fit the central FOV
        at first order (k = 0), compute the weighted residual W (A x - y)
        over all candidates, fit a three-component Gaussian mixture to it
        with the outer means held at least delta = 2 gamma_c from the central
        mean, keep voxels within tau = 3 gamma_c of the central mean, refit at
        the requested order, repeat; two iterations (K_max = 2) then a final
        fit on the surviving voxels. Steady flow and folded-in tissue are
        both rejected by the residual test. Pruitt validated this on 2D
        phase contrast in 22 adults; the 4D extension (volumetric polynomial,
        per-component fits, wrap-around also through-slab) follows their
        Discussion and ISMRM 2018 abstract 2940 (Pruitt AA, Jin N, Liu Y,
        Simonetti OP, Ahmad R. Background phase correction in the presence
        wrap-around artifact: application in 4D flow imaging. Proc ISMRM
        2018:2940). Pruitt's lambda = 50 is in
        the units of their normalised phase problem and does not transfer;
        see `l1` in correct_background_phase.

Shared front end: static_tissue_candidates() from magnitude, temporal velocity
variation, body-edge erosion, PC-MRA vessel estimate and optional lumen mask.
The offset is fitted on the time-averaged velocity (stationary for a given
gradient waveform) and subtracted from every frame; per_frame=True fits each
frame.

Conventions (see geometry.py): velocity_t (nt, nk, nj, ni, 3), magnitude_t
(nt, nk, nj, ni). No voxel_to_world is needed: a polynomial of a given order
in voxel coordinates is a polynomial of the same order in world coordinates
for any affine voxel-to-world mapping, and each velocity component is fitted
separately, so the components may be the encoded components or world
components. central_fraction refers to the array axes (k, j, i), which are
the acquisition axes. Only numpy and scipy are required.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Polynomial basis
# ---------------------------------------------------------------------------

def _poly_terms(order: int):
    return [(i, j, k)
            for i in range(order + 1)
            for j in range(order + 1 - i)
            for k in range(order + 1 - i - j)]


def design_matrix(coords: np.ndarray, order: int) -> np.ndarray:
    terms = _poly_terms(order)
    # each power of each coordinate once, each term (x^i * y^j) * z^k written
    # as a contiguous row, then one transposed copy: the same entries as the
    # column-by-column build at about half the time
    powers = [[coords[:, a] ** e for e in range(order + 1)] for a in range(3)]
    AT = np.empty((len(terms), coords.shape[0]))
    for n, (i, j, k) in enumerate(terms):
        np.multiply(powers[0][i], powers[1][j], out=AT[n])
        AT[n] *= powers[2][k]
    return np.ascontiguousarray(AT.T)


def normalised_coords(shape) -> np.ndarray:
    grids = np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


# ---------------------------------------------------------------------------
# Static tissue candidates (shared by all methods)
# ---------------------------------------------------------------------------

STATISTICS_SLAB_VOXELS = 1 << 16    # voxels per spatial slab in temporal_statistics (about 1.5 MB per vector buffer)


def _map_slabs(fn, n_slices, step, workers=None):
    """fn(slice) over the slabs of `step` slices of an axis of n_slices, one
    slab per thread (numpy releases the GIL on the slab-sized operations;
    the slabs write disjoint parts of the outputs)."""
    slabs = [slice(k0, k0 + step) for k0 in range(0, n_slices, step)]
    workers = workers or min(len(slabs), os.cpu_count() or 1)
    if workers <= 1 or len(slabs) <= 1:
        for sl in slabs:
            fn(sl)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, slabs))


def temporal_statistics(magnitude_t, velocity_t, pcmra=True):
    """
    Time-averaged magnitude, temporal mean and variance of each velocity
    component and (pcmra=True) the velocity-weighted PC-MRA (Bock 2010)
    sqrt(mean_t(M^2 |v|^2)), the same definition as unwrap.pcmra(). The
    velocity mean and variance are needed by body_mask,
    static_tissue_candidates and correct_background_phase, which share one
    result instead of each reducing the 4D array again.

    The sums run over spatial slabs of STATISTICS_SLAB_VOXELS voxels and,
    within a slab, over the frames in index order, each frame slab promoted
    to float64 on its own: the 4D arrays are never copied or promoted as a
    whole, the full-size temporaries of velocity_t.std(0) and
    velocity_t ** 2 are not formed, and the slab buffers stay in cache.
    numpy reduces axis 0 frame by frame in index order too, and the slabs
    only partition the voxels, so the results equal magnitude_t.mean(0),
    velocity_t.mean(0), velocity_t.var(0) and
    sqrt((magnitude_t ** 2 * speed_t ** 2).mean(0)) of the float64 arrays
    bit for bit (test_temporal_statistics_matches_numpy).

    Returns dict(magnitude (nk, nj, ni), mean (nk, nj, ni, 3),
    var (nk, nj, ni, 3), pcmra (nk, nj, ni) or None).
    """
    velocity_t = np.asarray(velocity_t)
    magnitude_t = np.asarray(magnitude_t)
    nt = len(velocity_t)
    if nt == 0 or len(magnitude_t) != nt:
        raise ValueError("velocity_t and magnitude_t must have the same, non-zero number of frames")
    shape = velocity_t.shape[1:]
    mag, mean, var = np.empty(shape[:3]), np.empty(shape), np.empty(shape)
    pcm = np.empty(shape[:3]) if pcmra else None
    step = max(1, STATISTICS_SLAB_VOXELS // int(np.prod(shape[1:3])))

    def slab(sl):
        v_sum = m_sum = p_sum = None
        for t in range(nt):
            vt = np.asarray(velocity_t[t, sl], float)
            mt = np.asarray(magnitude_t[t, sl], float)
            if t == 0:
                v_sum, m_sum = vt.copy(), mt.copy()
            else:
                v_sum += vt
                m_sum += mt
            if pcmra:
                speed2 = np.square(vt).sum(-1)          # |v|^2
                np.sqrt(speed2, out=speed2)             # speed_t
                np.square(speed2, out=speed2)           # speed_t ** 2
                term = np.square(mt)
                term *= speed2                          # magnitude_t ** 2 * speed_t ** 2
                if t == 0:
                    p_sum = term
                else:
                    p_sum += term
        np.divide(v_sum, nt, out=mean[sl])
        np.divide(m_sum, nt, out=mag[sl])
        if pcmra:
            np.divide(p_sum, nt, out=pcm[sl])
            np.sqrt(pcm[sl], out=pcm[sl])
        # variance: sum_t (v - mean)^2 / nt, as numpy's var computes it
        mean_sl, var_sl = mean[sl], var[sl]
        d = np.empty(mean_sl.shape)
        for t in range(nt):
            np.subtract(velocity_t[t, sl], mean_sl, out=d)
            d *= d
            if t == 0:
                var_sl[...] = d
            else:
                var_sl += d
        var_sl /= nt

    _map_slabs(slab, shape[0], step)
    return dict(magnitude=mag, mean=mean, var=var, pcmra=pcm)


def wrapped_temporal_differences(velocity_t, venc, workers=None):
    """
    The wrap-invariant temporal statistics of velocities that may be aliased,
    per voxel and component:

    first  : RMS of the frame-to-frame differences wrapped into
             [-venc, venc), over the nt - 1 differences of the sequence
             (the velocity-std rule of unwrap.noise_mask with venc);
    second : RMS of the periodic second differences of those wrapped
             differences, wrapped again, divided by sqrt(6) (the phase noise
             of body_mask with venc; the cardiac cycle of a gated
             acquisition is periodic, so the last frame neighbours the first).

    Computed frame by frame on spatial slabs, in threads: the 4D array is not
    copied or promoted as a whole, and no sequence-sized temporary is formed.
    The operations and the order of the sums over frames are those of the
    whole-array form (np.diff or np.roll, the modulo, np.mean over axis 0),
    so the results are the same, bit for bit
    (test_wrapped_temporal_differences_matches_numpy).

    Returns dict(first (nk, nj, ni, 3), second (nk, nj, ni, 3)).
    """
    velocity_t = np.asarray(velocity_t)
    nt = velocity_t.shape[0]
    if nt < 2:
        raise ValueError("wrapped temporal differences need at least two frames")
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    two_venc = 2 * venc
    shape = velocity_t.shape[1:]
    first, second = np.empty(shape), np.empty(shape)
    step = max(1, STATISTICS_SLAB_VOXELS // int(np.prod(shape[1:3])))

    def wrap(d):
        d += venc
        np.remainder(d, two_venc, out=d)
        d -= venc
        return d

    def slab(sl):
        frame = lambda t: np.asarray(velocity_t[t, sl], float)
        v0 = frame(0)
        prev = v0
        d_first = d_prev = None
        s1 = s2 = None
        for t in range(nt):
            nxt = frame(t + 1) if t + 1 < nt else v0
            d = wrap(nxt - prev)                    # wrapped first difference d[t] = W(v[t+1] - v[t]), periodic
            if t < nt - 1:                          # the nt - 1 differences of np.diff
                sq = d * d
                if s1 is None:
                    s1 = sq
                else:
                    s1 += sq
            if t == 0:
                d_first = d
            else:
                e = wrap(d - d_prev)                # second difference W(d[t] - d[t-1]) = row t-1 of roll(d, -1) - d
                e *= e
                if s2 is None:
                    s2 = e
                else:
                    s2 += e
            d_prev = d
            prev = nxt
        e = wrap(d_first - d_prev)                  # the last row of roll(d, -1) - d: W(d[0] - d[nt-1])
        e *= e
        if s2 is None:
            s2 = e
        else:
            s2 += e
        np.divide(s1, nt - 1, out=s1)
        np.sqrt(s1, out=first[sl])
        np.divide(s2, nt, out=s2)
        np.sqrt(s2, out=s2)
        np.divide(s2, np.sqrt(6), out=second[sl])

    _map_slabs(slab, shape[0], step, workers)
    return dict(first=first, second=second)


def otsu_threshold(x, bins=256):
    """
    Otsu's threshold of the values in `x` (the one that maximises the
    between-class variance of the histogram), the standard automatic choice
    for separating air from body in an MR magnitude image.

    Otsu N. A threshold selection method from gray-level histograms. IEEE
    Trans Syst Man Cybern 1979;9(1):62-66. doi:10.1109/TSMC.1979.4310076
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    h, edges = np.histogram(x, bins=bins)
    h = h.astype(float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s0 = np.cumsum(h * centres)
    s1 = s0[-1] - s0
    m0 = s0 / np.maximum(w0, 1e-12)
    m1 = s1 / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    i = int(np.argmax(between[:-1]))
    return float(edges[i + 1])


AIR_CLASS_MAX_RATIO = 0.35      # low Otsu class is "air" when its mean is below this fraction of the high class mean


def air_threshold(magnitude, bins=256):
    """
    Magnitude threshold separating air (and lung) from the body: Otsu's
    threshold when the histogram has an air class, i.e. the mean of the low
    class is below AIR_CLASS_MAX_RATIO times the mean of the high class.
    Otsu always returns a split, so on a field of view without air (a
    cropped ROI volume) it would separate tissue from the lumen instead; in
    that case -inf is returned (nothing is air) with a warning.
    """
    mag = np.asarray(magnitude, float)
    thr = otsu_threshold(mag, bins)
    low, high = mag[mag < thr], mag[mag >= thr]
    if low.size == 0 or high.size == 0 or low.mean() > AIR_CLASS_MAX_RATIO * high.mean():
        warnings.warn("magnitude histogram has no air class (low/high class mean ratio "
                      f"{(low.mean() / high.mean()) if low.size and high.size else float('nan'):.2f}); "
                      "treating the whole field of view as body. Pass an explicit magnitude threshold "
                      "if this is wrong.", stacklevel=3)
        return -np.inf
    return thr


BACKGROUND_SIGMAS = 4.0         # threshold = background mean + this many background stds (capped at the midpoint)


def minimum_error_threshold(x, bins=512, min_class_fraction=0.005):
    """
    Kittler-Illingworth minimum-error threshold of the values in `x`: the
    split at which a two-Gaussian model of the histogram, with each class
    fitted on its own side of the split, has the lowest Bayes classification
    error criterion

        J(T) = 1 + 2 [P1 log s1 + P2 log s2] - 2 [P1 log P1 + P2 log P2].

    This is the two-class model behind Otsu's threshold with the equal
    variance and equal weight assumptions removed; it is the calculator
    3D Slicer exposes as "Kittler-Illingworth" (vtkITKImageThresholdCalculator,
    itk::KittlerIllingworthThresholdCalculator), evaluated here exhaustively
    over the histogram rather than by ITK's iteration. The two classes are
    those of the minimum; the returned threshold is then placed
    BACKGROUND_SIGMAS standard deviations above the low-class mean, capped
    at the midpoint of the two means (see the comment in the code for why
    the equal-posterior point, returned as threshold_bayes, is not used).
    Returns a dict with threshold, threshold_bayes, criterion (the
    minimum), criterion_none (J of the unsplit histogram, 1 + log s^2),
    class weights, means and stds (each (2,), low class first), and
    ashman_d, the separation of the two class means in pooled standard
    deviations (Ashman 1994). Splits leaving a class below
    min_class_fraction of the voxels are not considered.

    Kittler J, Illingworth J. Minimum error thresholding. Pattern Recognit
    1986;19(1):41-47. doi:10.1016/0031-3203(86)90030-0
    Ashman KA, Bird CM, Zepf SE. Detecting bimodality in astronomical
    datasets. Astron J 1994;108:2348-2361. doi:10.1086/117248
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    h, edges = np.histogram(x, bins=bins)
    h = h.astype(float)
    c = 0.5 * (edges[:-1] + edges[1:])
    n = h.sum()
    w0 = np.cumsum(h)                    # low class: bins 0..T
    w1 = n - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    q0 = np.cumsum(h * c ** 2); q1 = q0[-1] - q0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0, m1 = s0 / w0, s1 / w1
        v0 = np.maximum(q0 / w0 - m0 ** 2, 1e-300)
        v1 = np.maximum(q1 / w1 - m1 ** 2, 1e-300)
        P0, P1 = w0 / n, w1 / n
        J = 1 + (P0 * np.log(v0) + P1 * np.log(v1)) - 2 * (P0 * np.log(P0) + P1 * np.log(P1))
    ok = (P0 >= min_class_fraction) & (P1 >= min_class_fraction)
    J = np.where(ok, J, np.inf)
    i = int(np.argmin(J))
    # Threshold placement. The equal-posterior point of the two fitted
    # Gaussians (the root ITK's iteration converges to) is the Bayes rule
    # only when both classes are Gaussian; the high class here is not (tissue
    # plus lumen, dark rims, coil shading), and when it is narrow the
    # crossing lands next to it and a lumen whose score lies between the
    # classes is lost to the background. The background class is close to
    # Gaussian, so the threshold is placed relative to it: BACKGROUND_SIGMAS
    # standard deviations above its mean (misclassifying < 0.01% of it),
    # but never above the midpoint of the two means.
    bayes = float(edges[i + 1])
    a = 0.5 / v0[i] - 0.5 / v1[i]
    b = m1[i] / v1[i] - m0[i] / v0[i]
    cc = (0.5 * m0[i] ** 2 / v0[i] - 0.5 * m1[i] ** 2 / v1[i]
          + 0.5 * np.log(v0[i] / v1[i]) - np.log(P0[i] / P1[i]))
    roots = np.roots([a, b, cc]) if abs(a) > 1e-300 else (np.array([-cc / b]) if abs(b) > 1e-300 else np.array([]))
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots > m0[i]) & (roots < m1[i])]
    if roots.size:
        bayes = float(roots[0])
    thr = float(min(0.5 * (m0[i] + m1[i]), m0[i] + BACKGROUND_SIGMAS * np.sqrt(v0[i])))
    v_all = x.var()
    out = dict(threshold=thr, threshold_bayes=bayes, criterion=float(J[i]),
               criterion_none=float(1 + np.log(max(v_all, 1e-300))),
               weights=np.array([P0[i], P1[i]]), means=np.array([m0[i], m1[i]]),
               stds=np.sqrt([v0[i], v1[i]]))
    out["ashman_d"] = float(abs(m1[i] - m0[i]) / np.sqrt(0.5 * (v0[i] + v1[i])))
    return out


MIN_MODE_SEPARATION = 2.0       # Ashman's D below which the two classes are one population
BACKGROUND_MAX_MAGNITUDE_RATIO = 0.5    # the background class must be at most this fraction as bright as the body
# with venc: the median phase noise of the background class must be at least this fraction of that of random phase
# (0.56 to 0.89 on 15 clinical cases; 0.10 for a split of a bright lumen from dark tissue without air)
BACKGROUND_MIN_RANDOM_PHASE_FRACTION = 0.3
# Noise-free input (synthetic phantoms): voxels whose velocity is constant over time (temporal std at most
# NOISE_FREE_RELATIVE_STD times the largest one) and that are bright (time-averaged magnitude above
# NOISE_FREE_MIN_MAGNITUDE_RATIO times the 90th percentile of the nonzero magnitude) are tissue without noise, not
# zero filling or velocity masked by the scanner in air (both dark). When they are at least NOISE_FREE_MIN_FRACTION of
# the nonzero-magnitude voxels the phase-noise score is degenerate and body_mask falls back to air_threshold.
NOISE_FREE_RELATIVE_STD = 1e-6
NOISE_FREE_MIN_MAGNITUDE_RATIO = 0.25
NOISE_FREE_MIN_FRACTION = 0.05


def body_mask(magnitude_t, velocity_t, min_component_voxels=50, threshold=None, return_score=False, venc=None,
              stats=None):
    """
    DRAFT (for review): replacement for the Otsu magnitude threshold, chosen
    on phantoms only; the guard thresholds and the choice of score are to be
    revisited on clinical data.

    Body (signal) versus background (air, lung, noise) from the signal to
    phase-noise score

        s = log(mean_t |M|) - log(sum_c std_t v_c),

    the log ratio of the time-averaged magnitude to the temporal velocity
    standard deviation. In air the phase is random, so the velocity std is
    about venc/sqrt(3) per component whatever the magnitude, while in tissue
    it is sqrt(2) venc / (pi SNR): the score separates the two classes by
    twice the log-SNR gap, and a multiplicative coil-sensitivity shading or a
    fat-suppressed rim, which pushes dark tissue into the air class of a
    magnitude-only threshold, moves the score much less. The score is split
    into two classes by the Kittler-Illingworth minimum-error criterion
    (minimum_error_threshold; Otsu is its equal-variance, equal-weight
    special case), the calculator 3D Slicer bundles, with the threshold
    placed relative to the background class.

    venc : scalar or (3,) per component; given for velocities that may be
        aliased (before unwrapping). The temporal std of an aliased voxel is
        that of its wrap jumps, which puts aliased flow into the background
        class; with venc the phase noise is instead the RMS of the second
        temporal differences (periodic over the gated cycle) of the
        frame-to-frame differences wrapped into [-venc, venc), wrapped
        again, divided by sqrt(6) (the std of white noise). Wraps do not
        change it, and unlike the std or the first differences it cancels
        most of the pulsatile flow, which on a low-noise field of view
        without random-phase air otherwise scores the lumen as background.
    threshold : None (default) uses minimum_error_threshold on the score; a
        number is a fixed score threshold; a callable f(score) -> float lets
        a host application supply its own calculator (in 3D Slicer,
        vtkITKImageThresholdCalculator with the Kittler-Illingworth method
        on the score volume gives the same split).

    No-background guard: on a field of view without air (a cropped ROI) the
    two classes describe one population, or tissue and a pulsatile lumen
    (whose temporal velocity std is flow, not noise). The split is accepted
    only when it lowers the minimum-error criterion below that of the
    unsplit histogram, the class means are at least MIN_MODE_SEPARATION
    pooled standard deviations apart (Ashman's D), and the low-score class
    is dark: its median magnitude is below BACKGROUND_MAX_MAGNITUDE_RATIO
    times that of the high-score class (air and lung are, a lumen is not),
    and, with venc, its median phase noise is at least
    BACKGROUND_MIN_RANDOM_PHASE_FRACTION times that of random phase (air
    has almost random phase; dark tissue next to a bright lumen, which the
    magnitude ratio does not tell from air, has not).
    Otherwise the whole field of view is body, with a warning. Signal
    islands of fewer than min_component_voxels voxels are background.

    Voxels without data (zero magnitude, or a velocity that is exactly
    constant over time while others vary: the zero filling outside the
    reconstructed field of view) are background and are left out of the
    fit; otherwise they form a class of their own, below air, and the split
    puts the air into the body.

    Noise-free input (synthetic phantoms, where the velocity of static
    tissue is exactly constant over time): the phase-noise score of the
    tissue is then infinite (clipped to the log floor) and the tissue would
    be taken for zero filling. When bright voxels (see
    NOISE_FREE_MIN_MAGNITUDE_RATIO) with constant velocity make up at least
    NOISE_FREE_MIN_FRACTION of the nonzero-magnitude voxels and no explicit
    threshold is given, the score is not used: the body is the magnitude
    above air_threshold (Otsu's threshold of the time-averaged magnitude, or
    all nonzero magnitude on a field of view without an air class), and the
    returned fit holds threshold (the magnitude threshold, -inf when there
    is no air class), method="otsu" and noise_free=True. Real data never has
    bright tissue with exactly constant velocity, so its result is unchanged.

    stats : optional temporal_statistics(magnitude_t, velocity_t) result,
        when the caller already has it; with venc it may also hold
        "wrapped", the wrapped_temporal_differences(velocity_t, venc)
        result.

    Returns (nk, nj, ni) bool, True = body; with return_score also the
    score and the threshold fit. Bock 2007 (noise masking by magnitude and
    velocity std); Kittler 1986; Ashman 1994 (citations in
    minimum_error_threshold).
    """
    velocity_t = np.asarray(velocity_t)
    wrapped_rule = venc is not None and velocity_t.shape[0] > 1
    if stats is None:
        stats = temporal_statistics(magnitude_t, velocity_t, pcmra=False)
    mag = stats["magnitude"]
    if wrapped_rule:
        venc = np.broadcast_to(np.asarray(venc, float), (3,))
        wrapped = stats.get("wrapped") or wrapped_temporal_differences(velocity_t, venc)
        vstd = wrapped["second"].sum(-1)
    else:
        vstd = np.sqrt(stats["var"]).sum(-1)        # velocity_t.std(0).sum(-1)
    floor = 1e-6 * max(float(np.nanmax(mag)), 1e-300)
    score = np.log(np.maximum(mag, floor)) - np.log(np.maximum(vstd, 1e-6 * max(float(np.nanmax(vstd)), 1e-300)))
    data = mag > 0
    if threshold is None and data.any():
        # noise-free input: bright voxels with constant velocity are tissue, the score is degenerate
        constant = vstd <= NOISE_FREE_RELATIVE_STD * float(np.nanmax(vstd))
        bright = mag > NOISE_FREE_MIN_MAGNITUDE_RATIO * float(np.percentile(mag[data], 90))
        n_bright_constant = int(np.count_nonzero(data & constant & bright))
        if n_bright_constant >= max(min_component_voxels, NOISE_FREE_MIN_FRACTION * np.count_nonzero(data)):
            thr = air_threshold(mag)
            body = (mag > thr) & data
            lab, nlab = ndimage.label(body)
            if nlab:
                sizes = ndimage.sum(body, lab, index=np.arange(1, nlab + 1))
                body &= ~np.isin(lab, np.flatnonzero(sizes < min_component_voxels) + 1)
            fit = dict(threshold=float(thr), method="otsu", noise_free=True)
            return (body, score, fit) if return_score else body
    if np.any(vstd[data] > 0):
        data &= vstd > 0
    if not data.any():
        data = np.ones(mag.shape, bool)
    fit = minimum_error_threshold(score[data])
    D = fit["ashman_d"]
    gain = fit["criterion_none"] - fit["criterion"]
    fit["criterion_gain"] = float(gain)
    low = (score <= fit["threshold"])[data]
    mag_data = mag[data]
    ratio = (float(np.median(mag_data[low]) / max(np.median(mag_data[~low]), 1e-300))
             if low.any() and (~low).any() else np.inf)
    fit["magnitude_ratio"] = ratio
    random_phase_fraction = np.inf
    if wrapped_rule and low.any():
        # second differences of random phase: uniform in [-venc, venc), RMS venc / sqrt(3), divided by sqrt(6)
        random_phase_fraction = float(np.median(vstd[data][low]) / (venc / np.sqrt(18)).sum())
        fit["background_random_phase_fraction"] = random_phase_fraction
    if callable(threshold):
        thr = float(threshold(score)); no_background = False
    elif threshold is not None:
        thr = float(threshold); no_background = False
    else:
        thr = fit["threshold"]
        no_background = (gain <= 0 or D < MIN_MODE_SEPARATION or ratio > BACKGROUND_MAX_MAGNITUDE_RATIO
                         or random_phase_fraction < BACKGROUND_MIN_RANDOM_PHASE_FRACTION)
    if no_background:
        warnings.warn("signal to phase-noise score has no background class (class separation "
                      f"D = {D:.1f}, criterion gain {gain:.2f}, low/high class magnitude ratio {ratio:.2f}, "
                      f"low class phase noise {random_phase_fraction:.2f} of random phase); "
                      "treating the whole field of view as body. Pass an explicit threshold if this is wrong.",
                      stacklevel=3)
        body = data.copy()
        fit["threshold"] = -np.inf
    else:
        fit["threshold"] = thr
        body = (score > thr) & data
        lab, nlab = ndimage.label(body)
        if nlab:
            sizes = ndimage.sum(body, lab, index=np.arange(1, nlab + 1))
            body &= ~np.isin(lab, np.flatnonzero(sizes < min_component_voxels) + 1)
    return (body, score, fit) if return_score else body


def static_tissue_candidates(
    magnitude_t, velocity_t, lumen_mask=None,
    magnitude_threshold="gmm", velocity_std_percentile=25.0,
    pcmra_percentile=90.0, edge_erosion_voxels=3, lumen_dilation_voxels=2,
    magnitude_percentile=None, stats=None,
):
    """
    Defaults: body = body_mask(), a Kittler-Illingworth minimum-error split
    of the signal to phase-noise score (log magnitude minus log temporal
    velocity std), which adapts to the air fraction of the field of view and,
    unlike a magnitude-only threshold, is not fooled by coil shading or
    fat-suppressed tissue (see body_mask for the no-air guard);
    magnitude_threshold="otsu" restores Otsu's threshold of the time-averaged
    magnitude (Otsu 1979; see air_threshold), a number is a fixed magnitude
    threshold, and magnitude_percentile= restores the earlier fixed-fraction
    rule (35th percentile suited a thoracic FOV); static = the quietest 25%
    of body voxels by temporal velocity std (Lankhaar 2005 used the 25% of
    pixels with the lowest temporal standard deviation; Medis QFlow 4D uses
    the 25% of the volume with the lowest velocity standard deviation,
    Burkhardt 2023); vessels = top 10% of PC-MRA inside the body; 3-voxel
    erosion of the body rim, where spatial wrap lives (this package's
    heuristic; the published wrap-around handling is the "arto" fitter);
    lumen segmentation dilated by 2 voxels to clear partial-volume edges.

    Lankhaar JW, Hofman MB, Marcus JT, Zwanenburg JJ, Faes TJ,
    Vonk-Noordegraaf A. Correction of phase offset errors in main pulmonary
    artery flow quantification. J Magn Reson Imaging 2005;22(1):73-79.
    PMID 15971181. doi:10.1002/jmri.20361.
    Burkhardt BEU, Kellenberger CJ, Callaghan FM, Valsangiacomo Buechel ER,
    Geiger J. Flow evaluation software for four-dimensional flow MRI: a
    reliability and validation study. Radiol Med 2023;128(10):1225-1235.
    PMID 37620674. doi:10.1007/s11547-023-01697-4.

    stats : optional temporal_statistics(magnitude_t, velocity_t) result,
    when the caller already has it. The statistics are accumulated in
    float64 whatever the input dtype.
    """
    if stats is None or stats.get("pcmra") is None:
        stats = temporal_statistics(magnitude_t, velocity_t)
    mag = stats["magnitude"]
    vstd = np.sqrt(stats["var"]).sum(-1)        # velocity_t.std(0).sum(-1)

    if magnitude_percentile is not None:
        body = mag > np.percentile(mag, magnitude_percentile)
    elif isinstance(magnitude_threshold, str):
        if magnitude_threshold == "gmm":
            body = body_mask(magnitude_t, velocity_t, stats=stats)
        elif magnitude_threshold == "otsu":
            body = mag > air_threshold(mag)
        else:
            raise ValueError("magnitude_threshold must be 'gmm', 'otsu' or a number")
    else:
        body = mag > float(magnitude_threshold)
    body = ndimage.binary_fill_holes(body)
    if edge_erosion_voxels > 0:
        # erode the outer rim only (where spatial wrap lives): internal
        # cavities such as the lungs, which the Otsu threshold excludes, are
        # closed in-plane before eroding so the tissue around them is kept
        outer = np.stack([ndimage.binary_fill_holes(b) for b in body])
        body &= ndimage.binary_erosion(outer, iterations=edge_erosion_voxels)
    if not body.any():
        return body
    quiet = vstd <= np.percentile(vstd[body], velocity_std_percentile)
    # velocity-weighted PC-MRA (Bock 2010), the same definition as unwrap.pcmra()
    pcmra = stats["pcmra"]
    vessel = pcmra > np.percentile(pcmra[body], pcmra_percentile)
    cand = body & quiet & ~vessel
    if lumen_mask is not None:
        lm = np.asarray(lumen_mask, bool)
        if lumen_dilation_voxels > 0:
            lm = ndimage.binary_dilation(lm, iterations=lumen_dilation_voxels)
        cand &= ~lm
    return cand


# ---------------------------------------------------------------------------
# Fitters. Each returns (coef, inlier_mask_over_candidates, extras)
# ---------------------------------------------------------------------------

def fit_ols(A, y, **_):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, np.ones(y.size, bool), {}


def fit_msac(A, y, threshold, n_iter=300, min_samples=None, rng=None, **_):
    rng = rng or np.random.default_rng(0)
    n, p = A.shape
    k = min(min_samples or max(4 * p, 60), n)
    if n <= 2 * p:
        # too few rows to sample subsets (can happen after wrap-around
        # exclusion): plain least squares
        return fit_ols(A, y)
    t2 = threshold ** 2
    best_cost, best_coef = np.inf, None
    r = np.empty(n)
    for _ in range(n_iter):
        pick = rng.choice(n, size=k, replace=False)
        coef, *_ = np.linalg.lstsq(A[pick], y[pick], rcond=None)
        # truncated squared residual, min((y - A c)^2, t^2).sum(), in one buffer
        np.matmul(A, coef, out=r)
        np.subtract(y, r, out=r)
        np.multiply(r, r, out=r)
        np.minimum(r, t2, out=r)
        cost = r.sum()
        if cost < best_cost:
            best_cost, best_coef = cost, coef
    inl = np.abs(y - A @ best_coef) < threshold
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[inl], y[inl], rcond=None)
        inl = np.abs(y - A @ coef) < threshold
    return coef, inl, {}


def _wrls_solve(A, y, w, ridge):
    """argmin sum w (y - A c)^2 + ridge ||c[1:]||^2 (intercept unpenalised)."""
    Aw = A * w[:, None]
    H = Aw.T @ A
    R = ridge * np.eye(A.shape[1])
    R[0, 0] = 0.0
    return np.linalg.solve(H + R, Aw.T @ y)


def _gmm2_1d(x, n_iter=100):
    """Two-component 1D Gaussian mixture by EM. Returns (means, stds, weights)."""
    mu = np.array([np.percentile(x, 40), np.percentile(x, 95)])
    sd = np.array([x.std(), x.std()]) + 1e-9
    pi = np.array([0.8, 0.2])
    # The same operations as
    #   ll = stack([pi_k / sd_k * exp(-0.5 ((x - mu_k) / sd_k)^2)], 1) + 1e-300
    #   r = ll / ll.sum(1); nk = r.sum(0) + 1e-9
    #   mu = (r x).sum(0) / nk; sd = sqrt((r (x - mu)^2).sum(0) / nk) + 1e-9
    # in two preallocated (n, 2) buffers (the column layout keeps the
    # reduction order of the axis-0 sums, so the iterates are unchanged).
    X = x[:, None]
    ll = np.empty((x.size, 2))
    r = np.empty_like(ll)
    tmp = np.empty(x.size)
    for _ in range(n_iter):
        for k in range(2):
            np.subtract(x, mu[k], out=tmp)
            np.divide(tmp, sd[k], out=tmp)
            np.multiply(tmp, tmp, out=tmp)
            np.multiply(tmp, -0.5, out=tmp)
            np.exp(tmp, out=tmp)
            np.multiply(tmp, pi[k] / sd[k], out=ll[:, k])
        ll += 1e-300
        np.divide(ll, ll.sum(1, keepdims=True), out=r)
        nk = r.sum(0) + 1e-9
        np.multiply(r, X, out=ll)
        mu = ll.sum(0) / nk
        np.subtract(X, mu, out=ll)
        np.multiply(ll, ll, out=ll)
        np.multiply(r, ll, out=ll)
        sd = np.sqrt(ll.sum(0) / nk) + 1e-9
        pi = nk / x.size
    return mu, sd, pi


def war_exclusion(A, y, weights, central, ridge=1e-3, war_sigma=2.0):
    """Wrap-around rejection (after Pruitt 2019): central-FOV WRLS fit,
    weighted residuals everywhere, two-component Gaussian mixture, keep the
    dominant component."""
    keep = np.ones(y.size, bool)
    if central is not None and central.sum() > A.shape[1] * 10:
        c0 = _wrls_solve(A[central], y[central], weights[central], ridge)
        r = np.sqrt(weights) * (y - A @ c0)
        mu, sd, pi = _gmm2_1d(r)
        k = int(np.argmax(pi))
        keep &= np.abs(r - mu[k]) < war_sigma * sd[k]
    return keep


def fit_wrls(A, y, weights, ridge=1e-3, central=None, outlier_z=1.0,
             cand_index=None, shape=None, war_sigma=2.0, **_):
    """
    Pruitt-style WRLS + OR + WaR.

    weights    : (n,) inverse temporal variance of y's voxels, normalised.
    central    : (n,) bool, True for candidates in the central FOV. If None,
                 WaR is skipped.
    cand_index : (n,) flat voxel index of each candidate (for morphology).
    shape      : volume shape (for morphology).
    """
    keep = war_exclusion(A, y, weights, central, ridge, war_sigma)

    # --- WRLS + OR: fit, z-score weighted residuals, drop |z| > 1, clean ---
    c = _wrls_solve(A[keep], y[keep], weights[keep], ridge)
    r = np.sqrt(weights) * (y - A @ c)
    z = (r - r[keep].mean()) / (r[keep].std() + 1e-12)
    reject = (np.abs(z) > outlier_z) & keep
    if cand_index is not None and shape is not None:
        vol = np.zeros(int(np.prod(shape)), bool)
        vol[cand_index[reject]] = True
        vol = vol.reshape(shape)
        vol = ndimage.binary_opening(vol, iterations=1)   # drop isolated voxels
        reject = vol.ravel()[cand_index]
    keep &= ~reject
    c = _wrls_solve(A[keep], y[keep], weights[keep], ridge)
    return c, keep, {}


# ---------------------------------------------------------------------------
# Pruitt 2019 as published: WRLS (l1) + ARTO
# ---------------------------------------------------------------------------

def _wrls_l1_solve(A, y, w, l1, n_iter=500, tol=1e-10):
    """
    argmin_x ||w (A x - y)||^2 + l1 ||x[1:]||_1 by FISTA (Beck and Teboulle
    2009, the solver Pruitt cites). w is 1/sigma per row. The columns of A
    are scaled to unit norm for the solve (Pruitt: "each column of matrix A
    has unit norm") and the coefficients mapped back. The intercept is not
    penalised: a constant offset is exactly what the fit must remove.
    """
    Aw = A * w[:, None]
    yw = y * w
    scale = np.linalg.norm(Aw, axis=0)
    scale[scale == 0] = 1.0
    An = Aw / scale
    L = np.linalg.eigvalsh(An.T @ An).max() * 2.0 + 1e-12   # Lipschitz constant of the gradient
    pen = np.full(A.shape[1], l1); pen[0] = 0.0
    x = np.linalg.lstsq(An, yw, rcond=None)[0]              # warm start at the unpenalised solution
    z, t = x.copy(), 1.0
    for _ in range(n_iter):
        g = 2.0 * (An.T @ (An @ z - yw))
        v = z - g / L
        x_new = np.sign(v) * np.maximum(np.abs(v) - pen / L, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + (t - 1.0) / t_new * (x_new - x)
        done = np.abs(x_new - x).max() < tol * max(np.abs(x_new).max(), 1e-12)
        x, t = x_new, t_new
        if done:
            break
    return x / scale


def _gmm3_1d(x, delta_factor=2.0, n_iter=100):
    """
    Three-component 1D Gaussian mixture by EM (Pruitt 2019, eq. 6): left,
    central and right components, the outer means held at least
    delta = delta_factor * gamma_c from the central mean so the fit cannot
    degenerate onto one population. Returns (means, stds, weights) ordered
    left, central, right. Only the central component is used downstream.
    """
    s = x.std() + 1e-9
    med = np.median(x)
    mu = np.array([med - 3 * s, med, med + 3 * s])
    sd = np.array([s, 0.5 * s, s]) + 1e-9
    pi = np.array([0.1, 0.8, 0.1])
    for _ in range(n_iter):
        ll = np.stack([pi[k] / sd[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2)
                       for k in range(3)], 1) + 1e-300
        r = ll / ll.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-9
        pi = nk / x.size
        delta = delta_factor * sd[1]
        mu[0] = min(mu[0], mu[1] - delta)
        mu[2] = max(mu[2], mu[1] + delta)
    return mu, sd, pi


def _order_columns(order_full, order_sub):
    """Column indices of design_matrix(., order_full) that belong to a
    polynomial of order order_sub <= order_full."""
    return np.array([n for n, (i, j, k) in enumerate(_poly_terms(order_full)) if i + j + k <= order_sub])


def fit_arto(A, y, weights, central, order, l1, tau=3.0, delta=2.0, n_iterations=2, init_order=1):
    """
    Pruitt 2019 WRLS + ARTO (their section 2.2, steps 1 to 8).

    A        : (n, p) design matrix of the requested `order`.
    y        : (n,) time-averaged velocity of the candidates.
    weights  : (n,) 1/sigma, inverse temporal standard deviation.
    central  : (n,) bool, candidates in the central FOV, the initial mask S0.
    l1       : l1 penalty in the units of the weighted residual.
    tau      : exclusion strength, voxels with |eps - mu_c| >= tau gamma_c are
               dropped (Pruitt: 3).
    delta    : minimum separation of the outer mixture means from the
               central one, in units of gamma_c (Pruitt: 2 gamma_c).
    n_iterations : K_max (Pruitt: 2). The initial fit (k = 0) is of order
               `init_order` (Pruitt: 1) on S0; later fits use `order`.
    Returns (coef, keep, extras) with extras holding the mixture parameters
    of each iteration.
    """
    n, p = A.shape
    if central is None or central.sum() < 10 * p:
        central = np.ones(n, bool)
    S = central.copy()
    cols0 = _order_columns(order, min(init_order, order))
    history = []
    for k in range(n_iterations):
        if k == 0:
            c = np.zeros(p)
            c[cols0] = _wrls_l1_solve(A[S][:, cols0], y[S], weights[S], l1)
        else:
            c = _wrls_l1_solve(A[S], y[S], weights[S], l1)
        eps = weights * (A @ c - y)                            # Pruitt eq. 5
        mu, sd, pi = _gmm3_1d(eps, delta_factor=delta)        # Pruitt eq. 6
        S = np.abs(eps - mu[1]) < tau * sd[1]                  # Pruitt eq. 7
        history.append(dict(iteration=k, means=mu.tolist(), stds=sd.tolist(), weights=pi.tolist(),
                            n_kept=int(S.sum())))
        if S.sum() < 2 * p:
            S = central.copy()
            break
    c = _wrls_l1_solve(A[S], y[S], weights[S], l1)
    return c, S, {"arto": history}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def correct_background_phase(
    velocity_t, magnitude_t, method="war+msac", order=2, threshold=None,
    candidates=None, lumen_mask=None, per_frame=False, ridge=1e-3,
    central_fraction=(0.65, 0.5, 0.5), n_iter=300, seed=0,
    l1=None, arto_tau=3.0, arto_delta=2.0, arto_iterations=2, **candidate_kwargs,
):
    """
    Subtract a polynomial background phase offset from every frame.

    method    : "ols", "msac", "wrls", "war+msac" or "arto" (see module docstring).
    order     : polynomial order. Default 2, the order used by Medis QFlow
                4D (Burkhardt 2023) and by Pruitt 2019; CAAS fits a first-order
                surface (Burkhardt 2023). Higher orders need ample static
                tissue.
    threshold : MSAC inlier threshold in velocity units; default 2.5 x robust
                sigma (1.4826 x MAD) of residuals from a first-pass linear fit,
                i.e. the usual 2.5-sigma outlier cut.
    central_fraction : fraction of each array axis (k, j, i) treated as the
                "central FOV" for wrap-around rejection. Pruitt 2019 (2D)
                used the central 50% of the field of view in the phase-encode
                direction; the fractions for the other axes are this
                package's choices. Set the frequency-encode axis to 1.0 if
                known.
    ridge     : coefficient penalty of "wrls", 1e-3 on coordinates scaled to
                [-1, 1] (Pruitt used an l1 penalty; at order 2 the choice is
                immaterial). "wrls" rejects |z| > 1 on weighted residuals and
                its wrap-around mixture keeps voxels within 2 SD of the
                dominant component; see the module docstring for how this
                departs from the published method.
    l1        : l1 penalty of "arto" in the units of the weighted residual.
                None (default) uses 1e-3 of the smallest penalty that would
                zero every non-intercept coefficient on the full candidate
                set, i.e. a mild regularisation in the spirit of Pruitt's
                finding that the result is insensitive to small changes in
                lambda at second order. Pruitt's lambda = 50 belongs to their
                normalised phase problem and does not transfer to velocity
                units; tune on held-out cases as they did.
    arto_tau, arto_delta, arto_iterations : the tau (3), delta (2 gamma_c)
                and K_max (2) of Pruitt's ARTO loop; the initial fit of the
                central FOV is first order, as in Pruitt.

    Returns (corrected, info); info holds offset, candidates, inliers, report.
    The arithmetic is in float64; the input arrays are read frame by frame
    (temporal_statistics, the final subtraction) and are not promoted or
    copied as a whole, so a float32 input costs no full-size temporary
    beyond the float64 result.
    """
    velocity_t = np.asarray(velocity_t)
    magnitude_t = np.asarray(magnitude_t)
    nt, nz, ny, nx, _ = velocity_t.shape
    shape = (nz, ny, nx)

    # time-averaged magnitude, temporal mean and variance of the velocity and
    # the PC-MRA, once, for the candidate selection, the weights and the target
    stats = temporal_statistics(magnitude_t, velocity_t, pcmra=candidates is None)
    if candidates is None:
        candidates = static_tissue_candidates(
            magnitude_t, velocity_t, lumen_mask=lumen_mask, stats=stats, **candidate_kwargs)
    candidates = np.asarray(candidates, bool)
    if candidates.sum() < 500:
        raise ValueError(f"only {candidates.sum()} static-tissue candidates; loosen thresholds")

    coords = normalised_coords(shape)
    A_all = design_matrix(coords, order)
    cflat = candidates.ravel()
    A = A_all[cflat]
    cand_index = np.flatnonzero(cflat)

    # inverse temporal variance weights (per component), normalised to mean 1
    var_t = stats["var"][candidates] + 1e-12                    # (n, 3)
    W = 1.0 / var_t
    W /= W.mean(axis=0, keepdims=True)
    # Pruitt's weights are 1/sigma, not 1/sigma^2
    W_sd = 1.0 / np.sqrt(var_t)
    W_sd /= W_sd.mean(axis=0, keepdims=True)

    central = np.ones(shape, bool)
    for ax, f in enumerate(central_fraction):
        n = shape[ax]
        lo, hi = int(round(n * (1 - f) / 2)), int(round(n * (1 + f) / 2))
        sl = [slice(None)] * 3
        sl[ax] = np.r_[0:lo, hi:n]
        central[tuple(sl)] = False
    central = central[candidates]

    rng = np.random.default_rng(seed)
    n_fits = nt if per_frame else 1
    offsets = np.zeros((n_fits,) + shape + (3,))
    inlier_mask = np.zeros(shape + (3,), bool)
    report = []

    for t in range(n_fits):
        target = np.asarray(velocity_t[t], float) if per_frame else stats["mean"]
        for comp in range(3):
            y = target[..., comp][candidates]
            thr = threshold
            if thr is None:
                # MSAC inlier threshold: 2.5 x robust sigma of the residuals
                # of a fit of the requested order. (A lower-order pre-fit
                # measures the unexplained polynomial, not the noise: with an
                # order-1 pre-fit under an order-2 offset the threshold was
                # 2-17x too large and MSAC degenerated to OLS.)
                r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
                thr = 2.5 * 1.4826 * max(np.median(np.abs(r - np.median(r))), 1e-9)

            if method == "ols":
                coef, inl, _ = fit_ols(A, y)
            elif method == "msac":
                coef, inl, _ = fit_msac(A, y, thr, n_iter=n_iter, rng=rng)
            elif method == "wrls":
                coef, inl, _ = fit_wrls(A, y, W[:, comp], ridge=ridge, central=central,
                                        cand_index=cand_index, shape=shape)
            elif method == "war+msac":
                # MSAC first to strip gross outliers, WaR on the survivors to
                # strip folded-in tissue, MSAC refit on what remains.
                _, inl0, _ = fit_msac(A, y, thr, n_iter=n_iter, rng=rng)
                i0 = np.flatnonzero(inl0)
                keep0 = war_exclusion(A[i0], y[i0], W[i0, comp], central[i0], ridge)
                i1 = i0[keep0]
                coef, inl1, _ = fit_msac(A[i1], y[i1], thr, n_iter=n_iter, rng=rng)
                inl = np.zeros(y.size, bool)
                inl[i1[inl1]] = True
            elif method == "arto":
                w = W_sd[:, comp]
                lam = l1
                if lam is None:
                    # 1e-3 of the lasso null penalty on unit-norm weighted columns
                    Aw = A * w[:, None]
                    An = Aw / np.maximum(np.linalg.norm(Aw, axis=0), 1e-12)
                    lam = 1e-3 * np.abs(2.0 * An.T @ (y * w))[1:].max()
                coef, inl, extra = fit_arto(A, y, w, central, order, lam, tau=arto_tau,
                                            delta=arto_delta, n_iterations=arto_iterations)
            else:
                raise ValueError(method)

            offsets[t][..., comp] = (A_all @ coef).reshape(shape)
            inlier_mask[..., comp][candidates] = inl
            resid = (y - A @ coef)[inl]
            entry = dict(frame=t if per_frame else None, component=comp,
                         method=method, order=order,
                         n_candidates=int(y.size), n_inliers=int(inl.sum()),
                         rms_resid=float(np.sqrt(np.mean(resid ** 2))))
            if method == "arto":
                entry["arto"] = extra["arto"]
                entry["l1"] = float(lam)
            report.append(entry)

    # subtract frame by frame into the float64 result (no broadcast temporary
    # and no float64 copy of the input)
    corrected = np.empty(velocity_t.shape, float)
    for t in range(nt):
        np.subtract(velocity_t[t], offsets[t if per_frame else 0], out=corrected[t])
    info = dict(offset=offsets if per_frame else offsets[0],
                candidates=candidates, inliers=inlier_mask, report=report)
    return corrected, info


def residual_static_velocity(velocity_t, static_mask):
    """Residual velocity in static tissue after correction, in the units of
    velocity_t (mean and RMS per component). Gatehouse 2010 treats < 0.6 cm/s
    (6 mm/s) as acceptable for shunt and regurgitant quantification."""
    v = velocity_t.mean(axis=0)[static_mask]
    return dict(mean=v.mean(0).tolist(), rms=np.sqrt((v ** 2).mean(0)).tolist())
