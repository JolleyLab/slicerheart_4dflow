"""
Background phase (eddy current) correction for 4D flow MRI, with a choice of
fitting strategy.

Methods (`method=` argument of correct_background_phase)
--------------------------------------------------------
"ols"   Walker 1993. Ordinary least squares polynomial fit to static-tissue
        candidates. The field standard; no robustness to candidate errors.

"msac"  Fischer et al. MRM 2022 (doi 10.1002/mrm.29363). M-estimate SAmple
        Consensus: many minimal-subset fits scored by a truncated squared
        residual, best model refitted on its inliers. Robust to vessels,
        lung and wrap that leak into the candidate set.

"war+msac"  MSAC to strip gross outliers (slow steady flow, leaked vessels),
        then Pruitt's wrap-around rejection on the survivors, then an MSAC
        refit. Recommended default: WaR handles folded-in tissue whose offset
        is smooth and close to the true surface (invisible to MSAC); MSAC
        first keeps gross outliers from confusing the WaR mixture model.

"wrls"  Pruitt et al. MRM 2019 (doi 10.1002/mrm.27572). Three stacked parts:
          WRLS  weighted regularized least squares, weights = inverse temporal
                variance of velocity, ridge penalty on polynomial coefficients;
          OR    outlier rejection for steady (venous) flow: z-score the
                weighted residuals, drop |z| > 1, remove isolated voxels, refit;
          WaR   wrap-around rejection: fit the central FOV first, compute
                weighted residuals everywhere, fit a two-component Gaussian
                mixture to them, keep only the dominant component, refit.
        Pruitt used an l1 penalty; ridge is used here for a closed-form solve
        and behaves the same at the low orders (1 to 3) that make sense for
        4D flow.

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

import warnings

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
    A = np.empty((coords.shape[0], len(terms)))
    for n, (i, j, k) in enumerate(terms):
        A[:, n] = coords[:, 0] ** i * coords[:, 1] ** j * coords[:, 2] ** k
    return A


def normalised_coords(shape) -> np.ndarray:
    grids = np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


# ---------------------------------------------------------------------------
# Static tissue candidates (shared by all methods)
# ---------------------------------------------------------------------------

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


def gaussian_mixture_1d(x, n_iter=200, bins=512):
    """
    Two-component Gaussian mixture of the values in `x`, fitted by
    expectation-maximisation on a histogram (Dempster 1977). Returns a dict
    with weights, means, variances (each (2,), sorted by mean), the log
    likelihood of the two-component fit and of a single Gaussian, and
    the equal-posterior threshold between the two components.

    Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete
    data via the EM algorithm. J R Stat Soc B 1977;39(1):1-38.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    h, edges = np.histogram(x, bins=bins)
    c = 0.5 * (edges[:-1] + edges[1:])
    h = h.astype(float)
    n = h.sum()
    mu = np.percentile(x, [20, 80]).astype(float)
    var = np.full(2, x.var() / 4 + 1e-12)
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        ll = -0.5 * (c[:, None] - mu) ** 2 / var - 0.5 * np.log(2 * np.pi * var) + np.log(w)
        m = ll.max(1, keepdims=True)
        r = np.exp(ll - m)
        tot = r.sum(1, keepdims=True)
        r /= tot
        nk = (h[:, None] * r).sum(0) + 1e-12
        mu_new = (h[:, None] * r * c[:, None]).sum(0) / nk
        var = (h[:, None] * r * (c[:, None] - mu_new) ** 2).sum(0) / nk + 1e-12
        w = nk / n
        converged = np.abs(mu_new - mu).max() < 1e-6 * (x.std() + 1e-12)
        mu = mu_new
        if converged:
            break
    loglik2 = float((h * (m[:, 0] + np.log(tot[:, 0]))).sum())
    loglik1 = float(-0.5 * n * (np.log(2 * np.pi * x.var() + 1e-300) + 1))
    order = np.argsort(mu)
    w, mu, var = w[order], mu[order], var[order]
    # equal-posterior crossing between the two means
    grid = np.linspace(mu[0], mu[1], 1024)
    post = [np.log(w[k]) - 0.5 * (grid - mu[k]) ** 2 / var[k] - 0.5 * np.log(var[k]) for k in (0, 1)]
    above = post[1] > post[0]
    threshold = float(grid[np.argmax(above)]) if above.any() else float(mu.mean())
    return dict(weights=w, means=mu, variances=var, loglik2=loglik2, loglik1=loglik1, n=n,
                threshold=threshold)


MIN_MODE_SEPARATION = 2.0       # Ashman's D below which the two mixture components are one population


def body_mask(magnitude_t, velocity_t, min_component_voxels=50, return_score=False):
    """
    DRAFT (for review): replacement for the Otsu magnitude threshold, chosen
    on phantoms only; the thresholds (MIN_MODE_SEPARATION, the BIC test) and
    the choice of score are to be revisited on clinical data.

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
    by a two-component Gaussian mixture (gaussian_mixture_1d) at the
    equal-posterior point, the modern replacement for Otsu's between-class
    criterion (Otsu is the equal-variance, equal-weight special case).

    No-background guard: on a field of view without air (a cropped ROI) the
    two components describe one population; when the mixture does not beat a
    single Gaussian by the Bayesian information criterion, or the modes are
    closer than MIN_MODE_SEPARATION pooled standard deviations (Ashman's D,
    Ashman 1994), the whole field of view is body, with a warning.
    Signal islands of fewer than min_component_voxels voxels are background.

    Returns (nk, nj, ni) bool, True = body; with return_score also the
    score and the mixture fit. Bock 2007 (noise masking by magnitude and
    velocity std); Ashman KM, Bird CM, Zepf SE. Detecting bimodality in
    astronomical datasets. Astron J 1994;108:2348. doi:10.1086/117248
    """
    mag = np.asarray(magnitude_t, float).mean(axis=0)
    vstd = np.asarray(velocity_t, float).std(axis=0).sum(-1)
    floor = 1e-6 * max(float(np.nanmax(mag)), 1e-300)
    score = np.log(np.maximum(mag, floor)) - np.log(np.maximum(vstd, 1e-6 * max(float(np.nanmax(vstd)), 1e-300)))
    fit = gaussian_mixture_1d(score)
    n = fit["n"]
    bic1 = -2 * fit["loglik1"] + 2 * np.log(n)
    bic2 = -2 * fit["loglik2"] + 5 * np.log(n)
    D = abs(fit["means"][1] - fit["means"][0]) / np.sqrt(0.5 * fit["variances"].sum())
    fit["ashman_d"] = float(D)
    fit["bic_gain"] = float(bic1 - bic2)
    if bic2 >= bic1 or D < MIN_MODE_SEPARATION:
        warnings.warn("signal to phase-noise score has no background class (mode separation "
                      f"D = {D:.1f}, BIC gain {bic1 - bic2:.0f}); treating the whole field of view as body. "
                      "Pass an explicit magnitude threshold if this is wrong.", stacklevel=3)
        body = np.ones(mag.shape, bool)
        fit["threshold"] = -np.inf
    else:
        body = score > fit["threshold"]
        lab, nlab = ndimage.label(body)
        if nlab:
            sizes = ndimage.sum(body, lab, index=np.arange(1, nlab + 1))
            body &= ~np.isin(lab, np.flatnonzero(sizes < min_component_voxels) + 1)
    return (body, score, fit) if return_score else body


def static_tissue_candidates(
    magnitude_t, velocity_t, lumen_mask=None,
    magnitude_threshold="gmm", velocity_std_percentile=25.0,
    pcmra_percentile=90.0, edge_erosion_voxels=3, lumen_dilation_voxels=2,
    magnitude_percentile=None,
):
    """
    Defaults: body = body_mask(), a two-component Gaussian mixture split of
    the signal to phase-noise score (log magnitude minus log temporal
    velocity std), which adapts to the air fraction of the field of view and,
    unlike a magnitude-only threshold, is not fooled by coil shading or
    fat-suppressed tissue (see body_mask for the no-air guard);
    magnitude_threshold="otsu" restores Otsu's threshold of the time-averaged
    magnitude (Otsu 1979; see air_threshold), a number is a fixed magnitude
    threshold, and magnitude_percentile= restores the earlier fixed-fraction
    rule (35th percentile suited a thoracic FOV); static = the quietest 25%
    of body voxels by temporal velocity std, the rule used by Medis QFlow 4D
    (Burkhardt 2023) and in the spirit of Walker 1993; vessels = top 10% of
    PC-MRA inside the body; 3-voxel erosion of the body rim, where spatial
    wrap lives (Pruitt 2019); lumen segmentation dilated by 2 voxels to clear
    partial-volume edges.
    """
    mag = magnitude_t.mean(axis=0)
    speed_t = np.sqrt((velocity_t ** 2).sum(-1))
    vstd = velocity_t.std(axis=0).sum(-1)

    if magnitude_percentile is not None:
        body = mag > np.percentile(mag, magnitude_percentile)
    elif isinstance(magnitude_threshold, str):
        if magnitude_threshold == "gmm":
            body = body_mask(magnitude_t, velocity_t)
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
    quiet = vstd <= np.percentile(vstd[body], velocity_std_percentile)
    # Bock 2010 velocity-weighted PC-MRA, the same definition as unwrap.pcmra()
    pcmra = np.sqrt((magnitude_t ** 2 * speed_t ** 2).mean(axis=0))
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
    for _ in range(n_iter):
        pick = rng.choice(n, size=k, replace=False)
        coef, *_ = np.linalg.lstsq(A[pick], y[pick], rcond=None)
        cost = np.minimum((y - A @ coef) ** 2, t2).sum()
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
    for _ in range(n_iter):
        ll = np.stack([pi[k] / sd[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2)
                       for k in range(2)], 1) + 1e-300
        r = ll / ll.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-9
        pi = nk / x.size
    return mu, sd, pi


def war_exclusion(A, y, weights, central, ridge=1e-3, war_sigma=2.0):
    """Pruitt wrap-around rejection: central-FOV WRLS fit, weighted residuals
    everywhere, two-component Gaussian mixture, keep the dominant component."""
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
# Main entry point
# ---------------------------------------------------------------------------

def correct_background_phase(
    velocity_t, magnitude_t, method="war+msac", order=2, threshold=None,
    candidates=None, lumen_mask=None, per_frame=False, ridge=1e-3,
    central_fraction=(0.65, 0.5, 0.5), n_iter=300, seed=0, **candidate_kwargs,
):
    """
    Subtract a polynomial background phase offset from every frame.

    method    : "ols", "msac", "wrls" or "war+msac" (see module docstring).
    order     : polynomial order. Default 2 (Medis, Pruitt in vivo); cvi42 and
                Walker allow up to 3, which needs ample static tissue; CAAS
                uses 1. Pruitt's 4th order is not advisable on pediatric FOVs.
    threshold : MSAC inlier threshold in velocity units; default 2.5 x robust
                sigma (1.4826 x MAD) of residuals from a first-pass linear fit,
                i.e. the usual 2.5-sigma outlier cut.
    central_fraction : fraction of each array axis (k, j, i) treated as the
                "central FOV" for wrap-around rejection. Pruitt 2019 used the
                central 65% through-slice and 50% phase-encode; 50% is also
                applied to the third axis here, which is harmless. Set the
                frequency-encode axis to 1.0 if known.
    ridge     : coefficient penalty, 1e-3 on coordinates scaled to [-1, 1]
                (Pruitt used an l1 penalty; at order 2 the choice is immaterial).
    Outlier rejection uses |z| > 1 on weighted residuals and the wrap-around
    mixture keeps voxels within 2 SD of the dominant component.

    Returns (corrected, info); info holds offset, candidates, inliers, report.
    """
    velocity_t = np.asarray(velocity_t, float)
    magnitude_t = np.asarray(magnitude_t, float)
    nt, nz, ny, nx, _ = velocity_t.shape
    shape = (nz, ny, nx)

    if candidates is None:
        candidates = static_tissue_candidates(
            magnitude_t, velocity_t, lumen_mask=lumen_mask, **candidate_kwargs)
    candidates = np.asarray(candidates, bool)
    if candidates.sum() < 500:
        raise ValueError(f"only {candidates.sum()} static-tissue candidates; loosen thresholds")

    coords = normalised_coords(shape)
    A_all = design_matrix(coords, order)
    cflat = candidates.ravel()
    A = A_all[cflat]
    cand_index = np.flatnonzero(cflat)

    # inverse temporal variance weights (per component), normalised to mean 1
    var_t = velocity_t.var(axis=0)[candidates] + 1e-12          # (n, 3)
    W = 1.0 / var_t
    W /= W.mean(axis=0, keepdims=True)

    central = np.ones(shape, bool)
    for ax, f in enumerate(central_fraction):
        n = shape[ax]
        lo, hi = int(round(n * (1 - f) / 2)), int(round(n * (1 + f) / 2))
        sl = [slice(None)] * 3
        sl[ax] = np.r_[0:lo, hi:n]
        central[tuple(sl)] = False
    central = central[candidates]

    rng = np.random.default_rng(seed)
    targets = velocity_t if per_frame else velocity_t.mean(axis=0)[None]
    offsets = np.zeros(targets.shape)
    inlier_mask = np.zeros(shape + (3,), bool)
    report = []

    for t in range(targets.shape[0]):
        for comp in range(3):
            y = targets[t][..., comp][candidates]
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
            else:
                raise ValueError(method)

            offsets[t][..., comp] = (A_all @ coef).reshape(shape)
            inlier_mask[..., comp][candidates] = inl
            resid = (y - A @ coef)[inl]
            report.append(dict(frame=t if per_frame else None, component=comp,
                               method=method, order=order,
                               n_candidates=int(y.size), n_inliers=int(inl.sum()),
                               rms_resid=float(np.sqrt(np.mean(resid ** 2)))))

    corrected = velocity_t - (offsets if per_frame else offsets[0][None])
    info = dict(offset=offsets if per_frame else offsets[0],
                candidates=candidates, inliers=inlier_mask, report=report)
    return corrected, info


def residual_static_velocity(velocity_t, static_mask):
    """Residual velocity in static tissue after correction, in the units of
    velocity_t (mean and RMS per component). Gatehouse 2010 treats < 0.6 cm/s
    (6 mm/s) as acceptable for shunt and regurgitant quantification."""
    v = velocity_t.mean(axis=0)[static_mask]
    return dict(mean=v.mean(0).tolist(), rms=np.sqrt((v ** 2).mean(0)).tolist())
