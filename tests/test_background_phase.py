"""
Background phase correction on a synthetic chest phantom with the failure
modes each fitting method was designed for (no external data needed):

  noise      static tissue noise, lungs and air with low SNR
  steady     a vein with slow constant (5 cm/s) flow, low temporal variance,
             which variance-based candidate selection mistakes for tissue
  wrap       a phase-encode wrap band at the FOV edge where folded-in tissue
             carries the offset of a different location (realistic wrap), not
             just large aliased values

Run as a script to print the full comparison table:
    python tests/test_background_phase.py
"""

import numpy as np
import pytest
from scipy import ndimage

from slicerheart_4dflow.background_phase import (
    air_threshold, body_mask, correct_background_phase, design_matrix, normalised_coords, otsu_threshold,
    residual_static_velocity, static_tissue_candidates)

METHODS = ("ols", "msac", "wrls", "war+msac", "arto")


def chest_phantom(rng, nt=20, shape=(40, 96, 96), steady=False, wrap=False):
    nz, ny, nx = shape
    Z, Y, X = np.meshgrid(np.linspace(-1, 1, nz), np.linspace(-1, 1, ny),
                          np.linspace(-1, 1, nx), indexing="ij")
    body = ((X / 0.85) ** 2 + (Y / 0.98) ** 2) < 1.0
    lungs = (((X - 0.35) ** 2 + (Y / 0.7) ** 2) < 0.22 ** 2) | (((X + 0.35) ** 2 + (Y / 0.7) ** 2) < 0.22 ** 2)
    artery = (X ** 2 + (Y + 0.2) ** 2) < 0.12 ** 2
    vein = ((X - 0.15) ** 2 + (Y + 0.45) ** 2) < 0.10 ** 2
    lumen = artery.copy()

    mag = np.where(body, 1.0, 0.05)
    mag[lungs] = 0.08
    mag[artery] = 1.5
    mag[vein] = 1.3
    mag_t = mag[None] * (1 + 0.05 * rng.normal(size=(nt,) + shape))

    t = np.linspace(0, 1, nt, endpoint=False)
    pulse = 80.0 * np.clip(np.sin(2 * np.pi * t), 0, None)[:, None, None, None] + 10.0
    v_true = np.zeros((nt,) + shape + (3,))
    v_true[..., 0] = pulse * artery[None]
    if steady:
        v_true[..., 0] += 5.0 * vein[None]          # slow constant venous flow
    noise = rng.normal(0, 1.0, v_true.shape)
    noise[:, lungs | ~body] *= 25.0
    v_true += noise

    coords = normalised_coords(shape)
    A = design_matrix(coords, 2)
    coef_true = np.array([
        [0.8, 1.5, -1.0, 0.6, 0.0, 0.9, -0.4, 0.5, 0.3, -0.7],
        [-0.5, 0.7, 1.2, -0.9, 0.3, 0.0, 0.8, -0.6, 0.2, 0.4],
        [0.3, -1.1, 0.6, 1.3, -0.5, 0.4, 0.0, 0.7, -0.8, 0.2],
    ])
    offset_true = np.stack([(A @ c).reshape(shape) for c in coef_true], -1)
    v_meas = v_true + offset_true[None]

    if wrap:
        # Phase-encode wrap along y: tissue from beyond the FOV folds into the
        # outer 12% band on each side. The folded tissue is static but carries
        # the offset polynomial evaluated at its true (out-of-FOV) location,
        # i.e. a smooth but different offset, with normal tissue magnitude.
        band = int(0.12 * ny)
        Yf = Y.copy()
        Yf[:, :band] = -1.0 - (Y[:, :band] + 1.0) - 0.3     # beyond the lower edge
        Yf[:, -band:] = 1.0 + (1.0 - Y[:, -band:]) + 0.3    # beyond the upper edge
        cf = np.stack([Z.ravel(), Yf.ravel(), X.ravel()], 1)
        Af = design_matrix(cf, 2)
        off_fold = np.stack([(Af @ c).reshape(shape) for c in coef_true], -1)
        wrapmask = np.zeros(shape, bool)
        wrapmask[:, :band] = True
        wrapmask[:, -band:] = True
        wrapmask &= body
        v_meas[:, wrapmask] = (noise + off_fold[None])[:, wrapmask]
        mag_t[:, wrapmask] = 0.9 * (1 + 0.05 * rng.normal(size=(nt, wrapmask.sum())))

    # ground-truth static tissue for the residual metric: body minus vessels,
    # lungs, and the wrap band
    true_static = body.copy()
    true_static &= ~(((X - 0.35) ** 2 + (Y / 0.7) ** 2) < 0.26 ** 2)
    true_static &= ~(((X + 0.35) ** 2 + (Y / 0.7) ** 2) < 0.26 ** 2)
    true_static &= ~((X ** 2 + (Y + 0.2) ** 2) < 0.15 ** 2)
    true_static &= ~(((X - 0.15) ** 2 + (Y + 0.45) ** 2) < 0.13 ** 2)
    band = int(0.12 * ny)
    true_static[:, :band] = False
    true_static[:, -band:] = False
    return v_meas, mag_t, offset_true, body, lumen, true_static


def evaluate(method, steady, wrap, erosion=3, pcmra=90.0):
    """Returns (max offset error in body, RMS offset error in body, residual
    static velocity RMS), all in cm/s."""
    rng = np.random.default_rng(3)
    v_meas, mag_t, off_true, body, lumen, true_static = chest_phantom(rng, steady=steady, wrap=wrap)
    corrected, info = correct_background_phase(
        v_meas, mag_t, method=method, order=2, lumen_mask=lumen, edge_erosion_voxels=erosion,
        pcmra_percentile=pcmra)
    err = (info["offset"] - off_true)[body]
    res = residual_static_velocity(corrected, true_static)
    return float(np.abs(err).max()), float(np.sqrt((err ** 2).mean())), float(np.mean(res["rms"]))


# (steady, wrap, erosion, pcmra) -> maximum allowed offset error (cm/s) per method.
# None means the method is not expected to handle that scenario.
# "arto" is Pruitt 2019 as published (3-component mixture, 3 SD cut, two
# iterations from a first-order central fit, 1/sigma weights, l1 penalty).
# On this phantom its 3 SD cut keeps about a fifth of the folded-in wrap-band
# candidates whose folded offset lies close to the surface, so it trails the
# package's 2 SD "wrls"/"war+msac" variants on the wrap scenarios while
# staying under the 0.6 cm/s acceptance; with a steady vein in the
# candidates and no PC-MRA exclusion its residual test does what "wrls"'s
# |z| > 1 step does not.
SCENARIOS = {
    "clean": ((False, False, 3, 90.0), {"ols": 0.12, "msac": 0.12, "wrls": 0.1, "war+msac": 0.1, "arto": 0.2}),
    "steady vein": ((True, False, 3, 90.0), {"ols": 0.1, "msac": 0.1, "wrls": 0.1, "war+msac": 0.1, "arto": 0.2}),
    "wrap band": ((False, True, 3, 90.0), {"ols": None, "msac": None, "wrls": 0.15, "war+msac": 0.15, "arto": 0.35}),
    "steady vein and wrap band": ((True, True, 3, 90.0), {"ols": None, "msac": None, "wrls": 0.15, "war+msac": 0.15, "arto": 0.35}),
    "no edge erosion": ((True, True, 0, 90.0), {"ols": None, "msac": None, "wrls": 0.2, "war+msac": 0.2, "arto": 0.4}),
    "steady vein, no PC-MRA exclusion": ((True, False, 3, 100.0), {"ols": None, "msac": 0.05, "wrls": 0.2, "war+msac": 0.05, "arto": 0.05}),
    "all artifacts, no PC-MRA exclusion": ((True, True, 3, 100.0), {"ols": None, "msac": None, "wrls": None, "war+msac": 0.1, "arto": 0.5}),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_background_phase_methods(scenario, metrics):
    (steady, wrap, erosion, pcmra), limits = SCENARIOS[scenario]
    for method, limit in limits.items():
        if limit is None:
            continue
        max_err, rms_err, residual = evaluate(method, steady, wrap, erosion, pcmra)
        metrics.record(f"background/{scenario}/{method}/max_offset_error_cm_s", max_err, atol=0.005)
        assert max_err < limit, f"{method}: max offset error {max_err:.3f} cm/s >= {limit}"
        # Gatehouse 2010 acceptance: residual static tissue velocity < 0.6 cm/s
        assert residual < 0.6, f"{method}: residual static velocity {residual:.3f} cm/s"


def test_body_threshold_otsu():
    """The body mask behind the static-tissue candidates uses Otsu's
    threshold: on the chest phantom it falls between lung/air (0.05-0.08) and
    tissue (0.9-1.5) whatever the air fraction, while the earlier 35th
    percentile only does so when air happens to be about a third of the FOV."""
    rng = np.random.default_rng(3)
    v_meas, mag_t, _, body, lumen, _ = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    mag = mag_t.mean(0)
    thr = otsu_threshold(mag)
    assert 0.06 < thr < 0.85
    assert ((mag > thr) == (mag > 0.5)).mean() > 0.99           # lung and air below, tissue above
    cand = static_tissue_candidates(mag_t, v_meas, lumen_mask=lumen)
    assert cand.sum() > 500 and not cand[~body].any()
    # same phantom padded with air so that air is 70% of the FOV: Otsu still
    # separates, the fixed percentile lands inside the air distribution
    pad = ((0, 0), (0, 48), (0, 48))
    mag_pad = np.pad(mag, pad, constant_values=0.05)
    mag_pad[:, 48:, :] *= 1 + 0.05 * rng.normal(size=(20, 48, 96))
    mag_pad[:, :48, 48:] *= 1 + 0.05 * rng.normal(size=(20, 48, 48))
    assert 0.06 < otsu_threshold(mag_pad) < 0.85
    assert np.percentile(mag_pad, 35.0) < 0.06
    # explicit overrides still work
    cand_p = static_tissue_candidates(mag_t, v_meas, lumen_mask=lumen, magnitude_percentile=35.0)
    cand_f = static_tissue_candidates(mag_t, v_meas, lumen_mask=lumen, magnitude_threshold=0.5)
    assert cand_p.sum() > 500 and cand_f.sum() > 500


def test_ols_fails_with_wrap():
    """Sanity check that the wrap scenario is a real challenge."""
    max_err, _, _ = evaluate("ols", steady=False, wrap=True)
    assert max_err > 0.3


def test_per_frame_and_explicit_candidates():
    rng = np.random.default_rng(3)
    v_meas, mag_t, off_true, body, lumen, true_static = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    corrected, info = correct_background_phase(
        v_meas, mag_t, method="ols", order=2, candidates=true_static, per_frame=True)
    assert info["offset"].shape == v_meas.shape
    err = (info["offset"].mean(axis=0) - off_true)[body]
    assert np.abs(err).max() < 0.5
    np.testing.assert_allclose(corrected, v_meas - info["offset"])


if __name__ == "__main__":
    for scenario, ((steady, wrap, erosion, pcmra), _) in SCENARIOS.items():
        print(f"\n{scenario}")
        for method in METHODS:
            max_err, rms_err, residual = evaluate(method, steady, wrap, erosion, pcmra)
            print(f"  {method:10s} offset error in body: RMS {rms_err:6.3f}  max {max_err:6.3f} cm/s   "
                  f"residual in true static tissue RMS {residual:.3f} cm/s")


def test_msac_threshold_rejects_slow_flow_contaminant():
    """The MSAC inlier threshold must come from residuals of a fit of the
    requested order: derived from an order-1 pre-fit under an order-2 offset
    it was 2-17x too large and MSAC kept every contaminant (same error as
    OLS). Here 15% of the candidates carry a one-sided bias of 4 sigma of
    the time-averaged noise (slow venous flow the static mask missed)."""
    from slicerheart_4dflow.background_phase import fit_msac
    rng = np.random.default_rng(5)
    nt, shape = 20, (20, 48, 48)
    coords = normalised_coords(shape)
    A = design_matrix(coords, 2)
    coef_true = np.array([0.8, 1.5, -1.0, 0.6, 0.0, 0.9, -0.4, 0.5, 0.3, -0.7])
    offset = (A @ coef_true).reshape(shape)
    sigma = 1.0
    v = offset[None] + rng.normal(0, sigma, (nt,) + shape)
    cand = np.zeros(shape, bool); cand[2:-2, 4:-4, 4:-4] = True
    cand &= rng.random(shape) < 0.3
    contaminated = cand & (rng.random(shape) < 0.15)
    v[:, contaminated] += 4.0 * sigma / np.sqrt(nt)
    v_t = np.zeros((nt,) + shape + (3,)); v_t[..., 0] = v
    mag_t = np.ones((nt,) + shape)
    errs = {}
    for method in ("ols", "msac"):
        _, info = correct_background_phase(v_t, mag_t, method=method, order=2, candidates=cand)
        errs[method] = np.abs(info["offset"][..., 0] - offset)[cand].max()
        if method == "msac":
            kept = info["inliers"][..., 0][contaminated].mean()
    assert kept < 0.3                                   # most contaminants rejected
    assert errs["msac"] < 0.6 * errs["ols"]
    # small candidate sets fall back to least squares instead of raising
    rows = rng.choice(A.shape[0], 30, replace=False)
    coef, inl, _ = fit_msac(A[rows], A[rows] @ coef_true, 0.1)
    assert inl.all() and np.allclose(coef, coef_true, atol=1e-6)


def test_wrls_l1_solve():
    """The FISTA lasso behind "arto": with no penalty it is weighted least
    squares; with a large penalty every non-intercept coefficient is zero
    and the intercept is the weighted mean."""
    from slicerheart_4dflow.background_phase import _wrls_l1_solve
    rng = np.random.default_rng(0)
    coords = normalised_coords((6, 8, 8))
    A = design_matrix(coords, 2)
    c_true = rng.normal(size=A.shape[1])
    w = rng.uniform(0.5, 2.0, A.shape[0])
    y = A @ c_true + 0.05 * rng.normal(size=A.shape[0]) / w
    c_wls = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    np.testing.assert_allclose(_wrls_l1_solve(A, y, w, 0.0, n_iter=2000, tol=1e-13), c_wls, atol=1e-6)
    c_big = _wrls_l1_solve(A, y, w, 1e9)
    assert np.abs(c_big[1:]).max() < 1e-9
    np.testing.assert_allclose(c_big[0], (w ** 2 * y).sum() / (w ** 2).sum(), rtol=1e-6)


def test_gmm3_separates_populations():
    """The three-component mixture of ARTO: the central component tracks the
    main population when outlier populations sit on either side, and the
    outer means are held at least delta from the central mean when there are
    no outliers at all."""
    from slicerheart_4dflow.background_phase import _gmm3_1d
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0.0, 1.0, 20000), rng.normal(-6.0, 1.0, 1500), rng.normal(5.0, 1.0, 1000)])
    mu, sd, pi = _gmm3_1d(x)
    assert abs(mu[1]) < 0.1 and abs(sd[1] - 1.0) < 0.1 and pi[1] > 0.85
    assert mu[0] < -5.0 and mu[2] > 4.0
    x0 = rng.normal(0.0, 1.0, 20000)
    mu0, sd0, _ = _gmm3_1d(x0)
    assert mu0[0] <= mu0[1] - 2 * sd0[1] and mu0[2] >= mu0[1] + 2 * sd0[1]
    assert abs(sd0[1] - 1.0) < 0.15


def test_arto_rejects_steady_flow_and_wrap_without_prior_masks(metrics):
    """Pruitt's claim, on the chest phantom: the residual test removes both a
    steady vein and folded-in tissue from the candidates with no PC-MRA
    exclusion and no rim erosion to help, and the correction stays under the
    Gatehouse 0.6 cm/s acceptance. Also records what each ARTO iteration
    excluded."""
    rng = np.random.default_rng(3)
    v_meas, mag_t, off_true, body, lumen, true_static = chest_phantom(rng, steady=True, wrap=True)
    ny = v_meas.shape[2]; band = int(0.12 * ny)
    wrapmask = np.zeros(body.shape, bool); wrapmask[:, :band] = True; wrapmask[:, -band:] = True; wrapmask &= body
    corrected, info = correct_background_phase(
        v_meas, mag_t, method="arto", order=2, lumen_mask=lumen, edge_erosion_voxels=0, pcmra_percentile=100.0)
    err = np.abs(info["offset"] - off_true)[body].max()
    metrics.record("background/arto_no_prior_masks/max_offset_error_cm_s", err, atol=0.005)
    assert err < 0.6
    assert np.mean(residual_static_velocity(corrected, true_static)["rms"]) < 0.6
    # rejection concentrates in the folded-in tissue: half of the wrap-band
    # candidates go (the half whose folded offset is far from the surface)
    # against well under 2% of true static tissue
    cand, inl = info["candidates"], info["inliers"][..., 0]
    rejected_wrap = 1 - (inl & wrapmask).sum() / (cand & wrapmask).sum()
    rejected_static = 1 - (inl & true_static).sum() / (cand & true_static).sum()
    assert rejected_wrap > 0.4 and rejected_static < 0.02
    hist = info["report"][0]["arto"]
    assert len(hist) == 2 and all(h["n_kept"] > 0 for h in hist)
    assert info["report"][0]["l1"] > 0


def _acquired_chest(snr=12.0, shading=0.0, dark_rim=False, nt=8, seed=3):
    """Chest geometry re-acquired through the complex phase-contrast model,
    with an optional anterior-to-posterior coil-sensitivity shading (1 at
    the anterior wall falling to 1 - shading at the posterior wall) and an
    optional dark subcutaneous rim (fat suppression). Returns magnitude_t,
    velocity_t and the body-minus-lungs truth."""
    from slicerheart_4dflow.phantom import acquire, chest
    rng = np.random.default_rng(seed)
    c = chest(nt=nt, rng=rng, wrap=False)
    nk, nj, ni = c["body"].shape
    mag = np.where(c["body"], 1.0, 0.03)
    mag[c["lungs"]] = 0.08; mag[c["lumen"]] = 1.5; mag[c["vein"]] = 1.3
    if dark_rim:
        mag[c["body"] & ~ndimage.binary_erosion(c["body"], iterations=4)] = 0.3
    Y = np.linspace(-1, 1, nj)[None, :, None]
    mag = mag * (1.0 - shading * (Y + 1) / 2)
    acq = [acquire(c["velocity_true"][t], 1500.0, magnitude=mag, snr=snr, rng=rng) for t in range(nt)]
    return (np.stack([a["magnitude"] for a in acq]), np.stack([a["velocity"] for a in acq]),
            c["body"] & ~c["lungs"])


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


@pytest.mark.parametrize("shading,snr,dark_rim,otsu_max", [
    (0.0, 12.0, False, 1.01),     # easy: both perfect
    (0.7, 5.0, False, 0.80),      # coil shading at low SNR: Otsu loses the posterior body
    (0.0, 12.0, True, 0.85),      # fat-suppressed rim: Otsu drops it with the air
])
def test_body_mask_beats_otsu_under_shading(shading, snr, dark_rim, otsu_max, metrics):
    """body_mask (Gaussian mixture on log magnitude / velocity std) keeps the
    whole body where a magnitude-only Otsu threshold cuts dark tissue off."""
    mag_t, v_t, truth = _acquired_chest(snr=snr, shading=shading, dark_rim=dark_rim)
    body, score, fit = body_mask(mag_t, v_t, return_score=True)
    body = ndimage.binary_fill_holes(body)
    otsu = ndimage.binary_fill_holes(mag_t.mean(0) > air_threshold(mag_t.mean(0)))
    d_gmm, d_otsu = _dice(body, truth), _dice(otsu, truth)
    name = f"shading{shading}_snr{snr:g}" + ("_rim" if dark_rim else "")
    metrics.record(f"body_mask/{name}/dice", d_gmm, direction="higher", atol=0.005)
    metrics.record(f"body_mask/{name}/otsu_dice", d_otsu, direction="record")
    assert d_gmm > 0.95
    assert d_otsu < otsu_max
    assert d_gmm >= d_otsu - 1e-9
    assert fit["ashman_d"] > 2.0 and fit["bic_gain"] > 0
    assert score.shape == truth.shape


def test_body_mask_guard_without_background():
    """A cropped ROI with no air: the mixture components are one population
    (Ashman's D < 2), so the whole field of view is body, with a warning."""
    rng = np.random.default_rng(0)
    nt, shape = 6, (16, 32, 32)
    Y, X = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
    lumen = np.broadcast_to((X ** 2 + Y ** 2) < 0.3 ** 2, shape).copy()
    mag_t = np.where(lumen, 1.5, 1.0)[None] * (1 + 0.05 * rng.normal(size=(nt,) + shape))
    v_t = rng.normal(0, 1.0, (nt,) + shape + (3,))
    v_t[..., 0] += 60.0 * lumen[None]
    with pytest.warns(UserWarning, match="no background class"):
        body = body_mask(mag_t, v_t)
    assert body.all()
    # the same ROI inside air is split
    mag_p = np.pad(mag_t, ((0, 0), (0, 0), (16, 16), (16, 16)), constant_values=0.03)
    mag_p *= 1 + 0.05 * rng.normal(size=mag_p.shape)
    v_p = rng.uniform(-1500, 1500, mag_p.shape + (3,))
    v_p[:, :, 16:48, 16:48] = v_t
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        body_p = body_mask(mag_p, v_p)
    truth = np.zeros(mag_p.shape[1:], bool); truth[:, 16:48, 16:48] = True
    assert _dice(body_p, truth) > 0.99
