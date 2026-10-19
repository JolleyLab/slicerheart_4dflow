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

from slicerheart_4dflow.background_phase import (
    correct_background_phase, design_matrix, normalised_coords, otsu_threshold, residual_static_velocity,
    static_tissue_candidates)

METHODS = ("ols", "msac", "wrls", "war+msac")


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
SCENARIOS = {
    "clean": ((False, False, 3, 90.0), {"ols": 0.12, "msac": 0.12, "wrls": 0.1, "war+msac": 0.1}),
    "steady vein": ((True, False, 3, 90.0), {"ols": 0.1, "msac": 0.1, "wrls": 0.1, "war+msac": 0.1}),
    "wrap band": ((False, True, 3, 90.0), {"ols": None, "msac": None, "wrls": 0.15, "war+msac": 0.15}),
    "steady vein and wrap band": ((True, True, 3, 90.0), {"ols": None, "msac": None, "wrls": 0.15, "war+msac": 0.15}),
    "no edge erosion": ((True, True, 0, 90.0), {"ols": None, "msac": None, "wrls": 0.2, "war+msac": 0.2}),
    "steady vein, no PC-MRA exclusion": ((True, False, 3, 100.0), {"ols": None, "msac": 0.05, "wrls": 0.2, "war+msac": 0.05}),
    "all artifacts, no PC-MRA exclusion": ((True, True, 3, 100.0), {"ols": None, "msac": None, "wrls": None, "war+msac": 0.1}),
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
