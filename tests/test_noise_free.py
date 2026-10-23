"""
Noise-free input (synthetic phantoms): the velocity of static tissue is exactly constant over time, so its temporal
std is 0, and the magnitude may have no contrast between tissue and air. Background phase correction and the lumen
estimate must still work: the phase-noise score of body_mask is then degenerate and the body is found by the
magnitude (air_threshold), and the velocity noise is estimated as 0.
"""

import warnings

import numpy as np
import pytest
from scipy import ndimage

from slicerheart_4dflow import correct_background_phase, estimate_lumen, pcmra
from slicerheart_4dflow.background_phase import body_mask, static_tissue_candidates
from slicerheart_4dflow.phantom import chest
from slicerheart_4dflow.segmentation import velocity_noise_in_tissue

VOXEL_TO_WORLD = np.diag([2.0, 2.0, 2.0, 1.0])


def noise_free_phantom(air=True, vessel_contrast=True, offset="linear", shape=(24, 40, 40), nt=12):
    """A cylindrical body (or the whole field of view without air) with a straight vessel along k carrying pulsatile
    Poiseuille flow, plus a background phase offset that is constant over time. Air has zero magnitude and zero
    velocity. Returns the measured velocity and magnitude and the truth (velocity without offset, offset, masks)."""
    nk, nj, ni = shape
    k, j, i = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    x, y, z = (i - ni / 2 + 0.5) / ni, (j - nj / 2 + 0.5) / nj, (k - nk / 2 + 0.5) / nk     # about [-0.5, 0.5]
    body = (x ** 2 + y ** 2 < 0.45 ** 2) if air else np.ones(shape, bool)
    r2 = ((i - ni / 2 - 4.0) ** 2 + (j - nj / 2 + 0.5) ** 2) / 5.0 ** 2
    vessel = r2 < 1.0
    wave = 0.3 + 0.7 * np.sin(np.pi * np.arange(nt) / nt) ** 2
    velocity = np.zeros((nt,) + shape + (3,))
    velocity[..., 2] = 600.0 * wave[:, None, None, None] * np.clip(1.0 - r2, 0.0, None)[None]
    off = np.stack([40.0 * x + 20.0 * y, -30.0 * y + 10.0 * z, 20.0 * z - 10.0 * x], axis=-1)
    if offset == "quadratic":
        off += np.stack([60.0 * x ** 2, 40.0 * x * y, -50.0 * y ** 2], axis=-1)
    off *= body[..., None]
    velocity *= body[None, ..., None]
    magnitude = np.where(body, 1.0, 0.0)
    if vessel_contrast:
        magnitude = np.where(vessel, 2.0, magnitude)
    magnitude_t = np.broadcast_to(magnitude, (nt,) + shape).copy()
    return dict(velocity_t=velocity + off[None], magnitude_t=magnitude_t, velocity_true_t=velocity, offset=off,
                body=body, vessel=vessel)


CASES = [dict(), dict(offset="quadratic"), dict(vessel_contrast=False), dict(air=False),
         dict(air=False, vessel_contrast=False, offset="quadratic")]
CASE_IDS = ["linear", "quadratic", "no-vessel-contrast", "no-air", "uniform-magnitude-quadratic"]


@pytest.mark.parametrize("kwargs", CASES, ids=CASE_IDS)
def test_body_mask_falls_back_to_magnitude(kwargs):
    p = noise_free_phantom(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # air_threshold warns on a field of view without air
        body, _, fit = body_mask(p["magnitude_t"], p["velocity_t"], return_score=True)
    assert fit.get("noise_free") is True
    np.testing.assert_array_equal(body, p["body"])
    assert np.isfinite(fit["threshold"]) == kwargs.get("air", True)


@pytest.mark.parametrize("method", ["war+msac", "ols", "wrls"])
@pytest.mark.parametrize("kwargs", CASES, ids=CASE_IDS)
def test_background_correction_removes_the_offset(kwargs, method):
    p = noise_free_phantom(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected, info = correct_background_phase(p["velocity_t"], p["magnitude_t"], method=method)
    assert info["candidates"].sum() >= 500
    assert not np.any(info["candidates"] & p["vessel"])
    # the offset is a quadratic polynomial, so the order-2 fit removes it everywhere in the body (up to the small
    # bias of the ridge penalty of wrls)
    error = np.abs(corrected - p["velocity_true_t"])[:, p["body"]]
    assert error.max() < (1e-3 if method == "wrls" else 1e-6) * np.abs(p["offset"]).max()


@pytest.mark.parametrize("kwargs", CASES, ids=CASE_IDS)
def test_estimate_lumen_grows_to_the_vessel(kwargs):
    p = noise_free_phantom(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected, _ = correct_background_phase(p["velocity_t"], p["magnitude_t"])
        assert velocity_noise_in_tissue(corrected, p["magnitude_t"], lumen_mask=p["vessel"]) < 1e-6
        angiogram = pcmra(p["magnitude_t"], corrected)
        threshold_only = estimate_lumen(angiogram, VOXEL_TO_WORLD)
        result = estimate_lumen(angiogram, VOXEL_TO_WORLD, velocity_t=corrected, magnitude_t=p["magnitude_t"])
    lumen, vessel = result["lumen"], p["vessel"]
    assert result["mask"].sum() >= threshold_only["mask"].sum()
    # the vessel has flow up to its wall, the static tissue around it has none after the correction: the lumen
    # covers the vessel and reaches at most one voxel (partial volume) beyond it
    assert np.all(lumen[vessel])
    assert not np.any(lumen & ~ndimage.binary_dilation(vessel, structure=np.ones((1, 3, 3))))


@pytest.mark.filterwarnings("ignore:signal to phase-noise score")
def test_velocity_noise_without_static_tissue():
    """When the dilated lumen covers the body, the noise is estimated outside the lumen instead of raising."""
    p = noise_free_phantom()
    rng = np.random.default_rng(0)
    velocity_t = p["velocity_true_t"] + rng.normal(0.0, 5.0, p["velocity_true_t"].shape) * p["body"][None, ..., None]
    # the rim of the body is outside the lumen, but within its 2-voxel dilation
    lumen = ndimage.binary_erosion(p["body"], structure=np.ones((1, 3, 3)), border_value=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not static_tissue_candidates(p["magnitude_t"], velocity_t, lumen_mask=lumen,
                                            velocity_std_percentile=100.0, edge_erosion_voxels=0).any()
    with pytest.warns(UserWarning, match="outside the lumen"):
        noise = velocity_noise_in_tissue(velocity_t, p["magnitude_t"], lumen_mask=lumen)
    assert 3.0 < noise < 7.0
    with pytest.raises(ValueError, match="no tissue outside the lumen"):
        velocity_noise_in_tissue(velocity_t, p["magnitude_t"], lumen_mask=np.ones_like(lumen))


def test_noisy_data_uses_the_phase_noise_score():
    """The noise-free fallback does not apply to data with noise."""
    c = chest(rng=np.random.default_rng(1), wrap=False)
    _, _, fit = body_mask(c["magnitude"], c["velocity"], return_score=True)
    assert "noise_free" not in fit
    assert np.isfinite(fit["ashman_d"])
