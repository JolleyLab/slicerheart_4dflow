"""
Velocity unwrapping, noise masking and PC-MRA.

The CFD tests alias the aortic CFD field (peak 0.85 m/s) at a venc of 0.5 m/s
in two cardiac-cycle scenarios:

  pulsatile  scale 0.2 .. 1.0 over 20 frames: aliasing appears mid-cycle, so
             temporal unwrapping can see the jump.
  persistent scale 0.75 .. 1.0: the jet exceeds venc in every frame, so there
             is no temporal jump to detect (stenosis / baffle case).

Metric: residual wrapped voxel-frames inside the lumen (|error| > venc) and
RMSE vs truth, for temporal, laplacian3d, laplacian4d and
laplacian_then_temporal.

Run as a script to print the full comparison table (5% and 15% noise):
    python tests/test_unwrap.py
"""

import time

import numpy as np
import pytest

from conftest import GEOMETRIES, oblique_voxel_to_world
from slicerheart_4dflow.geometry import array_axis_spacing, voxel_to_world_matrix
from slicerheart_4dflow.unwrap import (
    laplacian_unwrap_phase, noise_mask, pcmra, preprocess, temporal_unwrap, unwrap_velocity)

METHODS = ("temporal", "laplacian3d", "laplacian4d", "laplacian_then_temporal")


def wrap(v, venc):
    return (v + venc) % (2 * venc) - venc


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def test_laplacian_unwrap_smooth_phase_is_exact():
    shape, spacing = (20, 24, 28), (1.2, 1.0, 0.8)
    grids = np.meshgrid(*[np.arange(n) * h for n, h in zip(shape, spacing)], indexing="ij")
    phase = 0.25 * grids[0] - 0.2 * grids[1] + 0.3 * np.sin(grids[2] / 4.0) * 6
    wrapped = np.angle(np.exp(1j * phase))
    unwrapped = laplacian_unwrap_phase(wrapped, spacing)
    # equal up to a global multiple of 2 pi
    offset = np.round(np.median(unwrapped - phase) / (2 * np.pi)) * 2 * np.pi
    np.testing.assert_allclose(unwrapped - offset, phase, atol=1e-8)


def test_temporal_unwrap():
    venc = 1.0
    t = np.linspace(0, 1, 30)
    v_true = 2.5 * np.sin(np.pi * t)[:, None]       # up to 2.5 venc, starts at 0
    v_u = temporal_unwrap(wrap(v_true, venc), venc)
    np.testing.assert_allclose(v_u, v_true, atol=1e-12)


def synthetic_jet(nt=10, shape=(16, 16, 16)):
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r2 = grids[1] ** 2 + grids[2] ** 2
    profile = np.exp(-r2 / 0.3)
    scale = 1.6 * np.sin(np.linspace(0, np.pi, nt))
    v_true = np.zeros((nt,) + shape + (3,))
    v_true[..., 0] = scale[:, None, None, None] * profile[None]
    v_true[..., 1] = 0.3 * v_true[..., 0]
    return v_true


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("method", METHODS)
def test_unwrap_velocity_synthetic_jet(method, geometry):
    """Smooth jet aliased in mid-cycle frames; every method must recover it,
    whatever the voxel geometry."""
    venc = 1.0
    v_true = synthetic_jet()
    v_u = unwrap_velocity(wrap(v_true, venc), venc, GEOMETRIES[geometry](), method=method)
    assert (np.abs(v_u - v_true) > 1e-6).sum() == 0


@pytest.mark.parametrize("method", ("laplacian3d", "laplacian4d"))
def test_unwrap_velocity_uses_only_voxel_size(method):
    """Only the voxel size along each axis matters, not orientation or origin."""
    rng = np.random.default_rng(0)
    v = wrap(synthetic_jet() + 0.2 * rng.normal(size=(10, 16, 16, 16, 3)), 1.0)
    m = oblique_voxel_to_world()
    m_aligned = voxel_to_world_matrix(spacing=array_axis_spacing(m)[::-1])
    np.testing.assert_array_equal(unwrap_velocity(v, 1.0, m, method=method),
                                  unwrap_velocity(v, 1.0, m_aligned, method=method))


def test_unwrap_velocity_mask_leaves_outside_unchanged():
    rng = np.random.default_rng(0)
    v = rng.uniform(-1, 1, (4, 8, 8, 8, 3))
    mask = np.zeros((8, 8, 8), bool)
    mask[2:6, 2:6, 2:6] = True
    out = unwrap_velocity(v, 1.0, oblique_voxel_to_world(), mask=mask)
    np.testing.assert_array_equal(out[:, ~mask], v[:, ~mask])


def test_noise_mask_and_pcmra():
    rng = np.random.default_rng(0)
    nt, shape = 8, (20, 20, 20)
    body = np.zeros(shape, bool)
    body[4:16, 4:16, 4:16] = True
    mag_t = np.where(body, 50.0, 1.0)[None] * (1 + 0.05 * rng.normal(size=(nt,) + shape))
    v_t = 0.05 * rng.normal(size=(nt,) + shape + (3,))
    v_t[:, ~body] = rng.uniform(-1, 1, (nt, (~body).sum(), 3))   # random phase in air
    noise = noise_mask(mag_t, v_t, magnitude_percentile=75.0)   # air is 78% of the FOV
    assert noise.shape == shape
    assert not noise[body].any()
    assert noise[~body].mean() > 0.95
    angio = pcmra(mag_t, v_t)
    assert angio.shape == shape
    assert np.all(angio >= 0)


def test_preprocess_pipeline():
    from test_background_phase import chest_phantom
    rng = np.random.default_rng(0)
    venc = 100.0   # cm/s
    v_t, mag_t, _, _, lumen, _ = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    m = oblique_voxel_to_world()
    out = preprocess(v_t, mag_t, venc, m, lumen_mask=lumen, verbose=False)
    # encoded components are a signed permutation of the world components
    encoding_to_world = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out_e = preprocess(v_t, mag_t, venc, m, lumen_mask=lumen, encoding_to_world=encoding_to_world,
                       verbose=False)
    np.testing.assert_allclose(out_e["velocity"], out["velocity"] @ encoding_to_world.T)
    np.testing.assert_array_equal(out_e["noise"], out["noise"])
    out = preprocess(v_t, mag_t, venc, m, lumen_mask=lumen, divergence_method="fdm", verbose=False)
    assert out["velocity"].shape == v_t.shape
    assert out["noise"].shape == v_t.shape[1:4]
    assert out["pcmra"].shape == v_t.shape[1:4]
    assert set(out["report"]) >= {"background", "unwrap", "noise_fraction", "divergence"}


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

SCENARIOS = {
    "pulsatile": lambda nt: 0.2 + 0.8 * np.clip(np.sin(np.linspace(0, np.pi, nt)), 0, 1),
    "persistent": lambda nt: 0.75 + 0.25 * np.clip(np.sin(np.linspace(0, np.pi, nt)), 0, 1),
}


def make_aliased(cfd, scenario, noise_frac, nt=20, venc=0.5, seed=0):
    v_cfd, mask, _ = cfd
    rng = np.random.default_rng(seed)
    mag = np.where(mask, 60.0, 2.0)
    mag_t = mag[None] * (1 + 0.05 * rng.normal(size=(nt,) + mask.shape))
    v_true = np.stack([v_cfd * s for s in SCENARIOS[scenario](nt)])
    v_meas = v_true + rng.normal(0, noise_frac * venc, v_true.shape) * mask[None, ..., None]
    v_meas[:, ~mask] = rng.normal(0, 0.7 * venc, (nt, (~mask).sum(), 3))   # random phase in noise
    return v_true, v_meas, wrap(v_meas, venc), mag_t


def evaluate(cfd, v_true, v_meas, v_wrapped, mag_t, method, venc=0.5):
    _, mask, m = cfd
    signal = ~noise_mask(mag_t, v_wrapped)
    v_u = unwrap_velocity(v_wrapped, venc, m, method=method, mask=signal)
    residual = int((np.abs(v_u - v_meas)[:, mask] > venc).any(-1).sum())
    rmse = float(np.sqrt(np.mean((v_u - v_true)[:, mask] ** 2)) * 100)
    return residual, rmse, signal


@pytest.fixture(scope="module")
def pulsatile_5(cfd):
    return make_aliased(cfd, "pulsatile", 0.05)


@pytest.fixture(scope="module")
def persistent_15(cfd):
    return make_aliased(cfd, "persistent", 0.15)


@pytest.mark.cfd
def test_cfd_noise_mask_keeps_lumen(cfd, persistent_15):
    _, mask, _ = cfd
    v_true, v_meas, v_wrapped, mag_t = persistent_15
    signal = ~noise_mask(mag_t, v_wrapped)
    assert (signal & mask).sum() / mask.sum() > 0.98


@pytest.mark.cfd
@pytest.mark.parametrize("method,max_residual", [
    ("temporal", 0), ("laplacian3d", 800), ("laplacian4d", 300), ("laplacian_then_temporal", 0)])
def test_cfd_unwrap_pulsatile(cfd, pulsatile_5, method, max_residual, metrics):
    residual, rmse, _ = evaluate(cfd, *pulsatile_5, method)
    metrics.record(f"cfd/unwrap_pulsatile/{method}/residual_wrapped", residual, atol=20)
    metrics.record(f"cfd/unwrap_pulsatile/{method}/rmse_cm_s", rmse)
    assert residual <= max_residual
    assert rmse < 3.0


@pytest.mark.cfd
def test_cfd_unwrap_persistent_jet(cfd, persistent_15, metrics):
    """A jet aliased in every frame defeats temporal unwrapping alone."""
    _, mask, _ = cfd
    n_voxel_frames = persistent_15[0].shape[0] * mask.sum()
    res_temporal, rmse_temporal, _ = evaluate(cfd, *persistent_15, "temporal")
    res_combined, rmse_combined, _ = evaluate(cfd, *persistent_15, "laplacian_then_temporal")
    metrics.record("cfd/unwrap_persistent/temporal/residual_wrapped", res_temporal, atol=50)
    metrics.record("cfd/unwrap_persistent/laplacian_then_temporal/residual_wrapped", res_combined, atol=50)
    metrics.record("cfd/unwrap_persistent/laplacian_then_temporal/rmse_cm_s", rmse_combined)
    assert res_combined < 0.3 * res_temporal
    assert res_combined < 0.01 * n_voxel_frames
    assert rmse_combined < rmse_temporal


def main():
    import conftest
    cfd = conftest.load_cfd()
    _, mask, _ = cfd
    nt, venc = 20, 0.5
    print(f"CFD peak {np.sqrt((cfd[0] ** 2).sum(-1)).max():.2f} m/s, venc {venc} m/s, {mask.sum()} lumen voxels")
    for name in SCENARIOS:
        for noise_frac in (0.05, 0.15):
            v_true, v_meas, v_wrapped, mag_t = make_aliased(cfd, name, noise_frac, nt, venc)
            n_aliased = (np.abs(v_meas - v_wrapped)[:, mask] > 1e-9).any(-1).sum()
            print(f"\n{name}, noise {noise_frac:.0%} venc: {n_aliased} aliased lumen voxel-frames "
                  f"of {nt * mask.sum()} ({n_aliased / (nt * mask.sum()):.1%})")
            for method in METHODS:
                t0 = time.perf_counter()
                residual, rmse, signal = evaluate(cfd, v_true, v_meas, v_wrapped, mag_t, method, venc)
                dt = time.perf_counter() - t0
                print(f"  {method:24s} residual wrapped {residual:6d}  RMSE vs truth {rmse:6.2f} cm/s  {dt:5.2f} s")


if __name__ == "__main__":
    main()
