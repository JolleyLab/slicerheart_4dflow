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
    laplacian_unwrap_phase, noise_mask, otsu_threshold, pcmra, preprocess, temporal_unwrap, unwrap_velocity)

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
    # the Otsu default needs no knowledge of the air fraction
    noise_otsu = noise_mask(mag_t, v_t)
    assert not noise_otsu[body].any()
    assert noise_otsu[~body].mean() > 0.95
    # ... whereas the old 30th-percentile default under-masks this air-dominated FOV
    assert noise_mask(mag_t, v_t, magnitude_percentile=30.0)[~body].mean() < 0.5
    angio = pcmra(mag_t, v_t)
    assert angio.shape == shape
    assert np.all(angio >= 0)


def test_otsu_threshold_bimodal():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(2.0, 0.5, 20000), rng.normal(10.0, 1.5, 5000)])
    thr = otsu_threshold(x)
    assert 4.0 < thr < 8.0
    assert ((x < thr) == (x < 6.0)).mean() > 0.995


@pytest.mark.parametrize("phantom_name", ["chest", "bifurcation"])
def test_noise_mask_otsu_on_phantoms(phantom_name, metrics):
    """Air is 36% of the chest FOV and 87% of the bifurcation FOV; the Otsu
    default masks the air in both without flagging lumen."""
    from slicerheart_4dflow.phantom import acquire, bifurcation, chest
    if phantom_name == "chest":
        c = chest(nt=6)
        mag_t, v_t, lumen = c["magnitude"], c["velocity"], c["lumen"]
        air = mag_t.mean(0) < 0.3
    else:
        b = bifurcation(nt=4)
        rng = np.random.default_rng(0)
        acq = [acquire(b["velocity"][t], 1500.0, mask=b["mask"], snr=10.0, rng=rng) for t in range(4)]
        mag_t = np.stack([a["magnitude"] for a in acq]); v_t = np.stack([a["velocity"] for a in acq])
        lumen = b["mask"]; air = ~lumen
    noise = noise_mask(mag_t, v_t)
    lumen_flagged = noise[lumen].mean(); air_flagged = noise[air].mean()
    metrics.record(f"noise_mask/{phantom_name}/otsu_air_flagged", air_flagged, direction="higher")
    metrics.record(f"noise_mask/{phantom_name}/otsu_lumen_flagged", lumen_flagged, direction="lower", atol=0.005)
    assert lumen_flagged < 0.01
    assert air_flagged > 0.95
    assert noise_mask(mag_t, v_t, magnitude_percentile=30.0)[air].mean() <= air_flagged + 1e-9


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
    assert set(out["report"]) >= {"background", "unwrap", "noise_fraction", "divergence", "timing_s"}
    timing = out["report"]["timing_s"]
    assert set(timing) == {"background", "unwrap", "noise_mask", "pcmra", "divergence", "total"}
    assert all(t >= 0 for t in timing.values())
    assert timing["total"] >= sum(t for k, t in timing.items() if k != "total") - 1e-3   # values are rounded


@pytest.mark.cfd
def test_cfd_preprocess_timing(cfd, metrics):
    """Wall-clock cost of each preprocessing step on the aortic CFD (8 frames),
    recorded for reference (direction='record': machine dependent, never
    compared against the baseline)."""
    v_cfd, mask, m = cfd
    nt, venc = 8, 0.5
    rng = np.random.default_rng(0)
    v_true = np.stack([v_cfd * s for s in SCENARIOS["pulsatile"](nt)])
    # static tissue (for the background fit) in a slab away from the lumen
    tissue = np.zeros(mask.shape, bool)
    tissue[:, :, :mask.shape[2] // 6] = True
    tissue &= ~mask
    mag_t = np.where(mask, 60.0, np.where(tissue, 40.0, 2.0))[None] * (1 + 0.05 * rng.normal(size=(nt,) + mask.shape))
    v_meas = v_true + rng.normal(0, 0.05 * venc, v_true.shape) * mask[None, ..., None]
    v_meas[:, ~mask] = rng.normal(0, 0.7 * venc, (nt, (~mask).sum(), 3))
    v_meas[:, tissue] = rng.normal(0, 0.01 * venc, (nt, tissue.sum(), 3))
    out = preprocess(wrap(v_meas, venc), mag_t, venc, m, lumen_mask=mask, divergence_method="fdm", verbose=False)
    for step, seconds in out["report"]["timing_s"].items():
        metrics.record(f"cfd/preprocess_time_s/{step}", seconds, direction="record")
    assert out["report"]["timing_s"]["total"] > 0


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


# method label -> (method, extra keyword arguments of unwrap_velocity)
METHOD_VARIANTS = {
    "laplacian4d_periodic": ("laplacian4d", {"time_boundary": "periodic"}),
}


def evaluate(cfd, v_true, v_meas, v_wrapped, mag_t, method, venc=0.5):
    _, mask, m = cfd
    signal = ~noise_mask(mag_t, v_wrapped)
    method, kwargs = METHOD_VARIANTS.get(method, (method, {}))
    v_u = unwrap_velocity(v_wrapped, venc, m, method=method, mask=signal, **kwargs)
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
    ("temporal", 0), ("laplacian3d", 200), ("laplacian4d", 50), ("laplacian4d_periodic", 50),
    ("laplacian_then_temporal", 0)])
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
            for method in METHODS + tuple(METHOD_VARIANTS):
                t0 = time.perf_counter()
                residual, rmse, signal = evaluate(cfd, v_true, v_meas, v_wrapped, mag_t, method, venc)
                dt = time.perf_counter() - t0
                print(f"  {method:24s} residual wrapped {residual:6d}  RMSE vs truth {rmse:6.2f} cm/s  {dt:5.2f} s")


if __name__ == "__main__":
    main()


def test_otsu_guard_on_field_of_view_without_air():
    """A cropped ROI with no air: Otsu would split tissue from lumen and flag
    73% of the volume (including lumen) as noise; the air-class guard treats
    the whole FOV as body instead, with a warning."""
    from test_background_phase import chest_phantom
    from slicerheart_4dflow.background_phase import static_tissue_candidates, air_threshold
    rng = np.random.default_rng(0)
    # ROI block: tissue magnitude 1.0 with 5% noise, an artery of 1.5, no air
    nt, shape = 6, (16, 32, 32)
    Y, X = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
    lumen_c = np.broadcast_to((X ** 2 + Y ** 2) < 0.3 ** 2, shape).copy()
    mag_c = np.where(lumen_c, 1.5, 1.0)[None] * (1 + 0.05 * rng.normal(size=(nt,) + shape))
    v_c = rng.normal(0, 1.0, (nt,) + shape + (3,))
    v_c[..., 0] += 60.0 * lumen_c[None]
    assert mag_c.mean(0).min() > 0.5
    with pytest.warns(UserWarning, match="no air class"):
        assert air_threshold(mag_c.mean(0)) == -np.inf
    with pytest.warns(UserWarning):
        noise = noise_mask(mag_c, v_c, min_component_voxels=1)
    assert noise[lumen_c].mean() < 0.05 and noise.mean() < 0.1
    with pytest.warns(UserWarning):
        cand = static_tissue_candidates(mag_c, v_c, lumen_mask=lumen_c, edge_erosion_voxels=1)
    assert cand.sum() > 500
    _, mag_t, _, _, _, _ = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    # the uncropped phantom still gets the Otsu split without a warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thr = air_threshold(mag_t.mean(0))
    assert 0.06 < thr < 0.85


def test_temporal_unwrap_max_wraps_bounds_cumulative_count():
    venc = 1.0
    t = np.linspace(0, 1, 60)
    v_true = 6.5 * t[:, None]                        # ramps through 3 wraps
    v_u = temporal_unwrap(wrap(v_true, venc), venc, max_wraps=3)
    np.testing.assert_allclose(v_u, v_true, atol=1e-12)
    v_1 = temporal_unwrap(wrap(v_true, venc), venc, max_wraps=1)
    assert np.abs(np.round((v_1 - wrap(v_true, venc)) / (2 * venc))).max() == 1


def test_preprocess_counts_only_unwrapped_voxel_frames():
    from test_background_phase import chest_phantom
    rng = np.random.default_rng(0)
    v_t, mag_t, _, _, lumen, _ = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    out = preprocess(v_t, mag_t, 1e6, oblique_voxel_to_world(), lumen_mask=lumen, verbose=False)
    assert out["report"]["unwrap"]["voxel_frames_changed"] == 0     # nothing aliased at venc 1e6


# ---------------------------------------------------------------------------
# Periodic time axis (Loecher 2016)
# ---------------------------------------------------------------------------

def test_laplacian_unwrap_periodic_time_is_exact():
    """A 4D phase that is smooth in space and periodic in time is recovered
    exactly with a periodic time axis; the per-axis boundary choice does not
    disturb the spatial (Neumann) solve. Also with a second periodic axis (a
    complex FFT over both instead of the rfft of one) and with either
    estimator."""
    nt, shape, spacing = 12, (10, 12, 14), (1.0, 1.2, 1.0, 0.8)
    grids = np.meshgrid(np.arange(nt), *[np.arange(n) * h for n, h in zip(shape, spacing[1:])], indexing="ij")
    phase = 1.5 * np.sin(2 * np.pi * grids[0] / nt) + 0.25 * grids[1] - 0.2 * grids[2] + 1.5 * np.sin(grids[3] / 4)
    wrapped = np.angle(np.exp(1j * phase))
    for boundary in (None, ("periodic", "neumann", "neumann", "neumann"),
                     ("periodic", "neumann", "periodic", "neumann")):
        for estimator in ("schofield_zhu", "wrapped_difference"):
            unwrapped = laplacian_unwrap_phase(wrapped, spacing, estimator=estimator, boundary=boundary)
            offset = np.round(np.median(unwrapped - phase) / (2 * np.pi)) * 2 * np.pi
            np.testing.assert_allclose(unwrapped - offset, phase, atol=1e-8)
    with pytest.raises(ValueError):
        laplacian_unwrap_phase(wrapped, spacing, boundary="dirichlet")
    with pytest.raises(ValueError):
        unwrap_velocity(np.zeros((2, 4, 4, 4, 3)), 1.0, voxel_to_world_matrix(), method="laplacian4d",
                        time_boundary="mirror")


def plateau_jet(nt=12, shape=(12, 16, 16), centre_frame=0.0, width=5.0):
    """A jet with a flat profile and a wall one voxel wide (too steep for the
    spatial solve to cross, so the wrapped region fills the vessel, the
    failure mode Loecher describes) and a gentle pulse in time centred on
    `centre_frame`, measured circularly over the cycle."""
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = np.sqrt(grids[1] ** 2 + grids[2] ** 2)
    profile = 0.5 * (1 - np.tanh((r - 0.55) / 0.06))
    t = np.arange(nt)
    dt = (t - centre_frame + nt / 2) % nt - nt / 2
    scale = 1.7 * np.exp(-(dt / (width / 2.5)) ** 2)
    v = np.zeros((nt,) + shape + (3,))
    v[..., 0] = scale[:, None, None, None] * profile[None]
    return v


def test_laplacian4d_periodic_time_resolves_wrap_at_cycle_boundary(metrics):
    """Loecher 2016: the time axis of a retrospectively gated cycle is
    periodic, which matters when peak systole falls around the first and
    last frames. With the pulse mid-cycle the Neumann and periodic 4D solves
    both succeed where the 3D solve fails; with the pulse straddling the
    cycle boundary only the periodic solve clears every wrap."""
    venc = 1.0
    m = voxel_to_world_matrix(spacing=(1.0, 1.0, 1.0))
    residual = {}
    for where, centre in (("mid", 6.0), ("boundary", 0.0)):
        v_true = plateau_jet(centre_frame=centre)
        for name, kw in (("laplacian3d", dict(method="laplacian3d")),
                         ("laplacian4d", dict(method="laplacian4d")),
                         ("laplacian4d_periodic", dict(method="laplacian4d", time_boundary="periodic"))):
            v_u = unwrap_velocity(wrap(v_true, venc), venc, m, **kw)
            residual[where, name] = int((np.abs(v_u - v_true) > 1e-6).any(-1).sum())
            metrics.record(f"unwrap_cycle_boundary/{where}/{name}/residual_wrapped", residual[where, name], atol=10)
    assert residual["mid", "laplacian3d"] > 0
    assert residual["mid", "laplacian4d"] == 0 and residual["mid", "laplacian4d_periodic"] == 0
    assert residual["boundary", "laplacian4d"] > 0
    assert residual["boundary", "laplacian4d_periodic"] == 0


def test_preprocess_passes_method_kwargs():
    from test_background_phase import chest_phantom
    rng = np.random.default_rng(0)
    v_t, mag_t, _, _, lumen, _ = chest_phantom(rng, nt=6, shape=(20, 48, 48))
    out = preprocess(v_t, mag_t, 100.0, oblique_voxel_to_world(), lumen_mask=lumen, verbose=False,
                     background_method="arto", background_kwargs={"arto_iterations": 1},
                     unwrap_method="laplacian4d", unwrap_kwargs={"time_boundary": "periodic"})
    assert out["report"]["background"][0]["method"] == "arto"
    assert len(out["report"]["background"][0]["arto"]) == 1
    assert out["report"]["unwrap"]["method"] == "laplacian4d"


@pytest.mark.parametrize("field_of_view", ["random-phase air", "little random-phase air"])
def test_noise_mask_with_venc_keeps_aliased_flow(field_of_view, metrics):
    """On aliased data, the velocity-std rule without venc (97th percentile of the field of view) marks the voxels that
    wrap as noise wherever random-phase air is less than 3% of the field of view (the wrap jumps are 2 venc), which
    excludes them from unwrapping; with venc (wrapped frame-to-frame differences, relative to venc) it keeps them in
    both cases and still marks random-phase air."""
    from slicerheart_4dflow.phantom import alias
    from test_transfer_audit import synthetic_case
    if field_of_view == "random-phase air":
        case = synthetic_case()
    else:
        # the properties of Patient0002: tissue almost as bright as the lumen, low but not random signal around it
        case = synthetic_case(tissue_to_lumen_magnitude=0.87, air_to_lumen_magnitude=0.13, noise_over_venc=0.011)
    velocity, magnitude, venc = case["velocity_raw_encoded_t"], case["magnitude_t"], case["venc_encoded_mm_s"]
    lumen = case["lumen"]
    _, wraps = alias(case["velocity_true_t"], venc)
    wrapped = (wraps != 0).any(-1).any(0) & lumen
    assert wrapped.sum() > 50
    old = noise_mask(magnitude, velocity)
    new = noise_mask(magnitude, velocity, venc=venc)
    metrics.record(f"noise_mask/{field_of_view}/without_venc/wrapped_lumen_marked_fraction", old[wrapped].mean(), direction="record")
    metrics.record(f"noise_mask/{field_of_view}/venc/lumen_marked_fraction", new[lumen].mean(), atol=0.01)
    assert new[wrapped].mean() < 0.01 and new[lumen].mean() < 0.01
    if field_of_view == "random-phase air":
        assert new[magnitude.mean(0) < 0.2].mean() > 0.99
    else:
        assert old[wrapped].mean() > 0.9, "the rule without venc marks aliased voxels (if this changes, update the docs)"
