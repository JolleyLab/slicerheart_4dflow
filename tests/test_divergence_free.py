"""
Divergence-free filtering: exactness of the divergence operator and of the
Helmholtz-Hodge split on analytic fields for arbitrary voxel-to-world
mappings, invariance to the world coordinate system, and comparison of fdm,
relaxed, leray and dfw on the 4DFlowNet aortic CFD ground truth (0.59 mm, 56k
lumen voxels) with Gaussian noise at 10% of a 1.5 m/s venc.

Run as a script to print the full comparison table (5/10/20% noise, more
dfw variants):
    python tests/test_divergence_free.py [--lr]
"""

import time

import numpy as np
import pytest
from scipy import ndimage

from conftest import GEOMETRIES, WORLD_TRANSFORMS, oblique_voxel_to_world, world_transform
from slicerheart_4dflow.divergence_free import (
    build_divergence_operator, divergence_free_filter, divergence_free_sequence, divergence_stats,
    leray_split)
from slicerheart_4dflow.geometry import voxel_grid_world_points, voxel_spacing, voxel_to_world_matrix

VENC = 1.5  # m/s


# ---------------------------------------------------------------------------
# Analytic fields
# ---------------------------------------------------------------------------

def _ball_mask(shape, radius_fraction=0.7):
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    return sum(g ** 2 for g in grids) < radius_fraction ** 2


def _centred_world_points(shape, voxel_to_world):
    p = voxel_grid_world_points(shape, voxel_to_world)
    return p - p.reshape(-1, 3).mean(0)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_divergence_operator_linear_field_is_exact(geometry):
    """Central and one-sided differences are exact for a linear field, for
    any affine voxel-to-world mapping."""
    m = GEOMETRIES[geometry]()
    shape = (12, 14, 16)
    mask = _ball_mask(shape)
    x = voxel_grid_world_points(shape, m)
    A = np.array([[2.0, 1.0, -0.3], [0.4, -0.5, 1.0], [-1.0, 0.7, 3.0]])   # div = trace = 4.5
    v = x @ A.T + (0.1, -0.2, 0.3)
    D = build_divergence_operator(mask, m)
    d = D @ np.concatenate([v[..., k][mask] for k in range(3)])
    np.testing.assert_allclose(d, 4.5, atol=1e-9)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_projection_methods_on_analytic_field(geometry):
    """Every method that claims to denoise must get closer to the truth, not
    only reduce the divergence (a wrong projection can lower the divergence
    while moving away from the truth). The domain is a tube about 20 voxels
    across: dfw needs structure larger than its wavelet scales, and on a
    16^3 ball or a 10-voxel tube it made the field worse than the noisy
    input (0.32-0.39 vs 0.30), which the old assertion (divergence reduced)
    did not catch."""
    m = GEOMETRIES[geometry]()
    shape = (40, 40, 40)
    p = _centred_world_points(shape, m)
    axis = np.array([0.3, -1.0, 0.5]); axis /= np.linalg.norm(axis)
    radial = p - (p @ axis)[..., None] * axis
    mask = np.linalg.norm(radial, axis=-1) < 0.5 * np.linalg.norm(p, axis=-1).max() / np.sqrt(3)
    rng = np.random.default_rng(0)
    # solid-body rotation about the tube axis plus axial flow: divergence-free
    v_true = np.cross(axis, p) + 2.0 * axis
    v_noisy = v_true + 0.3 * rng.normal(size=v_true.shape)
    before = divergence_stats(v_noisy, mask, m)["div_rms"]
    err_noisy = np.sqrt(np.mean((v_noisy - v_true)[mask] ** 2))
    for method in ("fdm", "relaxed", "leray", "dfw"):
        out, info = divergence_free_filter(v_noisy, mask, m, method=method)
        np.testing.assert_array_equal(out[~mask], v_noisy[~mask])
        err = np.sqrt(np.mean((out - v_true)[mask] ** 2))
        if method == "leray":
            # alternating projections on a truncated box: no accuracy claim
            continue
        assert info["div_rms_after"] < before, method
        assert err < err_noisy, (method, err, err_noisy)
    out, info = divergence_free_filter(v_noisy, mask, m, method="fdm")
    assert info["div_rms_after"] < 1e-6 * before


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_leray_split_recovers_components(geometry):
    """A gradient field is curl-free; a rotation field is divergence-free."""
    m = GEOMETRIES[geometry]()
    shape = (32, 32, 32)
    p = _centred_world_points(shape, m)
    s = shape[0] * voxel_spacing(m).min() / 10                  # Gaussian width, decays within the box
    g = np.exp(-(p ** 2).sum(-1) / (2 * s ** 2))
    grad = -p / s ** 2 * g[..., None]                            # analytic gradient: curl-free
    rot = np.cross([0.3, -1.0, 0.5], p) * g[..., None] / s       # divergence-free
    df, cf = leray_split(grad + rot, m)
    assert np.abs(df - rot).max() < 1e-4 * np.abs(rot).max()
    assert np.abs(cf - grad).max() < 1e-4 * np.abs(grad).max()


@pytest.mark.parametrize("world", list(WORLD_TRANSFORMS))
@pytest.mark.parametrize("method,kw", [("fdm", {}), ("relaxed", {}), ("leray", {})])
def test_world_coordinate_system_invariance(method, kw, world):
    """Moving the world coordinate system (rotation, reflection, translation)
    rotates the result with it. (dfw is not exactly invariant, because the
    wavelet coefficients are thresholded per world component; see the CFD
    test.)"""
    m = oblique_voxel_to_world()
    T = WORLD_TRANSFORMS[world]()
    R = T[:3, :3]
    shape = (14, 16, 18)
    mask = _ball_mask(shape)
    rng = np.random.default_rng(2)
    v = np.cross([0.3, -1.0, 0.5], _centred_world_points(shape, m)) + 0.3 * rng.normal(size=shape + (3,))
    out, info = divergence_free_filter(v, mask, m, method=method, **kw)
    out_t, info_t = divergence_free_filter(v @ R.T, mask, T @ m, method=method, **kw)
    np.testing.assert_allclose(out_t, out @ R.T, atol=1e-6 * np.abs(v).max())
    np.testing.assert_allclose(info_t["div_rms_before"], info["div_rms_before"], rtol=1e-9)


def test_sequence_static_and_dynamic_mask():
    m = oblique_voxel_to_world()
    shape = (10, 10, 10)
    mask = _ball_mask(shape)
    rng = np.random.default_rng(1)
    v_t = rng.normal(size=(3,) + shape + (3,))
    out_s, infos_s = divergence_free_sequence(v_t, mask, m, method="fdm")
    out_d, infos_d = divergence_free_sequence(v_t, np.stack([mask] * 3), m, method="fdm")
    np.testing.assert_allclose(out_s, out_d, atol=1e-6)
    assert len(infos_s) == 3


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

def slice_flows(v, mask, voxel_to_world, axis=0, n=5):
    """Flow through axial slices of an axis-aligned volume (mL/s for m/s, mm).
    World component 2 - axis is along array axis `axis`."""
    spacing_kji = np.linalg.norm(voxel_to_world[:3, :3], axis=0)[::-1]
    idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    picks = np.linspace(idx[0] + 3, idx[-1] - 3, n).astype(int)
    area = np.prod([spacing_kji[a] for a in range(3) if a != axis]) * 1e-6
    q = []
    for k in picks:
        sl = [slice(None)] * 3
        sl[axis] = k
        q.append((v[tuple(sl)][..., 2 - axis] * mask[tuple(sl)]).sum() * area * 1e6)
    return np.array(q)


def metrics_of(v_ref, v, mask, D, voxel_to_world, q_ref):
    """RMSE (cm/s), mean angular error (deg), divergence RMS (/s), flow error (mL/s)."""
    e = (v - v_ref)[mask]
    rmse = np.sqrt(np.mean(e ** 2)) * 100
    speed = np.sqrt((v_ref[mask] ** 2).sum(-1))
    sel = speed > 0.1 * speed.max()
    a, b = v_ref[mask][sel], v[mask][sel]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    vv = np.concatenate([v[..., k][mask] for k in range(3)])
    div = np.sqrt(np.mean((D @ vv) ** 2)) * 1000      # per mm -> per m: /s
    qerr = np.abs(slice_flows(v, mask, voxel_to_world) - q_ref).mean()
    return rmse, ang, div, qerr


def run_method(v_noisy, mask, m, D, method, kw):
    if method is None:
        return v_noisy
    if method == "dfw+fdm":
        v, _ = divergence_free_filter(v_noisy, mask, m, method="dfw", **kw)
        v, _ = divergence_free_filter(v, mask, m, method="fdm", D=D)
        return v
    v, _ = divergence_free_filter(v_noisy, mask, m, method=method,
                                  D=D if method in ("fdm", "relaxed") else None, **kw)
    return v


@pytest.fixture(scope="module")
def cfd_noisy(cfd):
    v_hr, mask, m = cfd
    rng = np.random.default_rng(0)
    sigma = 0.10 * VENC
    v_noisy = v_hr + rng.normal(0, sigma, v_hr.shape) * mask[..., None]
    D = build_divergence_operator(mask, m)
    q_ref = slice_flows(v_hr, mask, m)
    return v_hr, v_noisy, mask, m, D, q_ref, sigma


# name -> (method, kwargs, max RMSE cm/s, max divergence RMS /s, max flow error mL/s)
CFD_CASES = {
    "fdm": ("fdm", {}, 12.6, 1e-3, 1.0),
    "relaxed": ("relaxed", {"lam": 0.5}, 13.0, 150, 1.5),
    "leray": ("leray", {"n_iter": 3}, 14.5, 250, None),   # leray alone leaks flow through the wall
    "dfw": ("dfw", {}, 3.0, 60, 1.5),
    "dfw k_df=2": ("dfw", {"k_df": 2.0, "k_cf": 3.0}, 2.5, 40, 1.5),
    "dfw then fdm": ("dfw+fdm", {}, 2.6, 1e-3, 1.5),
}


@pytest.mark.cfd
@pytest.mark.parametrize("name", list(CFD_CASES))
def test_cfd_divergence_free_methods(cfd_noisy, name, metrics):
    v_hr, v_noisy, mask, m, D, q_ref, sigma = cfd_noisy
    method, kw, max_rmse, max_div, max_qerr = CFD_CASES[name]
    rmse_noisy, _, _, _ = metrics_of(v_hr, v_noisy, mask, D, m, q_ref)
    v = run_method(v_noisy, mask, m, D, method, kw)
    rmse, ang, div, qerr = metrics_of(v_hr, v, mask, D, m, q_ref)
    metrics.record(f"cfd/{name}/rmse_cm_s", rmse)
    metrics.record(f"cfd/{name}/angular_error_deg", ang)
    metrics.record(f"cfd/{name}/flow_error_mL_s", qerr, atol=0.05)
    assert rmse < rmse_noisy
    assert rmse < max_rmse
    assert div < max_div
    if max_qerr is not None:
        assert qerr < max_qerr
    if method in ("fdm", "relaxed"):
        # pure projection cannot beat the sqrt(2/3) floor
        assert rmse > 0.95 * np.sqrt(2 / 3) * sigma * 100


@pytest.mark.cfd
def test_cfd_world_coordinate_system_invariance(cfd_noisy):
    """Same result in an oblique, reflected and translated world coordinate system."""
    v_hr, v_noisy, mask, m, D, q_ref, sigma = cfd_noisy
    T = world_transform()
    R = T[:3, :3]
    out, _ = divergence_free_filter(v_noisy, mask, m, method="fdm")
    out_t, _ = divergence_free_filter(v_noisy @ R.T, mask, T @ m, method="fdm")
    np.testing.assert_allclose(out_t, out @ R.T, atol=1e-5)


@pytest.mark.cfd
def test_cfd_dfw_world_orientation(cfd_noisy):
    """dfw thresholds the wavelet coefficients of each world component
    separately, so the result depends on the orientation of the world axes,
    but its accuracy does not."""
    v_hr, v_noisy, mask, m, D, q_ref, sigma = cfd_noisy
    T = world_transform()
    R = T[:3, :3]
    out, _ = divergence_free_filter(v_noisy, mask, m, method="dfw", sigma=sigma)
    out_t, _ = divergence_free_filter(v_noisy @ R.T, mask, T @ m, method="dfw", sigma=sigma)
    out_t = out_t @ R                                    # back to the original world axes
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    err, err_t = rms((out - v_hr)[mask]), rms((out_t - v_hr)[mask])
    assert abs(err_t - err) < 0.05 * err
    assert rms((out_t - out)[mask]) < 0.15 * sigma


@pytest.mark.cfd
def test_cfd_mask_perturbation(cfd_noisy, metrics):
    v_hr, v_noisy, mask, m, D, q_ref, sigma = cfd_noisy
    rng = np.random.default_rng(1)
    for label, mk in (("eroded", ndimage.binary_erosion(mask)), ("dilated", ndimage.binary_dilation(mask))):
        v_in = v_noisy.copy()
        if label == "dilated":
            ring = mk & ~mask
            v_in[ring] = rng.normal(0, sigma, (ring.sum(), 3))
        for method, max_rmse in (("fdm", 13.5), ("dfw", 7.0)):
            v, _ = divergence_free_filter(v_in, mk, m, method=method)
            rmse, _, _, _ = metrics_of(v_hr, v, mask, D, m, q_ref)
            metrics.record(f"cfd/mask_{label}/{method}/rmse_cm_s", rmse)
            assert rmse < max_rmse, f"{label} {method}: RMSE {rmse:.2f}"


def main(lr=False):
    import conftest
    v_hr, mask, m = conftest.load_cfd()
    if lr:
        # MRI resolution: block-average the CFD to 1.19 mm
        def block(a):
            return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).mean(axis=(1, 3, 5))
        v_hr = np.stack([block(v_hr[..., k]) for k in range(3)], -1)
        mask = block(mask.astype(float)) > 0.5
        m = voxel_to_world_matrix(spacing=2 * np.linalg.norm(m[:3, :3], axis=0))
    D = build_divergence_operator(mask, m)
    q_ref = slice_flows(v_hr, mask, m)
    rng = np.random.default_rng(0)
    print(f"Aortic CFD: {v_hr.shape[:3]}, {m[0, 0]:.3f} mm, {mask.sum()} lumen voxels, "
          f"peak {np.sqrt((v_hr**2).sum(-1)).max():.2f} m/s")
    print(f"{'':28s} {'RMSE cm/s':>10s} {'ang deg':>8s} {'div /s':>9s} {'flow err mL/s':>14s} {'time s':>7s}")
    configs = [
        ("noisy (no filter)", None, {}),
        ("fdm", "fdm", {}),
        ("relaxed lam=0.5", "relaxed", {"lam": 0.5}),
        ("leray x3", "leray", {"n_iter": 3}),
        ("dfw (default 1.5/3.0 soft)", "dfw", {}),
        ("dfw k_df=2.0 k_cf=3.0", "dfw", {"k_df": 2.0, "k_cf": 3.0}),
        ("dfw garrote 1.5/3.0", "dfw", {"shrink": "garrote"}),
        ("wavelet only, no split, k=2", "dfw", {"split": False, "k_df": 2.0}),
        ("wavelet only, no split, k=3", "dfw", {"split": False, "k_df": 3.0}),
        ("dfw then fdm", "dfw+fdm", {}),
    ]
    for frac in (0.05, 0.10, 0.20):
        sigma = frac * VENC
        v_noisy = v_hr + rng.normal(0, sigma, v_hr.shape) * mask[..., None]
        print(f"\nnoise {frac:.0%} venc = {sigma*100:.0f} cm/s   (projection floor = {np.sqrt(2/3)*sigma*100:.1f} cm/s)")
        for name, method, kw in configs:
            t0 = time.perf_counter()
            v = run_method(v_noisy, mask, m, D, method, kw)
            dt = time.perf_counter() - t0
            r, a, d, q = metrics_of(v_hr, v, mask, D, m, q_ref)
            print(f"  {name:26s} {r:10.2f} {a:8.2f} {d:9.3g} {q:14.2f} {dt:7.2f}")


if __name__ == "__main__":
    import sys
    main(lr="--lr" in sys.argv)
