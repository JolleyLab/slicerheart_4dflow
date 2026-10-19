"""
Divergence-free filtering: exactness of the divergence operator on analytic
fields, and comparison of fdm, relaxed, leray and dfw on the 4DFlowNet aortic
CFD ground truth (0.59 mm, 56k lumen voxels) with Gaussian noise at 10% of a
1.5 m/s venc.

Run as a script to print the full comparison table (5/10/20% noise, more
dfw variants):
    python tests/test_divergence_free.py [--lr]
"""

import time

import numpy as np
import pytest
from scipy import ndimage

from slicerheart_4dflow.divergence_free import (
    build_divergence_operator, divergence_free_filter, divergence_free_sequence, divergence_stats,
    leray_split)

VENC = 1.5  # m/s


# ---------------------------------------------------------------------------
# Analytic fields
# ---------------------------------------------------------------------------

def _ball_mask(shape, radius_fraction=0.7):
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    return sum(g ** 2 for g in grids) < radius_fraction ** 2


def _index_coords_mm(shape, spacing):
    grids = np.meshgrid(*[np.arange(n) * h for n, h in zip(shape, spacing)], indexing="ij")
    return np.stack(grids, -1)


def test_divergence_operator_linear_field_is_exact():
    """Central and one-sided differences are exact for a linear field."""
    shape, spacing = (12, 14, 16), (1.5, 1.0, 0.8)
    mask = _ball_mask(shape)
    p = _index_coords_mm(shape, spacing)
    # div = 2 - 0.5 + 3
    v = np.stack([2.0 * p[..., 0] + p[..., 1], -0.5 * p[..., 1] + p[..., 2], 3.0 * p[..., 2] - p[..., 0]], -1)
    D = build_divergence_operator(mask, spacing)
    d = D @ np.concatenate([v[..., k][mask] for k in range(3)])
    np.testing.assert_allclose(d, 4.5, atol=1e-10)


def test_projection_methods_on_analytic_field():
    shape, spacing = (16, 16, 16), (1.0, 1.0, 1.0)
    mask = _ball_mask(shape)
    p = _index_coords_mm(shape, spacing) - 7.5
    rng = np.random.default_rng(0)
    # solid-body rotation about the first axis: divergence-free
    v_true = np.stack([np.zeros(shape), -p[..., 2], p[..., 1]], -1)
    v_noisy = v_true + 0.3 * rng.normal(size=v_true.shape)
    before = divergence_stats(v_noisy, mask, spacing)["div_rms"]
    for method in ("fdm", "relaxed", "leray", "dfw"):
        out, info = divergence_free_filter(v_noisy, mask, spacing, method=method)
        assert info["div_rms_after"] < before, method
        np.testing.assert_array_equal(out[~mask], v_noisy[~mask])
    out, info = divergence_free_filter(v_noisy, mask, spacing, method="fdm")
    assert info["div_rms_after"] < 1e-6 * before


def test_leray_split_recovers_components():
    """A gradient field is curl-free; a rotation field is divergence-free."""
    shape, spacing = (24, 24, 24), (1.0, 1.0, 1.0)
    p = _index_coords_mm(shape, spacing) - 11.5
    g = np.exp(-(p ** 2).sum(-1) / 20.0)
    grad = np.stack(np.gradient(g, *spacing), -1)                           # curl-free
    rot = np.stack([np.zeros(shape), -p[..., 2] * g, p[..., 1] * g], -1)    # divergence-free
    df, cf = leray_split(grad + rot, spacing)
    assert np.abs(df - rot).max() < 0.1 * np.abs(rot).max()
    assert np.abs(cf - grad).max() < 0.1 * np.abs(grad).max()


def test_sequence_static_and_dynamic_mask():
    shape, spacing = (10, 10, 10), (1.0, 1.0, 1.0)
    mask = _ball_mask(shape)
    rng = np.random.default_rng(1)
    v_t = rng.normal(size=(3,) + shape + (3,))
    out_s, infos_s = divergence_free_sequence(v_t, mask, spacing, method="fdm")
    out_d, infos_d = divergence_free_sequence(v_t, np.stack([mask] * 3), spacing, method="fdm")
    np.testing.assert_allclose(out_s, out_d, atol=1e-6)
    assert len(infos_s) == 3


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

def slice_flows(v, mask, spacing, axis=0, n=5):
    idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    picks = np.linspace(idx[0] + 3, idx[-1] - 3, n).astype(int)
    area = np.prod([spacing[a] for a in range(3) if a != axis]) * 1e-6
    q = []
    for k in picks:
        sl = [slice(None)] * 3
        sl[axis] = k
        q.append((v[tuple(sl)][..., axis] * mask[tuple(sl)]).sum() * area * 1e6)
    return np.array(q)


def metrics(v_ref, v, mask, D, spacing, q_ref):
    """RMSE (cm/s), mean angular error (deg), divergence RMS (/s), flow error (mL/s)."""
    e = (v - v_ref)[mask]
    rmse = np.sqrt(np.mean(e ** 2)) * 100
    speed = np.sqrt((v_ref[mask] ** 2).sum(-1))
    sel = speed > 0.1 * speed.max()
    a, b = v_ref[mask][sel], v[mask][sel]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    vv = np.concatenate([v[..., k][mask] for k in range(3)])
    div = np.sqrt(np.mean((D @ vv) ** 2))
    qerr = np.abs(slice_flows(v, mask, spacing) - q_ref).mean()
    return rmse, ang, div, qerr


def run_method(v_noisy, mask, sp, D, method, kw):
    if method is None:
        return v_noisy
    if method == "dfw+fdm":
        v, _ = divergence_free_filter(v_noisy, mask, sp, method="dfw", **kw)
        v, _ = divergence_free_filter(v, mask, sp, method="fdm", D=D)
        return v
    v, _ = divergence_free_filter(v_noisy, mask, sp, method=method,
                                  D=D if method in ("fdm", "relaxed") else None, **kw)
    return v


@pytest.fixture(scope="module")
def cfd_noisy(cfd):
    v_hr, mask, sp = cfd
    rng = np.random.default_rng(0)
    sigma = 0.10 * VENC
    v_noisy = v_hr + rng.normal(0, sigma, v_hr.shape) * mask[..., None]
    D = build_divergence_operator(mask, sp)
    q_ref = slice_flows(v_hr, mask, sp)
    return v_hr, v_noisy, mask, sp, D, q_ref, sigma


# name -> (method, kwargs, max RMSE cm/s, max divergence RMS /s, max flow error mL/s)
CFD_CASES = {
    "fdm": ("fdm", {}, 12.6, 1e-6, 1.0),
    "relaxed": ("relaxed", {"lam": 0.5}, 13.0, 0.15, 1.5),
    "leray": ("leray", {"n_iter": 3}, 14.5, 0.25, None),   # leray alone leaks flow through the wall
    "dfw": ("dfw", {}, 3.0, 0.06, 1.5),
    "dfw k_df=2": ("dfw", {"k_df": 2.0, "k_cf": 3.0}, 2.5, 0.04, 1.5),
    "dfw then fdm": ("dfw+fdm", {}, 2.6, 1e-6, 1.5),
}


@pytest.mark.cfd
@pytest.mark.parametrize("name", list(CFD_CASES))
def test_cfd_divergence_free_methods(cfd_noisy, name):
    v_hr, v_noisy, mask, sp, D, q_ref, sigma = cfd_noisy
    method, kw, max_rmse, max_div, max_qerr = CFD_CASES[name]
    rmse_noisy, _, _, _ = metrics(v_hr, v_noisy, mask, D, sp, q_ref)
    v = run_method(v_noisy, mask, sp, D, method, kw)
    rmse, ang, div, qerr = metrics(v_hr, v, mask, D, sp, q_ref)
    assert rmse < rmse_noisy
    assert rmse < max_rmse
    assert div < max_div
    if max_qerr is not None:
        assert qerr < max_qerr
    if method in ("fdm", "relaxed"):
        # pure projection cannot beat the sqrt(2/3) floor
        assert rmse > 0.95 * np.sqrt(2 / 3) * sigma * 100


@pytest.mark.cfd
def test_cfd_mask_perturbation(cfd_noisy):
    v_hr, v_noisy, mask, sp, D, q_ref, sigma = cfd_noisy
    rng = np.random.default_rng(1)
    for label, m in (("eroded", ndimage.binary_erosion(mask)), ("dilated", ndimage.binary_dilation(mask))):
        v_in = v_noisy.copy()
        if label == "dilated":
            ring = m & ~mask
            v_in[ring] = rng.normal(0, sigma, (ring.sum(), 3))
        for method, max_rmse in (("fdm", 13.5), ("dfw", 7.0)):
            v, _ = divergence_free_filter(v_in, m, sp, method=method)
            rmse, _, _, _ = metrics(v_hr, v, mask, D, sp, q_ref)
            assert rmse < max_rmse, f"{label} {method}: RMSE {rmse:.2f}"


def main(lr=False):
    import conftest
    v_hr, mask, sp = conftest.load_cfd()
    if lr:
        # MRI resolution: block-average the CFD to 1.19 mm
        def block(a):
            return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).mean(axis=(1, 3, 5))
        v_hr = np.stack([block(v_hr[..., k]) for k in range(3)], -1)
        mask = block(mask.astype(float)) > 0.5
        sp = tuple(2 * s for s in sp)
    D = build_divergence_operator(mask, sp)
    q_ref = slice_flows(v_hr, mask, sp)
    rng = np.random.default_rng(0)
    print(f"Aortic CFD: {v_hr.shape[:3]}, {sp[0]:.3f} mm, {mask.sum()} lumen voxels, "
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
            v = run_method(v_noisy, mask, sp, D, method, kw)
            dt = time.perf_counter() - t0
            r, a, d, q = metrics(v_hr, v, mask, D, sp, q_ref)
            print(f"  {name:26s} {r:10.2f} {a:8.2f} {d:9.3g} {q:14.2f} {dt:7.2f}")


if __name__ == "__main__":
    import sys
    main(lr="--lr" in sys.argv)
