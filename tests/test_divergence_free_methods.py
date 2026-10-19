"""
Compare fdm, relaxed, leray and dfw on the 4DFlowNet aortic CFD ground truth
(0.59 mm, 56k lumen voxels) with Gaussian noise at 5, 10 and 20% of venc.

Metrics inside the lumen: RMSE vs truth (cm/s), mean angular error (deg) on
voxels above 10% of peak speed, divergence RMS (/s), through-plane flow
error at five axial slices (mL/s), wall time.

Usage: python test_divergence_free_methods.py /path/to/4DFlowNet/data [--lr]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flowfilters"))
import time
import numpy as np
import h5py
from divergence_free import divergence_free_filter, build_divergence_operator


def load(path):
    with h5py.File(path) as h:
        d = {k: h[k][0] for k in h.keys()}
    vel = np.stack([d["u"], d["v"], d["w"]], -1).astype(float)
    return vel, d["mask"] > 0.5, tuple(float(x) for x in d["dx"])


def slice_flows(v, mask, spacing, axis=0, n=5):
    idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    picks = np.linspace(idx[0] + 3, idx[-1] - 3, n).astype(int)
    area = np.prod([spacing[a] for a in range(3) if a != axis]) * 1e-6
    q = []
    for k in picks:
        sl = [slice(None)] * 3; sl[axis] = k
        q.append((v[tuple(sl)][..., axis] * mask[tuple(sl)]).sum() * area * 1e6)
    return np.array(q)


def metrics(v_ref, v, mask, D, spacing, q_ref):
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


def block(a):
    return a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2, a.shape[2]//2, 2).mean(axis=(1, 3, 5))


def main(data_dir):
    v_hr, mask, sp = load(f"{data_dir}/example_data_HR.h5")
    if "--lr" in sys.argv:
        # MRI resolution: block-average the CFD to 1.19 mm
        v_hr = np.stack([block(v_hr[..., k]) for k in range(3)], -1)
        mask = block(mask.astype(float)) > 0.5
        sp = tuple(2 * s for s in sp)
    D = build_divergence_operator(mask, sp)
    q_ref = slice_flows(v_hr, mask, sp)
    rng = np.random.default_rng(0)
    venc = 1.5
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
        sigma = frac * venc
        v_noisy = v_hr + rng.normal(0, sigma, v_hr.shape) * mask[..., None]
        print(f"\nnoise {frac:.0%} venc = {sigma*100:.0f} cm/s   (projection floor = {np.sqrt(2/3)*sigma*100:.1f} cm/s)")
        for name, method, kw in configs:
            t0 = time.perf_counter()
            if method is None:
                v = v_noisy
            elif method == "dfw+fdm":
                v, _ = divergence_free_filter(v_noisy, mask, sp, method="dfw", **kw)
                v, _ = divergence_free_filter(v, mask, sp, method="fdm", D=D)
            else:
                v, _ = divergence_free_filter(v_noisy, mask, sp, method=method,
                                              D=D if method in ("fdm", "relaxed") else None, **kw)
            dt = time.perf_counter() - t0
            r, a, d, q = metrics(v_hr, v, mask, D, sp, q_ref)
            print(f"  {name:26s} {r:10.2f} {a:8.2f} {d:9.3g} {q:14.2f} {dt:7.2f}")

    # mask sensitivity for the two leading methods at 10% noise
    from scipy import ndimage
    print("\nMask perturbation at 10% noise (error inside true lumen)")
    sigma = 0.10 * venc
    v_noisy = v_hr + rng.normal(0, sigma, v_hr.shape) * mask[..., None]
    for mlabel, m in [("true", mask), ("eroded 1", ndimage.binary_erosion(mask)),
                      ("dilated 1", ndimage.binary_dilation(mask))]:
        v_in = v_noisy.copy()
        if mlabel.startswith("dilated"):
            ring = m & ~mask
            v_in[ring] = rng.normal(0, sigma, (ring.sum(), 3))
        for name, method, kw in [("fdm", "fdm", {}), ("dfw", "dfw", {})]:
            v, _ = divergence_free_filter(v_in, m, sp, method=method, **kw)
            r, a, d, q = metrics(v_hr, v, mask, D, sp, q_ref)
            print(f"  {mlabel:10s} {name:5s} RMSE {r:6.2f}  ang {a:5.2f}  flow err {q:5.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
