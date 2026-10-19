"""
Test velocity unwrapping on the aortic CFD field (peak 0.85 m/s) aliased at a
venc of 0.5 m/s, in two cardiac-cycle scenarios:

  pulsatile  scale 0.2 .. 1.0 over 20 frames: aliasing appears mid-cycle, so
             temporal unwrapping can see the jump.
  persistent scale 0.75 .. 1.0: the jet exceeds venc in every frame, so there
             is no temporal jump to detect (stenosis / baffle case).

Noise at 5% and 15% of venc. Metric: residual wrapped voxels inside the
lumen (|error| > venc) and RMSE vs truth, for temporal, laplacian3d,
laplacian4d and laplacian_then_temporal.

Usage: python test_unwrap.py /path/to/4DFlowNet/data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flowfilters"))
import time
import numpy as np
import h5py
from unwrap import unwrap_velocity, noise_mask, pcmra


def load(path):
    with h5py.File(path) as h:
        d = {k: h[k][0] for k in h.keys()}
    vel = np.stack([d["u"], d["v"], d["w"]], -1).astype(float)
    return vel, d["mask"] > 0.5, tuple(float(x) for x in d["dx"])


def wrap(v, venc):
    return (v + venc) % (2 * venc) - venc


def main(data_dir):
    v_cfd, mask, sp = load(f"{data_dir}/example_data_HR.h5")
    rng = np.random.default_rng(0)
    nt = 20
    venc = 0.5
    peak = np.sqrt((v_cfd ** 2).sum(-1)).max()
    print(f"CFD peak {peak:.2f} m/s, venc {venc} m/s, {mask.sum()} lumen voxels")

    # a magnitude image: lumen bright, outside a noise floor
    mag = np.where(mask, 60.0, 2.0)
    mag_t = mag[None] * (1 + 0.05 * rng.normal(size=(nt,) + mask.shape))

    scenarios = {
        "pulsatile": 0.2 + 0.8 * np.clip(np.sin(np.linspace(0, np.pi, nt)), 0, 1),
        "persistent": 0.75 + 0.25 * np.clip(np.sin(np.linspace(0, np.pi, nt)), 0, 1),
    }
    for name, scale in scenarios.items():
        v_true = np.stack([v_cfd * s for s in scale])
        for noise_frac in (0.05, 0.15):
            v_meas = v_true + rng.normal(0, noise_frac * venc, v_true.shape) * mask[None, ..., None]
            v_meas[:, ~mask] = rng.normal(0, 0.7 * venc, (nt, (~mask).sum(), 3))   # random phase in noise
            v_wrapped = wrap(v_meas, venc)
            n_aliased = (np.abs(v_meas - v_wrapped)[:, mask] > 1e-9).any(-1).sum()
            print(f"\n{name}, noise {noise_frac:.0%} venc: {n_aliased} aliased lumen voxel-frames "
                  f"of {nt * mask.sum()} ({n_aliased / (nt * mask.sum()):.1%})")
            signal = ~noise_mask(mag_t, v_wrapped)
            print(f"  noise mask recovers lumen: {(signal & mask).sum() / mask.sum():.1%} of lumen kept, "
                  f"{(signal & ~mask).sum() / (~mask).sum():.1%} of outside kept")
            for method in ("temporal", "laplacian3d", "laplacian4d", "laplacian_then_temporal"):
                t0 = time.perf_counter()
                v_u = unwrap_velocity(v_wrapped, venc, sp, method=method, mask=signal)
                dt = time.perf_counter() - t0
                err = (v_u - v_meas)[:, mask]
                residual = (np.abs(err) > venc).any(-1).sum()
                rmse = np.sqrt(np.mean((v_u - v_true)[:, mask] ** 2)) * 100
                print(f"  {method:24s} residual wrapped {residual:6d}  RMSE vs truth {rmse:6.2f} cm/s  {dt:5.2f} s")


if __name__ == "__main__":
    main(sys.argv[1])
