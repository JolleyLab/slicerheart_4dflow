"""
Compare OLS (Walker), MSAC (Fischer 2022) and WRLS+OR+WaR (Pruitt 2019) on a
synthetic chest phantom with the three failure modes each was designed for:

  noise      static tissue noise, lungs and air with low SNR
  steady     a vein with slow constant (5 cm/s) flow, low temporal variance,
             which variance-based candidate selection mistakes for tissue
  wrap       a phase-encode wrap band at the FOV edge where folded-in tissue
             carries the offset of a different location (realistic wrap), not
             just large aliased values

Usage: python test_background_phase.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flowfilters"))
import numpy as np
from scipy import ndimage
from background_phase import (correct_background_phase, static_tissue_candidates,
                              design_matrix, normalised_coords, residual_static_velocity)


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
    return v_meas, mag_t, offset_true, body, lumen


def score(name, offset_est, offset_true, body, corrected, static):
    err = (offset_est - offset_true)[body]
    res = residual_static_velocity(corrected, static)
    print(f"  {name:22s} offset error in body: RMS {np.sqrt((err**2).mean()):6.3f}  "
          f"max {np.abs(err).max():6.3f} cm/s   residual in true static tissue RMS "
          f"{np.mean(res['rms']):.3f} cm/s")


def run(steady, wrap, erosion=3, pcmra=90.0):
    rng = np.random.default_rng(3)
    v_meas, mag_t, off_true, body, lumen = chest_phantom(rng, steady=steady, wrap=wrap)
    # ground-truth static tissue for the residual metric: body minus vessels,
    # lungs, and the wrap band
    nz, ny, nx = body.shape
    Z, Y, X = np.meshgrid(np.linspace(-1, 1, nz), np.linspace(-1, 1, ny),
                          np.linspace(-1, 1, nx), indexing="ij")
    true_static = body.copy()
    true_static &= ~(((X - 0.35) ** 2 + (Y / 0.7) ** 2) < 0.26 ** 2)
    true_static &= ~(((X + 0.35) ** 2 + (Y / 0.7) ** 2) < 0.26 ** 2)
    true_static &= ~((X ** 2 + (Y + 0.2) ** 2) < 0.15 ** 2)
    true_static &= ~(((X - 0.15) ** 2 + (Y + 0.45) ** 2) < 0.13 ** 2)
    band = int(0.12 * ny)
    true_static[:, :band] = False
    true_static[:, -band:] = False

    print(f"\nPhantom: steady vein {'on' if steady else 'off'}, wrap band {'on' if wrap else 'off'}, "
          f"edge erosion {erosion} vox, PC-MRA exclusion {'on' if pcmra < 100 else 'off'}. True offset range {off_true[body].min():.2f} to "
          f"{off_true[body].max():.2f} cm/s")
    for method in ("ols", "msac", "wrls", "war+msac"):
        corrected, info = correct_background_phase(
            v_meas, mag_t, method=method, order=2, lumen_mask=lumen, edge_erosion_voxels=erosion,
            pcmra_percentile=pcmra)
        r = info["report"]
        name = f"{method} (inl {np.mean([x['n_inliers']/x['n_candidates'] for x in r]):.0%})"
        score(name, info["offset"], off_true, body, corrected, true_static)


if __name__ == "__main__":
    run(steady=False, wrap=False)
    run(steady=True, wrap=False)
    run(steady=False, wrap=True)
    run(steady=True, wrap=True)
    # Erosion is the crude wrap defence; show what happens without it.
    run(steady=True, wrap=True, erosion=0)
    # Candidate selection by temporal variance only (no PC-MRA vessel
    # exclusion): the steady vein now enters the candidate set.
    run(steady=True, wrap=False, pcmra=100.0)
    run(steady=True, wrap=True, pcmra=100.0)
