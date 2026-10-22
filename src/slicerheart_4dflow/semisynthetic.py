"""
Semi-synthetic references built from measured data ("in-vivo-derived digital
phantoms").

A real 4D flow case has no ground truth. A reference with real anatomy, real
waveform and a real segmentation can be made from it: run the full pipeline
(background correction, unwrapping, divergence-free projection) and keep the
result as a clean field that is zero outside the lumen and divergence-free
inside it. Re-acquiring that reference with the phase-contrast model
(phantom.acquire) at the case's own venc, noise level, magnitude image and
background offset gives data that look like the case and have a known answer,
on which the filters and the derived quantities can be scored.

The reference is not the truth of the original case; it is a plausible flow
field on that anatomy. Errors measured against it say how well the pipeline
recovers a known field under the case's acquisition conditions, which the
phantoms with their simple geometry cannot tell.

Conventions (see geometry.py): velocity_t is (nt, nk, nj, ni, 3) in mm/s.
Measured and re-acquired velocities are the encoded components as acquired
(each aliased at its venc); encoding_to_world maps them to world components
(world = encoded @ encoding_to_world.T, a signed permutation for Siemens).
The reference is in world components.

References
----------
    Ferdian E, Suinesiaputra A, Dubowitz D, et al. 4DFlowNet: super-resolution
    4D flow MRI using deep learning and computational fluid dynamics. Front
    Phys 2020;8:138. doi:10.3389/fphy.2020.00138 (CFD-derived synthetic 4D
    flow with an acquisition model).
    Berhane H, Scott M, Elbaz M, et al. Fully automated 3D aortic
    segmentation of 4D flow MRI for hemodynamic analysis using deep
    learning. Magn Reson Med 2020;84(4):2204-2218. PMID 32167203.
    doi:10.1002/mrm.28257.
"""

from __future__ import annotations

import numpy as np

from .background_phase import correct_background_phase, static_tissue_candidates
from .divergence_free import divergence_free_sequence
from .phantom import acquire
from .unwrap import noise_mask, unwrap_velocity


def _encoding_to_world(encoding_to_world):
    return np.eye(3) if encoding_to_world is None else np.asarray(encoding_to_world, float)


def reference_from_measurement(velocity_t, magnitude_t, venc, voxel_to_world, lumen_mask, encoding_to_world=None,
                               background_method="war+msac", unwrap_method="laplacian_then_temporal",
                               divergence_method="fdm", smooth_sigma_vox=0.0):
    """
    Clean reference field from a measurement: background corrected, unwrapped
    (in the signal voxels and the whole lumen), optionally Gaussian-smoothed
    inside the lumen, zero outside the lumen and projected to divergence-free.
    Lumen voxels that the noise mask marks after unwrapping (signal voids,
    segmentation into air) carry no velocity information: the reference is
    zero there too, otherwise their random phase would be re-acquired as
    flow.

    velocity_t : encoded velocity components as acquired (aliased), mm/s.
    venc       : scalar or (3,) per encoded component, mm/s.
    Returns dict: reference (world components, mm/s), mask (the lumen voxels
    where the reference is defined: lumen minus noise), velocity_noise_mm_s
    (per encoded component, from magnitude-selected tissue), offset (the
    estimated background offset, encoded components), report (quality_report
    of the corrected and unwrapped velocity).
    """
    from scipy import ndimage
    from .report import quality_report
    v = np.asarray(velocity_t, float)
    mag = np.asarray(magnitude_t, float)
    m = np.asarray(lumen_mask, bool)
    E = _encoding_to_world(encoding_to_world)
    corrected, info = correct_background_phase(v, mag, method=background_method, lumen_mask=m)
    # the noise mask can mark fast pulsatile lumen voxels as noise; the reference must be unwrapped everywhere in
    # the lumen
    signal = ~noise_mask(mag, corrected, venc=venc) | m
    unwrapped = unwrap_velocity(corrected, venc, voxel_to_world, method=unwrap_method, mask=signal)
    rep = quality_report(v, unwrapped, mag, venc, voxel_to_world, lumen_mask=m)   # noise from tissue, before zeroing
    m = m & ~noise_mask(mag, unwrapped, venc=venc)
    world = unwrapped @ E.T
    if smooth_sigma_vox > 0:
        mf = m.astype(float)
        w = ndimage.gaussian_filter(mf, smooth_sigma_vox)
        smoothed = np.empty_like(world)
        for t in range(world.shape[0]):
            for c in range(3):
                smoothed[t, ..., c] = np.where(w > 1e-3, ndimage.gaussian_filter(world[t, ..., c] * mf, smooth_sigma_vox)
                                               / np.maximum(w, 1e-3), 0.0)
        world = smoothed
    world = world * m[None, ..., None]
    reference, _ = divergence_free_sequence(world, m, voxel_to_world, method=divergence_method)
    return {"reference": reference, "mask": m, "velocity_noise_mm_s": np.asarray(rep["velocity_noise_mm_s"], float),
            "offset": info["offset"], "report": rep}


def acquisition_snr(magnitude_t, lumen_mask, velocity_noise_mm_s, venc, tissue_mask=None):
    """
    SNR of the acquisition model (phantom.acquire, for unit magnitude) that
    reproduces a velocity noise sigma_v measured in tissue, on a magnitude
    image normalized by its lumen median: sigma_v = sqrt(2) venc / (pi SNR
    mag), so tissue of relative magnitude m_t has the noise sigma_lumen / m_t.
    tissue_mask: default from static_tissue_candidates, selected by magnitude
    only (as quality_report selects it for the noise estimate).
    Returns (snr, sigma_lumen).
    """
    mag_t = np.asarray(magnitude_t, float)
    mag = mag_t.mean(axis=0)
    lumen_median = float(np.median(mag[lumen_mask]))
    if tissue_mask is None:
        v0 = np.broadcast_to(np.zeros(3), mag_t.shape + (3,))       # a zero velocity, as a view
        tissue_mask = static_tissue_candidates(mag_t, v0, lumen_mask=lumen_mask, magnitude_threshold="otsu",
                                               velocity_std_percentile=100.0, edge_erosion_voxels=0)
    tissue_rel = float(np.median(mag[tissue_mask])) / max(lumen_median, 1e-9) if np.any(tissue_mask) else 1.0
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    noise_over_venc = float(np.mean(np.asarray(velocity_noise_mm_s, float) / venc))
    snr = float(np.sqrt(2.0) / (np.pi * max(noise_over_venc * tissue_rel, 1e-9)))
    return snr, float(noise_over_venc * tissue_rel * np.mean(venc))


def acquisition_noise(magnitude_t, lumen_mask, venc, snr):
    """
    Velocity noise (nt, nk, nj, ni, 3) of each voxel, frame and encoded
    component in a re-acquisition (reacquire) with this snr: sqrt(2) venc /
    (pi snr mag), with the magnitude normalized by its lumen median as
    reacquire does. Low-signal voxels are noisier, so errors of a pipeline on
    re-acquired data are compared with this rather than with one noise level.
    """
    mag_t = np.asarray(magnitude_t, float)
    lumen_median = max(float(np.median(mag_t.mean(axis=0)[lumen_mask])), 1e-9)
    relative = np.maximum(mag_t / lumen_median, 1e-3)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    return np.sqrt(2.0) * venc / (np.pi * snr * relative[..., None])


def reacquire(reference, venc, magnitude_t, lumen_mask, velocity_noise_mm_s, offset=None, encoding_to_world=None,
              rng=None, downsample=1):
    """
    Re-acquire a clean reference (world components) with the phase-contrast
    model at the case's venc and noise. The SNR is set from the velocity
    noise measured in the case's tissue (acquisition_snr), and the case's
    own magnitude image, normalized by its lumen median, provides the
    spatial SNR variation. An offset (encoded components, e.g. the one
    estimated from the case) is added before encoding, so that background
    correction is exercised too.
    Returns dict: velocity_t (encoded components, aliased), magnitude_t,
    wrap_count (per encoded component), snr.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    E = _encoding_to_world(encoding_to_world)
    snr, _ = acquisition_snr(magnitude_t, lumen_mask, velocity_noise_mm_s, venc)
    mag_t = np.asarray(magnitude_t, float)
    lumen_median = max(float(np.median(mag_t.mean(axis=0)[lumen_mask])), 1e-9)
    encoded = np.asarray(reference, float) @ np.linalg.inv(E).T      # world = encoded @ E.T
    out_v, out_m, out_k = [], [], []
    for t in range(encoded.shape[0]):
        field = encoded[t] + (0.0 if offset is None else offset)
        acq = acquire(field, venc, magnitude=mag_t[t] / lumen_median, snr=snr, rng=rng, downsample=downsample)
        out_v.append(acq["velocity"]); out_m.append(acq["magnitude"]); out_k.append(acq["wrap_count"])
    return {"velocity_t": np.stack(out_v), "magnitude_t": np.stack(out_m), "wrap_count": np.stack(out_k), "snr": snr}
