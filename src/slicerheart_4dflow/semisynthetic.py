"""
DRAFT: written before any clinical case was available. The case layout,
units and component frame follow the Slicer 5D convention from the
Discourse post; they must be aligned with what the SlicerHeart Import4DFlow
and Analyze4DFlow modules actually write before this is relied on.

Semi-synthetic references built from measured data ("in-vivo-derived
digital phantoms").

A real 4D flow case has no ground truth. A reference with real anatomy,
real waveform and a real segmentation can be made from it: run the full
pipeline (background correction, unwrapping, noise masking, divergence-free
projection) and keep the result as a clean field that is zero outside the
lumen and divergence-free inside it. Re-acquiring that reference with the
phase-contrast model at the case's own venc, noise level and resolution
gives data that look like the case and have a known answer, on which the
filters and the derived quantities (WSS, pressure) can be scored.

The reference is not the truth of the original case; it is a plausible
flow field on that anatomy. Errors measured against it say how well a
filter recovers a known field under the case's acquisition conditions,
which is the question the synthetic tier cannot answer with phantoms.

References
----------
    Ferdian E, Suinesiaputra A, Dubowitz D, et al. 4DFlowNet: super-resolution
    4D flow MRI using deep learning and computational fluid dynamics. Front
    Phys 2020;8:138. doi:10.3389/fphy.2020.00138 (CFD-derived synthetic 4D
    flow with an acquisition model).
    Berhane H, Scott M, Elbaz M, et al. Fully automated 3D aortic
    segmentation of 4D flow MRI for hemodynamic analysis using deep
    learning. Magn Reson Med 2020;84(4):2204-2218. PMID 32167203.
"""

from __future__ import annotations

import numpy as np

from .background_phase import correct_background_phase
from .divergence_free import divergence_free_sequence
from .phantom import acquire
from .unwrap import unwrap_velocity


def reference_from_measurement(velocity_t, magnitude_t, venc, voxel_to_world, lumen_mask,
                               background_method="war+msac", unwrap_method="laplacian_then_temporal",
                               divergence_method="fdm", smooth_sigma_vox=0.0):
    """
    Clean reference field (nt, nk, nj, ni, 3) mm/s from a measurement:
    corrected, unwrapped, optionally Gaussian-smoothed inside the lumen,
    projected to divergence-free, zero outside the lumen. Also returns the
    estimated velocity noise (mm/s, per component, from static tissue) and
    the background offset, so the re-acquisition can match the case.
    """
    from scipy import ndimage
    from .report import quality_report
    v = np.asarray(velocity_t, float)
    m = np.asarray(lumen_mask, bool)
    corrected, info = correct_background_phase(v, magnitude_t, method=background_method, lumen_mask=m)
    unwrapped = unwrap_velocity(corrected, venc, voxel_to_world, method=unwrap_method, mask=m)
    if smooth_sigma_vox > 0:
        mf = m.astype(float)
        w = ndimage.gaussian_filter(mf, smooth_sigma_vox)
        sm = np.empty_like(unwrapped)
        for t in range(v.shape[0]):
            for c in range(3):
                sm[t, ..., c] = np.where(w > 1e-3, ndimage.gaussian_filter(unwrapped[t, ..., c] * mf, smooth_sigma_vox)
                                         / np.maximum(w, 1e-3), 0.0)
        unwrapped = sm
    rep = quality_report(v, unwrapped, magnitude_t, venc, voxel_to_world, lumen_mask=m)   # noise from tissue, before zeroing
    unwrapped = unwrapped * m[None, ..., None]
    reference, _ = divergence_free_sequence(unwrapped, m, voxel_to_world, method=divergence_method)
    return {"reference": reference, "velocity_noise_mm_s": np.asarray(rep["velocity_noise_mm_s"], float),
            "offset": info["offset"], "report": rep}


def acquisition_snr(magnitude_t, lumen_mask, velocity_noise_mm_s, venc, tissue_mask=None):
    """
    SNR of the acquisition model (for unit magnitude) that reproduces a
    velocity noise sigma_v measured in tissue of a given magnitude, on an
    image whose magnitude is normalised by the lumen median:
    sigma_v = sqrt(2) venc / (pi SNR mag), so the noise seen in tissue of
    relative magnitude m_t is sigma_lumen / m_t. Returns (snr, sigma_lumen).
    """
    from .background_phase import static_tissue_candidates
    mag = np.asarray(magnitude_t, float).mean(axis=0)
    lumen_med = float(np.median(mag[lumen_mask]))
    if tissue_mask is None:
        v0 = np.zeros(mag.shape + (3,))[None]
        tissue_mask = static_tissue_candidates(np.asarray(magnitude_t, float), v0, lumen_mask=lumen_mask,
                                               velocity_std_percentile=100.0, edge_erosion_voxels=0)
    tissue_rel = float(np.median(mag[tissue_mask])) / max(lumen_med, 1e-9) if tissue_mask.any() else 1.0
    sigma_tissue = float(np.mean(velocity_noise_mm_s))
    sigma_lumen = sigma_tissue * tissue_rel
    snr = float(np.sqrt(2.0) * np.mean(np.broadcast_to(np.asarray(venc, float), (3,))) / (np.pi * max(sigma_lumen, 1e-6)))
    return snr, sigma_lumen


def reacquire(reference, venc, magnitude_t, lumen_mask, velocity_noise_mm_s, offset=None, rng=None,
              downsample=1):
    """
    Re-acquire a clean reference with the phase-contrast model at the
    case's venc and noise. The SNR of the acquisition model is set from
    the velocity noise measured in the case's tissue (see acquisition_snr),
    and the case's own magnitude image, normalised by its lumen median,
    provides the spatial SNR variation. An offset (e.g. the one estimated
    from the case) is added before encoding so background correction is
    exercised too.
    Returns velocity_t (measured, aliased), magnitude_t, wrap_count, snr.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    snr, _ = acquisition_snr(magnitude_t, lumen_mask, velocity_noise_mm_s, venc)
    mag_t = np.asarray(magnitude_t, float)
    lumen_med = max(float(np.median(mag_t.mean(axis=0)[lumen_mask])), 1e-9)
    out_v, out_m, out_k = [], [], []
    for t in range(reference.shape[0]):
        field = reference[t] + (0.0 if offset is None else offset)
        acq = acquire(field, venc, magnitude=mag_t[t] / lumen_med, snr=snr, rng=rng, downsample=downsample)
        out_v.append(acq["velocity"]); out_m.append(acq["magnitude"]); out_k.append(acq["wrap_count"])
    return {"velocity_t": np.stack(out_v), "magnitude_t": np.stack(out_m), "wrap_count": np.stack(out_k),
            "snr": snr}
