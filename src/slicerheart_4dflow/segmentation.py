"""
Automatic lumen (blood pool) segmentation of 4D flow data.

estimate_lumen() is a starting point for review and editing, not a final
segmentation:

1. Threshold the velocity-weighted PC-MRA (Bock 2010) at Otsu's threshold of
   the log PC-MRA within the bright half of the image, on a grid finer than
   the image (oversampling, default 2 along each axis; the PC-MRA is
   resampled linearly with VTK in physical space), open and close it by
   about one image voxel, keep connected components of at least
   minimum_component_ml that are not mostly at the image border, and fill
   holes slice by slice.
2. The PC-MRA is magnitude times speed, and the speed falls to zero at the
   vessel wall, so the threshold cuts inside the wall: on 15 pediatric cases
   (1.4 to 1.6 mm voxels), dilating this mask by one image voxel increased
   the net flow through the measurement planes by 10% (median). The mask is
   therefore grown, by at most growth_mm (default 1.5 image voxels), into
   voxels where the RMS speed over the cycle exceeds growth_noise_factor
   times that of the velocity noise (sqrt(3) sigma_v): it reaches the wall
   where there is flow, but not static tissue. After that, a further
   dilation by one image voxel increased the net flow by 1.3% (median over
   the cases).

The partial-volume undershoot scales with the voxel size, so the default
growth and the opening and closing radius are relative to it; distances are
measured in mm with a spacing-aware distance map, so they are the same along
all axes of an anisotropic grid.

The mask is static: vessels that move during the cycle (aortic root, main
pulmonary artery) are covered where they carry flow at any time, which
includes static tissue in the frames when the vessel is elsewhere.

Conventions (see geometry.py): arrays are [k, j, i], voxel_to_world maps
(i, j, k) to world. The fine grid is aligned with the image voxels: each
image voxel holds oversampling^3 fine voxels (vtk_image.fine_voxel_to_world).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .background_phase import otsu_threshold, static_tissue_candidates
from .geometry import array_axis_spacing
from .vtk_image import block_average, supersample


def _distance_outside(mask, spacing):
    """Distance (mm) of each voxel to the nearest voxel of mask (0 inside), along the array axes with spacing."""
    return ndimage.distance_transform_edt(~mask, sampling=spacing)


def dilate_mm(mask, spacing, radius_mm):
    """Voxels within radius_mm (Euclidean, spacing-aware) of the mask."""
    return _distance_outside(mask, spacing) <= radius_mm + 1e-6


def erode_mm(mask, spacing, radius_mm):
    """Voxels of the mask that are farther than radius_mm from every voxel outside it."""
    return ndimage.distance_transform_edt(mask, sampling=spacing) > radius_mm + 1e-6


def _keep_components(mask, voxel_to_world, minimum_ml, border_mm):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    voxel_ml = abs(np.linalg.det(np.asarray(voxel_to_world, float)[:3, :3])) / 1000.0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1)) * voxel_ml
    # Regions that are mostly at the image border are artifacts (vessels that leave the field of view are mostly
    # inside)
    spacing = array_axis_spacing(voxel_to_world)
    border = np.ones(mask.shape, bool)
    inner = tuple(slice(int(np.ceil(border_mm / s)), n - int(np.ceil(border_mm / s))) for s, n in zip(spacing, mask.shape))
    border[inner] = False
    border_fractions = ndimage.mean(border, labels, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, bool)
    keep[1:] = (sizes >= minimum_ml) & (border_fractions < 0.3)
    return keep[labels]


def velocity_noise_in_tissue(velocity_t, magnitude_t, lumen_mask=None):
    """Velocity noise sigma_v (mm/s): median temporal std of the velocity components in tissue selected by magnitude
    (static_tissue_candidates without the velocity rule)."""
    tissue = static_tissue_candidates(np.asarray(magnitude_t, float), np.asarray(velocity_t, float), lumen_mask=lumen_mask,
                                      velocity_std_percentile=100.0, edge_erosion_voxels=0)
    if not tissue.any():
        raise ValueError("no tissue selected by magnitude to estimate the velocity noise")
    return float(np.median(np.asarray(velocity_t, float)[:, tissue].std(axis=0)))


def estimate_lumen(pcmra, voxel_to_world, velocity_t=None, magnitude_t=None, noise_mm_s=None, oversampling=2,
                   growth_mm=None, growth_noise_factor=2.0, smoothing_mm=None, minimum_component_ml=1.0):
    """
    Automatic lumen segmentation (see the module docstring).

    pcmra        : (nk, nj, ni) velocity-weighted PC-MRA (unwrap.pcmra).
    velocity_t   : (nt, nk, nj, ni, 3) corrected velocity, mm/s; without it the
                   PC-MRA threshold is not grown toward the wall.
    magnitude_t  : (nt, nk, nj, ni) magnitude, for the noise estimate (or give
                   noise_mm_s, the velocity noise sigma_v).
    growth_mm    : largest growth toward the wall; default 1.5 times the mean
                   image voxel size.
    smoothing_mm : radius of the opening and closing; default the mean image
                   voxel size.
    Returns dict: mask (fine grid, bool), voxel_to_world (of the fine grid),
    oversampling, fraction (float32, fraction of each image voxel in the
    mask), lumen (image voxels at least half in the mask).
    """
    f = int(oversampling)
    voxel_size = float(np.mean(array_axis_spacing(voxel_to_world)))
    growth_mm = 1.5 * voxel_size if growth_mm is None else float(growth_mm)
    smoothing_mm = voxel_size if smoothing_mm is None else float(smoothing_mm)
    pcmra = np.asarray(pcmra, float)
    log_pcmra = np.log1p(pcmra)
    threshold = otsu_threshold(log_pcmra[pcmra > np.percentile(pcmra, 50)])
    fine, fine_vtw = supersample(log_pcmra, voxel_to_world, f, "linear")
    spacing = array_axis_spacing(fine_vtw)
    mask = fine > threshold
    mask = dilate_mm(erode_mm(mask, spacing, smoothing_mm), spacing, smoothing_mm)      # opening
    mask = _keep_components(mask, fine_vtw, minimum_component_ml, 3 * voxel_size)
    mask = erode_mm(dilate_mm(mask, spacing, smoothing_mm), spacing, smoothing_mm)      # closing
    for k in range(mask.shape[0]):
        mask[k] = ndimage.binary_fill_holes(mask[k])

    if velocity_t is not None and growth_mm > 0:
        velocity_t = np.asarray(velocity_t, float)
        if noise_mm_s is None:
            if magnitude_t is None:
                raise ValueError("magnitude_t or noise_mm_s is needed to grow the lumen")
            noise_mm_s = velocity_noise_in_tissue(velocity_t, magnitude_t, lumen_mask=block_average(mask, f) > 0)
        speed = np.sqrt(np.mean(np.sum(velocity_t ** 2, axis=-1), axis=0))        # RMS speed over the cycle
        speed, _ = supersample(speed, voxel_to_world, f, "linear")
        # grow through voxels with flow, connected to the mask, but not farther than growth_mm from it
        allowed = (speed > growth_noise_factor * np.sqrt(3.0) * noise_mm_s) & dilate_mm(mask, spacing, growth_mm)
        steps = int(np.ceil(growth_mm / spacing.min()))
        for _ in range(steps):
            grown = mask | (ndimage.binary_dilation(mask) & allowed)
            if np.array_equal(grown, mask):
                break
            mask = grown
    fraction = block_average(mask, f)
    return {"mask": mask, "voxel_to_world": fine_vtw, "oversampling": f, "fraction": fraction, "lumen": fraction >= 0.5}
