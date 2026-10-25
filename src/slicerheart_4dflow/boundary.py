"""
Vessel boundary in each frame, from an approximate segmentation and the magnitude images.

A segmentation of a vessel (drawn by hand, or the automatic lumen) is approximate and static, while the vessel wall
moves over the cardiac cycle (the large arteries widen in systole), and the flow near the wall is easily lost: a region
that is too small by half a voxel can miss 10-20% of the flow of a vessel a few voxels across.
refine_vessel_boundary moves the boundary of the approximate region in or out in each frame, by at most a few voxels,
smoothly along the boundary, so that its shape is kept and no region that is not connected to it can be added:

- the magnitude I of a voxel near the boundary is taken as a mix of blood and of the surrounding tissue,
  I = f I_in + (1 - f) I_out, so N = (I - I_out) / (I_in - I_out), clipped to [0, 1], is the fraction of blood in it
  (partial volume). I_in and I_out are local levels in the frame: Gaussian-weighted means of I over the voxels at least
  one voxel inside the region, and over a ring outside the band around the boundary (other vessels excluded).
- the offset of the boundary is the excess blood volume (the excess blood fraction N - [inside the region], in a band
  around the boundary) per area of the boundary, both summed with a Gaussian weight along the boundary: the mean
  displacement of the boundary in the neighbourhood (positive: the vessel is larger than the region), clamped to the
  maximum offset. Where the levels cannot be estimated from nearby voxels or the contrast is too small, the region is
  kept.
- the vessel in a frame is the set of points whose signed distance from the approximate boundary is at most the offset;
  the fraction of each voxel in it (partial volume, subvoxel boundary) can be used as the lumen mask of plane_flow, which
  accepts one mask per frame.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import array_axis_spacing, world_to_array_coordinates, voxel_grid_world_points


def signed_distance_mm(mask, voxel_to_world):
    """Signed distance (mm) of each voxel centre from the boundary of a mask: negative inside, positive outside."""
    mask = np.asarray(mask, bool)
    spacing = array_axis_spacing(voxel_to_world)
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    # the boundary is half a voxel from the centres of the voxels next to it
    return (np.where(mask, -inside, outside) + np.where(mask, 0.5, -0.5) * spacing.min()).astype(np.float32)


def _signed_distance_on_grid(region, region_voxel_to_world, shape, voxel_to_world, margin_mm=0.0):
    """Signed distance (mm) from the boundary of region (on its own grid) at the voxel centres of another grid.
    The region grid is padded by margin_mm, so that the distance is known up to margin_mm outside the region also when its
    grid is cropped to the region (as the labelmap of a segment)."""
    pad = np.ceil(margin_mm / array_axis_spacing(region_voxel_to_world)).astype(int) + 1     # (k, j, i)
    region = np.pad(region, [(p, p) for p in pad])
    region_voxel_to_world = np.array(region_voxel_to_world, float)
    region_voxel_to_world[:3, 3] -= region_voxel_to_world[:3, :3] @ pad[::-1]                # columns are i, j, k
    distance = signed_distance_mm(region, region_voxel_to_world)
    points = voxel_grid_world_points(shape, voxel_to_world).reshape(-1, 3)
    coordinates = world_to_array_coordinates(region_voxel_to_world, points)
    far = float(np.abs(distance[np.isfinite(distance)]).max()) if np.isfinite(distance).any() else 1e6
    values = ndimage.map_coordinates(distance, coordinates, order=1, mode="constant", cval=far)
    return values.reshape(shape).astype(np.float32)


def refine_vessel_boundary(region, magnitude_t, voxel_to_world, region_voxel_to_world=None, exclude=None,
                           max_offset_voxels=2.0, sigma_mm=4.0, levels_sigma_mm=4.0, min_level_weight=0.02,
                           min_contrast=0.1):
    """
    Boundary of a vessel in each frame, from an approximate segmentation of it and the magnitude images.

    region             : approximate segmentation of the vessel, bool (or fraction >= 0.5), on the grid of
                         region_voxel_to_world (default: the image grid; for example the lumen within a branch segment,
                         on the grid of the segmentation, which may be finer than the image)
    magnitude_t        : (nt, nk, nj, ni) magnitude images
    voxel_to_world     : 4x4 voxel to world matrix of the images
    exclude            : optional (nk, nj, ni) bool on the image grid: other vessels, which the boundary does not move
                         into and which are not used as the outside level
    max_offset_voxels  : how far the boundary may move in or out, in image voxels (largest voxel size)
    sigma_mm           : width of the Gaussian that averages the offset along the boundary
    levels_sigma_mm    : width of the Gaussian of the local inside and outside levels
    min_level_weight   : minimum Gaussian-weighted fraction of inside and of outside voxels for the levels to be used
                         (where it is smaller, the offset is 0)
    min_contrast       : minimum difference of the inside and outside levels, relative to the inside level (where it
                         is smaller, there is no edge to find, and the offset is 0)

    Returns a dict:
      offset_mm          : (nt, nk, nj, ni) float32 offset of the boundary at each voxel (positive outward)
      signed_distance_mm : (nk, nj, ni) float32 signed distance of each voxel centre from the approximate boundary
      fraction           : (nt, nk, nj, ni) float32 fraction of each voxel in the vessel in each frame (the region
                           where the signed distance is at most the offset; a linear ramp over one voxel), for the
                           lumen_mask of plane_flow
    """
    magnitude_t = np.asarray(magnitude_t)
    shape = magnitude_t.shape[1:4]
    voxel_to_world = np.asarray(voxel_to_world, float)
    spacing = array_axis_spacing(voxel_to_world)
    voxel = float(spacing.max())
    max_offset = max_offset_voxels * voxel
    half_band = max_offset + voxel
    ring_width = 2.0 * voxel
    if region_voxel_to_world is None:
        phi_full = signed_distance_mm(np.asarray(region) >= 0.5, voxel_to_world)
    else:
        # the region may be cropped to its extent: the distance is needed up to the outer edge of the Gaussian weights
        margin = half_band + ring_width + 3.0 * max(sigma_mm, levels_sigma_mm) + voxel
        phi_full = _signed_distance_on_grid(np.asarray(region) >= 0.5, np.asarray(region_voxel_to_world, float), shape,
                                            voxel_to_world, margin_mm=margin)
    nt = magnitude_t.shape[0]
    offset_full = np.zeros((nt,) + shape, np.float32)
    near = phi_full <= half_band + ring_width + 3.0 * max(sigma_mm, levels_sigma_mm)
    if not near.any() or not (phi_full < 0).any():
        return {"offset_mm": offset_full, "signed_distance_mm": phi_full, "fraction": _fraction(phi_full, offset_full, spacing)}
    nz = np.nonzero(near)
    box = tuple(slice(int(a.min()), int(a.max()) + 1) for a in nz)
    phi = phi_full[box]
    excluded = np.zeros(phi.shape, bool) if exclude is None else (np.asarray(exclude, bool)[box] & (phi > 0))
    band = (np.abs(phi) <= half_band) & ~excluded
    # Core: voxels at least one voxel inside (vessels a few voxels across have them; a deeper core would be missing
    # there, and the inside level would come from distant parts of the region)
    core = phi < -voxel
    for depth in (0.5 * voxel, 0.0):
        if core.sum() >= 4:
            break
        core = phi < -depth
    ring = (phi > half_band) & (phi <= half_band + ring_width) & ~excluded
    sigma_levels = levels_sigma_mm / spacing
    sigma = sigma_mm / spacing
    weight_in = ndimage.gaussian_filter(core.astype(np.float32), sigma_levels)
    weight_out = ndimage.gaussian_filter(ring.astype(np.float32), sigma_levels)
    reliable = (weight_in > min_level_weight) & (weight_out > min_level_weight)
    weight_in, weight_out = np.maximum(weight_in, 1e-6), np.maximum(weight_out, 1e-6)
    # Boundary area: the shell of voxels within half a voxel of the approximate boundary, whose thickness is one voxel
    shell = ndimage.gaussian_filter((np.abs(phi) <= 0.5 * voxel).astype(np.float32), sigma)
    inside = (phi < 0).astype(np.float32)
    for t in range(nt):
        image = np.asarray(magnitude_t[t][box], np.float32)
        level_in = ndimage.gaussian_filter(image * core, sigma_levels) / weight_in
        level_out = ndimage.gaussian_filter(image * ring, sigma_levels) / weight_out
        contrast = level_in - level_out
        edge = contrast > min_contrast * np.abs(level_in)
        blood = np.clip(np.where(edge, (image - level_out) / np.where(edge, contrast, 1.0), 0.0), 0.0, 1.0)
        evidence = band & reliable & edge
        excess = (blood - inside) * evidence
        # mean displacement: excess blood volume per boundary area (the shell has the thickness of one voxel)
        known = ndimage.gaussian_filter(((np.abs(phi) <= 0.5 * voxel) & reliable & edge).astype(np.float32), sigma)
        offset = voxel * ndimage.gaussian_filter(excess, sigma) / np.maximum(known, 1e-6)
        offset = np.where(known > 0.5 * shell, offset, 0.0)
        offset_full[(t,) + box] = np.clip(offset, -max_offset, max_offset)
    fraction = _fraction(phi_full, offset_full, spacing)
    if exclude is not None:
        fraction[:, np.asarray(exclude, bool) & (phi_full > 0)] = 0.0
    return {"offset_mm": offset_full, "signed_distance_mm": phi_full, "fraction": fraction}


def _fraction(phi, offset_t, spacing):
    """Fraction of each voxel inside the boundary phi = offset: a linear ramp over one voxel (the mean voxel size)."""
    width = float(np.cbrt(np.prod(spacing)))
    return np.clip(0.5 + (offset_t - phi[None]) / width, 0.0, 1.0).astype(np.float32)
