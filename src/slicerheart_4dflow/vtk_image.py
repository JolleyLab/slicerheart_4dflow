"""
Image resampling in physical space with VTK (vtkImageData, vtkImageReslice, vtkImageShrink3D).

Images are numpy arrays [k, j, i] with a voxel_to_world matrix, as everywhere
in this package. They are converted to vtkImageData with the origin, spacing
and direction of voxel_to_world, so that VTK resamples between grids in
physical (world) space: the output grid is given by its own voxel_to_world
and shape, and may have any spacing, origin and orientation. voxel_to_world
must be a rotation (with flips) times a scaling, the geometry of an image
file; sheared grids are not supported. Requires the vtk extra.
"""

from __future__ import annotations

import numpy as np


def _vtk():
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as e:      # pragma: no cover
        raise ImportError("image resampling needs VTK: pip install slicerheart_4dflow[vtk]") from e
    return vtk, numpy_support


def image_geometry(voxel_to_world):
    """(origin, spacing, direction 3x3) of a voxel_to_world matrix. Raises ValueError if the voxel axes are not
    orthogonal."""
    m = np.asarray(voxel_to_world, float)
    spacing = np.linalg.norm(m[:3, :3], axis=0)
    direction = m[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("voxel_to_world has non-orthogonal axes (shear), which vtkImageData cannot represent")
    return m[:3, 3].copy(), spacing, direction


def _vtk_matrix3(direction):
    vtk, _ = _vtk()
    matrix = vtk.vtkMatrix3x3()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, float(direction[r, c]))
    return matrix


def to_vtk_image(array, voxel_to_world):
    """vtkImageData with the scalars of a [k, j, i] array and the geometry of voxel_to_world."""
    vtk, numpy_support = _vtk()
    array = np.ascontiguousarray(array)
    origin, spacing, direction = image_geometry(voxel_to_world)
    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    image.SetDirectionMatrix(_vtk_matrix3(direction))
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array.ravel(), deep=True))
    return image


def from_vtk_image(image):
    """[k, j, i] array of the scalars of a vtkImageData."""
    _, numpy_support = _vtk()
    ni, nj, nk = image.GetDimensions()
    return numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(nk, nj, ni).copy()


def resample(array, voxel_to_world, output_shape, output_voxel_to_world, interpolation="linear"):
    """
    Resample a [k, j, i] array on the grid of voxel_to_world to the grid of
    output_voxel_to_world and output_shape (nk, nj, ni), in world space
    (vtkImageReslice). interpolation: "linear", "cubic" or "nearest". Output
    points within half a voxel outside the input take the border value,
    points further out are 0.
    """
    vtk, _ = _vtk()
    origin, spacing, direction = image_geometry(output_voxel_to_world)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(to_vtk_image(array, voxel_to_world))
    reslice.SetOutputOrigin(*origin)
    reslice.SetOutputSpacing(*spacing)
    reslice.SetOutputDirection(*direction.ravel())
    nk, nj, ni = output_shape
    reslice.SetOutputExtent(0, ni - 1, 0, nj - 1, 0, nk - 1)
    {"linear": reslice.SetInterpolationModeToLinear, "cubic": reslice.SetInterpolationModeToCubic,
     "nearest": reslice.SetInterpolationModeToNearestNeighbor}[interpolation]()
    reslice.SetBackgroundLevel(0.0)
    reslice.BorderOn()
    reslice.Update()
    return from_vtk_image(reslice.GetOutput())


def fine_voxel_to_world(voxel_to_world, factors):
    """voxel_to_world of a grid factors = (f_i, f_j, f_k) (or one number) times finer than the grid of voxel_to_world,
    aligned with its voxels: each voxel holds f_i x f_j x f_k fine voxels, fine voxel f is at index
    (f + 0.5) / factor - 0.5 of the coarse grid."""
    factors = np.broadcast_to(np.asarray(factors, float), (3,))
    scale = np.eye(4)
    scale[:3, :3] = np.diag(1.0 / factors)
    scale[:3, 3] = 0.5 / factors - 0.5
    return np.asarray(voxel_to_world, float) @ scale


def supersample(array, voxel_to_world, factor, interpolation="linear"):
    """Resample a [k, j, i] array to the aligned grid factor times finer along each axis.
    Returns (array, voxel_to_world of the fine grid)."""
    fine_vtw = fine_voxel_to_world(voxel_to_world, factor)
    shape = tuple(int(n * factor) for n in np.shape(array))
    return resample(np.asarray(array, np.float32), voxel_to_world, shape, fine_vtw, interpolation), fine_vtw


def block_average(array, factors):
    """Mean of each block of f_k x f_j x f_i voxels of a [k, j, i] array on an aligned fine grid (vtkImageShrink3D
    with averaging): the value of each voxel of the coarse grid. factors = (f_i, f_j, f_k) or one number."""
    vtk, _ = _vtk()
    f_i, f_j, f_k = (int(f) for f in np.broadcast_to(np.asarray(factors), (3,)))
    array = np.asarray(array, np.float32)
    nk, nj, ni = (n // f for n, f in zip(array.shape, (f_k, f_j, f_i)))
    shrink = vtk.vtkImageShrink3D()
    shrink.SetInputData(to_vtk_image(array[:nk * f_k, :nj * f_j, :ni * f_i], np.eye(4)))
    shrink.SetShrinkFactors(f_i, f_j, f_k)
    shrink.AveragingOn()
    shrink.Update()
    return from_vtk_image(shrink.GetOutput()).astype(np.float32)


def mask_fraction(mask, mask_voxel_to_world, shape, voxel_to_world):
    """
    Fraction (float32, 0 to 1) of each voxel of the grid (shape (nk, nj, ni),
    voxel_to_world) inside a mask on another grid (mask_voxel_to_world), for
    example a segmentation on a finer grid. The mask is resampled in world
    space (nearest neighbour) to the grid F_i x F_j x F_k times finer than the
    target grid and aligned with its voxels, F the number of mask voxels per
    target voxel along each axis (at least 1), and the fine voxels of each
    target voxel are averaged. Exact for a finer grid aligned with the target
    voxels (and for the same grid or a crop of it).
    """
    voxel_to_world = np.asarray(voxel_to_world, float)
    target_to_mask = np.linalg.inv(np.asarray(mask_voxel_to_world, float)) @ voxel_to_world
    factors = np.maximum(1, np.round(np.linalg.norm(target_to_mask[:3, :3], axis=0)).astype(int))   # (F_i, F_j, F_k)
    fine_shape = tuple(int(n * f) for n, f in zip(shape, factors[::-1]))
    fine = resample(np.asarray(mask).astype(np.uint8), mask_voxel_to_world, fine_shape,
                    fine_voxel_to_world(voxel_to_world, factors), "nearest")
    return block_average(fine, factors)
