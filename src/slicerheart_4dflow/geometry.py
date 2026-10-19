"""
Image geometry: mapping between voxel and world (physical) coordinates.

Conventions used throughout the package
---------------------------------------
Image arrays are indexed [k, j, i] (numpy C order, i varies fastest), as
returned by ITK GetArrayFromImage, slicer.util.arrayFromVolume or a VTK
image's point data reshaped to (nk, nj, ni). Time-resolved arrays add a
leading time axis: (nt, nk, nj, ni) and (nt, nk, nj, ni, 3).

voxel_to_world is a 4x4 homogeneous matrix that maps voxel coordinates
(i, j, k, 1) (column vector; i is the last array axis) to world coordinates
(x, y, z, 1) in mm. Any affine mapping is allowed: anisotropic spacing,
oblique or flipped axes, even shear.

The world (physical) coordinate system is not assumed to be any particular
one (e.g. LPS or RAS): all computations depend only on the input being
consistent. Velocity vectors are given by their world components
(vx, vy, vz), i.e., in the same coordinate system as voxel_to_world. Points (plane
centres, seeds, centerlines, pathline points) are world (x, y, z) positions
in mm, and directions (plane normals) are world vectors.
"""

from __future__ import annotations

import numpy as np


def voxel_to_world_matrix(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), directions=None):
    """
    Build a 4x4 voxel-to-world matrix.

    spacing    : (si, sj, sk) voxel size along the i, j, k axes in mm.
    origin     : world position (x, y, z) of voxel (0, 0, 0).
    directions : optional 3x3 matrix whose columns are the world directions
                 of the i, j, k axes (unit vectors). Identity if None.
    """
    m = np.eye(4)
    d = np.eye(3) if directions is None else np.asarray(directions, float)
    m[:3, :3] = d * np.asarray(spacing, float)[None, :]
    m[:3, 3] = np.asarray(origin, float)
    return m


def as_voxel_to_world(voxel_to_world) -> np.ndarray:
    """Validate and return voxel_to_world as a float (4, 4) array. A (3, 3)
    matrix is accepted as the linear part with zero origin."""
    m = np.asarray(voxel_to_world, float)
    if m.shape == (3, 3):
        full = np.eye(4)
        full[:3, :3] = m
        m = full
    if m.shape != (4, 4):
        raise ValueError(f"voxel_to_world must be a 4x4 matrix, got shape {m.shape}")
    if not np.allclose(m[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("voxel_to_world must be an affine matrix (last row 0, 0, 0, 1)")
    if abs(np.linalg.det(m[:3, :3])) < 1e-12:
        raise ValueError("voxel_to_world is singular")
    return m


def voxel_spacing(voxel_to_world) -> np.ndarray:
    """Voxel size (si, sj, sk) in mm: lengths of the voxel axes in world space."""
    m = as_voxel_to_world(voxel_to_world)
    return np.linalg.norm(m[:3, :3], axis=0)


def array_axis_spacing(voxel_to_world) -> np.ndarray:
    """Voxel size along the array axes (k, j, i), in mm."""
    return voxel_spacing(voxel_to_world)[::-1]


def voxel_to_world_points(voxel_to_world, ijk) -> np.ndarray:
    """(n, 3) voxel coordinates (i, j, k) -> (n, 3) world points (x, y, z)."""
    m = as_voxel_to_world(voxel_to_world)
    ijk = np.asarray(ijk, float)
    return ijk @ m[:3, :3].T + m[:3, 3]


def world_to_voxel_points(voxel_to_world, xyz) -> np.ndarray:
    """(n, 3) world points (x, y, z) -> (n, 3) continuous voxel coordinates (i, j, k)."""
    m = np.linalg.inv(as_voxel_to_world(voxel_to_world))
    xyz = np.asarray(xyz, float)
    return xyz @ m[:3, :3].T + m[:3, 3]


def world_to_array_coordinates(voxel_to_world, xyz) -> np.ndarray:
    """(n, 3) world points -> (3, n) continuous array coordinates (k, j, i),
    as expected by scipy.ndimage.map_coordinates."""
    return world_to_voxel_points(voxel_to_world, np.atleast_2d(xyz))[:, ::-1].T


def voxel_grid_world_points(shape, voxel_to_world) -> np.ndarray:
    """World position of every voxel centre, shape (nk, nj, ni, 3)."""
    m = as_voxel_to_world(voxel_to_world)
    k, j, i = np.meshgrid(*[np.arange(n, dtype=float) for n in shape[:3]], indexing="ij")
    ijk = np.stack([i, j, k], -1)
    return ijk @ m[:3, :3].T + m[:3, 3]


def world_to_array_axis_vectors(voxel_to_world) -> np.ndarray:
    """
    3x3 matrix P that maps a world vector (vx, vy, vz) to its contravariant
    components along the array axes (k, j, i), in voxels per unit of the
    vector (e.g. voxels/s for a velocity): u = P @ v, with u[0] along axis 0
    (k), u[1] along axis 1 (j), u[2] along axis 2 (i).
    """
    m = as_voxel_to_world(voxel_to_world)
    return np.linalg.inv(m[:3, :3])[::-1]
