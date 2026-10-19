import numpy as np
import pytest

from conftest import GEOMETRIES
from slicerheart_4dflow.geometry import (
    array_axis_spacing, transform_points, world_to_axis_aligned, as_voxel_to_world, voxel_grid_world_points, voxel_spacing, voxel_to_world_matrix,
    voxel_to_world_points, world_to_array_axis_vectors, world_to_array_coordinates, world_to_voxel_points)


def test_voxel_to_world_matrix():
    d = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    m = voxel_to_world_matrix(spacing=(0.5, 2.0, 3.0), origin=(1.0, 2.0, 3.0), directions=d)
    np.testing.assert_allclose(voxel_to_world_points(m, [[1.0, 0.0, 0.0]]), [[1.0, 2.5, 3.0]])
    np.testing.assert_allclose(voxel_to_world_points(m, [[0.0, 1.0, 0.0]]), [[-1.0, 2.0, 3.0]])
    np.testing.assert_allclose(voxel_spacing(m), (0.5, 2.0, 3.0))
    np.testing.assert_allclose(array_axis_spacing(m), (3.0, 2.0, 0.5))


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_round_trip(geometry):
    m = GEOMETRIES[geometry]()
    ijk = np.random.default_rng(0).uniform(-5, 20, (10, 3))
    np.testing.assert_allclose(world_to_voxel_points(m, voxel_to_world_points(m, ijk)), ijk, atol=1e-10)
    # array coordinates are (k, j, i)
    np.testing.assert_allclose(world_to_array_coordinates(m, voxel_to_world_points(m, ijk)), ijk[:, ::-1].T,
                               atol=1e-10)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_voxel_grid_world_points(geometry):
    m = GEOMETRIES[geometry]()
    shape = (3, 4, 5)
    pts = voxel_grid_world_points(shape, m)
    assert pts.shape == shape + (3,)
    np.testing.assert_allclose(pts[2, 1, 3], voxel_to_world_points(m, [[3, 1, 2]])[0])


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_world_to_array_axis_vectors(geometry):
    m = GEOMETRIES[geometry]()
    P = world_to_array_axis_vectors(m)
    # the world vector of one voxel step along array axis 0 (k) is column 2 of M
    np.testing.assert_allclose(P @ m[:3, 2], (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(P @ m[:3, 0], (0.0, 0.0, 1.0), atol=1e-12)


def test_as_voxel_to_world_validation():
    np.testing.assert_allclose(as_voxel_to_world(np.diag([1.0, 2.0, 3.0]))[:3, :3], np.diag([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        as_voxel_to_world(np.eye(2))
    with pytest.raises(ValueError):
        as_voxel_to_world(np.zeros((4, 4)))
    bad = np.eye(4)
    bad[3, 0] = 1.0
    with pytest.raises(ValueError):
        as_voxel_to_world(bad)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_world_to_axis_aligned(geometry):
    m = GEOMETRIES[geometry]()
    a = world_to_axis_aligned(m)
    ijk = np.random.default_rng(1).uniform(-3, 12, (8, 3))
    np.testing.assert_allclose(transform_points(a, voxel_to_world_points(m, ijk)), ijk * voxel_spacing(m),
                               atol=1e-10)
    if geometry != "sheared":
        # orthogonal voxel axes: rotation, possibly with reflection
        np.testing.assert_allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3), atol=1e-12)
