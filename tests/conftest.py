"""
Shared fixtures and helpers.

The aortic CFD example of the 4DFlowNet repository (Ferdian et al., Front Phys
2020, MIT license) is used as ground truth. It is downloaded on first use into
tests/data (or the folder in the SLICERHEART_4DFLOW_TEST_DATA environment
variable). Tests that need it are skipped if it cannot be obtained.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from slicerheart_4dflow.geometry import voxel_to_world_matrix

CFD_URL = "https://raw.githubusercontent.com/EdwardFerdian/4DFlowNet/master/data/example_data_HR.h5"
CFD_FILENAME = "example_data_HR.h5"


class TestDataUnavailable(RuntimeError):
    pass


def cfd_path():
    path = Path(os.environ.get("SLICERHEART_4DFLOW_TEST_DATA", Path(__file__).parent / "data")) / CFD_FILENAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(CFD_URL, tmp)
        except Exception as exc:  # network unavailable
            raise TestDataUnavailable(f"cannot download test data from {CFD_URL}: {exc}") from exc
        tmp.replace(path)
    return path


def load_cfd():
    """Aortic CFD velocity field in m/s.

    Returns (velocity (nk, nj, ni, 3) with world components (vx, vy, vz),
    lumen mask (nk, nj, ni), voxel_to_world 4x4). The world axes x, y, z are
    aligned with the voxel axes i, j, k, origin at voxel (0, 0, 0).
    """
    import h5py
    with h5py.File(cfd_path(), "r") as h:
        d = {k: h[k][0] for k in h.keys()}
    # u is along array axis 0 (k, z), v along axis 1 (j, y), w along axis 2
    # (i, x): the divergence of the CFD field is by far the smallest with
    # this assignment
    velocity = np.stack([d["w"], d["v"], d["u"]], -1).astype(float)
    spacing_kji = [float(x) for x in d["dx"]]
    return velocity, d["mask"] > 0.5, voxel_to_world_matrix(spacing=spacing_kji[::-1])


@pytest.fixture(scope="session")
def cfd():
    pytest.importorskip("h5py")
    try:
        return load_cfd()
    except TestDataUnavailable as exc:
        pytest.skip(str(exc))


def aortic_waveform(nt):
    """Systolic pulse plus a small diastolic wave, peak about 1."""
    t = np.arange(nt) / nt
    return 0.1 + 0.9 * np.exp(-((t - 0.25) / 0.09) ** 2) + 0.15 * np.exp(-((t - 0.55) / 0.08) ** 2)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotation_matrix(axis, angle_deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def world_transform(reflect=True):
    """A rigid world transform (rotation, optional reflection as in RAS <-> LPS,
    translation), 4x4."""
    t = np.eye(4)
    t[:3, :3] = rotation_matrix((1.0, -2.0, 0.5), 37.0)
    if reflect:
        t[:3, :3] = t[:3, :3] @ np.diag([-1.0, -1.0, 1.0]) @ np.diag([1.0, 1.0, -1.0])
    t[:3, 3] = (12.5, -40.0, 7.0)
    return t


def oblique_voxel_to_world(spacing=(0.8, 1.1, 1.6), shear=False):
    """Anisotropic voxels, oblique and flipped axes, non-zero origin; with
    shear=True the voxel axes are not orthogonal."""
    directions = rotation_matrix((0.3, 1.0, -0.4), 28.0) @ np.diag([1.0, -1.0, 1.0])
    m = voxel_to_world_matrix(spacing=spacing, origin=(-20.0, 15.0, 30.0), directions=directions)
    if shear:
        m[:3, :3] = m[:3, :3] @ np.array([[1.0, 0.25, 0.0], [0.0, 1.0, -0.15], [0.0, 0.0, 1.0]])
    return m


GEOMETRIES = {
    "identity": lambda: voxel_to_world_matrix(),
    "anisotropic": lambda: voxel_to_world_matrix(spacing=(0.7, 1.0, 1.5), origin=(3.0, -2.0, 5.0)),
    "oblique": lambda: oblique_voxel_to_world(),
    "sheared": lambda: oblique_voxel_to_world(shear=True),
}
