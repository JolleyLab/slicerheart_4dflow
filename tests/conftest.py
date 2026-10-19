"""
Shared fixtures.

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

    Returns (velocity (nz, ny, nx, 3) with components (vz, vy, vx),
    lumen mask (nz, ny, nx), spacing (dz, dy, dx) in mm).
    """
    import h5py
    with h5py.File(cfd_path(), "r") as h:
        d = {k: h[k][0] for k in h.keys()}
    # u is along array axis 0, v along axis 1, w along axis 2 (the divergence
    # of the CFD field is by far the smallest with this assignment)
    velocity = np.stack([d["u"], d["v"], d["w"]], -1).astype(float)
    return velocity, d["mask"] > 0.5, tuple(float(x) for x in d["dx"])


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
