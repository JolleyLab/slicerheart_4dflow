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
    """A rigid world transform (rotation, optional reflection, translation), 4x4."""
    t = np.eye(4)
    t[:3, :3] = rotation_matrix((1.0, -2.0, 0.5), 37.0)
    if reflect:
        t[:3, :3] = t[:3, :3] @ np.diag([-1.0, -1.0, 1.0]) @ np.diag([1.0, 1.0, -1.0])
    t[:3, 3] = (12.5, -40.0, 7.0)
    return t


# Changes of the world coordinate system that must not change any result
WORLD_TRANSFORMS = {
    "LPS to RAS": lambda: np.diag([-1.0, -1.0, 1.0, 1.0]),
    "rotation, reflection, translation": lambda: world_transform(),
}


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


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
#
# Tests record the numbers behind their assertions (RMSE, residual wrapped
# voxels, WSS ratios, ...) through the `metrics` fixture. Each recorded value
# is compared with tests/metrics/baseline.json: an error metric may not get
# worse than the baseline by more than the tolerance, so a change that stays
# inside a test's hard limit but degrades a method still fails. All values of
# a run are written to tests/metrics/latest.json.
#
#     python -m pytest                     # compare with the baseline
#     python -m pytest --update-baseline   # accept the current values
#     python -m pytest --metrics-tolerance 0.02
#
# A metric that is absent from the baseline is recorded without comparison.

import json

METRICS_DIR = Path(__file__).parent / "metrics"
BASELINE_PATH = METRICS_DIR / "baseline.json"
LATEST_PATH = METRICS_DIR / "latest.json"


def pytest_addoption(parser):
    parser.addoption("--update-baseline", action="store_true", default=False,
                     help="write the metrics recorded in this run into tests/metrics/baseline.json")
    parser.addoption("--metrics-tolerance", type=float, default=0.05,
                     help="relative tolerance for regression of recorded metrics (default 0.05)")


class MetricRecorder:
    """record(name, value, direction="lower", tol=None, atol=0.0, tier="synthetic")

    direction: "lower" (an error: may not rise above baseline (1 + tol) + atol),
               "higher" (a score: may not fall below baseline (1 - tol) - atol),
               "exact" (must stay within baseline +/- (|baseline| tol + atol)),
               "record" (stored for reference, never compared: timings and
               other machine-dependent values).
    tier:      "synthetic" (phantoms, CFD, mock cases; tests/metrics/baseline.json)
               or "clinical" (real scanner cases; tests/metrics/clinical_baseline.json).
               The two baselines are kept apart so that numbers measured on
               data with a known answer are never mixed with numbers measured
               on data without one.
    """

    TIERS = ("synthetic", "clinical")

    def __init__(self, baselines: dict, tolerance: float, update: bool):
        self.baselines, self.tolerance, self.update = baselines, tolerance, update
        self.values = {t: {} for t in self.TIERS}

    def record(self, name, value, direction="lower", tol=None, atol=0.0, tier="synthetic"):
        if tier not in self.TIERS:
            raise ValueError(f"tier must be one of {self.TIERS}")
        value = float(value)
        self.values[tier][name] = value
        baseline = self.baselines[tier]
        if direction == "record" or self.update or name not in baseline:
            return value
        ref = float(baseline[name])
        tol = self.tolerance if tol is None else tol
        margin = abs(ref) * tol + atol
        if direction == "lower":
            ok = value <= ref + margin
        elif direction == "higher":
            ok = value >= ref - margin
        elif direction == "exact":
            ok = abs(value - ref) <= margin
        else:
            raise ValueError(direction)
        assert ok, (f"{tier} metric {name!r} regressed: {value:.6g} vs baseline {ref:.6g} "
                    f"(direction {direction}, tolerance {tol:g} + {atol:g})")
        return value


BASELINE_PATHS = {"synthetic": BASELINE_PATH, "clinical": METRICS_DIR / "clinical_baseline.json"}
LATEST_PATHS = {"synthetic": LATEST_PATH, "clinical": METRICS_DIR / "clinical_latest.json"}


@pytest.fixture(scope="session")
def metrics(request):
    baselines = {t: (json.loads(p.read_text()) if p.exists() else {}) for t, p in BASELINE_PATHS.items()}
    rec = MetricRecorder(baselines, request.config.getoption("--metrics-tolerance"),
                         request.config.getoption("--update-baseline"))
    yield rec
    METRICS_DIR.mkdir(exist_ok=True)
    for tier in MetricRecorder.TIERS:
        if not rec.values[tier]:
            continue
        LATEST_PATHS[tier].write_text(json.dumps(dict(sorted(rec.values[tier].items())), indent=2) + "\n")
        if rec.update:
            merged = dict(baselines[tier]); merged.update(rec.values[tier])
            BASELINE_PATHS[tier].write_text(json.dumps(dict(sorted(merged.items())), indent=2) + "\n")


def pytest_collection_modifyitems(config, items):
    """Every test that is not on clinical data is on synthetic data."""
    for item in items:
        if item.get_closest_marker("clinical") is None:
            item.add_marker(pytest.mark.synthetic)
