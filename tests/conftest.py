"""
Shared fixtures and helpers.

Test data files are downloaded on first use from the TestingData release of this
repository (its description lists the source and license of each file) into
tests/data (or the folder in the SLICERHEART_4DFLOW_TEST_DATA environment
variable), and verified by their sha256 hash. Tests that need a file are skipped if
it cannot be downloaded.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from slicerheart_4dflow.geometry import voxel_to_world_matrix

TEST_DATA_URL = "https://github.com/JolleyLab/slicerheart_4dflow/releases/download/TestingData/"
# Aortic CFD velocity field with lumen mask, used as ground truth by the tests marked `cfd`
CFD_FILENAME = "example_data_HR.h5"
CFD_SHA256 = "e3da2444c1aa30604c5e6550544caab672ed087988331874c0838b7f384befd8"


class TestDataUnavailable(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_data_path(filename, sha256):
    """Path of a test data file, downloaded from the TestingData release on first use and verified by its sha256 hash."""
    path = Path(os.environ.get("SLICERHEART_4DFLOW_TEST_DATA", Path(__file__).parent / "data")) / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(TEST_DATA_URL + filename, tmp)
        except Exception as exc:  # network unavailable
            raise TestDataUnavailable(f"cannot download test data from {TEST_DATA_URL + filename}: {exc}") from exc
        if _sha256(tmp) != sha256:
            tmp.unlink()
            raise RuntimeError(f"downloaded {filename} does not match its sha256 hash {sha256}")
        tmp.replace(path)
    elif _sha256(path) != sha256:
        raise RuntimeError(f"{path} does not match its sha256 hash {sha256}: delete it to download it again")
    return path


def cfd_path():
    return test_data_path(CFD_FILENAME, CFD_SHA256)


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
# a run are written to latest.json in the test report folder (REPORT_DIR).
#
#     python -m pytest                     # compare with the baseline
#     python -m pytest --update-baseline   # accept the current values
#     python -m pytest --metrics-tolerance 0.02
#
# A metric that is absent from the baseline is recorded without comparison.
#
# Metrics of the clinical tier describe local data, so their baseline is kept with
# the data, in one file: <clinical data folder>/tests/slicerheart_4dflow_baseline.json (see
# CaseMetricRecorder), and the values of a run go to clinical_latest.json in the test report folder.
#
# Test results are written to the test report folder, .test-reports at the repository root (not committed),
# or the folder set in the SLICERHEART_4DFLOW_TEST_REPORT_DIR environment variable. tests/ holds only source
# and expected values, the clinical data folder only data and its baselines.

import hashlib
import json
import time
import warnings

METRICS_DIR = Path(__file__).parent / "metrics"
BASELINE_PATH = METRICS_DIR / "baseline.json"
REPORT_DIR = Path(os.environ.get("SLICERHEART_4DFLOW_TEST_REPORT_DIR", Path(__file__).parent.parent / ".test-reports"))
LATEST_PATH = REPORT_DIR / "latest.json"
CLINICAL_LATEST_PATH = REPORT_DIR / "clinical_latest.json"
# Data set of the clinical test cases (inputs/<case>/, outputs/<case>/, manifest.json), and the baseline of all cases,
# relative to it: the local data set folder of SLICERHEART_4DFLOW_CLINICAL_DATA if set, else the revision of the
# manifest kept in this folder (CLINICAL_MANIFEST, which gives the cases without a download, and whose hash is the key
# of the CI cache), downloaded by slicerheart_4dflow.clinical_test_data when a test first needs a case. The base folder
# is .testingdata_cache/ in the repository, or the folder of SLICERHEART_4DFLOW_TESTDATA.
from slicerheart_4dflow.testing_data import ClinicalTestingData, TestingDataUnavailable, clinical_test_data  # noqa: E402
CLINICAL_MANIFEST = Path(__file__).parent / "clinical_manifest_rev001.json"
TESTING_DATA_CACHE = Path(os.environ.get("SLICERHEART_4DFLOW_TESTDATA",
                                         Path(__file__).parent.parent / ".testingdata_cache"))
_LOCAL_CLINICAL_DATA = os.environ.get("SLICERHEART_4DFLOW_CLINICAL_DATA")
CLINICAL_TESTING_DATA = None if _LOCAL_CLINICAL_DATA else ClinicalTestingData(CLINICAL_MANIFEST, TESTING_DATA_CACHE)
CLINICAL_DATA_DIR = Path(_LOCAL_CLINICAL_DATA) if _LOCAL_CLINICAL_DATA else CLINICAL_TESTING_DATA.root
CLINICAL_BASELINE_PATH = Path("tests") / "slicerheart_4dflow_baseline.json"


def clinical_case_ids():
    """Case IDs of the clinical data set: the case folders of the local data set, or the cases of the manifest of the
    cached data set (no download)."""
    if CLINICAL_TESTING_DATA is None:
        from slicerheart_4dflow.slicer_io import dataset_cases
        return dataset_cases(CLINICAL_DATA_DIR)
    return CLINICAL_TESTING_DATA.cases()


def ensure_clinical_case(case_id, prefix=None):
    """Make the files of a case available (download them into the testing data cache if needed; nothing to do for a
    local data set). prefix limits it to the files below a folder of the data set (for example
    "inputs/<case>/acquisition/"). Skips the test if the data cannot be downloaded."""
    if CLINICAL_TESTING_DATA is None:
        return
    try:
        if prefix is None:
            clinical_test_data(CLINICAL_TESTING_DATA.revision, cases=[case_id], base_dir=TESTING_DATA_CACHE)
        else:
            CLINICAL_TESTING_DATA.ensure_common()
            CLINICAL_TESTING_DATA.ensure(CLINICAL_TESTING_DATA.assets_of(prefix))
    except TestingDataUnavailable as exc:
        pytest.skip(str(exc))
# The clinical metrics of a case depend on every file in its inputs/<case> and outputs/<case> folders (the baseline of
# the case is stale if any of them changes)


def pytest_collection_modifyitems(config, items):
    """Mark every test that is not marked clinical as synthetic, so that `pytest -m synthetic` runs them.

    The tests that take a case (parameter case_ref) are run case by case, in the order of the tests within each case,
    instead of test by test over all cases: test_clinical.load caches the last cases only, so a test-by-test order
    loaded every case once per test (2 to 7 s each). The other tests keep their places."""
    for item in items:
        if item.get_closest_marker("clinical") is None:
            item.add_marker(pytest.mark.synthetic)
    positions = [i for i, item in enumerate(items)
                 if getattr(item, "callspec", None) is not None and "case_ref" in item.callspec.params]
    cases = list(dict.fromkeys(tuple(items[i].callspec.params["case_ref"]) for i in positions))
    ordered = sorted((items[i] for i in positions), key=lambda item: cases.index(tuple(item.callspec.params["case_ref"])))
    for i, item in zip(positions, ordered):
        items[i] = item


# ---------------------------------------------------------------------------
# Progress log
# ---------------------------------------------------------------------------
#
# The clinical tier runs for an hour or more, and pytest prints its summary only at the end (and its output may be
# held back by a pipe). Each run writes <test report folder>/progress.log while it runs: one line when a test starts
# (so a long or hanging test is visible while it runs) and one with its result and duration, the number of tests
# done, the outcomes so far and an estimate of the remaining time (from the durations of the previous run, kept in
# durations.json in the same folder, or from the mean duration so far). Every line is flushed at once. Follow it with
#
#     tail -f .test-reports/progress.log                      (Git Bash)
#     Get-Content .test-reports/progress.log -Wait            (PowerShell)
#
# Runs at the same time must use different report folders (SLICERHEART_4DFLOW_TEST_REPORT_DIR).

PROGRESS_PATH = REPORT_DIR / "progress.log"
DURATIONS_PATH = REPORT_DIR / "durations.json"


class ProgressLog:
    def __init__(self, path, nodeids, description):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.remaining_ids = list(nodeids)
        self.total, self.done, self.start = len(nodeids), 0, time.time()
        self.outcomes = {}
        try:
            self.previous = json.loads(DURATIONS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.previous = {}
        self.durations = {}
        known = sum(1 for nodeid in nodeids if nodeid in self.previous)
        estimate = (f", about {sum(self.previous.get(n, 0.0) for n in nodeids) / 60:.0f} min by the previous run"
                    f" ({known} of them)") if known else ""
        self.write(f"{self.total} tests: {description}{estimate}")

    def write(self, text):
        self.file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        self.file.flush()

    def counts(self):
        return ", ".join(f"{n} {o}" for o, n in sorted(self.outcomes.items()))

    def started(self, nodeid):
        self.write(f"[{self.done + 1}/{self.total}] start   {nodeid}")

    def finished(self, nodeid, outcome, duration, message=""):
        self.done += 1
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        self.durations[nodeid] = duration
        if nodeid in self.remaining_ids:
            self.remaining_ids.remove(nodeid)
        elapsed = time.time() - self.start
        mean = elapsed / self.done
        remaining = sum(self.previous.get(n, mean) for n in self.remaining_ids)
        self.write(f"[{self.done}/{self.total}] {outcome.upper():7s} {nodeid} ({duration:.1f} s){message}"
                   f" | {self.counts()} | elapsed {elapsed / 60:.1f} min, about {remaining / 60:.0f} min remaining")

    def close(self, exitstatus):
        self.write(f"finished (exit status {int(exitstatus)}): {self.counts()} in {(time.time() - self.start) / 60:.1f} min")
        self.file.close()
        DURATIONS_PATH.write_text(json.dumps(dict(self.previous, **self.durations), indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")


_progress = None


def pytest_collection_finish(session):
    global _progress
    if session.config.option.collectonly:
        return
    args = " ".join(session.config.invocation_params.args)
    _progress = ProgressLog(PROGRESS_PATH, [item.nodeid for item in session.items], f"pytest {args}".strip())


def pytest_runtest_logstart(nodeid, location):
    if _progress:
        _progress.started(nodeid)


def pytest_runtest_logreport(report):
    """One result line per test: of the call phase, or of setup if it failed or skipped (the test did not run)."""
    if not _progress or not (report.when == "call" or (report.when == "setup" and not report.passed)):
        return
    message = ""
    if report.failed:
        # the exception message (for example, of the failed assertion), not only the line where it was raised
        crash = getattr(report.longrepr, "reprcrash", None)
        text = crash.message if crash is not None else str(report.longrepr).strip().splitlines()[-1:]
        message = f": {' '.join(str(text).split())[:300]}" if text else ""
    elif report.skipped and isinstance(report.longrepr, tuple):
        message = f": {report.longrepr[2]}"
    outcome = "error" if report.failed and report.when == "setup" else report.outcome
    _progress.finished(report.nodeid, outcome, report.duration, message)


def pytest_sessionfinish(session, exitstatus):
    global _progress
    if _progress:
        _progress.close(exitstatus)
        _progress = None


def pytest_addoption(parser):
    parser.addoption("--update-baseline", action="store_true", default=False,
                     help="write the metrics recorded in this run into tests/metrics/baseline.json"
                          " (clinical tier: into tests/slicerheart_4dflow_baseline.json of the clinical data folder)")
    parser.addoption("--metrics-tolerance", type=float, default=0.05,
                     help="relative tolerance for regression of recorded metrics (default 0.05)")


class MetricRecorder:
    """record(name, value, direction="lower", tol=None, atol=0.0)

    direction: "lower" (an error: may not rise above baseline (1 + tol) + atol),
               "higher" (a score: may not fall below baseline (1 - tol) - atol),
               "exact" (must stay within baseline +/- (|baseline| tol + atol)),
               "record" (stored for reference, never compared: timings and
               other machine-dependent values).
    """

    def __init__(self, baseline: dict, tolerance: float, update: bool):
        self.baseline, self.tolerance, self.update = baseline, tolerance, update
        self.values = {}

    def record(self, name, value, direction="lower", tol=None, atol=0.0):
        value = float(value)
        self.values[name] = value
        if direction == "record" or self.update or name not in self.baseline:
            return value
        ref = float(self.baseline[name])
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
        assert ok, (f"metric {name!r} regressed: {value:.6g} vs baseline {ref:.6g} "
                    f"(direction {direction}, tolerance {tol:g} + {atol:g})")
        return value


@pytest.fixture(scope="session")
def metrics(request):
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    rec = MetricRecorder(baseline, request.config.getoption("--metrics-tolerance"),
                         request.config.getoption("--update-baseline"))
    yield rec
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(dict(sorted(rec.values.items())), indent=2) + "\n")
    if rec.update:
        merged = dict(baseline); merged.update(rec.values)
        BASELINE_PATH.write_text(json.dumps(dict(sorted(merged.items())), indent=2) + "\n")


def case_file_hashes(data_dir, case_id):
    """sha256 of every file of a case (inputs/<case> and outputs/<case> of the data set), {path relative to the data
    set: hash}."""
    data_dir = Path(data_dir)
    hashes = {}
    for folder in (data_dir / "inputs" / case_id, data_dir / "outputs" / case_id):
        for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
            if path.is_file():
                digest = hashlib.sha256()
                with open(path, "rb") as file:
                    for block in iter(lambda: file.read(1 << 22), b""):
                        digest.update(block)
                hashes[path.relative_to(data_dir).as_posix()] = digest.hexdigest()
    return hashes


class CaseMetricRecorder:
    """Records clinical metrics named "clinical/<case_id>/<metric>" and compares them with the baseline, which is kept
    with the data, in one file for all cases: <clinical data folder>/tests/slicerheart_4dflow_baseline.json

        {"package": "slicerheart_4dflow",
         "cases": {"<case id>": {"package_version": ..., "created": ...,
                                 "file_hashes": {path relative to the data set: sha256}, "metrics": {metric: value}}}}

    For each case the baseline records the hash of each case file it was computed from (inputs/<case> and
    outputs/<case>). If any of them changed (the case was exported again, the outputs were regenerated, the lumen or
    planes were edited) then the baseline of that case is stale: its metrics are recorded but not compared, and a warning names
    the changed files. With --update-baseline the metrics of each case are written to its section (merged with the
    stored metrics if the case files are unchanged, replacing them otherwise); sections of other cases are kept."""

    def __init__(self, tolerance: float, update: bool, data_dir=None):
        self.tolerance, self.update = tolerance, update
        self.data_dir = Path(data_dir if data_dir is not None else CLINICAL_DATA_DIR)
        self.baseline_path = self.data_dir / CLINICAL_BASELINE_PATH
        self._stored_cases = None   # read when the first metric is recorded, after the data set is available
        self.case_ids = set()
        self.recorders = {}  # case_id: MetricRecorder
        self.states = {}  # case_id: (current file hashes, stale)
        self.values = {}

    @property
    def stored_cases(self):
        if self._stored_cases is None:
            stored = json.loads(self.baseline_path.read_text(encoding="utf-8")) if self.baseline_path.exists() else {}
            self._stored_cases = stored.get("cases", {})
        return self._stored_cases

    def register_case(self, case_id):
        self.case_ids.add(case_id)

    def _recorder(self, case_id):
        if case_id in self.recorders:
            return self.recorders[case_id]
        key = case_id
        stored = self.stored_cases.get(key, {})
        hashes = case_file_hashes(self.data_dir, case_id)
        stored_hashes = stored.get("file_hashes", {})
        changed = sorted(set(hashes) ^ set(stored_hashes) | {f for f in hashes if stored_hashes.get(f) not in (None, hashes[f])})
        stale = bool(stored) and bool(changed)
        if stale and not self.update:
            warnings.warn(f"{key}: the baseline of the case in {self.baseline_path} is stale, its metrics are not compared"
                          f" (case files changed since it was recorded: {', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''});"
                          " run with --update-baseline to record a new baseline")
        baseline_metrics = {} if stale else stored.get("metrics", {})
        self.recorders[case_id] = MetricRecorder(baseline_metrics, self.tolerance, self.update)
        self.states[case_id] = (hashes, stale)
        return self.recorders[case_id]

    def record(self, name, value, direction="lower", tol=None, atol=0.0):
        prefix, case_id, metric = name.split("/", 2)
        if prefix != "clinical" or case_id not in self.case_ids:
            raise ValueError(f"clinical metric {name!r} must be named clinical/<registered case id>/<metric>")
        self.values[name] = float(value)
        return self._recorder(case_id).record(metric, value, direction, tol, atol)

    def write(self):
        if not self.values:
            return
        CLINICAL_LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLINICAL_LATEST_PATH.write_text(json.dumps(dict(sorted(self.values.items())), indent=2) + "\n")
        if not self.update:
            return
        import slicerheart_4dflow
        # Read the file again, so that sections of cases that were not tested in this run are kept as they are now
        stored = json.loads(self.baseline_path.read_text(encoding="utf-8")) if self.baseline_path.exists() else {}
        cases = stored.get("cases", {})
        for case_id, recorder in self.recorders.items():
            key = case_id
            hashes, stale = self.states[case_id]
            metrics = {} if stale else dict(cases.get(key, {}).get("metrics", {}))
            metrics.update(recorder.values)
            cases[key] = {
                "package_version": slicerheart_4dflow.__version__,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_hashes": hashes,
                "metrics": dict(sorted(metrics.items())),
                }
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(json.dumps({"package": "slicerheart_4dflow", "cases": dict(sorted(cases.items()))},
                                                 indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def clinical_metrics(request):
    rec = CaseMetricRecorder(request.config.getoption("--metrics-tolerance"), request.config.getoption("--update-baseline"))
    yield rec
    rec.write()
