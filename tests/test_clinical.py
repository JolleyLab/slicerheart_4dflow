"""
Clinical tier (docs/VALIDATION_STRATEGY.md).

Two groups of tests share the check functions in tests/clinical/checks.py:

  mock case   a phantom written to disk in the exact case layout (Slicer 5D
              sequences, segmentation, planes, manifest, reference flows) by
              tests/clinical/casegen.py. Exercises the loader, every check
              and the transfer audit in CI, with a known answer. SYNTHETIC
              tier: synthetic baseline, `synthetic` marker.
  real cases  vendor cases found under tests/data/clinical/<vendor>/ (or
              SLICERHEART_4DFLOW_CLINICAL_DATA). CLINICAL tier: clinical
              baseline, `clinical` marker, skipped when absent.

    python -m pytest -m clinical            # only the clinical tier
    python -m pytest -m "not clinical"      # synthetic tier (default CI)
"""

import os
from pathlib import Path

import numpy as np
import pytest

from clinical import checks
from clinical.casegen import make_mock_bifurcation_case
from clinical.transfer_audit import transfer_audit, to_markdown
from slicerheart_4dflow.io import load_case, read_flow_sequence, write_flow_sequence, planes_to_frame, to_ras

VENDORS = ("siemens", "philips", "ge")


def clinical_root(vendor):
    return Path(os.environ.get("SLICERHEART_4DFLOW_CLINICAL_DATA", Path(__file__).parent / "data" / "clinical")) / vendor


def find_cases(vendor):
    root = clinical_root(vendor)
    return sorted(p for p in root.glob("*/manifest.json")) if root.exists() else []


CASES = [(v, m) for v in VENDORS for m in find_cases(v)]
CASE_IDS = [m.parent.name for _, m in CASES]


# ---------------------------------------------------------------------------
# Mock case: the clinical path on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock_clinical")
    return load_case(make_mock_bifurcation_case(root))


@pytest.fixture(scope="module")
def mock_pipeline(mock_case):
    return checks.run_pipeline(mock_case)


def test_mock_case_layout_and_manifest(mock_case):
    checks.check_manifest(mock_case)
    man = mock_case["manifest"]
    assert man["vendor"] == "mock" and man["aliasing_present"]
    assert mock_case["truth"] is not None
    assert mock_case["times_ms"][1] == pytest.approx(man["rr_ms"] / man["n_phases"])


def test_mock_case_consistency(mock_case, mock_pipeline, metrics):
    checks.check_consistency(mock_case, mock_pipeline, metrics, tier="synthetic")


def test_mock_case_reference(mock_case, mock_pipeline, metrics):
    checks.check_reference(mock_case, mock_pipeline, metrics, tier="synthetic")


def test_mock_case_stability(mock_case, mock_pipeline, metrics):
    checks.check_stability(mock_case, mock_pipeline, metrics, tier="synthetic")


def test_mock_case_against_truth(mock_case, mock_pipeline, metrics):
    """Only the mock has a truth: the unwrapped field must be within the
    acquisition noise of it and the offset must be recovered."""
    truth, lumen = mock_case["truth"], mock_case["lumen"]
    err = np.sqrt(np.mean((mock_pipeline["unwrapped"] - truth["velocity_true"])[:, lumen] ** 2))
    noise = float(np.mean(mock_pipeline["report"]["velocity_noise_mm_s"]))
    metrics.record("mock_bifurcation_001/unwrapped_vs_truth_rmse_mm_s", err, atol=1.0)
    assert err < 1.3 * noise
    off_err = np.abs(mock_pipeline["background"]["offset"] - truth["offset"])[truth["tissue"] | lumen].max()
    metrics.record("mock_bifurcation_001/offset_max_error_mm_s", off_err, atol=0.5)
    assert off_err < 6.0                                        # 0.6 cm/s


def test_mock_case_semisynthetic(mock_case, metrics):
    checks.check_semisynthetic(mock_case, metrics, tier="synthetic")


def test_transfer_audit_self_consistent(mock_case):
    """A mock case audited against a synthetic case built at its own
    settings must agree on every paired quantity (the audit machinery
    itself must not flag)."""
    rows = transfer_audit(mock_case)
    md = to_markdown(rows, mock_case["manifest"]["case_id"])
    assert "| static_residual_rms_mm_s |" in md
    flagged = [r for r in rows if r[4]]
    assert not flagged, flagged
    finite = [r for r in rows if np.isfinite(r[3])]
    assert len(finite) >= 8
    assert all(0.7 < r[3] < 1.4 for r in finite), [(r[0], r[3]) for r in finite]


def test_io_units_and_component_frames(tmp_path, mock_case):
    """cm/s on disk and RAS components must come back as mm/s LPS."""
    v, M, times = mock_case["velocity_t"][:2], mock_case["voxel_to_world"], mock_case["times_ms"][:2]
    p = tmp_path / "v_cm_ras.seq.nrrd"
    write_flow_sequence(p, v, M, times, velocity_units="cm/s", component_frame="RAS")
    r = read_flow_sequence(p, velocity_units="cm/s", component_frame="RAS")
    np.testing.assert_allclose(r["velocity_t"], v, atol=0.05)
    np.testing.assert_allclose(r["voxel_to_world"], M)
    np.testing.assert_allclose(r["times_ms"], times)
    # reading it as LPS mm/s would be wrong by a factor 10 and two signs: the manifest fields matter
    wrong = read_flow_sequence(p)["velocity_t"]
    assert np.allclose(wrong[..., 2] * 10.0, v[..., 2], atol=0.05) and np.allclose(wrong[..., 0] * -10.0, v[..., 0], atol=0.05)
    # RAS conversion of a data set and of planes
    v_ras, M_ras, pts = to_ras(v, M, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(pts, [[-1.0, -2.0, 3.0]])
    np.testing.assert_allclose(M_ras[:3, 3], M[:3, 3] * [-1, -1, 1])
    pl = planes_to_frame({"a": {"centre": (1.0, 2.0, 3.0), "normal": (0.0, 0.0, 1.0), "radius": 5.0}}, "RAS", "LPS")
    assert pl["a"]["centre"] == (-1.0, -2.0, 3.0) and pl["a"]["radius"] == 5.0


# ---------------------------------------------------------------------------
# Real cases
# ---------------------------------------------------------------------------

@pytest.mark.clinical
@pytest.mark.parametrize("vendor", VENDORS)
def test_vendor_case_present(vendor):
    if not find_cases(vendor):
        pytest.skip(f"no {vendor} clinical case; see tests/clinical/{vendor}/README.md")


@pytest.fixture(scope="module")
def real_case(request):
    return load_case(request.param)


@pytest.fixture(scope="module")
def real_pipeline(real_case):
    return checks.run_pipeline(real_case)


def _real(fn):
    return pytest.mark.clinical(pytest.mark.parametrize("real_case", [m for _, m in CASES], ids=CASE_IDS,
                                                        indirect=True)(fn))


@_real
def test_clinical_manifest(real_case):
    checks.check_manifest(real_case)
    assert real_case["manifest"]["vendor"] in VENDORS and real_case["manifest"]["license"].startswith("CC BY")


@_real
def test_clinical_consistency(real_case, real_pipeline, metrics):
    checks.check_consistency(real_case, real_pipeline, metrics, tier="clinical")


@_real
def test_clinical_reference(real_case, real_pipeline, metrics):
    checks.check_reference(real_case, real_pipeline, metrics, tier="clinical")


@_real
def test_clinical_stability(real_case, real_pipeline, metrics):
    checks.check_stability(real_case, real_pipeline, metrics, tier="clinical")


@_real
def test_clinical_semisynthetic(real_case, metrics):
    checks.check_semisynthetic(real_case, metrics, tier="clinical")


@_real
def test_clinical_transfer_audit(real_case, metrics):
    """Records the audit (never fails on a flag: a flag is a finding to act
    on, see docs/VALIDATION_STRATEGY.md section 3) and writes audit.md next
    to the case."""
    cid = real_case["manifest"]["case_id"]
    rows = transfer_audit(real_case)
    for name, c, s, ratio, flagged in rows:
        metrics.record(f"{cid}/audit/{name}/clinical", c, direction="record", tier="clinical")
        metrics.record(f"{cid}/audit/{name}/synthetic_matched", s, direction="record", tier="clinical")
    (real_case["path"] / "audit.md").write_text(to_markdown(rows, cid) + "\n")
    print(to_markdown(rows, cid))
