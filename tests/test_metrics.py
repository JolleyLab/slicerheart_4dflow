"""The regression recorder itself: a worse value fails, a better one passes,
an unknown metric is recorded without comparison, directions are honoured."""

import pytest

from conftest import MetricRecorder


def test_recorder_directions():
    rec = MetricRecorder({"err": 1.0, "score": 0.9, "ratio": 1.5}, tolerance=0.05, update=False)
    rec.record("err", 1.04)                                  # within 5%
    rec.record("err", 0.5)                                   # improvement passes
    with pytest.raises(AssertionError):
        rec.record("err", 1.06)
    rec.record("score", 0.86, direction="higher")
    with pytest.raises(AssertionError):
        rec.record("score", 0.84, direction="higher")
    rec.record("ratio", 1.56, direction="exact")
    with pytest.raises(AssertionError):
        rec.record("ratio", 1.40, direction="exact")
    rec.record("new_metric", 123.0)                          # no baseline: recorded only
    assert rec.values["new_metric"] == 123.0
    rec.record("tiny", 0.0)
    rec2 = MetricRecorder({"tiny": 0.0}, 0.05, False)
    rec2.record("tiny", 0.0005, atol=0.001)                  # absolute floor for near-zero metrics
    with pytest.raises(AssertionError):
        rec2.record("tiny", 0.002, atol=0.001)


def test_update_mode_never_fails():
    rec = MetricRecorder({"err": 1.0}, 0.05, update=True)
    rec.record("err", 100.0)
    assert rec.values["err"] == 100.0


def test_clinical_baseline_is_stored_with_the_data_and_detects_changed_case_files(tmp_path, monkeypatch):
    """The clinical baseline is one file in the clinical data folder, with a section per case that stores the hashes of
    the case files; a section of changed case files is not compared (stale), and --update-baseline replaces it."""
    import json
    import warnings
    import conftest
    monkeypatch.setattr(conftest, "CLINICAL_LATEST_PATH", tmp_path / "reports" / "clinical_latest.json")
    data_dir = tmp_path / "clinical"
    case_dirs = {}
    for case in ("Case1", "Case2"):
        case_dirs[case] = data_dir / "siemens" / case
        (case_dirs[case] / "inputs").mkdir(parents=True)
        (case_dirs[case] / "inputs" / "velocity_raw.seq.nrrd").write_bytes(f"raw velocity {case}".encode())
    baseline_path = data_dir / "tests" / "slicerheart_4dflow_baseline.json"

    def recorder(update):
        rec = conftest.CaseMetricRecorder(tolerance=0.05, update=update, data_dir=data_dir)
        for case, case_dir in case_dirs.items():
            rec.register_case(case, case_dir)
        return rec

    rec = recorder(update=True)
    rec.record("clinical/Case1/flow_error", 1.0)
    rec.record("clinical/Case2/flow_error", 5.0)
    rec.write()
    cases = json.loads(baseline_path.read_text())["cases"]
    assert sorted(cases) == ["siemens/Case1", "siemens/Case2"]
    assert cases["siemens/Case1"]["metrics"] == {"flow_error": 1.0}
    assert list(cases["siemens/Case1"]["file_hashes"]) == ["inputs/velocity_raw.seq.nrrd"]

    rec = recorder(update=False)
    rec.record("clinical/Case1/flow_error", 1.04)              # within tolerance
    with pytest.raises(AssertionError, match="regressed"):
        rec.record("clinical/Case1/flow_error", 2.0)

    (case_dirs["Case1"] / "inputs" / "velocity_raw.seq.nrrd").write_bytes(b"exported again")
    rec = recorder(update=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec.record("clinical/Case1/flow_error", 2.0)           # stale baseline: not compared
    assert any("stale" in str(w.message) and "velocity_raw" in str(w.message) for w in caught)
    with pytest.raises(AssertionError, match="regressed"):
        rec.record("clinical/Case2/flow_error", 9.0)           # other cases are still compared

    rec = recorder(update=True)
    rec.record("clinical/Case1/other_metric", 3.0)             # only Case1 is tested in this run
    rec.write()
    cases = json.loads(baseline_path.read_text())["cases"]
    assert cases["siemens/Case1"]["metrics"] == {"other_metric": 3.0}   # stale metrics are replaced, not merged
    assert cases["siemens/Case2"]["metrics"] == {"flow_error": 5.0}     # cases not tested in the run are kept
    with pytest.raises(ValueError):
        rec.record("clinical/UnknownCase/flow_error", 1.0)
