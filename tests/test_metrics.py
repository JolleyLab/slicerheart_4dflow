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
