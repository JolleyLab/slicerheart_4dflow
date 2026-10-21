# Tests

    pip install -e ".[test]"
    python -m pytest                 # all tests
    python -m pytest -m "not cfd"    # only tests that need no downloaded data

Tests marked `cfd` use an aortic CFD velocity field with lumen mask as ground
truth. Test data files are downloaded on first use from the
[TestingData release](https://github.com/JolleyLab/slicerheart_4dflow/releases/tag/TestingData)
(its description lists the source and license of each file) into `tests/data`,
or into the folder set in the `SLICERHEART_4DFLOW_TEST_DATA` environment
variable, and verified by their sha256 hash (in `conftest.py`). These tests are
skipped if a file cannot be downloaded.

Each test file can also be run as a script to print the full method comparison
tables (more noise levels and method variants than the assertions cover):

    python tests/test_background_phase.py        # synthetic chest phantom
    python tests/test_divergence_free.py [--lr]  # --lr: block-average to 1.19 mm
    python tests/test_unwrap.py
    python tests/test_qa.py
    python tests/test_pathlines.py

Note that the magnitude image of the CFD example has no static tissue (noise floor
outside the lumen), so background phase correction is tested on a synthetic
phantom instead.

## Regression metrics

Tests record the numbers behind their assertions (CFD RMSE per method,
residual wrapped voxels, WSS and pressure errors, phantom conservation
errors, ...) through the `metrics` fixture in `conftest.py`. Every recorded
value is compared with `tests/metrics/baseline.json`: an error metric may not
get worse than its baseline by more than 5% (relative, plus a small absolute
floor for near-zero values), so a change that stays inside a test's hard
limit but degrades a method still fails. The values of each run are written
to `.test-reports/latest.json` at the repository root (ignored by git; the
`SLICERHEART_4DFLOW_TEST_REPORT_DIR` environment variable sets another folder).

    python -m pytest                       # compare with the baseline
    python -m pytest --update-baseline     # accept the current values as the new baseline
    python -m pytest --metrics-tolerance 0.02

When a change improves a metric the comparison passes; run with
`--update-baseline` and commit the new baseline so the improvement is
protected too. Metrics for tests that are skipped (no CFD data) are left
unchanged in the baseline.

Processing times (`cfd/preprocess_time_s/*`, from `preprocess()`'s per-step
timing) are recorded with `direction="record"`: they are stored in the
baseline for reference and never compared, since they depend on the
machine. Compare them by hand when a change is meant to be faster.

## Clinical cases

Real scanner data per vendor live outside git and are described in
`tests/clinical/README.md`. They are planned to be published on Zenodo
(CC BY 4.0), but are not released publicly yet, so they are available only
locally. Tests marked `clinical` skip until a
case's `manifest.json` is present under `tests/data/clinical/<vendor>/` or the
folder named by `SLICERHEART_4DFLOW_CLINICAL_DATA`.
