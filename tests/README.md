# Tests

    pip install -e ".[test]"
    python -m pytest                 # all tests
    python -m pytest -m "not cfd"    # only tests that need no downloaded data

Tests marked `cfd` use the aortic CFD example shipped in the 4DFlowNet repository
(Ferdian et al., Front Phys 2020, MIT license): a 0.59 mm aortic CFD ground truth
with lumen mask. `example_data_HR.h5` (2 MB) is downloaded on first use into
`tests/data`, or into the folder set in the `SLICERHEART_4DFLOW_TEST_DATA`
environment variable. These tests are skipped if the file cannot be downloaded.

Each test file can also be run as a script to print the full method comparison
tables (more noise levels and method variants than the assertions cover):

    python tests/test_background_phase.py        # synthetic chest phantom
    python tests/test_divergence_free.py [--lr]  # --lr: block-average to 1.19 mm
    python tests/test_unwrap.py
    python tests/test_qa.py
    python tests/test_pathlines.py

Note that the 4DFlowNet magnitude image has no static tissue (noise floor
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
to `tests/metrics/latest.json` (ignored by git).

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

Metrics measured on real cases go to a separate baseline,
`tests/metrics/clinical_baseline.json` (`tier="clinical"` in
`metrics.record`), so numbers from data with a known answer and numbers
from data without one are never mixed. Every test not marked `clinical`
carries the `synthetic` marker automatically:

    python -m pytest -m synthetic          # phantoms, CFD, mock cases (CI)
    python -m pytest -m clinical           # real cases only

## Clinical cases

Real scanner data per vendor live outside git (Zenodo, CC BY 4.0) and are
described in `tests/clinical/README.md`. Tests marked `clinical` skip until a
case's `manifest.json` is present under `tests/data/clinical/<vendor>/` or the
folder named by `SLICERHEART_4DFLOW_CLINICAL_DATA`.
