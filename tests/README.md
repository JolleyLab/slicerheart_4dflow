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

## Progress of a run

pytest prints its summary only at the end, which is an hour or more for the
clinical tier. Every run writes `.test-reports/progress.log` while it runs: a
line when each test starts, so a slow or hanging test is visible, and one
with its result, duration, the outcomes so far and an estimate of the
remaining time (from `durations.json` of the previous run in the same folder).
Follow it in another terminal:

    tail -f .test-reports/progress.log
    powershell Get-Content .test-reports\progress.log -Wait

Runs at the same time need different report folders
(`SLICERHEART_4DFLOW_TEST_REPORT_DIR`). To get the stack of a hanging test,
add `-o faulthandler_timeout=1200`: pytest then writes the stack of a test
that runs longer than 20 minutes.

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

The clinical tests also run on a synthetic case in the clinical case layout
(`clinical_case.py`: a bifurcation phantom with aliasing, background offset and
noise, the automatic lumen on the twice-finer grid, encoded components that are
a signed permutation of the world components, plane flows computed as the
SlicerHeart plane probe does, and the true flows as an independent reference).
It is written into the test report folder once per session; its tests carry
the `synthetic` marker, so they run in CI and check the clinical test code
without clinical data, and their metrics are in the synthetic baseline.

Real scanner data per vendor live outside git and are described in
`tests/clinical/README.md`. They are planned to be published on Zenodo
(CC BY 4.0), but are not released publicly yet, so they are available only
locally. Tests marked `clinical` skip until a
case's `manifest.json` is present under `tests/data/clinical/<vendor>/` or the
folder named by `SLICERHEART_4DFLOW_CLINICAL_DATA`.

## Transfer audit

`transfer_audit.py` checks whether the synthetic validation carries over to a
clinical case: it builds a synthetic case matched to the clinical one (venc,
voxel size, frames, noise, vessel radius, peak velocity, waveform,
background offset), runs the same pipeline on both and compares quantities
that need no ground truth (background residual, noise mask, unwrapping
errors, divergence, encoding check, sensitivity of plane flow to the lumen).
A quantity that differs by more than a factor of two is flagged: the
phantoms miss an effect of real data for that step (see section 3 of
`docs/VALIDATION_STRATEGY.md`). The clinical test `test_transfer_audit`
writes a report per case and `summary.md` to `.test-reports/transfer_audit/`;
one case can be audited directly:

    python tests/transfer_audit.py D:\Data\4Dflow\_test_cases_anonymized_\siemens\Patient0001
