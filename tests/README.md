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
