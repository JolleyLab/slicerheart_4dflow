# Tests

All tests run against the two example files shipped in the 4DFlowNet repository
(Ferdian et al., Front Phys 2020): a 0.59 mm aortic CFD ground truth with lumen
mask, and a simulated 1.19 mm 4D flow MRI of the same geometry.

    git clone --depth 1 https://github.com/EdwardFerdian/4DFlowNet
    pip install h5py
    python tests/test_background_phase.py            # synthetic chest phantom, no data needed
    python tests/test_divergence_free_methods.py 4DFlowNet/data [--lr]
    python tests/test_unwrap.py 4DFlowNet/data
    python tests/test_qa.py 4DFlowNet/data
    python tests/test_pathlines.py 4DFlowNet/data

Note that the 4DFlowNet magnitude image has no static tissue (noise floor
outside the lumen), so background phase correction is tested on a synthetic
phantom instead.
