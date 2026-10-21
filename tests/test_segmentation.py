"""
Automatic lumen segmentation (segmentation.estimate_lumen) on a bifurcation phantom acquired with noise: the lumen on
the twice-finer grid covers the phantom lumen, its growth toward the wall recovers the flow that the PC-MRA threshold
alone misses, and the results do not depend on the voxels being anisotropic.
"""

import numpy as np
import pytest

from slicerheart_4dflow import estimate_lumen, pcmra, plane_flow
from slicerheart_4dflow.segmentation import dilate_mm

from test_transfer_audit import synthetic_case


def net_flows(case, velocity, lumen):
    return np.array([plane_flow(velocity, case["voxel_to_world"], lumen_mask=lumen, **p)["flow"].mean()
                     for p in case["planes"].values()])


@pytest.mark.parametrize("voxel_size_mm,limit", [([1.6, 1.6, 1.6], 0.05), ([1.4, 1.4, 2.4], 0.08)],
                         ids=["isotropic", "anisotropic"])
def test_estimate_lumen_reaches_the_wall(voxel_size_mm, limit, metrics):
    """The lumen is estimated from the measured data (PC-MRA and speed with noise and background offset); its quality
    is judged by the flow of the true velocity through it against the flow through the true lumen, which separates
    the error of the mask from the errors of the measurement."""
    case = synthetic_case(venc_mm_s=[1500.0] * 3, voxel_size_mm=voxel_size_mm)
    velocity, magnitude, truth = case["velocity_true_t"], case["magnitude_t"].astype(float), case["lumen"]
    measured = case["velocity_raw_encoded_t"].astype(float)       # not aliased at this venc
    angiogram = pcmra(magnitude, measured)
    threshold_only = estimate_lumen(angiogram, case["voxel_to_world"])
    grown = estimate_lumen(angiogram, case["voxel_to_world"], velocity_t=measured, magnitude_t=magnitude)
    assert grown["mask"].shape == tuple(2 * n for n in truth.shape)
    assert grown["mask"].sum() >= threshold_only["mask"].sum()

    true_flows = net_flows(case, velocity, truth)
    errors = {}
    for name, result in (("threshold", threshold_only), ("grown", grown)):
        flows = net_flows(case, velocity, result["fraction"])
        errors[name] = float(np.max(np.abs(flows / true_flows - 1)))
        metrics.record(f"segmentation/{voxel_size_mm[2]}mm/{name}/max_flow_error", errors[name], atol=0.01)
    assert errors["grown"] < limit, errors
    assert errors["grown"] < 0.5 * errors["threshold"], errors

    # covers the phantom lumen, and does not reach far into the static tissue around it
    covered = (grown["fraction"] > 0)[truth].mean()
    outside = (grown["lumen"] & ~truth).sum() / truth.sum()
    metrics.record(f"segmentation/{voxel_size_mm[2]}mm/lumen_covered", covered, direction="higher", atol=0.01)
    metrics.record(f"segmentation/{voxel_size_mm[2]}mm/volume_outside_over_lumen", outside, atol=0.02)
    assert covered > 0.95 and outside < 0.5, (covered, outside)


def test_morphology_is_in_mm_on_anisotropic_grid():
    """dilate_mm grows the same distance along every axis of an anisotropic grid."""
    mask = np.zeros((21, 21, 21), bool)
    mask[10, 10, 10] = True
    grown = dilate_mm(mask, np.array([3.0, 1.0, 1.0]), 3.0)         # k axis voxels are 3 mm
    assert grown[10, 10, 7] and grown[10, 10, 13] and not grown[10, 10, 6]
    assert grown[9, 10, 10] and grown[11, 10, 10] and not grown[8, 10, 10]
