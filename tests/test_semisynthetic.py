"""
Semi-synthetic references (semisynthetic.py) on a phantom with known truth: the reference built from an acquisition
is close to the truth, a re-acquisition reproduces the velocity noise of the acquisition, and the default pipeline
(preprocess) recovers the reference from the re-acquired data.
"""

import numpy as np

from slicerheart_4dflow import acquisition_snr, preprocess, quality_report, reacquire, reference_from_measurement

from test_transfer_audit import synthetic_case


def lumen_rmse(a, b, lumen):
    return float(np.sqrt(np.mean((a - b)[:, lumen] ** 2)))


def test_reference_reacquisition_and_recovery(metrics):
    case = synthetic_case()
    v, mag, vtw, lumen = case["velocity_raw_encoded_t"], case["magnitude_t"], case["voxel_to_world"], case["lumen"]
    venc, truth = case["venc_encoded_mm_s"], case["velocity_true_t"]
    built = reference_from_measurement(v, mag, venc, vtw, lumen)
    reference, noise = built["reference"], built["velocity_noise_mm_s"]
    sigma = float(np.mean(noise))
    # the reference is the measurement cleaned: within the acquisition noise of the truth
    reference_error = lumen_rmse(reference, truth, lumen)
    metrics.record("semisynthetic/reference_vs_truth_rmse_over_noise", reference_error / sigma, atol=0.05)
    assert reference_error < 1.3 * sigma, (reference_error, sigma)

    # the re-acquisition has the noise of the acquisition
    snr, _ = acquisition_snr(mag, lumen, noise, venc)
    re = reacquire(reference, venc, mag, lumen, noise, offset=built["offset"], rng=np.random.default_rng(5))
    assert re["snr"] == snr
    re_noise = np.mean(quality_report(re["velocity_t"], re["velocity_t"], re["magnitude_t"], venc, vtw,
                                      lumen_mask=lumen)["velocity_noise_mm_s"])
    metrics.record("semisynthetic/reacquired_noise_ratio", abs(re_noise / sigma - 1), atol=0.02)
    assert abs(re_noise / sigma - 1) < 0.15, (re_noise, sigma)
    assert np.any(re["wrap_count"][:, lumen] != 0), "the re-acquisition should alias, as the acquisition does"

    # the default pipeline recovers the reference from the re-acquisition, where it keeps the velocity (preprocess sets
    # the voxels of its noise mask to zero, which includes some fast pulsatile lumen voxels: recorded separately)
    result = preprocess(re["velocity_t"], re["magnitude_t"], venc, vtw, lumen_mask=lumen, verbose=False)
    kept = built["mask"] & ~result["noise"]
    error = lumen_rmse(result["velocity"], reference, kept)
    metrics.record("semisynthetic/recovery_rmse_over_noise", error / sigma, atol=0.05)
    metrics.record("semisynthetic/lumen_fraction_set_to_zero", 1.0 - kept.sum() / lumen.sum(), direction="record")
    assert error < 1.5 * sigma, (error, sigma)
