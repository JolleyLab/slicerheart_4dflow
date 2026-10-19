"""
DRAFT: written before any clinical case was available. The case layout,
units and component frame follow the Slicer 5D convention from the
Discourse post; they must be aligned with what the SlicerHeart Import4DFlow
and Analyze4DFlow modules actually write before this is relied on.

The clinical tier's checks, as functions of a loaded case so that the same
code runs on a real case (tier "clinical", clinical baseline) and on a
mock case written in the same layout (tier "synthetic", synthetic
baseline). Nothing here knows whether the data are real; the caller says
which tier the numbers belong to.

Four kinds of check (docs/VALIDATION_STRATEGY.md, section 2):
  consistency   quantities that must hold without a ground truth
  reference     agreement with the reference measurements of the case
  stability     the result must not move under changes that should not matter
  semisynthetic a known answer on the case's own anatomy (reference built
                from the case, re-acquired at its venc and noise)
plus a manifest check and the pipeline itself.
"""

from __future__ import annotations

import numpy as np

from slicerheart_4dflow import (correct_background_phase, unwrap_velocity, divergence_free_sequence,
                                encoding_check, reference_from_measurement, reacquire)
from slicerheart_4dflow.qa import plane_flow, net_volume, mass_conservation
from slicerheart_4dflow.report import quality_report
from slicerheart_4dflow.unwrap import noise_mask
from slicerheart_4dflow.geometry import voxel_to_world_points

MANIFEST_FIELDS = ("manifest_version", "case_id", "vendor", "venc_mm_s", "rr_ms", "n_phases", "voxel_size_mm",
                   "velocity_units", "velocity_component_frame", "gating", "aliasing_present",
                   "reference_software", "license", "files")
STATIC_RESIDUAL_LIMIT_MM_S = 6.0        # Gatehouse 2010
NET_VOLUME_LIMIT_ML = 7.0               # manuscript limits of agreement vs cvi42
CONSERVATION_LIMIT_PCT = 5.0
RATIO_LIMIT = 0.10


def check_manifest(case):
    man = case["manifest"]
    missing = [f for f in MANIFEST_FIELDS if f not in man]
    assert not missing, f"manifest missing {missing}"
    v = case["velocity_t"]
    assert v.ndim == 5 and v.shape[-1] == 3
    assert v.shape[0] == man["n_phases"]
    if case["magnitude_t"] is not None:
        assert case["magnitude_t"].shape == v.shape[:4]
    if case["lumen"] is not None:
        assert case["lumen"].shape == v.shape[1:4] and case["lumen"].sum() > 100
    if case["reference"] is not None and case["planes"] is not None:
        assert set(case["planes"]) >= set(case["reference"]["planes"])
    assert man["velocity_units"] in ("mm/s", "cm/s", "m/s")
    assert man["velocity_component_frame"] in ("LPS", "RAS")
    assert man["gating"] in ("retrospective", "prospective")


def run_pipeline(case, divergence_method=None):
    """Background correction, unwrapping, (optional) divergence filter, report, plane flows."""
    v, mag, M, lumen, venc = (case["velocity_t"].astype(float), case["magnitude_t"].astype(float),
                              case["voxel_to_world"], case["lumen"], case["venc_mm_s"])
    corrected, bg_info = correct_background_phase(v, mag, method="war+msac", lumen_mask=lumen)
    signal = ~noise_mask(mag, corrected)
    unwrapped = unwrap_velocity(corrected, venc, M, method="laplacian_then_temporal", mask=signal)
    processed = unwrapped
    if divergence_method and lumen is not None:
        processed, _ = divergence_free_sequence(unwrapped, lumen, M, method=divergence_method)
    planes = case["planes"] or {}
    rep = quality_report(v, processed, mag, venc, M, lumen_mask=lumen, planes=planes or None,
                         checks=[tuple(c) for c in (case["reference"] or {}).get("conservation_checks", [])] or None,
                         rr_ms=case["rr_ms"], study_id=case["manifest"]["case_id"])
    flows = {name: plane_flow(processed, M, lumen_mask=lumen, **p)["flow"] for name, p in planes.items()}
    return {"corrected": corrected, "unwrapped": unwrapped, "processed": processed, "report": rep,
            "flows": flows, "background": bg_info, "signal": signal}


def check_consistency(case, pipe, metrics, tier):
    cid = case["manifest"]["case_id"]
    rep = pipe["report"]
    res = np.max(np.abs(rep["background_offset"]["residual_mean_processed_mm_s"]))
    metrics.record(f"{cid}/static_residual_mm_s", res, atol=0.2, tier=tier)
    assert rep["background_offset"]["pass"], f"static residual {res:.2f} mm/s"
    assert rep["venc"]["venc_adequate"] == (not case["manifest"]["aliasing_present"]), rep["venc"]["note"]
    e = rep["encoding"]
    metrics.record(f"{cid}/encoding_discrimination", e["discrimination"] or 0.0, direction="higher", tier=tier)
    assert e["consistent"], e["note"]
    # inconclusive is allowed (short segments) but recorded
    metrics.record(f"{cid}/encoding_conclusive", float(e["conclusive"]), direction="record", tier=tier)
    sv = np.asarray(rep["velocity_noise_mm_s"], float)
    metrics.record(f"{cid}/velocity_noise_mm_s", sv.mean(), direction="record", tier=tier)
    assert np.all(sv > 0) and np.all(sv < 0.2 * case["venc_mm_s"]), "noise estimate out of range"
    if "mass_conservation" in rep:
        for c in rep["mass_conservation"]["processed"]:
            metrics.record(f"{cid}/{c['check']}/abs_pct_error", abs(c["percent_error"]), atol=0.5, tier=tier)
            assert abs(c["percent_error"]) < CONSERVATION_LIMIT_PCT, c
    for label, num, den, expected in (case["reference"] or {}).get("ratio_checks", []):
        ratio = net_volume(pipe["flows"][num], rr_ms=case["rr_ms"]) / net_volume(pipe["flows"][den], rr_ms=case["rr_ms"])
        metrics.record(f"{cid}/{label}/ratio", ratio, direction="exact", atol=0.02, tier=tier)
        assert abs(ratio / expected - 1) < RATIO_LIMIT, (label, ratio, expected)


def check_reference(case, pipe, metrics, tier):
    cid = case["manifest"]["case_id"]
    ref = case["reference"]["planes"]
    for name, r in ref.items():
        net_ml = net_volume(pipe["flows"][name], rr_ms=case["rr_ms"]) / 1000.0
        metrics.record(f"{cid}/{name}/net_volume_mL", net_ml, direction="exact", atol=0.5, tier=tier)
        assert abs(net_ml - r["net_volume_mL"]) < NET_VOLUME_LIMIT_ML, (name, net_ml, r["net_volume_mL"])
    peaks = [r["peak_velocity_mm_s"] for r in ref.values() if "peak_velocity_mm_s" in r]
    if peaks and case["lumen"] is not None:
        peak = float(np.linalg.norm(pipe["processed"][:, case["lumen"]], axis=-1).max())
        metrics.record(f"{cid}/peak_speed_over_reference_peak", peak / max(peaks), direction="record", tier=tier)


def check_stability(case, pipe, metrics, tier):
    """Flows must not move more than 5% under a one-voxel dilation of the
    segmentation, nor at all under a rigid change of the world frame."""
    from scipy import ndimage
    cid = case["manifest"]["case_id"]
    M, lumen = case["voxel_to_world"], case["lumen"]
    v = pipe["processed"]
    for name, p in (case["planes"] or {}).items():
        q0 = plane_flow(v, M, lumen_mask=lumen, **p)["flow"].mean()
        q1 = plane_flow(v, M, lumen_mask=ndimage.binary_dilation(lumen), **p)["flow"].mean()
        change = abs(q1 / q0 - 1) if abs(q0) > 1e-9 else 0.0
        metrics.record(f"{cid}/{name}/flow_change_mask_dilation", change, atol=0.01, tier=tier)
        assert change < 0.05, (name, change)
    # world frame: rotate by 30 degrees about an oblique axis and translate
    axis = np.array([1.0, 2.0, 0.5]); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(0.5) * K + (1 - np.cos(0.5)) * K @ K
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [12.0, -7.0, 3.0]
    name, p = next(iter(case["planes"].items()))
    q0 = plane_flow(v, M, lumen_mask=lumen, **p)["flow"]
    p2 = dict(p, centre=tuple(R @ np.asarray(p["centre"]) + T[:3, 3]), normal=tuple(R @ np.asarray(p["normal"])))
    q1 = plane_flow(v @ R.T, T @ M, lumen_mask=lumen, **p2)["flow"]
    assert np.allclose(q0, q1, rtol=1e-6, atol=1e-6)


def check_semisynthetic(case, metrics, tier, rng_seed=1):
    """Build the reference from the case, re-acquire at its venc and noise,
    run the pipeline on the re-acquired data and score against the reference.
    For a mock case with stored truth, also score the reference itself."""
    cid = case["manifest"]["case_id"]
    v, mag, M, lumen, venc = (case["velocity_t"].astype(float), case["magnitude_t"].astype(float),
                              case["voxel_to_world"], case["lumen"], case["venc_mm_s"])
    built = reference_from_measurement(v, mag, venc, M, lumen)
    ref = built["reference"]
    re = reacquire(ref, venc, mag, lumen, built["velocity_noise_mm_s"], offset=built["offset"],
                   rng=np.random.default_rng(rng_seed))
    corrected, _ = correct_background_phase(re["velocity_t"], re["magnitude_t"], method="war+msac", lumen_mask=lumen)
    unwrapped = unwrap_velocity(corrected, venc, M, method="laplacian_then_temporal", mask=lumen)
    err = np.sqrt(np.mean((unwrapped - ref)[:, lumen] ** 2))
    noise = float(np.mean(built["velocity_noise_mm_s"]))
    metrics.record(f"{cid}/semisynthetic/rmse_mm_s", err, atol=1.0, tier=tier)
    metrics.record(f"{cid}/semisynthetic/noise_mm_s", noise, direction="record", tier=tier)
    # the pipeline must not add error beyond the acquisition noise (the RMSE
    # over voxels and components of white noise is sigma per component)
    assert err < 1.3 * noise + 1.0, (err, noise)
    wrapped = int((re["wrap_count"][:, lumen] != 0).any(-1).sum())
    residual = int((np.abs(unwrapped - ref)[:, lumen] > venc.min()).any(-1).sum())
    metrics.record(f"{cid}/semisynthetic/residual_wrapped_voxel_frames", residual, atol=5, tier=tier)
    assert residual <= 0.02 * max(wrapped, 1) + 5, (residual, wrapped)
    if case["truth"] is not None:
        truth = case["truth"]["velocity_true"].astype(float)
        ref_err = np.sqrt(np.mean((ref - truth)[:, lumen] ** 2))
        metrics.record(f"{cid}/reference_vs_truth_rmse_mm_s", ref_err, atol=1.0, tier=tier)
        assert ref_err < 1.3 * noise + 1.0
    return {"rmse": err, "noise": noise, "wrapped": wrapped, "residual": residual}
