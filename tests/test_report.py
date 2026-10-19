"""
Quality report on the chest phantom: background correction pass/fail,
venc adequacy, mass conservation between two planes along the artery,
JSON round trip and Markdown rendering.
"""

import json

import numpy as np

from slicerheart_4dflow.phantom import chest, alias
from slicerheart_4dflow.background_phase import correct_background_phase
from slicerheart_4dflow.unwrap import unwrap_velocity, noise_mask
from slicerheart_4dflow.report import quality_report, to_json, to_markdown


def _chest_case():
    c = chest(nt=8, wrap=False)
    v_raw = c["velocity"]
    corrected, _ = correct_background_phase(v_raw, c["magnitude"], method="war+msac", lumen_mask=c["lumen"])
    return c, v_raw, corrected


def test_report_background_and_consistency():
    c, v_raw, v_proc = _chest_case()
    vtw = c["voxel_to_world"]
    # two planes along the artery (world z = array k axis), no branches between
    kj = np.argwhere(c["lumen"][20]).mean(0)                      # (j, i) centroid at slice 20
    centre = lambda k: (kj[1], kj[0], float(k))
    planes = {"prox": dict(centre=centre(12), normal=(0, 0, 1), radius=14.0),
              "dist": dict(centre=centre(28), normal=(0, 0, 1), radius=14.0)}
    rep = quality_report(v_raw, v_proc, c["magnitude"], venc=1500.0, voxel_to_world=vtw,
                         lumen_mask=c["lumen"], static_mask=c["static"], noise_mask=noise_mask(c["magnitude"], v_proc),
                         planes=planes, checks=[("prox vs dist", ["prox"], ["dist"])], study_id="phantom")
    assert rep["background_offset"]["pass"]
    assert not np.all(np.abs(rep["background_offset"]["residual_mean_raw_mm_s"]) < 6.0)   # raw fails
    assert rep["venc"]["venc_adequate"]
    mc = rep["mass_conservation"]
    assert abs(mc["processed"][0]["percent_error"]) < 3.0
    # the offset biases raw flow: raw error larger than processed
    assert abs(mc["raw"][0]["percent_error"]) >= abs(mc["processed"][0]["percent_error"]) - 1e-9
    s = to_json(rep)
    assert json.loads(s)["study_id"] == "phantom"
    md = to_markdown(rep)
    assert "PASS" in md and "Mass conservation" in md and "prox vs dist" in md


def test_report_processing_time():
    from slicerheart_4dflow.unwrap import preprocess
    c = chest(nt=6, wrap=False)
    out = preprocess(c["velocity"], c["magnitude"], 1500.0, c["voxel_to_world"], lumen_mask=c["lumen"], verbose=False)
    rep = quality_report(c["velocity"], out["velocity"], c["magnitude"], 1500.0, c["voxel_to_world"],
                         lumen_mask=c["lumen"], static_mask=c["static"], timing=out["report"]["timing_s"])
    pt = rep["processing_time_s"]
    assert {"background", "unwrap", "total", "report"} <= set(pt)
    assert pt["report"] > 0
    assert "Processing time (s)" in to_markdown(rep)
    assert json.loads(to_json(rep))["processing_time_s"]["total"] == pt["total"]


def test_report_detects_aliasing():
    # 16 frames so the true velocity changes by less than venc per frame
    # (the condition temporal unwrapping needs); artery pulses 100..900 mm/s
    c = chest(nt=16, wrap=False)
    v_proc, _ = correct_background_phase(c["velocity"], c["magnitude"], method="war+msac", lumen_mask=c["lumen"])
    venc = 500.0
    wrapped, k = alias(v_proc, venc)
    unwrapped = unwrap_velocity(wrapped, venc, c["voxel_to_world"], method="laplacian_then_temporal",
                                mask=c["lumen"])
    rep = quality_report(wrapped, unwrapped, c["magnitude"], venc=venc, voxel_to_world=c["voxel_to_world"],
                         lumen_mask=c["lumen"], static_mask=c["static"], wrap_count=k)
    v = rep["venc"]
    assert not v["venc_adequate"]
    assert v["peak_component_over_venc"][2] > 1.5
    assert v["lumen_voxel_frames_aliased"] > 0.05
    assert v["unwrapped_voxel_frames"] > 0 and v["max_wraps"] == 1
    assert "exceeded" in to_markdown(rep)
