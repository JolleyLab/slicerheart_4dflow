"""
Quality report on the chest phantom: background correction pass/fail,
venc adequacy, mass conservation between two planes along the artery,
values of the raw (input) and of the processed (output) velocity, selection
of sections, JSON round trip and Markdown rendering.
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
    assert rep["background_offset"]["pass_processed"]
    assert not rep["background_offset"]["pass_raw"]
    assert not np.all(np.abs(rep["background_offset"]["residual_mean_raw_mm_s"]) < 6.0)   # raw fails
    assert rep["venc"]["venc_adequate"]
    mc = rep["mass_conservation"]
    assert abs(mc["processed"][0]["percent_error"]) < 3.0
    # the offset biases raw flow: raw error larger than processed
    assert abs(mc["raw"][0]["percent_error"]) >= abs(mc["processed"][0]["percent_error"]) - 1e-9
    s = to_json(rep)
    assert json.loads(s)["study_id"] == "phantom"
    md = to_markdown(rep)
    assert "Background offset  (input FAIL, output PASS)" in md
    assert "Mass conservation" in md and "prox vs dist" in md


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
    assert v["peak_component_over_venc"]["processed"][2] > 1.5
    # the raw velocity is aliased: no component can be above venc, and its peak speed is below the true one
    assert max(v["peak_component_over_venc"]["raw"]) <= 1.0
    assert v["peak_speed_mm_s"]["raw"] < v["peak_speed_mm_s"]["processed"]
    assert v["lumen_voxel_frames_aliased"] > 0.05
    assert v["unwrapped_voxel_frames"] > 0 and v["max_wraps"] == 1
    md = to_markdown(rep)
    assert "exceeded" in md and "Aliased in the input and unwrapped in the output" in md


def test_velocity_noise_estimate_is_unbiased():
    """The chest phantom carries 10 mm/s Gaussian velocity noise in tissue
    (x25 in lung and air). The old estimate (mean temporal std, ddof=0, over
    the static-tissue candidates, which are selected for LOW temporal std)
    gave 7.3; the magnitude-selected, median-based estimate must be within
    5% of the truth at 6 and 20 frames."""
    for nt in (6, 20):
        c = chest(nt=nt, wrap=False)
        v = c["velocity_true"]
        rep = quality_report(v, v, c["magnitude"], 1500.0, c["voxel_to_world"], lumen_mask=c["lumen"])
        for stage in ("raw", "processed"):
            assert np.allclose(rep["velocity_noise_mm_s"][stage], 10.0, rtol=0.05), (nt, rep["velocity_noise_mm_s"])
        assert rep["velocity_noise_voxels"] > 1000


def test_venc_check_uses_encoded_components():
    """A world velocity (800, 800, 800) with venc 1000 per encoding axis is
    not aliased on an axis-aligned grid (each image-axis component is 800),
    but on a grid rotated 45 degrees about z the i-axis component is 1131
    and the check must say the venc was exceeded."""
    from slicerheart_4dflow.geometry import voxel_to_world_matrix
    shape = (8, 8, 8)
    mask = np.zeros(shape, bool); mask[2:6, 2:6, 2:6] = True
    v = np.zeros((2,) + shape + (3,)); v[:, mask] = 800.0
    mag = np.ones((2,) + shape)
    static = ~mask
    rep = quality_report(v, v, mag, 1000.0, np.eye(4), lumen_mask=mask, static_mask=static)
    assert rep["venc"]["venc_adequate"]
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rep = quality_report(v, v, mag, 1000.0, voxel_to_world_matrix(directions=R), lumen_mask=mask, static_mask=static)
    assert not rep["venc"]["venc_adequate"]
    assert max(rep["venc"]["peak_component_mm_s"]["processed"]) > 1100.0


def test_report_gives_input_and_output():
    """Every value of a velocity field is given for the raw (input) and for
    the processed (output) velocity, and the Markdown says which is which."""
    c, v_raw, v_proc = _chest_case()
    rep = quality_report(v_raw, v_proc, c["magnitude"], venc=1500.0, voxel_to_world=c["voxel_to_world"],
                         lumen_mask=c["lumen"], static_mask=c["static"])
    stages = {"raw", "processed"}
    assert set(rep["velocity_noise_mm_s"]) == set(rep["venc_to_noise_ratio"]) == stages
    assert set(rep["encoding"]) == stages
    for key in ("peak_speed_mm_s", "peak_component_mm_s", "peak_component_over_venc", "lumen_voxel_frames_near_venc"):
        assert set(rep["venc"][key]) == stages, key
    b = rep["background_offset"]
    assert np.max(np.abs(b["residual_mean_raw_mm_s"])) > np.max(np.abs(b["residual_mean_processed_mm_s"]))
    assert np.max(b["residual_rms_raw_mm_s"]) > np.max(b["residual_rms_processed_mm_s"])
    md = to_markdown(rep)
    assert md.count("| | input | output |") == 3                     # velocity noise, background offset, venc
    assert "Input is the velocity before processing" in md
    assert "Encoding directions  (input " in md and "- Input: " in md and "- Output: " in md
    assert "raw" not in md.replace("(raw)", "") and "processed" not in md


def test_report_sections():
    """Only the sections that are asked for are computed and rendered."""
    import pytest
    from slicerheart_4dflow.report import SECTIONS
    c, v_raw, v_proc = _chest_case()
    keys = {"velocity_noise": "velocity_noise_mm_s", "background_offset": "background_offset", "venc": "venc",
            "divergence": "divergence_rms_1_s", "encoding": "encoding"}
    titles = {"velocity_noise": "## Velocity noise", "background_offset": "## Background offset", "venc": "## venc",
              "divergence": "## Divergence", "encoding": "## Encoding directions"}
    assert set(keys) == set(SECTIONS)
    report = lambda **kwargs: quality_report(v_raw, v_proc, c["magnitude"], venc=1500.0, voxel_to_world=c["voxel_to_world"],
                                             lumen_mask=c["lumen"], **kwargs)
    full = report()
    for selected in (["venc"], ["velocity_noise", "encoding"], "divergence", []):
        rep = report(sections=selected)
        names = {selected} if isinstance(selected, str) else set(selected)
        md = to_markdown(rep)
        for section in SECTIONS:
            assert (keys[section] in rep) == (section in names), (selected, section)
            assert (titles[section] in md) == (section in names), (selected, section)
            if section in names:
                assert rep[keys[section]] == full[keys[section]], (selected, section)
    # sections of the lumen are left out without a lumen mask
    rep = quality_report(v_raw, v_proc, c["magnitude"], venc=1500.0, voxel_to_world=c["voxel_to_world"])
    assert not {"venc", "divergence_rms_1_s", "encoding"} & set(rep) and "background_offset" in rep
    with pytest.raises(ValueError, match="unknown sections"):
        report(sections=["venc", "noise"])
    # the magnitude is only needed to select the tissue of the velocity noise and of the background offset
    lumen_sections = ["venc", "divergence", "encoding"]
    rep = quality_report(v_raw, v_proc, None, venc=1500.0, voxel_to_world=c["voxel_to_world"], lumen_mask=c["lumen"],
                         sections=lumen_sections)
    assert all(rep[keys[section]] == full[keys[section]] for section in lumen_sections)
    for section in ("velocity_noise", "background_offset"):
        with pytest.raises(ValueError, match="magnitude_t"):
            quality_report(v_raw, v_proc, None, venc=1500.0, voxel_to_world=c["voxel_to_world"], sections=[section])
