"""
A synthetic clinical test case: a bifurcation phantom written in the layout of a clinical case (format 3, as
SlicerHeart Import4DFlow exports it), so that the clinical tests (test_clinical.py) run in CI without clinical data.

The phantom (test_transfer_audit.synthetic_case: Murray radii, 60/40 split, aortic waveform) is acquired at a venc below
its peak velocity (aliasing), with a second-order background offset, noise and static tissue, in encoded components
that are a signed permutation of the world components (as Siemens encodes them). The corrections are the pipeline of
this package (war+msac background, noise mask, laplacian_then_temporal unwrapping), the lumen is the automatic one
(estimate_lumen, on a grid twice as fine as the image), the planes are those of the phantom plus a second parent plane
4 mm downstream (a plane pair), the plane flows of outputs/flows.json are computed as the SlicerHeart plane probe does
(plain interpolation, samples tested against the segmentation), and references/phantom/flows.json holds the true flows.

This is synthetic data: its tests carry the synthetic marker and its metrics go to the synthetic baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CASE_ID = "Synthetic0001"
VENDOR = "synthetic"
# Columns: world (LPS) directions of the encoded components (signed permutation, as for Siemens)
ENCODING_TO_WORLD = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SAMPLING_DISTANCE_MM = 0.5


def _nrrd():
    import nrrd
    return nrrd


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))


def _write_sequence(path, array_t, vtw, times_ms, vector=False, spatial=True):
    """Like Slicer writes a volume sequence: (component, i, j, k, time), LPS, spatial vectors with a measurement frame."""
    data = np.ascontiguousarray(array_t.transpose(*(([4] if vector else []) + [3, 2, 1, 0])))
    header = {"space": "left-posterior-superior", "space origin": vtw[:3, 3],
              "kinds": (["vector"] if vector else []) + ["domain"] * 3 + ["list"],
              "space directions": np.vstack(([np.full(3, np.nan)] if vector else []) + [vtw[:3, :3].T, np.full((1, 3), np.nan)]),
              "axis 3 index values": " ".join(f"{t:g}" for t in times_ms)}
    if vector and spatial:
        header["measurement frame"] = np.eye(3)
    _nrrd().write(str(path), data, header)


def _write_volume(path, array, vtw):
    header = {"space": "left-posterior-superior", "space origin": vtw[:3, 3], "kinds": ["domain"] * 3,
              "space directions": vtw[:3, :3].T}
    _nrrd().write(str(path), np.ascontiguousarray(array.T), header)


def _write_segmentation(path, mask, vtw):
    header = {"space": "left-posterior-superior", "space origin": vtw[:3, 3], "kinds": ["domain"] * 3,
              "space directions": vtw[:3, :3].T, "Segment0_ID": "lumen", "Segment0_Name": "lumen",
              "Segment0_LabelValue": "1", "Segment0_Layer": "0"}
    _nrrd().write(str(path), np.ascontiguousarray(mask.astype(np.uint8).T), header)


def _write_plane(path, name, plane):
    radius = float(plane["radius"])
    _write_json(path, {"markups": [{"type": "Plane", "name": name, "coordinateSystem": "LPS",
                                    "center": [float(x) for x in plane["centre"]], "normal": [float(x) for x in plane["normal"]],
                                    "size": [2 * radius, 2 * radius, 0.0]}]})


def _probe_flows(velocity_t, vtw, labelmap, labelmap_vtw, planes):
    """Flow rate (ml/s) through each plane as the SlicerHeart plane probe computes it: plain trilinear interpolation of
    the velocity at the disc samples that are inside the segmentation."""
    from scipy import ndimage
    from slicerheart_4dflow.geometry import world_to_array_coordinates
    from slicerheart_4dflow.qa import plane_samples
    flows = {}
    for name, plane in planes.items():
        points, coordinates, normal, area, _ = plane_samples(velocity_t.shape[1:4], vtw, plane["centre"], plane["normal"],
                                                             plane["radius"], SAMPLING_DISTANCE_MM)
        nearest = np.round(world_to_array_coordinates(labelmap_vtw, points)).astype(int)
        valid = np.all((nearest >= 0) & (nearest < np.array(labelmap.shape)[:, None]), axis=0)
        inside = np.zeros(len(points), bool)
        inside[valid] = labelmap[nearest[0, valid], nearest[1, valid], nearest[2, valid]]
        through_plane = velocity_t @ normal
        flows[name] = np.array([ndimage.map_coordinates(through_plane[t], coordinates[:, inside], order=1).sum() * area
                                for t in range(velocity_t.shape[0])]) / 1000.0
    return flows


def _flows_json(flows, rr_ms):
    planes = {}
    for name, q in flows.items():
        planes[name] = {"net_volume_mL": float(np.mean(q) * rr_ms / 1000.0),
                        "forward_volume_mL": float(np.mean(np.clip(q, 0, None)) * rr_ms / 1000.0),
                        "flow_rate_ml_s": [float(x) for x in q]}
    return planes


def write_synthetic_case(root):
    """Write the synthetic case to root/<VENDOR>/<CASE_ID>/ and return the path of its manifest."""
    from slicerheart_4dflow import (correct_background_phase, estimate_lumen, noise_mask, pcmra, plane_flow, quality_report,
                                    to_json, unwrap_velocity)
    from slicerheart_4dflow.phantom import aortic_waveform
    from test_transfer_audit import synthetic_case

    base = Path(root) / VENDOR / CASE_ID
    for folder in ("inputs", "outputs", "annotations"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    case = synthetic_case(venc_mm_s=[1000.0] * 3)
    vtw, lumen_true, rr_ms = case["voxel_to_world"], case["lumen"], float(case["rr_ms"])
    magnitude = case["magnitude_t"].astype(np.float32)
    venc = np.asarray(case["venc_encoded_mm_s"], float)
    nt = magnitude.shape[0]
    times_ms = np.arange(nt) * rr_ms / nt
    E = ENCODING_TO_WORLD
    raw_world = case["velocity_raw_encoded_t"].astype(float)          # the phantom is acquired in world components
    raw_encoded = raw_world @ E                                      # world = encoded @ E.T

    # Corrections (encoded components): background, noise mask, unwrapping
    corrected, _ = correct_background_phase(raw_encoded, magnitude, method="war+msac")
    signal = ~noise_mask(magnitude, corrected, venc=venc)
    unwrapped = unwrap_velocity(corrected, venc, vtw, method="laplacian_then_temporal", mask=signal)
    wraps = np.rint((unwrapped - corrected) / (2 * venc)).astype(np.int16)
    velocity = (unwrapped @ E.T).astype(np.float32)
    angiogram = pcmra(magnitude, velocity).astype(np.float32)

    _write_sequence(base / "inputs" / "magnitude.seq.nrrd", magnitude, vtw, times_ms)
    _write_sequence(base / "inputs" / "velocity_raw.seq.nrrd", raw_world.astype(np.float32), vtw, times_ms, vector=True)
    _write_sequence(base / "outputs" / "velocity.seq.nrrd", velocity, vtw, times_ms, vector=True)
    _write_sequence(base / "outputs" / "wrap_count.seq.nrrd", wraps, vtw, times_ms, vector=True, spatial=False)
    _write_volume(base / "outputs" / "pcmra.nrrd", angiogram, vtw)

    lumen = estimate_lumen(angiogram, vtw, velocity_t=velocity, magnitude_t=magnitude)
    _write_segmentation(base / "annotations" / "lumen.seg.nrrd", lumen["mask"], lumen["voxel_to_world"])

    planes = dict(case["planes"])
    parent = planes["parent"]
    planes["parent_b"] = dict(parent, centre=tuple(np.asarray(parent["centre"]) + 4.0 * np.asarray(parent["normal"])))
    plane_files = []
    for name, plane in planes.items():
        _write_plane(base / "annotations" / f"{name}.mrk.json", name, plane)
        plane_files.append(f"{name}.mrk.json")

    probe = _probe_flows(velocity, vtw, lumen["mask"], lumen["voxel_to_world"], planes)
    split = 0.6
    _write_json(base / "outputs" / "flows.json", {
        "planes": _flows_json(probe, rr_ms), "sampling_distance_mm": SAMPLING_DISTANCE_MM,
        "conservation_checks": [["parent pair", ["parent"], ["parent_b"]],
                                ["parent vs daughters", ["parent"], ["daughter1", "daughter2"]]],
        "ratio_checks": [["daughter1 : daughter2", "daughter1", "daughter2", split / (1 - split)]]})
    truth = {name: plane_flow(case["velocity_true_t"], vtw, lumen_mask=lumen_true, **plane)["flow"] / 1000.0
             for name, plane in planes.items()}
    _write_json(base / "references" / "phantom" / "flows.json", {"planes": _flows_json(truth, rr_ms),
                                                                "source": "phantom ground truth"})
    report = quality_report(raw_encoded, unwrapped, magnitude, venc, vtw, lumen_mask=lumen["lumen"], wrap_count=wraps,
                            rr_ms=rr_ms, study_id=CASE_ID)
    (base / "outputs" / "quality_report.json").write_text(to_json(report))

    _write_json(base / "outputs" / "processing.json", {
        "software": {"name": "slicerheart_4dflow test suite (tests/clinical_case.py)"},
        "processing": {"backgroundSubtraction": "QuadraticFit", "phaseUnwrapping": "LaplacianThenTemporalUnwrap"},
        "phase_unwrapping_warnings": []})
    _write_json(base / "inputs" / "acquisition.json", {
        "vendor": VENDOR, "scanner": "phantom.acquire", "software_version": "", "rr_ms": rr_ms,
        "venc_encoded_mm_s": venc.tolist(), "encoding_to_world": E.tolist(),
        "component_encoding": [{"direction_code": f"encoded{c}", "max_value": 4096.0, "invert_to_ras": False} for c in range(3)]})
    _write_json(base / "annotations" / "annotations.json", {
        "lumen": {"file": "lumen.seg.nrrd", "segment": "lumen", "method": "automatic", "reviewed": True,
                  "algorithm": {"name": "slicerheart_4dflow.estimate_lumen"}},
        "planes": {"files": plane_files, "method": "automatic", "reviewed": True,
                   "algorithm": {"name": "phantom planes"}}})
    _write_json(base / "manifest.json", {"format_version": 3, "case_id": CASE_ID, "license": "CC BY 4.0", "doi": None,
                                         "diagnosis_category": "normal",
                                         "description": "synthetic bifurcation phantom (tests/clinical_case.py)"})
    return base / "manifest.json"


if __name__ == "__main__":
    import sys
    print(write_synthetic_case(sys.argv[1] if len(sys.argv) > 1 else "synthetic_clinical_case"))
