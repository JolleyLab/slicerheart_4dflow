"""
A synthetic clinical test case: a bifurcation phantom written as a data set with one case in the layout of the clinical
cases (format 4, as SlicerHeart Import4DFlow exports them), so that the clinical tests (test_clinical.py) run in CI without clinical data.

The phantom (test_transfer_audit.synthetic_case: Murray radii, 60/40 split, aortic waveform) is acquired at a venc below
its peak velocity (aliasing), with a second-order background offset, noise and static tissue, in encoded components
that are a signed permutation of the world components (as Siemens encodes them). The corrections are the pipeline of
this package (war+msac background, noise mask, laplacian_then_temporal unwrapping), the lumen is the automatic one
(estimate_lumen, on a grid twice as fine as the image), the planes are those of the phantom plus a second parent plane
4 mm downstream (a plane pair), the plane flows of outputs/<case>/flows.json are computed as the SlicerHeart plane probe does
(plain interpolation, samples tested against the segmentation), and inputs/<case>/references/phantom/flows.json holds the true flows.
The planes have a branch segmentation (vessels.seg.nrrd: the true lumen dilated by one voxel, so the measurement region of
each plane is its lumen), and the reference of one plane is along the opposite of the plane normal (plane_normal_sign -1)
and lacks the last frame, as references of planes placed by hand can.

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


def probe_plane_samples(voxel_to_world, plane, sampling_mm, world="LPS"):
    """Sample points of the SlicerHeart plane probe as the flows of the test case outputs are computed
    (Analyze4DFlowLogic.writeTestCaseFlows): a new point-normal plane markup of size 2 * radius at the centre, sampled
    on a regular grid at multiples of the sampling distance from the centre (truncated to whole steps) along the in-plane
    axes of the markup, each sample representing sampling_mm^2. The in-plane axes are those Slicer gives a new plane
    when its normal is set: the axes of RAS turned by the smallest rotation from (0, 0, 1) to the normal (a half turn
    about -A for the opposite normal). Returns (points_world (n, 3), array_coords (3, n), unit_normal, area_per_sample)."""
    import math
    from slicerheart_4dflow.geometry import world_to_array_coordinates
    world_to_ras = np.diag([-1.0, -1.0, 1.0]) if world == "LPS" else np.eye(3)
    normal = np.asarray(plane["normal"], float)
    normal = normal / np.linalg.norm(normal)
    n = world_to_ras @ normal
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    if np.linalg.norm(axis) < 1e-4:
        axis = np.array([0.0, -1.0, 0.0])          # vtkMath::Perpendiculars of (0, 0, 1)
    axis = axis / np.linalg.norm(axis)
    angle = math.atan2(np.linalg.norm(np.cross(z, n)), z @ n)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    x_axis, y_axis = world_to_ras @ rotation[:, 0], world_to_ras @ rotation[:, 1]    # world_to_ras is its own inverse
    steps = math.floor(plane["radius"] / sampling_mm)
    u, v = np.meshgrid(np.arange(-steps, steps + 1) * sampling_mm, np.arange(-steps, steps + 1) * sampling_mm, indexing="ij")
    points = np.asarray(plane["centre"], float) + u.reshape(-1, 1) * x_axis + v.reshape(-1, 1) * y_axis
    return points, world_to_array_coordinates(voxel_to_world, points), normal, sampling_mm ** 2


def probe_flows(velocity_t, vtw, regions, labelmap_vtw, planes, sampling_mm=None, world="LPS"):
    """Flow rate (ml/s) through each plane as the SlicerHeart plane probe computes the flows of the test case outputs:
    plain trilinear interpolation of the velocity at the samples of the probe (probe_plane_samples) whose nearest voxel
    of the segmentation grid (which may be finer than the image) is in the measurement region of the plane, within the
    disc of the plane (the voxel centre at most the plane radius from the plane axis).
    regions: the measurement region (bool, on the grid of labelmap_vtw) of every plane, or {plane name: region}."""
    from scipy import ndimage
    from slicerheart_4dflow.geometry import world_to_array_coordinates
    sampling_mm = SAMPLING_DISTANCE_MM if sampling_mm is None else sampling_mm
    labelmap_vtw = np.asarray(labelmap_vtw, float)
    flows = {}
    for name, plane in planes.items():
        labelmap = regions[name] if isinstance(regions, dict) else regions
        points, coordinates, normal, area = probe_plane_samples(vtw, plane, sampling_mm, world)
        nearest = np.round(world_to_array_coordinates(labelmap_vtw, points)).astype(int)
        valid = np.all((nearest >= 0) & (nearest < np.array(labelmap.shape)[:, None]), axis=0)
        inside = np.zeros(len(points), bool)
        inside[valid] = labelmap[nearest[0, valid], nearest[1, valid], nearest[2, valid]]
        voxel_centres = (labelmap_vtw[:3, :3] @ nearest[::-1] + labelmap_vtw[:3, 3:4]).T
        offsets = voxel_centres - np.asarray(plane["centre"], float)
        inside &= np.linalg.norm(offsets - (offsets @ normal)[:, None] * normal, axis=1) <= plane["radius"]
        through_plane = velocity_t @ normal
        flows[name] = np.array([ndimage.map_coordinates(through_plane[t], coordinates[:, inside], order=1).sum() * area
                                for t in range(velocity_t.shape[0])]) / 1000.0
    return flows


def _flows_json(flows, rr_ms):
    planes = {}
    for name, q in flows.items():
        planes[name] = {"net_volume_mL": float(np.mean(q) * rr_ms / 1000.0),
                        "forward_volume_mL": float(np.mean(np.clip(q, 0, None)) * rr_ms / 1000.0),
                        "backward_volume_mL": float(np.mean(np.clip(q, None, 0)) * rr_ms / 1000.0),
                        "flow_rate_ml_s": [float(x) for x in q]}
    return planes


def _along_opposite_normal(values, times_ms):
    """A reference of a plane whose positive flow direction is opposite to the plane normal, without its last frame."""
    return {"net_volume_mL": -values["net_volume_mL"], "forward_volume_mL": -values["backward_volume_mL"],
            "backward_volume_mL": -values["forward_volume_mL"], "plane_normal_sign": -1,
            "flow_rate_ml_s": [-q for q in values["flow_rate_ml_s"][:-1]], "time_ms": [float(t) for t in times_ms[:-1]]}


def write_synthetic_case(root):
    """Write a data set with the synthetic case (CASE_ID) into root: manifest.json, inputs/<CASE_ID>/, outputs/<CASE_ID>/.
    Returns (root, CASE_ID)."""
    from slicerheart_4dflow import (correct_background_phase, estimate_lumen, noise_mask, pcmra, plane_flow, quality_report,
                                    to_json, unwrap_velocity)
    from slicerheart_4dflow.phantom import aortic_waveform
    from test_transfer_audit import synthetic_case

    root = Path(root)
    acquisition_dir = root / "inputs" / CASE_ID / "acquisition"
    annotations_dir = root / "inputs" / CASE_ID / "annotations"
    references_dir = root / "inputs" / CASE_ID / "references"
    outputs_dir = root / "outputs" / CASE_ID
    for folder in (acquisition_dir, annotations_dir, outputs_dir):
        folder.mkdir(parents=True, exist_ok=True)
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

    _write_sequence(acquisition_dir / "magnitude.seq.nrrd", magnitude, vtw, times_ms)
    _write_sequence(acquisition_dir / "velocity_raw.seq.nrrd", raw_world.astype(np.float32), vtw, times_ms, vector=True)
    _write_sequence(outputs_dir / "velocity.seq.nrrd", velocity, vtw, times_ms, vector=True)
    _write_sequence(outputs_dir / "wrap_count.seq.nrrd", wraps, vtw, times_ms, vector=True, spatial=False)
    _write_volume(outputs_dir / "pcmra.nrrd", angiogram, vtw)

    lumen = estimate_lumen(angiogram, vtw, velocity_t=velocity, magnitude_t=magnitude)
    _write_segmentation(annotations_dir / "lumen.seg.nrrd", lumen["mask"], lumen["voxel_to_world"])
    from scipy import ndimage
    _write_segmentation(annotations_dir / "vessels.seg.nrrd", ndimage.binary_dilation(lumen_true), vtw)

    planes = dict(case["planes"])
    parent = planes["parent"]
    planes["parent_b"] = dict(parent, centre=tuple(np.asarray(parent["centre"]) + 4.0 * np.asarray(parent["normal"])))
    plane_files = []
    for name, plane in planes.items():
        _write_plane(annotations_dir / f"{name}.mrk.json", name, plane)
        plane_files.append(f"{name}.mrk.json")

    probe = probe_flows(velocity, vtw, lumen["mask"], lumen["voxel_to_world"], planes)
    split = 0.6
    _write_json(outputs_dir / "flows.json", {
        "planes": _flows_json(probe, rr_ms), "sampling_distance_mm": SAMPLING_DISTANCE_MM,
        "conservation_checks": [["parent pair", ["parent"], ["parent_b"]],
                                ["parent vs daughters", ["parent"], ["daughter1", "daughter2"]]],
        "ratio_checks": [["daughter1 : daughter2", "daughter1", "daughter2", split / (1 - split)]]})
    truth = {name: plane_flow(case["velocity_true_t"], vtw, lumen_mask=lumen_true, **plane)["flow"] / 1000.0
             for name, plane in planes.items()}
    reference = _flows_json(truth, rr_ms)
    reference["daughter2"] = _along_opposite_normal(reference["daughter2"], times_ms)
    _write_json(references_dir / "phantom" / "flows.json", {"planes": reference, "source": "phantom ground truth"})
    report = quality_report(raw_encoded, unwrapped, magnitude, venc, vtw, lumen_mask=lumen["lumen"], wrap_count=wraps,
                            rr_ms=rr_ms, study_id=CASE_ID)
    (outputs_dir / "quality_report.json").write_text(to_json(report))

    _write_json(outputs_dir / "processing.json", {
        "software": {"name": "slicerheart_4dflow test suite (tests/clinical_case.py)"},
        "processing": {"backgroundSubtraction": "QuadraticFit", "phaseUnwrapping": "LaplacianThenTemporalUnwrap"},
        "phase_unwrapping_warnings": []})
    _write_json(acquisition_dir / "acquisition.json", {
        "vendor": VENDOR, "scanner": "phantom.acquire", "software_version": "", "rr_ms": rr_ms,
        "subject": {"diagnosis_category": "normal"},
        "venc_encoded_mm_s": venc.tolist(), "encoding_to_world": E.tolist(),
        "component_encoding": [{"direction_code": f"encoded{c}", "max_value": 4096.0, "invert_to_ras": False} for c in range(3)]})
    _write_json(annotations_dir / "annotations.json", {
        "lumen": {"file": "lumen.seg.nrrd", "segment": "lumen", "method": "automatic", "reviewed": True,
                  "algorithm": {"name": "slicerheart_4dflow.estimate_lumen"}},
        "planes": {"files": plane_files, "method": "automatic", "reviewed": True,
                   "algorithm": {"name": "phantom planes"},
                   "segmentation": {"file": "vessels.seg.nrrd", "method": "manual", "reviewed": True,
                                    "segments": {name: "lumen" for name in planes}}}})
    _write_json(root / "manifest.json", {"format_version": 4, "name": "synthetic bifurcation phantom (tests/clinical_case.py)",
                                         "license": "CC BY 4.0", "doi": None})
    return root, CASE_ID


if __name__ == "__main__":
    import sys
    print(write_synthetic_case(sys.argv[1] if len(sys.argv) > 1 else "synthetic_clinical_case"))
