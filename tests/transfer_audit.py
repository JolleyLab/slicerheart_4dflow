"""
Transfer audit: does what the synthetic tests show carry over to a clinical case?

A clinical case has no ground truth, so the methods are validated on phantoms
(synthetic tier). The audit checks, per processing step, whether the phantoms
behave like the clinical case. It computes quantities that need no ground truth
on the clinical case and on a synthetic case matched to it, and compares them:

1. Measure the clinical case: venc, voxel size, number of frames, R-R
   interval, velocity noise, magnitude of lumen, tissue and air, vessel
   radius, peak speed, flow waveform and background phase offset.
2. Build a matched synthetic case: a bifurcation phantom on a grid of the same
   voxel size and frames, with the vessel radius, peak speed and waveform of
   the case, in static tissue with the magnitudes of the case, the clinical
   background offset (resampled to the phantom grid), acquired at the venc of
   the case with the noise of the case (phantom.acquire).
3. Run the same pipeline on both (the fixed order of preprocess(): war+msac
   background, noise mask, laplacian_then_temporal unwrapping, FDM divergence
   filter) and compute the paired quantities (PAIRED_QUANTITIES).
4. Compare: a pair whose ratio is outside a factor of BAND (2) is flagged.

The paired quantities are normalized (for example by the velocity noise) so
that they depend as little as possible on the size of the field of view and of
the vessel, which the phantom does not reproduce exactly. The matched
properties are measured on the synthetic case the same way as on the clinical
case and reported too: if they do not agree, the synthetic case is not matched
and its comparison is not meaningful.

A flagged pair means the synthetic model misses an effect that matters for that
step on real data (moving tissue, intravoxel dephasing, segmentation error,
...). The response is not to tune the method on the clinical case but to add
the effect to the phantom or the acquisition model, and a synthetic test that
reproduces the clinical value, so that the synthetic tier covers it.

The audit is a report, not a pass/fail test: test_clinical.test_transfer_audit
writes one report per case into the test report folder and records the values
as record-only metrics. Run it on one case:

    python tests/transfer_audit.py <data set folder> <case id> [--band 2.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slicerheart_4dflow import (correct_background_phase, divergence_free_sequence, encoding_check, noise_mask,
                                plane_flow, unwrap_velocity)
from slicerheart_4dflow.background_phase import body_mask, residual_static_velocity, static_tissue_candidates
from slicerheart_4dflow.geometry import voxel_spacing, voxel_to_world_matrix
from slicerheart_4dflow.phantom import acquire, bifurcation

BAND = 2.0
# Matched properties must agree within this relative difference for the comparison to be meaningful
MATCH_TOLERANCE = 0.25

# name: (description, floor); values below the floor count as the floor in the ratio, so that two negligible values
# (0.0001 against 0.0003) are not flagged
PAIRED_QUANTITIES = {
    "background/static_residual_over_noise": (
        "RMS of the time-averaged velocity in static tissue after background correction, over the noise of a time "
        "average (sigma_v / sqrt(frames)); about 1 for a perfect correction", 0.1),
    "background/rejected_candidate_fraction": ("static tissue candidates rejected as outliers by the war+msac fit", 0.01),
    "noise_mask/lumen_fraction": ("lumen voxels marked as noise before unwrapping (should be about 0; they are not "
                                  "unwrapped)", 0.01),
    "noise_mask/tissue_fraction": ("tissue voxels (selected by magnitude) marked as noise", 0.01),
    "unwrap/residual_temporal_jump_fraction": (
        "lumen voxel-frames whose velocity jumps by more than venc from the previous frame after unwrapping "
        "(unwrapping errors)", 0.001),
    "unwrap/residual_spatial_jump_fraction": (
        "pairs of neighbor lumen voxels whose velocity differs by more than venc after unwrapping "
        "(unwrapping errors)", 0.001),
    "divergence/rms_over_noise": ("lumen divergence RMS times voxel size, over sigma_v", 0.1),
    "divergence/fdm_reduction": ("fraction of the lumen divergence RMS removed by the FDM filter", 0.01),
    "divergence/fdm_energy_removed_over_noise": (
        "velocity energy removed by the FDM filter, over the noise energy (about 1/3 for a projection of white "
        "noise; more where the lumen or the flow model is wrong)", 0.1),
    "encoding/declared_to_best_ratio": ("divergence of the declared component mapping over the best one", 0.1),
    "encoding/discrimination": ("divergence of the second best mapping over the best one", 0.1),
    "plane_flow/lumen_dilation_change": (
        "largest relative change of net flow through a plane when the lumen is dilated by one voxel", 0.01),
}

MATCHED_PROPERTIES = ("noise_over_venc", "peak_mean_velocity_over_venc", "vessel_radius_mm", "background_offset_rms_mm_s",
                      "tissue_to_lumen_magnitude")

# Number of frames (of highest flow) in which the divergence quantities are computed
DIVERGENCE_FRAMES = 3


# ---------------------------------------------------------------------------
# Pipeline and measurements (the same for the clinical and the synthetic case)
# ---------------------------------------------------------------------------

def run_pipeline(case):
    """The fixed order of preprocess() on the encoded components, keeping the intermediate results."""
    raw = np.asarray(case["velocity_raw_encoded_t"], float)
    magnitude = np.asarray(case["magnitude_t"], float)
    venc, vtw, lumen = case["venc_encoded_mm_s"], case["voxel_to_world"], case["lumen"]
    corrected, background = correct_background_phase(raw, magnitude, method="war+msac", lumen_mask=lumen)
    signal = ~noise_mask(magnitude, corrected, venc=venc)
    unwrapped = unwrap_velocity(corrected, venc, vtw, method="laplacian_then_temporal", mask=signal)
    noise = noise_mask(magnitude, unwrapped, venc=venc)
    wraps = np.rint((unwrapped - corrected) / (2 * venc)).astype(np.int16)
    E = np.asarray(case["encoding_to_world"], float)
    return {"corrected": corrected, "background": background, "signal": signal, "unwrapped": unwrapped, "wraps": wraps,
            "noise": noise, "velocity": unwrapped @ E.T, "magnitude": magnitude}


def tissue_mask(case, pipe):
    """Tissue for the noise estimate, selected by magnitude only (as quality_report does)."""
    return static_tissue_candidates(pipe["magnitude"], pipe["corrected"], lumen_mask=case["lumen"],
                                    velocity_std_percentile=100.0, edge_erosion_voxels=0)


def velocity_noise(pipe, tissue):
    """sigma_v per encoded component: median temporal variance in tissue, unbiased for Gaussian noise."""
    from scipy.stats import chi2
    v = pipe["unwrapped"]
    nt = v.shape[0]
    var = v.var(axis=0, ddof=1)[tissue]
    return np.sqrt(np.median(var, axis=0) / (chi2.median(nt - 1) / (nt - 1)))


def flow_planes(case):
    """Plane flows (mm3/s per frame) of the case, the plane with the largest net flow first."""
    v, vtw, lumen = case["velocity_for_planes"], case["voxel_to_world"], case["lumen"]
    results = {name: plane_flow(v, vtw, lumen_mask=lumen, **p) for name, p in case["planes"].items()}
    return dict(sorted(results.items(), key=lambda item: -abs(np.mean(item[1]["flow"]))))


def measure_properties(case, pipe):
    """The acquisition and anatomy properties that the matched synthetic case reproduces."""
    lumen = case["lumen"]
    venc = np.asarray(case["venc_encoded_mm_s"], float)
    tissue = tissue_mask(case, pipe)
    sigma = velocity_noise(pipe, tissue)
    mean_magnitude = pipe["magnitude"].mean(axis=0)
    lumen_magnitude = float(np.median(mean_magnitude[lumen]))
    # body and air from body_mask (air = its background, of random phase), as the pipeline selects them, not from the
    # Otsu threshold of the magnitude: below that threshold the clinical cases have mostly dark tissue with the phase
    # noise of tissue, which gave the air of the synthetic case the phase of tissue, and on the synthetic case the
    # threshold separates the lumen from the tissue
    in_body = body_mask(pipe["magnitude"], pipe["corrected"], venc=venc)
    air = ~in_body & (mean_magnitude > 0)
    speed = np.linalg.norm(pipe["velocity"], axis=-1)[:, lumen]
    planes = flow_planes(dict(case, velocity_for_planes=pipe["velocity"]))
    # the plane with the largest flow (the parent vessel of the phantom)
    largest = next(iter(planes.values()))
    radius = float(np.sqrt(largest["area"] / np.pi))
    flow = largest["flow"]
    offset = background_offset(pipe)
    body = in_body & ~lumen
    return {
        "frames": int(pipe["unwrapped"].shape[0]),
        "rr_ms": float(case["rr_ms"]),
        "voxel_size_mm": voxel_spacing(case["voxel_to_world"]).tolist(),
        "venc_mm_s": venc.tolist(),
        "noise_mm_s": sigma.tolist(),
        "noise_over_venc": float(np.mean(sigma / venc)),
        "peak_speed_over_venc": float(np.percentile(speed, 99.0) / np.mean(venc)),
        "peak_mean_velocity_mm_s": float(np.abs(largest["mean_velocity"]).max()),
        "peak_mean_velocity_over_venc": float(np.abs(largest["mean_velocity"]).max() / np.mean(venc)),
        # not matched: sensitive to the velocity profile when the peak is near venc; a property of the data, not of
        # the methods
        "aliased_lumen_fraction": float((pipe["wraps"][:, lumen] != 0).any(-1).mean()),
        "vessel_radius_mm": radius,
        "waveform": (flow / flow[np.argmax(np.abs(flow))]).tolist(),
        "background_offset_rms_mm_s": float(np.sqrt(np.mean(offset[body] ** 2))),
        "tissue_to_lumen_magnitude": float(np.median(mean_magnitude[tissue]) / lumen_magnitude),
        "air_to_lumen_magnitude": float(np.median(mean_magnitude[air]) / lumen_magnitude) if air.any() else 0.0,
    }


def background_offset(pipe):
    offset = pipe["background"]["offset"]
    return offset.mean(axis=0) if offset.ndim == 5 else offset


def paired_quantities(case, pipe):
    """The quantities of PAIRED_QUANTITIES for one case (clinical or synthetic)."""
    lumen, vtw = case["lumen"], case["voxel_to_world"]
    venc = np.asarray(case["venc_encoded_mm_s"], float)
    unwrapped, wraps, noise = pipe["unwrapped"], pipe["wraps"], pipe["noise"]
    nt = unwrapped.shape[0]
    tissue = tissue_mask(case, pipe)
    sigma = velocity_noise(pipe, tissue)
    q = {}

    static = pipe["background"]["candidates"] & pipe["background"]["inliers"].all(axis=-1)
    residual = np.asarray(residual_static_velocity(unwrapped, static)["rms"])
    q["background/static_residual_over_noise"] = float(np.mean(residual / (sigma / np.sqrt(nt))))
    report = pipe["background"]["report"]
    q["background/rejected_candidate_fraction"] = 1.0 - sum(r["n_inliers"] for r in report) / max(sum(r["n_candidates"] for r in report), 1)

    q["noise_mask/lumen_fraction"] = float((~pipe["signal"])[lumen].mean())
    q["noise_mask/tissue_fraction"] = float(noise[tissue].mean()) if tissue.any() else float("nan")

    # unwrapping errors are counted where the unwrapping is applied: in the lumen voxels that the noise mask keeps
    # (the noise mask quantities cover the others, which are left wrapped)
    unwrapped_lumen = lumen & pipe["signal"]
    jumps = np.abs(np.diff(unwrapped[:, unwrapped_lumen], axis=0)) > venc
    q["unwrap/residual_temporal_jump_fraction"] = float(jumps.any(-1).mean()) if nt > 1 else 0.0
    pairs = jumps_total = 0
    for axis in range(3):
        both = np.logical_and(np.take(unwrapped_lumen, range(lumen.shape[axis] - 1), axis=axis),
                              np.take(unwrapped_lumen, range(1, lumen.shape[axis]), axis=axis))
        difference = np.abs(np.diff(unwrapped, axis=axis + 1))[:, both]
        pairs += difference.shape[0] * difference.shape[1]
        jumps_total += int((difference > venc).any(-1).sum())
    q["unwrap/residual_spatial_jump_fraction"] = jumps_total / max(pairs, 1)

    # the frames of highest flow through the plane of the largest flow
    flow = next(iter(flow_planes(dict(case, velocity_for_planes=pipe["velocity"])).values()))["flow"]
    frames = sorted(np.argsort(-np.abs(flow))[:DIVERGENCE_FRAMES].tolist())
    velocity = pipe["velocity"][frames]
    filtered, infos = divergence_free_sequence(velocity, lumen, vtw, method="fdm")
    sigma_world = float(np.sqrt(np.mean(sigma ** 2)))
    spacing = float(np.mean(voxel_spacing(vtw)))
    before = np.array([info["div_rms_before"] for info in infos])
    after = np.array([info["div_rms_after"] for info in infos])
    q["divergence/rms_over_noise"] = float(np.mean(before) * spacing / sigma_world)
    q["divergence/fdm_reduction"] = float(1.0 - np.mean(after) / np.mean(before))
    removed = np.sum((filtered - velocity)[:, lumen] ** 2)
    q["divergence/fdm_energy_removed_over_noise"] = float(removed / (len(frames) * lumen.sum() * 3 * sigma_world ** 2))

    check = encoding_check(pipe["velocity"], lumen, vtw)
    # not compared where the check is inconclusive (too little flow, or flow along few directions)
    conclusive = bool(check.get("conclusive", True))
    q["encoding/declared_to_best_ratio"] = float(check["ratio"]) if conclusive else float("nan")
    q["encoding/discrimination"] = float(check["discrimination"]) if conclusive else float("nan")

    dilated = ndimage.binary_dilation(lumen)
    changes = []
    for plane in case["planes"].values():
        q0 = plane_flow(pipe["velocity"], vtw, lumen_mask=lumen, **plane)["flow"].mean()
        q1 = plane_flow(pipe["velocity"], vtw, lumen_mask=dilated, **plane)["flow"].mean()
        changes.append(abs(q1 - q0) / max(abs(q0), 1e-9))
    q["plane_flow/lumen_dilation_change"] = float(max(changes))
    return q


# ---------------------------------------------------------------------------
# Matched synthetic case
# ---------------------------------------------------------------------------

def bifurcation_extent_mm(radius_mm, angle_deg=30.0, split=0.6, margin_mm=20.0):
    """Half extents (x, y, z) in mm of a box that holds the bifurcation planes, plus a margin of static tissue."""
    a = np.radians(angle_deg)
    r1, r2 = radius_mm * split ** (1 / 3), radius_mm * (1 - split) ** (1 / 3)
    overlap = max(r1, r2) / np.sin(a)
    plane_distance = max(25.0, overlap + 2.0, (r1 + r2 + 3.0) / np.sin(2 * a) + 2.0)
    disc = radius_mm + 3.0
    return np.array([plane_distance * np.sin(a) + disc, disc, plane_distance + disc]) + margin_mm


def matched_synthetic_case(properties, offset=None, seed=0, air_border_voxels=3):
    """A synthetic case with the properties of measure_properties, in the format of slicer_io.load_case
    (encoded components = world components). offset: background offset (nk, nj, ni, 3) of the clinical case,
    resampled to the phantom grid (stretched to its extent) and scaled to the RMS of the clinical offset in static
    tissue; default: none."""
    rng = np.random.default_rng(seed)
    spacing = np.asarray(properties["voxel_size_mm"], float)              # (si, sj, sk)
    venc = np.asarray(properties["venc_mm_s"], float)
    nt = properties["frames"]
    radius = properties["vessel_radius_mm"]
    half = bifurcation_extent_mm(radius) + air_border_voxels * spacing
    shape = tuple(int(n) for n in np.ceil(2 * half / spacing)[::-1])      # (nk, nj, ni)
    vtw = voxel_to_world_matrix(spacing=tuple(spacing), origin=tuple(-half))
    waveform = np.asarray(properties["waveform"], float)

    # vessel radius and peak velocity measured as on the clinical case, in the plane of the parent vessel: the radius
    # from the lumen area (smaller than the radius of the phantom because of the voxelization) and the peak of the
    # mean through-plane velocity
    unit = bifurcation(shape=shape, voxel_to_world=vtw, radius_mm=radius, v_max_parent_mm_s=1.0, nt=nt, waveform=waveform)
    area = plane_flow(unit["velocity"][:1], vtw, lumen_mask=unit["mask"], **unit["planes"]["parent"])["area"]
    radius *= radius / np.sqrt(area / np.pi)
    half = bifurcation_extent_mm(radius) + air_border_voxels * spacing
    shape = tuple(int(n) for n in np.ceil(2 * half / spacing)[::-1])
    vtw = voxel_to_world_matrix(spacing=tuple(spacing), origin=tuple(-half))
    unit = bifurcation(shape=shape, voxel_to_world=vtw, radius_mm=radius, v_max_parent_mm_s=1.0, nt=nt, waveform=waveform)
    unit_peak = np.abs(plane_flow(unit["velocity"], vtw, lumen_mask=unit["mask"], **unit["planes"]["parent"])["mean_velocity"]).max()
    phantom = bifurcation(shape=shape, voxel_to_world=vtw, radius_mm=radius, nt=nt, waveform=waveform,
                          v_max_parent_mm_s=properties["peak_mean_velocity_mm_s"] / unit_peak)
    lumen = phantom["mask"]

    tissue_magnitude = properties["tissue_to_lumen_magnitude"]
    magnitude = np.where(lumen, 1.0, tissue_magnitude)
    body = np.zeros(shape, bool)
    body[air_border_voxels:-air_border_voxels, air_border_voxels:-air_border_voxels, air_border_voxels:-air_border_voxels] = True
    # air around the tissue; the vessels keep their signal where they leave the field of view
    magnitude = np.where(body | lumen, magnitude, max(properties["air_to_lumen_magnitude"], 0.01))

    # snr of a unit-magnitude voxel such that the tissue velocity noise is that of the clinical case:
    # sigma_v = sqrt(2) venc / (pi snr magnitude)
    snr = np.sqrt(2.0) / (np.pi * tissue_magnitude * properties["noise_over_venc"])

    if offset is not None:
        coordinates = np.meshgrid(*[np.linspace(0, n_in - 1, n_out) for n_in, n_out in zip(offset.shape[:3], shape)],
                                  indexing="ij")
        offset = np.stack([ndimage.map_coordinates(offset[..., c], coordinates, order=1) for c in range(3)], axis=-1)
        rms = np.sqrt(np.mean(offset[body & ~lumen] ** 2))
        offset *= properties["background_offset_rms_mm_s"] / max(rms, 1e-9)
    else:
        offset = np.zeros(shape + (3,))

    acquired = acquire(phantom["velocity"] + offset[None], venc, magnitude=np.broadcast_to(magnitude, (nt,) + shape),
                       snr=snr, rng=rng)
    velocity = acquired["velocity"].astype(np.float32)
    return {
        "manifest": {"case_id": "matched synthetic"},
        "velocity_raw_encoded_t": velocity,
        "magnitude_t": acquired["magnitude"].astype(np.float32),
        "voxel_to_world": vtw,
        "lumen": lumen,
        "venc_encoded_mm_s": venc,
        "encoding_to_world": np.eye(3),
        "rr_ms": properties["rr_ms"],
        "planes": phantom["planes"],
        "velocity_true_t": phantom["velocity"],
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def transfer_audit(case, band=BAND, seed=0):
    """Audit one case (loaded by slicer_io.load_case, or a matched_synthetic_case). Returns a dict with the
    measured and matched properties, and rows {name: dict(case, synthetic, ratio, flagged)}."""
    pipe = run_pipeline(case)
    properties = measure_properties(case, pipe)
    synthetic = matched_synthetic_case(properties, offset=background_offset(pipe), seed=seed)
    synthetic_pipe = run_pipeline(synthetic)
    synthetic_properties = measure_properties(synthetic, synthetic_pipe)

    matching = {}
    for name in MATCHED_PROPERTIES:
        c, s = properties[name], synthetic_properties[name]
        difference = abs(s - c) / max(abs(c), 1e-9)
        matching[name] = {"case": c, "synthetic": s, "relative_difference": difference,
                          "matched": bool(difference <= MATCH_TOLERANCE)}

    q_case, q_synthetic = paired_quantities(case, pipe), paired_quantities(synthetic, synthetic_pipe)
    rows = {}
    for name in PAIRED_QUANTITIES:
        c, s = q_case[name], q_synthetic[name]
        # small values are compared with a floor, so that two negligible values are not flagged
        floor = PAIRED_QUANTITIES[name][1]
        ratio = (max(abs(c), floor) / max(abs(s), floor)) if np.isfinite(c) and np.isfinite(s) else float("nan")
        rows[name] = {"case": c, "synthetic": s, "ratio": ratio,
                      "flagged": bool(np.isfinite(ratio) and (ratio > band or ratio < 1.0 / band))}
    return {"case_id": case["manifest"]["case_id"], "band": band, "properties": properties,
            "synthetic_properties": synthetic_properties, "matching": matching, "rows": rows}


def to_markdown(audit):
    lines = [f"# Transfer audit: {audit['case_id']}", "",
             "Synthetic case: bifurcation phantom matched to the case (see tests/transfer_audit.py).", "",
             "## Matched properties", "",
             "| property | case | synthetic | difference | matched |", "|---|---|---|---|---|"]
    for name, m in audit["matching"].items():
        lines.append(f"| {name} | {m['case']:.4g} | {m['synthetic']:.4g} | {100 * m['relative_difference']:.0f}% | "
                     f"{'yes' if m['matched'] else '**no**'} |")
    p = audit["properties"]
    lines += ["", f"Frames {p['frames']}, R-R {p['rr_ms']:.0f} ms, voxel size "
              f"{' x '.join(f'{s:.2f}' for s in p['voxel_size_mm'])} mm, venc "
              f"{', '.join(f'{v:.0f}' for v in p['venc_mm_s'])} mm/s. Lumen voxel-frames with wraps: "
              f"{100 * p['aliased_lumen_fraction']:.2f}% (synthetic {100 * audit['synthetic_properties']['aliased_lumen_fraction']:.2f}%).", "",
              f"## Paired quantities (flagged outside a factor of {audit['band']:g})", "",
              "| quantity | case | synthetic | ratio | flag |", "|---|---|---|---|---|"]
    for name, r in audit["rows"].items():
        lines.append(f"| {name} | {r['case']:.4g} | {r['synthetic']:.4g} | {r['ratio']:.2f} | {'**FLAG**' if r['flagged'] else ''} |")
    lines += ["", "## Quantities", ""] + [f"- `{name}`: {text}" for name, (text, _) in PAIRED_QUANTITIES.items()]
    return "\n".join(lines) + "\n"


def to_json(audit):
    return json.dumps({k: v for k, v in audit.items() if k not in ("properties", "synthetic_properties")}
                      | {"properties": {k: v for k, v in audit["properties"].items() if k != "waveform"}},
                      indent=2, default=float)


def write_summary(folder):
    """summary.md in folder: the ratios of all case reports (*.json) in it, flagged ones in bold."""
    folder = Path(folder)
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
    if not audits:
        return
    names = list(PAIRED_QUANTITIES)
    lines = ["# Transfer audit summary", "",
             f"Ratio of each quantity on the case to that on its matched synthetic case; **bold**: outside a factor of "
             f"{audits[0]['band']:g}. Unmatched: properties that could not be matched (comparison not meaningful). "
             "Quantities are described in the case reports.", "",
             "| case | unmatched | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for audit in audits:
        unmatched = ", ".join(name for name, m in audit["matching"].items() if not m["matched"]) or ""
        cells = []
        for name in names:
            row = audit["rows"].get(name)
            cells.append("" if row is None else (f"**{row['ratio']:.2f}**" if row["flagged"] else f"{row['ratio']:.2f}"))
        lines.append(f"| {audit['case_id']} | {unmatched} | " + " | ".join(cells) + " |")
    flagged = {name: sum(a["rows"][name]["flagged"] for a in audits if name in a["rows"]) for name in names}
    lines.append("| flagged cases | | " + " | ".join(str(flagged[name]) for name in names) + " |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="folder of the data set (with manifest.json, inputs/, outputs/)")
    parser.add_argument("case", help="case ID (folder name in inputs/)")
    parser.add_argument("--band", type=float, default=BAND)
    parser.add_argument("--output", help="write the report (markdown) to this file")
    args = parser.parse_args()
    from slicerheart_4dflow.slicer_io import load_case
    audit = transfer_audit(load_case(args.dataset, args.case), band=args.band)
    text = to_markdown(audit)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
