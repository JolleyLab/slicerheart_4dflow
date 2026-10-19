"""
DRAFT: written before any clinical case was available. The case layout,
units and component frame follow the Slicer 5D convention from the
Discourse post; they must be aligned with what the SlicerHeart Import4DFlow
and Analyze4DFlow modules actually write before this is relied on.

Transfer audit: paired quantities that both tiers can report without a
ground truth, computed on a clinical case and on a synthetic case built at
the clinical case's own settings (venc, voxel size, frames, velocity
noise), printed side by side. A pair outside the band says the synthetic
model is missing an effect for that filter (docs/VALIDATION_STRATEGY.md,
section 3). The response is to add the effect to the acquisition model or
a phantom, not to tune the filter on the clinical case.

    python -m tests.clinical.transfer_audit path/to/manifest.json [--band 2.0]
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np

from slicerheart_4dflow import load_case, encoding_check
from slicerheart_4dflow.background_phase import correct_background_phase
from slicerheart_4dflow.divergence_free import divergence_free_sequence, divergence_stats
from slicerheart_4dflow.report import quality_report
from slicerheart_4dflow.unwrap import noise_mask, unwrap_velocity, temporal_unwrap

BAND = 2.0


def paired_quantities(case):
    """The audit quantities of one loaded case (real or synthetic)."""
    v, mag, M, lumen, venc = (case["velocity_t"].astype(float), case["magnitude_t"].astype(float),
                              case["voxel_to_world"], case["lumen"], case["venc_mm_s"])
    out = {}
    corrected, info = correct_background_phase(v, mag, method="war+msac", lumen_mask=lumen)
    rep = quality_report(v, corrected, mag, venc, M, lumen_mask=lumen)
    out["static_residual_rms_mm_s"] = float(np.mean(rep["background_offset"]["residual_rms_processed_mm_s"]))
    n_c = sum(r["n_candidates"] for r in info["report"]); n_i = sum(r["n_inliers"] for r in info["report"])
    out["background_candidates_rejected_fraction"] = 1.0 - n_i / max(n_c, 1)
    out["velocity_noise_over_venc"] = float(np.mean(rep["velocity_noise_mm_s"]) / np.mean(venc))
    noise = noise_mask(mag, corrected)
    out["noise_mask_fov_fraction"] = float(noise.mean())
    out["noise_mask_lumen_fraction"] = float(noise[lumen].mean())
    unwrapped = unwrap_velocity(corrected, venc, M, method="laplacian_then_temporal", mask=~noise)
    k = np.round((unwrapped - corrected) / (2 * venc))
    out["aliased_lumen_voxel_frames_fraction"] = float((k[:, lumen] != 0).any(-1).mean())
    # residual temporal jumps larger than venc after unwrapping (should be ~0)
    jumps = np.abs(np.diff(unwrapped[:, lumen], axis=0)) > venc
    out["residual_temporal_jumps_fraction"] = float(jumps.any(-1).mean()) if unwrapped.shape[0] > 1 else 0.0
    out["lumen_component_over_venc_fraction"] = float((np.abs(unwrapped[:, lumen]) > venc).any(-1).mean())
    div0 = np.mean([divergence_stats(unwrapped[t], lumen, M)["div_rms"] for t in range(v.shape[0])])
    proj, _ = divergence_free_sequence(unwrapped, lumen, M, method="fdm")
    div1 = np.mean([divergence_stats(proj[t], lumen, M)["div_rms"] for t in range(v.shape[0])])
    out["divergence_rms_1_s"] = float(div0)
    out["divergence_reduction"] = float(1.0 - div1 / max(div0, 1e-12))
    e_removed = np.sum((proj - unwrapped)[:, lumen] ** 2); e_total = np.sum(unwrapped[:, lumen] ** 2)
    out["projection_energy_removed_fraction"] = float(e_removed / max(e_total, 1e-12))
    e = encoding_check(unwrapped, lumen, M)
    out["encoding_discrimination"] = float(e["discrimination"]) if np.isfinite(e["discrimination"]) else float("nan")
    out["encoding_ratio"] = float(e["ratio"])
    return out


def matched_synthetic_case(case, root):
    """A mock bifurcation case at the clinical case's venc, voxel size, frame
    count and velocity noise (SNR from sigma_v = sqrt(2) venc / (pi SNR))."""
    from .casegen import make_mock_bifurcation_case
    v, mag, M, lumen, venc = (case["velocity_t"].astype(float), case["magnitude_t"].astype(float),
                              case["voxel_to_world"], case["lumen"], case["venc_mm_s"])
    from slicerheart_4dflow.semisynthetic import acquisition_snr
    rep = quality_report(v, v, mag, venc, M, lumen_mask=lumen)
    snr, _ = acquisition_snr(mag, lumen, rep["velocity_noise_mm_s"], venc)
    spacing = tuple(float(x) for x in case["manifest"]["voxel_size_mm"])
    return make_mock_bifurcation_case(root, case_id="matched_synthetic", spacing_mm=spacing,
                                      nt=int(v.shape[0]), venc_mm_s=float(np.mean(venc)), snr=snr,
                                      rr_ms=case["rr_ms"])


def transfer_audit(case, band=BAND, root=None):
    """Rows (name, clinical value, synthetic value, ratio, flagged)."""
    root = tempfile.mkdtemp() if root is None else root
    synth = load_case(matched_synthetic_case(case, root))
    q_c, q_s = paired_quantities(case), paired_quantities(synth)
    rows = []
    for name in q_c:
        c, s = q_c[name], q_s[name]
        ratio = (c / s) if (s not in (0.0,) and np.isfinite(c) and np.isfinite(s)) else float("nan")
        flagged = bool(np.isfinite(ratio) and (ratio > band or ratio < 1.0 / band) and max(abs(c), abs(s)) > 1e-3)
        rows.append((name, c, s, ratio, flagged))
    return rows


def to_markdown(rows, case_id):
    L = [f"Transfer audit: {case_id}", "", "| quantity | clinical | synthetic (matched) | ratio | flag |", "|---|---|---|---|---|"]
    for name, c, s, r, f in rows:
        L.append(f"| {name} | {c:.4g} | {s:.4g} | {r:.2f} | {'FLAG' if f else ''} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--band", type=float, default=BAND)
    args = ap.parse_args()
    case = load_case(args.manifest)
    print(to_markdown(transfer_audit(case, args.band), case["manifest"]["case_id"]))


if __name__ == "__main__":
    main()
