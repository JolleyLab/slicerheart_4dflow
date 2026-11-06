"""
Per-study quality report for 4D flow pre-processing.

quality_report() gathers in one structure quality-assurance checks of the
kind addressed by the 2023 4D flow consensus statement. Every check that a
velocity field allows is computed on both the raw velocity (the input of the
processing) and the processed velocity (its output):

  venc adequacy      peak lumen speed and peak component against venc,
                     fraction of lumen voxel-frames aliased in the raw
                     velocity and the number of wraps recovered (a venc
                     below the peak velocity is the most common acquisition
                     problem).
  velocity SNR       noise estimate from the temporal velocity standard
                     deviation in static tissue (velocity noise is
                     sigma_v = sqrt(2) venc / (pi SNR_magnitude), so static
                     tissue velocity std is a direct noise measure).
  background offset  residual static-tissue velocity, against the 0.6 cm/s
                     acceptance threshold for shunt and regurgitant
                     quantification.
  mass conservation  percent inflow/outflow error at user-defined plane
                     sets, and Qp:Qs when planes are labelled for it.
  noise mask         fraction of the field of view masked.
  divergence         RMS divergence in the lumen.
  encoding           consistency of the encoding directions.

In the dict the two are named "raw" and "processed" (as keys of a value, or
in the name of the key); to_markdown() calls them input and output. Values
that describe the acquisition (venc, frames), the masks, or what the
processing changed (wraps) are given once.

The velocity noise, background offset, venc, divergence and encoding checks
are the sections of the report (SECTIONS); quality_report(sections=)
computes only the sections that are asked for.

The report is a plain dict (JSON serialisable) with a to_markdown() helper
for a one-page summary, so a study coordinator or reviewer can read it
without opening the data.

References
----------
    Bissell MM, Raimondi F, Ait Ali L, et al. 4D Flow cardiovascular
    magnetic resonance consensus statement: 2023 update. J Cardiovasc Magn
    Reson 2023;25(1):40. PMID 37474977. doi:10.1186/s12968-023-00942-z.
    (Acquisition, post-processing and quality assurance.)
    Gatehouse PD, Rolf MP, Graves MJ, et al. Flow measurement by
    cardiovascular magnetic resonance: a multi-centre multi-vendor study of
    background phase offset errors that can compromise the accuracy of
    derived regurgitant or shunt flow measurements. J Cardiovasc Magn Reson
    2010;12:5. PMID 20074359. doi:10.1186/1532-429X-12-5. (0.6 cm/s
    threshold.)
    Burkhardt BEU, Kellenberger CJ, Callaghan FM, Valsangiacomo Buechel
    ER, Geiger J. Flow evaluation software for four-dimensional flow MRI: a
    reliability and validation study. Radiol Med 2023;128(10):1225-1235.
    PMID 37620674. doi:10.1007/s11547-023-01697-4. (Conservation of mass
    used as an internal control when comparing software packages.)

The velocity noise relation sigma_v = sqrt(2) venc / (pi SNR) follows from
first principles: the phase of each acquisition has noise 1/SNR (radians)
in a voxel of magnitude SNR, the velocity phase is the difference of two
acquisitions (noise sqrt(2)/SNR), and venc maps a phase of pi to velocity.

Conventions as elsewhere (velocity mm/s, world components, voxel_to_world).
"""

from __future__ import annotations

import json
import time
import numpy as np

from .background_phase import static_tissue_candidates, residual_static_velocity
from .divergence_free import divergence_stats
from .segmentation import noise_tissue_mask as select_noise_tissue, velocity_noise
from .qa import plane_flow, mass_conservation

RESIDUAL_STATIC_THRESHOLD_MM_S = 6.0        # 0.6 cm/s, Gatehouse 2010
NEAR_VENC_FRACTION = 0.98                   # a component at or above this fraction of venc is "near venc"
# Sections that quality_report(sections=) can select; the others are computed when their arguments are given
SECTIONS = ("velocity_noise", "background_offset", "venc", "divergence", "encoding")
LUMEN_SECTIONS = ("venc", "divergence", "encoding")    # computed in the lumen: left out without lumen_mask


def _f(x):
    return float(x)


def quality_report(velocity_raw_t, velocity_processed_t, magnitude_t, venc, voxel_to_world,
                   lumen_mask=None, static_mask=None, noise_mask=None, planes=None, checks=None,
                   heart_rate_bpm=None, wrap_count=None, study_id=None, extra=None, timing=None,
                   noise_tissue_mask=None, rr_ms=None, multivenc=None, sections=None):
    """
    velocity_raw_t, velocity_processed_t : (nt, nk, nj, ni, 3) mm/s, before and
        after preprocessing (same frames).
    magnitude_t : (nt, nk, nj, ni) magnitude, from which the tissue of the
                  velocity noise and background offset sections is selected;
                  None if neither is computed (or their masks are given).
    venc        : scalar or (3,) mm/s.
    lumen_mask  : (nk, nj, ni) bool; enables the sections computed in the
                  lumen (LUMEN_SECTIONS: venc, divergence, encoding).
    sections    : names of the sections to compute, of SECTIONS (default:
                  all). The planes, mass conservation, multi-venc and noise
                  mask entries do not depend on it: they are computed when
                  their arguments are given.
    static_mask : (nk, nj, ni) bool; default from static_tissue_candidates.
    noise_tissue_mask : (nk, nj, ni) bool, tissue for the velocity-noise
                  estimate; must not be selected by velocity (default:
                  segmentation.noise_tissue_mask, the same tissue as the
                  velocity noise of the lumen estimate).
    rr_ms       : cycle length for per-beat volumes (alternative to heart_rate_bpm).
    multivenc   : DRAFT. Optional dict for a dual-/multi-venc acquisition:
                  the 'multivenc' entry of preprocess()['report'] (method,
                  vencs, effective venc, fractions changed and corrected),
                  optionally with 'velocity_high_t' (nt, nk, nj, ni, 3), the
                  highest-venc series, from which the static-tissue noise
                  ratio of the raw and of the processed (combined) velocity
                  to the high-venc image is reported (the velocity-to-noise
                  gain of the low venc).
    noise_mask  : (nk, nj, ni) bool, True = noise; optional.
    planes      : {name: dict(centre=, normal=, radius=)} in world mm; optional.
    checks      : [(label, inflow_names, outflow_names)] for mass_conservation.
    wrap_count  : optional (nt, nk, nj, ni, 3) integer wraps applied by unwrapping.
    extra       : any JSON-serialisable dict to include (acquisition metadata).
    timing      : optional {step: seconds} from preprocess()['report']['timing_s'];
                  reported as processing_time_s together with the time this
                  report itself took.
    """
    t_report = time.perf_counter()
    sections = set(SECTIONS if sections is None else [sections] if isinstance(sections, str) else sections)
    if sections - set(SECTIONS):
        raise ValueError(f"unknown sections {sorted(sections - set(SECTIONS))}; the sections are {', '.join(SECTIONS)}")
    v0 = np.asarray(velocity_raw_t, float); v1 = np.asarray(velocity_processed_t, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    rep = {"study_id": study_id, "n_frames": int(v0.shape[0]), "venc_mm_s": venc.tolist()}
    stages = (("raw", v0), ("processed", v1))
    mag_t = None if magnitude_t is None else np.asarray(magnitude_t, float)

    # Velocity noise of the raw and of the processed velocity
    # (segmentation.velocity_noise) in tissue selected by MAGNITUDE only
    # (segmentation.noise_tissue_mask, selected on the raw velocity): the
    # static-tissue candidates are selected for low temporal std, so their
    # std is biased low (7.3 vs a true 10 mm/s on the chest phantom). The same
    # definition is used for the velocity noise of the lumen estimate
    # (velocity_noise_in_tissue).
    high_venc_noise = bool(multivenc) and multivenc.get("velocity_high_t") is not None
    if "velocity_noise" in sections or high_venc_noise:
        if noise_tissue_mask is None:
            if mag_t is None:
                raise ValueError("magnitude_t (or noise_tissue_mask) is needed for the velocity noise")
            noise_tissue_mask = select_noise_tissue(mag_t, v0, lumen_mask=lumen_mask)
        noise_tissue_mask = np.asarray(noise_tissue_mask, bool)
    if "velocity_noise" in sections:
        sigma_v = {stage: velocity_noise(v, noise_tissue_mask) for stage, v in stages}
        rep["velocity_noise_mm_s"] = {stage: sigma_v[stage].tolist() for stage, _ in stages}
        rep["velocity_noise_voxels"] = int(noise_tissue_mask.sum())
        rep["venc_to_noise_ratio"] = {stage: (venc / np.maximum(sigma_v[stage], 1e-9)).tolist() for stage, _ in stages}

    # background offset residual
    if "background_offset" in sections:
        if static_mask is None:
            if mag_t is None:
                raise ValueError("magnitude_t (or static_mask) is needed for the background offset")
            static_mask = static_tissue_candidates(mag_t, v0, lumen_mask=lumen_mask)
        static_mask = np.asarray(static_mask, bool)
        r0 = residual_static_velocity(v0, static_mask); r1 = residual_static_velocity(v1, static_mask)
        rep["background_offset"] = {
            "static_voxels": int(static_mask.sum()),
            "residual_mean_raw_mm_s": r0["mean"], "residual_mean_processed_mm_s": r1["mean"],
            "residual_rms_raw_mm_s": r0["rms"], "residual_rms_processed_mm_s": r1["rms"],
            "threshold_mm_s": RESIDUAL_STATIC_THRESHOLD_MM_S,
            "pass_raw": bool(np.max(np.abs(r0["mean"])) < RESIDUAL_STATIC_THRESHOLD_MM_S),
            "pass_processed": bool(np.max(np.abs(r1["mean"])) < RESIDUAL_STATIC_THRESHOLD_MM_S)}

    if lumen_mask is not None:
        lm = np.asarray(lumen_mask, bool)
        rep["lumen_voxels"] = int(lm.sum())
    if lumen_mask is not None and "venc" in sections:
        # aliasing is per ENCODED component, i.e. along the image axes (i, j, k)
        # for the usual in-plane / through-plane encoding, not per world
        # component: project onto the unit voxel axes before comparing to venc
        M = np.asarray(voxel_to_world, float)[:3, :3]
        axes = M / np.linalg.norm(M, axis=0, keepdims=True)              # columns: unit i, j, k in world
        peak, comp_peak, near_venc = {}, {}, {}
        for stage, v in stages:
            enc = v[:, lm] @ axes
            peak[stage] = _f(np.percentile(np.linalg.norm(v[:, lm], axis=-1), 99.9))
            comp_peak[stage] = np.abs(enc).max(axis=0).max(axis=0)
            near_venc[stage] = _f((np.abs(enc) >= NEAR_VENC_FRACTION * venc).any(-1).mean())
        rep["venc"] = {
            "peak_speed_mm_s": peak,
            "peak_component_mm_s": {stage: comp_peak[stage].tolist() for stage, _ in stages},
            "peak_component_over_venc": {stage: (comp_peak[stage] / venc).tolist() for stage, _ in stages},
            "lumen_voxel_frames_near_venc": near_venc,
            "venc_adequate": bool((comp_peak["processed"] < venc).all()),
            "note": "components are along the image axes (i, j, k), the encoding directions; a component "
                    "peak of the processed velocity above venc means that the raw velocity was aliased and "
                    "unwrapping was required (venc_adequate is false); the near-venc fraction of the raw "
                    "velocity is a weak indicator of aliasing (wrapped values fall anywhere in range), "
                    "the aliased fraction from wrap_count is the reliable one"}
        if wrap_count is not None:
            # what the processing changed: aliased in the raw velocity, unwrapped in the processed velocity
            k = np.asarray(wrap_count)
            changed = (k[:, lm] != 0).any(-1)
            rep["venc"]["unwrapped_voxel_frames"] = int(changed.sum())
            rep["venc"]["lumen_voxel_frames_aliased"] = _f(changed.mean())
            rep["venc"]["max_wraps"] = int(np.abs(k).max())
    if lumen_mask is not None and "divergence" in sections:
        d0 = [divergence_stats(v0[t], lm, voxel_to_world)["div_rms"] for t in range(v0.shape[0])]
        d1 = [divergence_stats(v1[t], lm, voxel_to_world)["div_rms"] for t in range(v1.shape[0])]
        rep["divergence_rms_1_s"] = {"raw_mean": _f(np.mean(d0)), "processed_mean": _f(np.mean(d1))}
    if lumen_mask is not None and "encoding" in sections:
        # encoding-direction self-check
        from .encoding import encoding_check
        rep["encoding"] = {}
        for stage, v in stages:
            e = encoding_check(v, lm, voxel_to_world)
            rep["encoding"][stage] = {
                "consistent": e["consistent"], "conclusive": e["conclusive"], "ratio_to_minimum": _f(e["ratio"]),
                "discrimination": _f(e["discrimination"]) if np.isfinite(e["discrimination"]) else None,
                "excluded_voxels": e["excluded_voxels"],
                "best_permutation": list(e["best"][0]), "best_signs": list(e["best"][1]),
                "note": e["note"]}

    if noise_mask is not None:
        rep["noise_mask_fraction"] = _f(np.asarray(noise_mask, bool).mean())

    if multivenc:
        mv = {k: val for k, val in multivenc.items() if k not in ("margin", "velocity_high_t")}
        if high_venc_noise:
            from .multivenc import noise_ratio
            mv["noise_ratio_to_high_venc"] = {stage: _f(noise_ratio(v, multivenc["velocity_high_t"], noise_tissue_mask))
                                              for stage, v in stages}
        rep["multivenc"] = mv

    if planes:
        flows = {}
        for name, p in planes.items():
            flows[name] = {}
            for label, v in stages:
                r = plane_flow(v, voxel_to_world, lumen_mask=lumen_mask, **p)
                flows[name][label] = {"mean_flow_mm3_s": _f(r["flow"].mean()), "area_mm2": _f(r["area"]),
                                      "peak_mean_velocity_mm_s": _f(np.abs(r["mean_velocity"]).max())}
        rep["planes"] = flows
        if checks:
            cons = {}
            for label, v in stages:
                fl = {n: plane_flow(v, voxel_to_world, lumen_mask=lumen_mask, **planes[n])["flow"]
                      for n in planes}
                cons[label] = mass_conservation(fl, checks, heart_rate_bpm, rr_ms)
            rep["mass_conservation"] = cons
    if extra:
        rep["extra"] = extra
    rep["processing_time_s"] = dict(timing or {})
    rep["processing_time_s"]["report"] = round(time.perf_counter() - t_report, 4)
    return rep


def to_json(report, path=None, indent=2):
    s = json.dumps(report, indent=indent, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if path:
        with open(path, "w") as f:
            f.write(s)
    return s


def _input_output_table(rows):
    """Markdown table of (name, {"raw": value, "processed": value}, format)
    rows, with the raw value in the input column and the processed value in
    the output column. A value is a number or a list of numbers."""
    def cell(x, fmt):
        return ", ".join(format(c, fmt) for c in x) if isinstance(x, (list, tuple)) else format(x, fmt)
    L = ["| | input | output |", "|---|---|---|"]
    for name, value, fmt in rows:
        L.append(f"| {name} | {cell(value['raw'], fmt)} | {cell(value['processed'], fmt)} |")
    return L


def to_markdown(rep):
    """One-page Markdown summary of a quality_report() dict. The raw velocity
    is called input and the processed velocity output; every value says which
    of the two it describes, unless it describes the acquisition, a mask or
    the processing."""
    L = [f"# 4D flow quality report{': ' + str(rep['study_id']) if rep.get('study_id') else ''}", ""]
    L.append(f"Frames: {rep['n_frames']}. venc (mm/s): {', '.join(f'{v:.0f}' for v in rep['venc_mm_s'])}.")
    L.append("")
    L.append("Input is the velocity before processing (raw), output is the velocity after processing.")
    if "velocity_noise_mm_s" in rep:
        L.append("")
        L.append("## Velocity noise in static tissue")
        L += _input_output_table([("Velocity noise (mm/s)", rep["velocity_noise_mm_s"], ".1f"),
                                  ("venc / noise", rep["venc_to_noise_ratio"], ".0f")])
    if "background_offset" in rep:
        b = rep["background_offset"]
        verdict = lambda ok: "PASS" if ok else "FAIL"
        L.append("")
        L.append(f"## Background offset  (input {verdict(b['pass_raw'])}, output {verdict(b['pass_processed'])})")
        L += _input_output_table([
            ("Mean residual static velocity (mm/s)",
             {"raw": b["residual_mean_raw_mm_s"], "processed": b["residual_mean_processed_mm_s"]}, "+.2f"),
            ("RMS residual static velocity (mm/s)",
             {"raw": b["residual_rms_raw_mm_s"], "processed": b["residual_rms_processed_mm_s"]}, ".2f")])
        L.append("")
        L.append(f"Threshold of the mean: {b['threshold_mm_s']:.1f} mm/s. Static voxels: {b['static_voxels']}.")
    if "venc" in rep:
        v = rep["venc"]
        L.append("")
        L.append(f"## venc  ({'adequate' if v['venc_adequate'] else 'exceeded: the input was aliased'})")
        percent = lambda x: {stage: 100 * value for stage, value in x.items()}
        L += _input_output_table([
            ("Peak speed in the lumen (mm/s)", v["peak_speed_mm_s"], ".0f"),
            ("Peak component / venc", v["peak_component_over_venc"], ".2f"),
            (f"Lumen voxel-frames with a component at or above {100 * NEAR_VENC_FRACTION:.0f}% of venc (%)",
             percent(v["lumen_voxel_frames_near_venc"]), ".2f")])
        if "unwrapped_voxel_frames" in v:
            L.append("")
            L.append(f"Aliased in the input and unwrapped in the output: {100 * v['lumen_voxel_frames_aliased']:.2f}% "
                     f"of the lumen voxel-frames ({v['unwrapped_voxel_frames']}). "
                     f"Largest number of wraps of a voxel in the image: {v['max_wraps']}.")
    if "divergence_rms_1_s" in rep:
        d = rep["divergence_rms_1_s"]
        L.append("")
        L.append(f"## Divergence (lumen RMS, 1/s)\nInput {d['raw_mean']:.3f}, output {d['processed_mean']:.3f}.")
    if "encoding" in rep:
        e = rep["encoding"]
        status = lambda r: "FAIL" if not r["consistent"] else ("PASS" if r["conclusive"] else "INCONCLUSIVE")
        L.append("")
        L.append(f"## Encoding directions  (input {status(e['raw'])}, output {status(e['processed'])})")
        L.append(f"- Input: {e['raw']['note']}.")
        L.append(f"- Output: {e['processed']['note']}.")
    if "multivenc" in rep:
        mv = rep["multivenc"]
        L.append("")
        L.append(f"## Multi-venc unwrapping ({mv.get('method')}, DRAFT)")
        line = (f"vencs {mv.get('vencs', [mv.get('venc_low'), mv.get('venc_high')])}, effective venc "
                f"{mv.get('effective_venc')}; {100 * mv.get('fraction_changed', 0):.1f}% of low-venc voxel-frames "
                f"unwrapped, {100 * mv.get('fraction_corrected', 0):.2f}% corrected")
        if "noise_ratio_to_high_venc" in mv:
            nr = mv["noise_ratio_to_high_venc"]
            line += (f"; static-tissue velocity noise relative to that of the high venc: "
                     f"input {nr['raw']:.2f}, output {nr['processed']:.2f}")
        L.append(line + ".")
    if rep.get("processing_time_s"):
        pt = rep["processing_time_s"]
        L.append("")
        L.append("Processing time (s): " + ", ".join(f"{k} {val:.2f}" for k, val in pt.items()) + ".")
    if "noise_mask_fraction" in rep:
        L.append(f"\nNoise mask covers {100 * rep['noise_mask_fraction']:.0f}% of the field of view.")
    if "mass_conservation" in rep:
        L.append("")
        L.append("## Mass conservation")
        L.append("| check | % error, input | % error, output | ratio, input | ratio, output |")
        L.append("|---|---|---|---|---|")
        for r0, r1 in zip(rep["mass_conservation"]["raw"], rep["mass_conservation"]["processed"]):
            L.append(f"| {r1['check']} | {r0['percent_error']:+.1f} | {r1['percent_error']:+.1f} "
                     f"| {r0['ratio']:.2f} | {r1['ratio']:.2f} |")
    if "planes" in rep:
        L.append("")
        L.append("## Planes")
        L.append("| plane | area mm2 | mean flow mL/s, input | mean flow mL/s, output "
                 "| peak mean velocity mm/s, input | peak mean velocity mm/s, output |")
        L.append("|---|---|---|---|---|---|")
        for n, p in rep["planes"].items():
            q0, q1 = p["raw"], p["processed"]
            L.append(f"| {n} | {q1['area_mm2']:.0f} | {q0['mean_flow_mm3_s'] / 1000:.1f} | {q1['mean_flow_mm3_s'] / 1000:.1f} "
                     f"| {q0['peak_mean_velocity_mm_s']:.0f} | {q1['peak_mean_velocity_mm_s']:.0f} |")
    return "\n".join(L) + "\n"
