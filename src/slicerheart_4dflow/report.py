"""
Per-study quality report for 4D flow pre-processing.

quality_report() gathers in one structure quality-assurance checks of the
kind addressed by the 2023 4D flow consensus statement, computed on the raw
and the processed data:

  venc adequacy      peak lumen speed against venc, fraction of lumen
                     voxel-frames aliased before unwrapping and the number
                     of wraps recovered (a venc below the peak velocity is
                     the most common acquisition problem).
  velocity SNR       noise estimate from the temporal velocity standard
                     deviation in static tissue (velocity noise is
                     sigma_v = sqrt(2) venc / (pi SNR_magnitude), so static
                     tissue velocity std is a direct noise measure).
  background offset  residual static-tissue velocity after correction,
                     against the 0.6 cm/s acceptance threshold for shunt and
                     regurgitant quantification.
  mass conservation  percent inflow/outflow error at user-defined plane
                     sets, and Qp:Qs when planes are labelled for it.
  noise mask         fraction of the field of view masked.
  divergence         RMS divergence in the lumen before and after filtering.

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
from .qa import plane_flow, mass_conservation

RESIDUAL_STATIC_THRESHOLD_MM_S = 6.0        # 0.6 cm/s, Gatehouse 2010


def _f(x):
    return float(x)


def quality_report(velocity_raw_t, velocity_processed_t, magnitude_t, venc, voxel_to_world,
                   lumen_mask=None, static_mask=None, noise_mask=None, planes=None, checks=None,
                   heart_rate_bpm=None, wrap_count=None, study_id=None, extra=None, timing=None,
                   noise_tissue_mask=None, rr_ms=None):
    """
    velocity_raw_t, velocity_processed_t : (nt, nk, nj, ni, 3) mm/s, before and
        after preprocessing (same frames).
    venc        : scalar or (3,) mm/s.
    lumen_mask  : (nk, nj, ni) bool; enables peak-speed, aliasing and divergence checks.
    static_mask : (nk, nj, ni) bool; default from static_tissue_candidates.
    noise_tissue_mask : (nk, nj, ni) bool, tissue for the velocity-noise
                  estimate; must not be selected by velocity (default: body
                  by magnitude minus vessels and the dilated lumen).
    rr_ms       : cycle length for per-beat volumes (alternative to heart_rate_bpm).
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
    v0 = np.asarray(velocity_raw_t, float); v1 = np.asarray(velocity_processed_t, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    rep = {"study_id": study_id, "n_frames": int(v0.shape[0]), "venc_mm_s": venc.tolist()}

    mag_t = np.asarray(magnitude_t, float)
    if static_mask is None:
        static_mask = static_tissue_candidates(mag_t, v0, lumen_mask=lumen_mask)
    static_mask = np.asarray(static_mask, bool)

    # Velocity noise: pooled temporal variance (ddof=1) over tissue selected
    # by MAGNITUDE only (body minus vessels and the dilated lumen). The
    # static-tissue candidates are selected for low temporal std, so their
    # std is biased low (7.3 vs a true 10 mm/s on the chest phantom), and
    # ddof=0 over few frames biases it further.
    if noise_tissue_mask is None:
        noise_tissue_mask = static_tissue_candidates(mag_t, v0, lumen_mask=lumen_mask,
                                                     velocity_std_percentile=100.0, edge_erosion_voxels=0)
    noise_tissue_mask = np.asarray(noise_tissue_mask, bool)
    nt = v1.shape[0]
    if nt > 1 and noise_tissue_mask.any():
        # median of the per-voxel sample variances, rescaled to be unbiased
        # for Gaussian noise (median of chi2_{nt-1} / (nt-1)), so a few
        # high-variance voxels (lung, moving tissue) do not dominate
        from scipy.stats import chi2
        var = v1.var(axis=0, ddof=1)[noise_tissue_mask]
        sigma_v = np.sqrt(np.median(var, axis=0) / (chi2.median(nt - 1) / (nt - 1)))
    else:
        sigma_v = np.full(3, np.nan)
    rep["velocity_noise_mm_s"] = sigma_v.tolist()
    rep["velocity_noise_voxels"] = int(noise_tissue_mask.sum())
    rep["venc_to_noise_ratio"] = (venc / np.maximum(sigma_v, 1e-9)).tolist()

    # background offset residual
    r0 = residual_static_velocity(v0, static_mask); r1 = residual_static_velocity(v1, static_mask)
    rep["background_offset"] = {
        "static_voxels": int(static_mask.sum()),
        "residual_mean_raw_mm_s": r0["mean"], "residual_mean_processed_mm_s": r1["mean"],
        "residual_rms_processed_mm_s": r1["rms"],
        "threshold_mm_s": RESIDUAL_STATIC_THRESHOLD_MM_S,
        "pass": bool(np.max(np.abs(r1["mean"])) < RESIDUAL_STATIC_THRESHOLD_MM_S)}

    if lumen_mask is not None:
        lm = np.asarray(lumen_mask, bool)
        speed1 = np.linalg.norm(v1[:, lm], axis=-1)
        peak = _f(np.percentile(speed1, 99.9))
        # aliasing is per ENCODED component, i.e. along the image axes (i, j, k)
        # for the usual in-plane / through-plane encoding, not per world
        # component: project onto the unit voxel axes before comparing to venc
        M = np.asarray(voxel_to_world, float)[:3, :3]
        axes = M / np.linalg.norm(M, axis=0, keepdims=True)              # columns: unit i, j, k in world
        enc1 = v1[:, lm] @ axes; enc0 = v0[:, lm] @ axes
        comp_peak = np.abs(enc1).max(axis=0).max(axis=0)
        saturated_raw = (np.abs(enc0) >= 0.98 * venc).any(-1).mean()
        rep["venc"] = {
            "peak_speed_mm_s": peak,
            "peak_component_mm_s": comp_peak.tolist(),
            "peak_component_over_venc": (comp_peak / venc).tolist(),
            "lumen_voxel_frames_near_venc_raw": _f(saturated_raw),
            "venc_adequate": bool((comp_peak < venc).all()),
            "note": "components are along the image axes (i, j, k), the encoding directions; a component "
                    "peak above venc means aliasing was present and unwrapping was required; "
                    "near-venc fraction is a weak indicator (wrapped values fall anywhere in range), "
                    "aliased fraction from wrap_count is the reliable one"}
        if wrap_count is not None:
            k = np.asarray(wrap_count)
            changed = (k[:, lm] != 0).any(-1)
            rep["venc"]["unwrapped_voxel_frames"] = int(changed.sum())
            rep["venc"]["lumen_voxel_frames_aliased"] = _f(changed.mean())
            rep["venc"]["max_wraps"] = int(np.abs(k).max())
        d0 = [divergence_stats(v0[t], lm, voxel_to_world)["div_rms"] for t in range(v0.shape[0])]
        d1 = [divergence_stats(v1[t], lm, voxel_to_world)["div_rms"] for t in range(v1.shape[0])]
        rep["divergence_rms_1_s"] = {"raw_mean": _f(np.mean(d0)), "processed_mean": _f(np.mean(d1))}
        rep["lumen_voxels"] = int(lm.sum())
        # encoding-direction self-check on the processed (unwrapped) field
        from .encoding import encoding_check
        e = encoding_check(v1, lm, voxel_to_world)
        rep["encoding"] = {
            "consistent": e["consistent"], "conclusive": e["conclusive"], "ratio_to_minimum": _f(e["ratio"]),
            "discrimination": _f(e["discrimination"]) if np.isfinite(e["discrimination"]) else None,
            "excluded_voxels": e["excluded_voxels"],
            "best_permutation": list(e["best"][0]), "best_signs": list(e["best"][1]),
            "note": e["note"]}

    if noise_mask is not None:
        rep["noise_mask_fraction"] = _f(np.asarray(noise_mask, bool).mean())

    if planes:
        flows = {}
        for name, p in planes.items():
            flows[name] = {}
            for label, v in (("raw", v0), ("processed", v1)):
                r = plane_flow(v, voxel_to_world, lumen_mask=lumen_mask, **p)
                flows[name][label] = {"mean_flow_mm3_s": _f(r["flow"].mean()), "area_mm2": _f(r["area"]),
                                      "peak_mean_velocity_mm_s": _f(np.abs(r["mean_velocity"]).max())}
        rep["planes"] = flows
        if checks:
            cons = {}
            for label, v in (("raw", v0), ("processed", v1)):
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


def to_markdown(rep):
    """One-page Markdown summary of a quality_report() dict."""
    L = [f"# 4D flow quality report{': ' + str(rep['study_id']) if rep.get('study_id') else ''}", ""]
    L.append(f"Frames: {rep['n_frames']}. venc (mm/s): {', '.join(f'{v:.0f}' for v in rep['venc_mm_s'])}.")
    sv = rep["velocity_noise_mm_s"]; vnr = rep["venc_to_noise_ratio"]
    L.append(f"Velocity noise in static tissue (mm/s): {', '.join(f'{x:.1f}' for x in sv)}; "
             f"venc/noise: {', '.join(f'{x:.0f}' for x in vnr)}.")
    b = rep["background_offset"]
    L.append("")
    L.append(f"## Background offset  ({'PASS' if b['pass'] else 'FAIL'})")
    L.append(f"Residual static velocity after correction (mm/s): "
             f"{', '.join(f'{x:+.2f}' for x in b['residual_mean_processed_mm_s'])} "
             f"(threshold {b['threshold_mm_s']:.1f} mm/s; before correction "
             f"{', '.join(f'{x:+.2f}' for x in b['residual_mean_raw_mm_s'])}). "
             f"Static voxels: {b['static_voxels']}.")
    if "venc" in rep:
        v = rep["venc"]
        L.append("")
        L.append(f"## venc  ({'adequate' if v['venc_adequate'] else 'exceeded, unwrapping required'})")
        L.append(f"Peak speed {v['peak_speed_mm_s']:.0f} mm/s; peak component / venc: "
                 f"{', '.join(f'{x:.2f}' for x in v['peak_component_over_venc'])}; "
                 f"lumen voxel-frames near venc in raw data: {100 * v['lumen_voxel_frames_near_venc_raw']:.2f}%.")
        if "unwrapped_voxel_frames" in v:
            L.append(f"Aliased lumen voxel-frames (unwrapped): {100 * v['lumen_voxel_frames_aliased']:.1f}% "
                     f"({v['unwrapped_voxel_frames']}), max wraps {v['max_wraps']}.")
    if "divergence_rms_1_s" in rep:
        d = rep["divergence_rms_1_s"]
        L.append("")
        L.append(f"## Divergence (lumen RMS, 1/s)\nraw {d['raw_mean']:.3f}, processed {d['processed_mean']:.3f}.")
    if "encoding" in rep:
        e = rep["encoding"]
        L.append("")
        status = "FAIL" if not e["consistent"] else ("PASS" if e["conclusive"] else "INCONCLUSIVE")
        L.append(f"## Encoding directions  ({status})")
        L.append(e["note"].capitalize() + ".")
    if rep.get("processing_time_s"):
        pt = rep["processing_time_s"]
        L.append("")
        L.append("Processing time (s): " + ", ".join(f"{k} {val:.2f}" for k, val in pt.items()) + ".")
    if "noise_mask_fraction" in rep:
        L.append(f"\nNoise mask covers {100 * rep['noise_mask_fraction']:.0f}% of the field of view.")
    if "mass_conservation" in rep:
        L.append("")
        L.append("## Mass conservation")
        L.append("| check | raw % error | processed % error | ratio (processed) |")
        L.append("|---|---|---|---|")
        for r0, r1 in zip(rep["mass_conservation"]["raw"], rep["mass_conservation"]["processed"]):
            L.append(f"| {r1['check']} | {r0['percent_error']:+.1f} | {r1['percent_error']:+.1f} | {r1['ratio']:.2f} |")
    if "planes" in rep:
        L.append("")
        L.append("## Planes (processed)")
        L.append("| plane | mean flow mL/s | area mm2 | peak mean velocity mm/s |")
        L.append("|---|---|---|---|")
        for n, p in rep["planes"].items():
            q = p["processed"]
            L.append(f"| {n} | {q['mean_flow_mm3_s'] / 1000:.1f} | {q['area_mm2']:.0f} | {q['peak_mean_velocity_mm_s']:.0f} |")
    return "\n".join(L) + "\n"
