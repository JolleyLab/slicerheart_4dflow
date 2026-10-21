"""
Self-test of the transfer audit (transfer_audit.py): auditing a synthetic case must find nothing. The case is a
matched synthetic case itself (bifurcation phantom with typical clinical properties); the audit measures it, builds
its own matched synthetic case with another noise realization and compares. Every property must be matched and no
paired quantity flagged, otherwise the audit would flag clinical cases for its own reasons.
"""

import numpy as np

from slicerheart_4dflow.background_phase import design_matrix, normalised_coords

from transfer_audit import MATCHED_PROPERTIES, PAIRED_QUANTITIES, matched_synthetic_case, to_markdown, transfer_audit

PROPERTIES = {
    "frames": 16, "rr_ms": 700.0, "voxel_size_mm": [1.6, 1.6, 1.8], "venc_mm_s": [1000.0, 1000.0, 1000.0],
    "noise_over_venc": 0.025, "peak_mean_velocity_mm_s": 850.0, "vessel_radius_mm": 6.0,
    "waveform": None, "background_offset_rms_mm_s": 10.0, "tissue_to_lumen_magnitude": 0.5, "air_to_lumen_magnitude": 0.03,
}


def synthetic_case(seed=1, **properties):
    """A synthetic case with typical clinical properties (PROPERTIES, overridden by properties), aortic waveform and a
    second-order background offset; also used by test_semisynthetic."""
    from slicerheart_4dflow.phantom import aortic_waveform
    properties = dict(PROPERTIES, waveform=aortic_waveform(PROPERTIES["frames"]).tolist(), **properties)
    shape = (20, 20, 20)
    coefficients = np.random.default_rng(3).normal(size=(3, 10))
    A = design_matrix(normalised_coords(shape), 2)
    offset = np.stack([(A @ c).reshape(shape) for c in coefficients], axis=-1)
    case = matched_synthetic_case(properties, offset=offset, seed=seed)
    case["manifest"]["case_id"] = "synthetic"
    return case


def test_audit_of_synthetic_case_finds_nothing(metrics):
    case = synthetic_case()
    audit = transfer_audit(case, seed=2)
    print(to_markdown(audit))
    assert set(audit["matching"]) == set(MATCHED_PROPERTIES)
    assert set(audit["rows"]) == set(PAIRED_QUANTITIES)
    for name, m in audit["matching"].items():
        metrics.record(f"transfer_audit/self/matching/{name}", m["relative_difference"], atol=0.02)
    for name, r in audit["rows"].items():
        if np.isfinite(r["ratio"]):        # not compared: inconclusive encoding check
            metrics.record(f"transfer_audit/self/ratio/{name}", abs(np.log(r["ratio"])), atol=0.05)
    assert all(m["matched"] for m in audit["matching"].values()), audit["matching"]
    assert not any(r["flagged"] for r in audit["rows"].values()), {n: r for n, r in audit["rows"].items() if r["flagged"]}
