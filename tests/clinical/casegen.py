"""
DRAFT: written before any clinical case was available. The case layout,
units and component frame follow the Slicer 5D convention from the
Discourse post; they must be aligned with what the SlicerHeart Import4DFlow
and Analyze4DFlow modules actually write before this is relied on.

Mock clinical cases: a phantom written to disk in exactly the layout a
real case has (Slicer 5D sequences, segmentation, planes, manifest,
reference flows), so that the clinical tier's loader, checks and audit are
exercised in CI before any scanner data exist, and so that a real case can
be compared with a synthetic one of the same layout.

The mock is SYNTHETIC data: its metrics go to the synthetic baseline and
its tests carry the `synthetic` marker, not `clinical`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slicerheart_4dflow.background_phase import design_matrix, normalised_coords
from slicerheart_4dflow.geometry import voxel_to_world_matrix
from slicerheart_4dflow.io import write_flow_sequence
from slicerheart_4dflow.phantom import acquire, bifurcation

MANIFEST_VERSION = 2


def make_mock_bifurcation_case(root, case_id="mock_bifurcation_001", spacing_mm=(1.5, 1.5, 1.5), nt=12,
                               venc_mm_s=600.0, snr=15.0, rr_ms=800.0, offset_cm_s=1.0, seed=0):
    """
    Bifurcation phantom (Murray radii, 60/40 split, parent peak 800 mm/s)
    acquired at a clinical venc below the parent peak (aliasing present),
    with static tissue around the vessel, a second-order background offset
    and complex noise. Written to root/<case_id>/ in the case layout.
    Returns the manifest path.
    """
    rng = np.random.default_rng(seed)
    base = Path(root) / case_id
    (base / "derived").mkdir(parents=True, exist_ok=True)
    (base / "reference").mkdir(parents=True, exist_ok=True)

    shape = (56, 28, 48)
    vtw = voxel_to_world_matrix(spacing=spacing_mm, origin=(-40.0, -25.0, -45.0))
    b = bifurcation(shape=shape, voxel_to_world=vtw, nt=nt)
    mask = b["mask"]
    v_true = b["velocity"]

    # static tissue: everything within 12 mm of the lumen, air beyond
    from scipy import ndimage
    tissue = ndimage.binary_dilation(mask, iterations=8) & ~mask
    magnitude = np.where(mask, 1.0, np.where(tissue, 0.7, 0.03))

    # background offset: second-order polynomial, |offset| up to ~offset_cm_s
    coords = normalised_coords(shape)
    A = design_matrix(coords, 2)
    coef = rng.normal(0, 1.0, (3, A.shape[1])) * offset_cm_s * 10.0 / 3.0       # mm/s
    offset = np.stack([(A @ c).reshape(shape) for c in coef], -1)

    frames = []
    mags = []
    for t in range(nt):
        acq = acquire(v_true[t] + offset, venc_mm_s, magnitude=magnitude, snr=snr, rng=rng)
        frames.append(acq["velocity"]); mags.append(acq["magnitude"])
    velocity_t = np.stack(frames).astype(np.float32)
    magnitude_t = np.stack(mags).astype(np.float32)
    times_ms = np.arange(nt) * rr_ms / nt

    write_flow_sequence(base / "derived" / "velocity.seq.nrrd", velocity_t, vtw, times_ms,
                        magnitude_t=magnitude_t, magnitude_path=base / "derived" / "magnitude.seq.nrrd")
    import nrrd
    seg = np.ascontiguousarray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
    nrrd.write(str(base / "derived" / "lumen.seg.nrrd"), seg, {
        "space": "left-posterior-superior",
        "space directions": np.array([vtw[:3, a].tolist() for a in range(3)]),
        "space origin": vtw[:3, 3].tolist(), "kinds": ["domain", "domain", "domain"], "encoding": "gzip"})

    planes = {name: {"centre": list(p["centre"]), "normal": list(p["normal"]), "radius": p["radius"]}
              for name, p in b["planes"].items()}
    (base / "derived" / "planes.json").write_text(json.dumps({"coordinate_system": "LPS", "planes": planes}, indent=2))

    # reference flows: the phantom's own (net volume per beat = mean flow x RR)
    ref_planes = {}
    for name, br in b["branches"].items():
        q = br["flow_mm3_s"] * b["waveform"]                       # mm3/s per frame
        ref_planes[name] = {"net_volume_mL": float(q.mean() * rr_ms / 1e6),
                            "forward_volume_mL": float(np.clip(q, 0, None).mean() * rr_ms / 1e6),
                            "peak_velocity_mm_s": float(br["v_max_mm_s"] * b["waveform"].max())}
    flows = {"planes": ref_planes,
             "conservation_checks": [list(c) for c in b["checks"][:1]],
             "ratio_checks": [["daughter1 : daughter2", "daughter1", "daughter2", b["expected_ratio_d1_d2"]]],
             "source": "phantom ground truth (bifurcation, Murray radii, split 0.6)"}
    (base / "reference" / "flows.json").write_text(json.dumps(flows, indent=2))

    # the truth, for the semi-synthetic and audit machinery to score against
    np.savez_compressed(base / "reference" / "truth.npz", velocity_true=v_true.astype(np.float32),
                        offset=offset.astype(np.float32), tissue=tissue)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "case_id": case_id,
        "vendor": "mock",
        "scanner": "none (phantom.bifurcation + phantom.acquire)",
        "software_version": "slicerheart_4dflow test suite",
        "sequence": "simulated four-point phase contrast, retrospective gating",
        "gating": "retrospective",
        "voxel_size_mm": list(spacing_mm),
        "n_phases": nt,
        "rr_ms": rr_ms,
        "venc_mm_s": [venc_mm_s] * 3,
        "velocity_units": "mm/s",
        "velocity_component_frame": "LPS",
        "aliasing_present": bool(np.abs(v_true).max() > venc_mm_s),
        "on_scanner_corrections": {"maxwell": True, "eddy_current": False, "gradient_nonlinearity": True},
        "contrast": "none",
        "age_group": "n/a",
        "diagnosis_category": "phantom",
        "deidentification": {"tool": "n/a", "date": "n/a", "private_tags": "n/a", "dates": "n/a"},
        "reference_software": "phantom ground truth",
        "license": "CC BY 4.0",
        "doi": "n/a",
        "files": {"velocity": "derived/velocity.seq.nrrd", "magnitude": "derived/magnitude.seq.nrrd",
                  "lumen": "derived/lumen.seg.nrrd", "planes": "derived/planes.json",
                  "reference_flows": "reference/flows.json", "truth": "reference/truth.npz"},
        "acquisition_model": {"snr": snr, "offset_cm_s": offset_cm_s, "seed": seed},
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return base / "manifest.json"
