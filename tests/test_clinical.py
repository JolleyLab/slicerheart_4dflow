"""
Clinical tier: real scanner data against a commercial reference. Skipped
until a vendor case exists (see tests/clinical/README.md).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

VENDORS = ("siemens", "philips", "ge")
MANIFEST_FIELDS = ("case_id", "vendor", "venc_mm_s", "rr_ms", "n_phases", "voxel_size_mm",
                   "aliasing_present", "reference_software", "license", "doi", "files")


def clinical_root(vendor):
    return Path(os.environ.get("SLICERHEART_4DFLOW_CLINICAL_DATA", Path(__file__).parent / "data" / "clinical")) / vendor


def find_cases(vendor):
    root = clinical_root(vendor)
    return sorted(p for p in root.glob("*/manifest.json")) if root.exists() else []


def load_case(manifest_path):
    man = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    d = np.load(base / man["files"]["velocity"])
    planes = json.loads((base / man["files"]["planes"]).read_text())
    ref = json.loads((base / man["files"]["reference_flows"]).read_text())
    try:
        import nrrd
        lumen, _ = nrrd.read(str(base / man["files"]["lumen"]))
        lumen = np.asarray(lumen).T > 0                      # nrrd is Fortran order: (i, j, k) -> (k, j, i)
    except ImportError:
        lumen = None
    return man, d, planes, ref, lumen


CASES = [(v, m) for v in VENDORS for m in find_cases(v)]


@pytest.mark.clinical
@pytest.mark.parametrize("vendor", VENDORS)
def test_vendor_case_present(vendor):
    if not find_cases(vendor):
        pytest.skip(f"no {vendor} clinical case; see tests/clinical/{vendor}/README.md")


@pytest.mark.clinical
@pytest.mark.parametrize("vendor,manifest", CASES, ids=[m.parent.name for _, m in CASES])
def test_manifest_and_arrays(vendor, manifest):
    man, d, planes, ref, lumen = load_case(manifest)
    for f in MANIFEST_FIELDS:
        assert f in man, f
    assert man["vendor"] == vendor and man["license"].startswith("CC BY")
    v, mag, vtw = d["velocity_t"], d["magnitude_t"], d["voxel_to_world"]
    assert v.ndim == 5 and v.shape[-1] == 3 and mag.shape == v.shape[:4] and vtw.shape == (4, 4)
    assert v.shape[0] == man["n_phases"]
    if lumen is not None:
        assert lumen.shape == v.shape[1:4]
    assert set(planes) >= set(ref["planes"])


@pytest.mark.clinical
@pytest.mark.parametrize("vendor,manifest", CASES, ids=[m.parent.name for _, m in CASES])
def test_pipeline_against_reference(vendor, manifest, metrics):
    from slicerheart_4dflow import correct_background_phase, unwrap_velocity, divergence_free_sequence
    from slicerheart_4dflow.qa import plane_flow, net_volume, mass_conservation
    from slicerheart_4dflow.report import quality_report
    man, d, planes, ref, lumen = load_case(manifest)
    v, mag, vtw = d["velocity_t"].astype(float), d["magnitude_t"].astype(float), d["voxel_to_world"]
    venc = np.asarray(man["venc_mm_s"], float)
    corrected, _ = correct_background_phase(v, mag, method="war+msac", lumen_mask=lumen)
    unwrapped = unwrap_velocity(corrected, venc, vtw, method="laplacian_then_temporal", mask=lumen)
    rep = quality_report(v, unwrapped, mag, venc, vtw, lumen_mask=lumen, study_id=man["case_id"])
    assert rep["background_offset"]["pass"]
    assert rep["venc"]["venc_adequate"] == (not man["aliasing_present"])
    hr = 60000.0 / man["rr_ms"]
    flows = {name: plane_flow(unwrapped, vtw, lumen_mask=lumen, **p)["flow"] for name, p in planes.items()}
    for name, r in ref["planes"].items():
        net_ml = net_volume(flows[name], hr) / 1000.0
        metrics.record(f"clinical/{man['case_id']}/{name}/net_volume_mL", net_ml, direction="exact", atol=0.5)
        assert abs(net_ml - r["net_volume_mL"]) < 7.0           # manuscript limits of agreement vs cvi42
    for c in mass_conservation(flows, [tuple(x) for x in ref.get("conservation_checks", [])], hr):
        metrics.record(f"clinical/{man['case_id']}/{c['check']}/abs_pct_error", abs(c["percent_error"]), atol=0.5)
        assert abs(c["percent_error"]) < 5.0                     # inflow = outflow
    for label, num, den, expected in ref.get("ratio_checks", []):
        ratio = net_volume(flows[num], hr) / net_volume(flows[den], hr)
        metrics.record(f"clinical/{man['case_id']}/{label}/ratio", ratio, direction="exact", atol=0.02)
        assert abs(ratio / expected - 1) < 0.10                  # Qp:Qs type ratio vs reference
