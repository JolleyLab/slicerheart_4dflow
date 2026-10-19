"""
Bifurcation phantom: parent and two daughters with Murray's-law radii and a
known flow split. Gives mass conservation and flow-ratio (Qp:Qs type) checks
a ground-truth answer, under any image geometry, on clean and on acquired
(noisy, partial-volume) data, and through the quality report.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES
from slicerheart_4dflow.phantom import bifurcation, acquire
from slicerheart_4dflow.qa import plane_flow, mass_conservation
from slicerheart_4dflow.divergence_free import divergence_free_sequence
from slicerheart_4dflow.report import quality_report, to_markdown


def _flows(b, v):
    return {n: plane_flow(v, b["voxel_to_world"], lumen_mask=b["mask"], **p)["flow"] for n, p in b["planes"].items()}


@pytest.mark.parametrize("geometry", ["identity", "anisotropic", "oblique"])
def test_flow_split_and_conservation(geometry):
    b = bifurcation(voxel_to_world=GEOMETRIES[geometry]())
    flows = _flows(b, b["velocity"])
    for n, q in flows.items():
        assert abs(q[0] / b["branches"][n]["flow_mm3_s"] - 1) < 0.03
    res = {r["check"]: r for r in mass_conservation(flows, b["checks"])}
    assert abs(res["parent vs daughters"]["percent_error"]) < 3.0
    assert abs(res["daughter1 : daughter2"]["ratio"] / b["expected_ratio_d1_d2"] - 1) < 0.03


def test_murray_radii_and_velocities():
    b = bifurcation(split=0.6)
    r = {n: x["radius_mm"] for n, x in b["branches"].items()}
    assert np.isclose(r["parent"] ** 3, r["daughter1"] ** 3 + r["daughter2"] ** 3)
    assert np.isclose(b["branches"]["daughter1"]["flow_mm3_s"] / b["flow_mm3_s"][0], 0.6)
    assert b["mask"].sum() > 20000 and b["junction"].sum() > 0


def test_split_preserved_through_acquisition_and_filtering():
    nt = 6
    b = bifurcation(nt=nt)
    v, m, vtw = b["velocity"], b["mask"], b["voxel_to_world"]
    rng = np.random.default_rng(0)
    acq = [acquire(v[t], 1500.0, mask=m, snr=20.0, rng=rng) for t in range(nt)]
    vm = np.stack([a["velocity"] for a in acq])
    res_noisy = {r["check"]: r for r in mass_conservation(_flows(b, vm), b["checks"])}
    assert abs(res_noisy["parent vs daughters"]["percent_error"]) < 5.0
    assert abs(res_noisy["daughter1 : daughter2"]["ratio"] / b["expected_ratio_d1_d2"] - 1) < 0.05
    vf, _ = divergence_free_sequence(vm, m, vtw, method="fdm")
    res_f = {r["check"]: r for r in mass_conservation(_flows(b, vf), b["checks"])}
    assert abs(res_f["parent vs daughters"]["percent_error"]) < 5.0
    assert abs(res_f["daughter1 : daughter2"]["ratio"] / b["expected_ratio_d1_d2"] - 1) < 0.05


def test_quality_report_on_bifurcation():
    nt = 6
    b = bifurcation(nt=nt)
    v, m, vtw = b["velocity"], b["mask"], b["voxel_to_world"]
    rng = np.random.default_rng(1)
    acq = [acquire(v[t], 1500.0, mask=m, snr=20.0, rng=rng) for t in range(nt)]
    vm = np.stack([a["velocity"] for a in acq]); mag = np.stack([a["magnitude"] for a in acq])
    static = ~m & (mag.mean(0) > 0.0)                 # air only: residual-offset check runs on it
    rep = quality_report(vm, vm, mag, venc=1500.0, voxel_to_world=vtw, lumen_mask=m, static_mask=static,
                         planes=b["planes"], checks=b["checks"], study_id="bifurcation")
    mc = {r["check"]: r for r in rep["mass_conservation"]["processed"]}
    assert abs(mc["parent vs daughters"]["percent_error"]) < 5.0
    assert abs(mc["daughter1 : daughter2"]["ratio"] / b["expected_ratio_d1_d2"] - 1) < 0.05
    assert rep["venc"]["venc_adequate"]
    md = to_markdown(rep)
    assert "daughter1 : daughter2" in md and "parent" in md
