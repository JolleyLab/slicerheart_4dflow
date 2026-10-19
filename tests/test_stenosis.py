"""
Stenosis phantom: a cosine-constricted tube with a jet above a clinical venc
throughout the cardiac cycle. Exercises the four things a straight tube
cannot: persistent aliasing, a convection-dominated pressure drop, a large
wall-shear-stress contrast, and flow conservation through a changing section.

Pressure note. The phantom velocity is the lubrication approximation (a
parabolic profile scaled to the local radius), not a Navier-Stokes solution.
Its convective acceleration varies across the section, averaging one third of
the centreline value, so the volumetric least-squares pressure drop is about
one third of the centreline Bernoulli drop. The test checks the solver
against the field's own section-averaged momentum balance (what the solver
is defined to recover) and records the Bernoulli ratio; the simplified
Bernoulli gradient from peak velocity (the clinical estimate, Hatle 1978) is
checked against the centreline value separately.
"""

import numpy as np
import pytest

from slicerheart_4dflow.phantom import stenosis, acquire
from slicerheart_4dflow.divergence_free import divergence_stats
from slicerheart_4dflow.derivatives import velocity_gradient
from slicerheart_4dflow.qa import plane_flow
from slicerheart_4dflow.unwrap import unwrap_velocity
from slicerheart_4dflow.pressure import relative_pressure, pressure_gradient, MMHG_PER_PA
from slicerheart_4dflow.wss import wall_shear_stress


@pytest.fixture(scope="module")
def sten():
    return stenosis()


def test_geometry_and_conservation(sten):
    v, m, vtw = sten["velocity"], sten["mask"], sten["voxel_to_world"]
    thr = sten["throat"]
    assert abs(thr["radius_mm"] / 5.0 - 1) < 0.02
    assert np.isclose(thr["peak_velocity_mm_s"][0], 2 * thr["mean_velocity_mm_s"][0])
    assert abs(np.linalg.norm(v[0], axis=-1).max() / thr["peak_velocity_mm_s"][0] - 1) < 0.05
    # discrete divergence small relative to the shear rate
    G = velocity_gradient(v[0], m, vtw)
    shear = np.abs(G).max()
    assert divergence_stats(v[0], m, vtw)["div_rms"] < 0.02 * shear
    # flow conserved through the constriction (plane sampling at 1 mm voxels)
    q = [plane_flow(v, vtw, centre=(19.5, 19.5, float(k)), normal=(0, 0, 1), radius=12.0,
                    lumen_mask=m)["flow"][0] for k in (5, 20, 31, 42, 58)]
    assert np.all(np.abs(np.array(q) / sten["flow_mm3_s"][0] - 1) < 0.03)


def test_persistent_aliasing_needs_spatial_unwrapping():
    nt = 12
    wave = 0.55 + 0.45 * np.sin(np.linspace(0, np.pi, nt)) ** 2       # jet 1.75 to 3.2 m/s
    s = stenosis(nt=nt, waveform=wave)
    v, m, vtw = s["velocity"], s["mask"], s["voxel_to_world"]
    venc = 1000.0
    assert (s["throat"]["peak_velocity_mm_s"] > venc).all()           # aliased in every frame
    rng = np.random.default_rng(0)
    acq = [acquire(v[t], venc, mask=m, snr=30.0, rng=rng) for t in range(nt)]
    vm = np.stack([a["velocity"] for a in acq]); k = np.stack([a["wrap_count"] for a in acq])
    assert np.abs(k).max() == 2
    resid = {}
    for method in ("temporal", "laplacian3d", "laplacian_then_temporal"):
        u = unwrap_velocity(vm, venc, vtw, method=method, mask=m)
        resid[method] = (np.abs(u - v)[:, m] > venc).any(-1).mean()
    assert resid["temporal"] > 0.01                                     # cannot see a wrap that never jumps
    assert resid["laplacian3d"] < 0.003
    assert resid["laplacian_then_temporal"] < 0.0005
    assert resid["laplacian_then_temporal"] < 0.1 * resid["temporal"]


def test_pressure_drop(sten):
    v, m, vtw = sten["velocity"], sten["mask"], sten["voxel_to_world"]
    z, r = sten["z_mm"], sten["r_mm"]
    # field-consistent reference: section-averaged axial NS gradient integrated along z
    g = pressure_gradient(v, m, vtw, 800.0)[0][..., 2]
    ks = np.arange(v.shape[1])
    zk = np.array([z[k][m[k]].mean() for k in ks]); gk = np.array([g[k][m[k]].mean() for k in ks])
    p_sec = np.concatenate([[0.0], np.cumsum(0.5 * (gk[1:] + gk[:-1]) * np.diff(zk))])
    it = int(np.argmin(np.abs(zk - sten["throat"]["z_mm"])))
    drop_field = p_sec[0] - p_sec[it]
    bern = sten["bernoulli_drop_pa"][0]
    assert abs(drop_field / (bern / 3) - 1) < 0.05                      # lubrication profile: 1/3 of centreline
    P, info = relative_pressure(v, m, vtw, 800.0)
    assert info["cg"][0]["cg_status"] == 0
    centre = m & (r < 1.0)
    zc, pc = z[centre], P[0][centre]
    drop_solver = pc[np.argsort(zc)][:4].mean() - pc[np.abs(zc - sten["throat"]["z_mm"]) < 1.0].mean()
    assert abs(drop_solver / drop_field - 1) < 0.10
    # clinical estimate: simplified Bernoulli from peak velocity vs centreline analytic drop
    sb_mmhg = sten["simplified_bernoulli_mmhg"][0]
    assert abs(sb_mmhg / (sten["pressure_drop_pa"][0] * MMHG_PER_PA) - 1) < 0.10
    assert 30 < sb_mmhg < 50                                            # coarctation-range gradient


def test_wss_contrast(sten):
    out = wall_shear_stress(sten["velocity"], sten["mask"], sten["voxel_to_world"])
    zp = (out["points"] - out["points"].mean(0)) @ sten["axis_world"]
    inlet = np.abs(zp + 24.0) < 3.0
    throat = np.abs(zp - sten["throat"]["z_mm"]) < 1.5
    w_in = np.median(out["wss_magnitude"][0][inlet]); w_th = np.median(out["wss_magnitude"][0][throat])
    i_th = int(np.argmin(sten["radius_of_z"][:, 1]))
    assert abs(w_in / sten["wss_pa"][0][0] - 1) < 0.15
    assert abs(w_th / sten["wss_pa"][0][i_th] - 1) < 0.15
    assert abs((w_th / w_in) / (10.0 / 5.0) ** 3 - 1) < 0.15              # WSS ~ Q / R^3
