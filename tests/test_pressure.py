"""
Relative pressure by the pressure Poisson equation against analytic
solutions: Poiseuille (viscous term only, dp/dz = -4 mu v_max / R^2) and
Womersley (inertial terms dominate, prescribed harmonic pressure gradient).

Poiseuille recovers the gradient within 8% with uniform weights (the
wall-layer viscous term is biased by the no-slip condition; wall_weight=0.1
brings it to 3% but is not the default, see pressure.py). Womersley recovers
the pressure-gradient time course with correlation > 0.999 and slope within
8%, which is the regime of aortic 4D flow where inertial terms dominate
(Bock et al. 2011).
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, WORLD_TRANSFORMS
from slicerheart_4dflow.phantom import poiseuille, womersley, add_noise
from slicerheart_4dflow.pressure import pressure_gradient, relative_pressure, pressure_drop, MMHG_PER_PA


def _axial_slope(P, z, inner):
    return np.polyfit(z[inner], P[inner], 1)[0]


def test_wall_weight_option_improves_viscous_flow():
    p = poiseuille(shape=(32, 48, 48))
    inner = p["mask"] & (p["r_mm"] < 9.0)
    errs = {}
    for ww in (1.0, 0.1):
        P, _ = relative_pressure(p["velocity"], p["mask"], p["voxel_to_world"], 800.0, wall_weight=ww)
        errs[ww] = abs(_axial_slope(P[0], p["z_mm"], inner) / p["pressure_gradient_pa_mm"][0] - 1)
    assert errs[0.1] < errs[1.0] < 0.08



@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_gradient_and_pressure(geometry, metrics):
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, v_max_mm_s=800.0, axis_world=n)
    v, m = p["velocity"], p["mask"]
    inner = m & (p["r_mm"] < 9.0)
    grad = pressure_gradient(v, m, vtw, rr_ms=800.0)
    gz = (grad[0] @ n)[inner]
    assert np.allclose(np.median(gz), p["pressure_gradient_pa_mm"][0], rtol=1e-6)      # exact in the interior
    assert np.abs(pressure_gradient(v, m, vtw, 800.0, include_viscous=False)).max() < 1e-9  # no convection
    P, info = relative_pressure(v, m, vtw, 800.0, grad=grad)
    assert info["cg"][0]["cg_status"] == 0
    slope = _axial_slope(P[0], p["z_mm"], inner)
    metrics.record(f"pressure/poiseuille/{geometry}/abs_relative_error", abs(slope / p["pressure_gradient_pa_mm"][0] - 1), atol=0.005)
    assert abs(slope / p["pressure_gradient_pa_mm"][0] - 1) < 0.08
    assert abs(P[0][m].mean()) < 1e-9                                                   # zero-mean reference


def test_womersley_pressure_time_course(metrics):
    w = womersley(shape=(24, 48, 48))
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    P, _ = relative_pressure(v, m, vtw, rr_ms=800.0)
    inner = m & (w["r_mm"] < 9.0)
    est = np.array([_axial_slope(P[t], w["z_mm"], inner) for t in range(P.shape[0])])
    truth = w["pressure_gradient_pa_mm"]
    metrics.record("pressure/womersley/slope", np.polyfit(truth, est, 1)[0], direction="exact", atol=0.01)
    metrics.record("pressure/womersley/max_relative_error", np.abs(est - truth).max() / np.abs(truth).max(), atol=0.005)
    assert np.corrcoef(est, truth)[0, 1] > 0.999
    assert abs(np.polyfit(truth, est, 1)[0] - 1) < 0.08
    assert np.abs(est - truth).max() < 0.10 * np.abs(truth).max()
    # pressure drop over 20 mm along the axis at peak gradient, in mmHg
    za, zb = w["z_mm"] < -8, w["z_mm"] > 8
    dp = pressure_drop(P, m & za & inner, m & zb & inner) * MMHG_PER_PA
    assert np.abs(dp).max() > 0.2                        # a few tenths of mmHg over 2 cm, as expected


@pytest.mark.parametrize("transform", list(WORLD_TRANSFORMS))
def test_world_frame_invariance(transform):
    T = WORLD_TRANSFORMS[transform]()
    vtw = GEOMETRIES["oblique"]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    w = womersley(shape=(16, 40, 40), voxel_to_world=vtw, axis_world=n, nt=6)
    R = T[:3, :3]
    P1, _ = relative_pressure(w["velocity"], w["mask"], vtw, 800.0)
    P2, _ = relative_pressure(w["velocity"] @ R.T, w["mask"], T @ vtw, 800.0)
    assert np.allclose(P1, P2, atol=1e-6 * np.abs(P1).max())


def test_noise_sensitivity_documented():
    """5% venc noise degrades the Womersley gradient estimate; dfw filtering
    recovers most of it. Documents why pressure needs filtered input."""
    from slicerheart_4dflow.divergence_free import divergence_free_sequence
    w = womersley(shape=(16, 40, 40), nt=12)
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    inner = m & (w["r_mm"] < 7.0)
    truth = w["pressure_gradient_pa_mm"]
    def est(vv):
        P, _ = relative_pressure(vv, m, vtw, 800.0)
        return np.array([_axial_slope(P[t], w["z_mm"], inner) for t in range(P.shape[0])])
    vn = add_noise(v, 0.05 * 1500.0, m, rng=np.random.default_rng(0))
    e_noisy = np.abs(est(vn) - truth).max() / np.abs(truth).max()
    vf, _ = divergence_free_sequence(vn, m, vtw, method="dfw")
    e_dfw = np.abs(est(vf) - truth).max() / np.abs(truth).max()
    assert e_dfw < e_noisy


def test_temporal_term_frame_duration_and_periodic():
    """Frames covering only part of the RR interval: with frame_duration_ms
    and periodic=False the temporal term uses the acquired frame duration
    and one-sided differences at the ends, instead of wrapping a 90% cycle
    onto itself."""
    from slicerheart_4dflow.pressure import pressure_gradient, BLOOD_DENSITY_KG_M3
    shape = (6, 10, 10)
    mask = np.zeros(shape, bool); mask[1:-1, 2:-2, 2:-2] = True
    nt, rr, frame = 8, 1000.0, 100.0                        # 8 x 100 ms = 80% of RR
    t = np.arange(nt) * frame
    vz = 500.0 + 300.0 * np.sin(2 * np.pi * t / rr)         # mm/s, uniform in space
    v = np.zeros((nt,) + shape + (3,)); v[..., 2] = vz[:, None, None, None]
    v *= mask[None, ..., None]
    g = pressure_gradient(v, mask, np.eye(4), rr, frame_duration_ms=frame, periodic=False,
                          include_viscous=False, include_convective=False)
    centre = (3, 5, 5)
    dvdt_true = 300.0 * 2 * np.pi / rr * np.cos(2 * np.pi * t / rr) * 1e-3 * 1e3   # (m/s)/s
    expected = -BLOOD_DENSITY_KG_M3 * dvdt_true * 1e-3                               # Pa/mm
    got = g[:, centre[0], centre[1], centre[2], 2]
    # interior frames: central difference of a sinusoid, accurate to ~sinc
    assert np.allclose(got[1:-1], expected[1:-1], rtol=0.07)
    # end frames: one-sided, same sign and order of magnitude
    assert np.sign(got[0]) == np.sign(expected[0]) and np.sign(got[-1]) == np.sign(expected[-1])
    # the default (periodic over rr/nt) would be wrong here
    g_wrong = pressure_gradient(v, mask, np.eye(4), rr, include_viscous=False, include_convective=False)
    assert not np.allclose(g_wrong[:, centre[0], centre[1], centre[2], 2][1:-1], expected[1:-1], rtol=0.07)
