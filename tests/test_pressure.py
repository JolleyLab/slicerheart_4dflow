"""
Relative pressure by the pressure Poisson equation against analytic
solutions: Poiseuille (viscous term only, dp/dz = -4 mu v_max / R^2) and
Womersley (inertial terms dominate, prescribed harmonic pressure gradient).

Poiseuille recovers the gradient within 4% (the wall-layer viscous term is
biased by the no-slip condition; wall_weight limits its effect). Womersley
recovers the pressure-gradient time course with correlation > 0.999 and
slope within 3%, which is the regime of aortic 4D flow where inertial terms
dominate (Bock et al. 2011).
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, WORLD_TRANSFORMS
from slicerheart_4dflow.phantom import poiseuille, womersley, add_noise
from slicerheart_4dflow.pressure import pressure_gradient, relative_pressure, pressure_drop, MMHG_PER_PA


def _axial_slope(P, z, inner):
    return np.polyfit(z[inner], P[inner], 1)[0]


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_gradient_and_pressure(geometry):
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
    assert abs(slope / p["pressure_gradient_pa_mm"][0] - 1) < 0.04
    assert abs(P[0][m].mean()) < 1e-9                                                   # zero-mean reference


def test_womersley_pressure_time_course():
    w = womersley(shape=(24, 48, 48))
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    P, _ = relative_pressure(v, m, vtw, rr_ms=800.0)
    inner = m & (w["r_mm"] < 9.0)
    est = np.array([_axial_slope(P[t], w["z_mm"], inner) for t in range(P.shape[0])])
    truth = w["pressure_gradient_pa_mm"]
    assert np.corrcoef(est, truth)[0, 1] > 0.999
    assert abs(np.polyfit(truth, est, 1)[0] - 1) < 0.03
    assert np.abs(est - truth).max() < 0.05 * np.abs(truth).max()
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
