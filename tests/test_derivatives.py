"""
Velocity-gradient quantities against the Poiseuille analytic solution
(vorticity |omega| = 2 v_max r / R^2, dissipation mu (2 v_max r / R^2)^2,
zero helicity, zero divergence) for every image geometry, plus
world-coordinate invariance and a noise-sensitivity check on the CFD field.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, WORLD_TRANSFORMS
from slicerheart_4dflow.derivatives import (velocity_gradient, vorticity, helicity_density,
                                            viscous_dissipation, kinetic_energy_density, lumen_integrals)
from slicerheart_4dflow.phantom import poiseuille, add_noise, BLOOD_VISCOSITY_PA_S, BLOOD_DENSITY_KG_M3


def _poiseuille(vtw, nt=1):
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    return poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, v_max_mm_s=800.0,
                      axis_world=n, nt=nt)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_vorticity_and_dissipation_exact(geometry):
    vtw = GEOMETRIES[geometry]()
    p = _poiseuille(vtw)
    v, m, r = p["velocity"][0], p["mask"], p["r_mm"]
    G = velocity_gradient(v, m, vtw)
    inner = m & (r < 9.0)                       # central differences everywhere
    w = np.linalg.norm(vorticity(v, m, vtw, G), axis=-1)
    w_true = 2 * 800.0 * r / 12.0 ** 2
    assert np.allclose(w[inner], w_true[inner], rtol=1e-9, atol=1e-6)
    phi = viscous_dissipation(v, m, vtw, G)
    phi_true = BLOOD_VISCOSITY_PA_S * (2 * 0.8 * r * 1e-3 / 0.012 ** 2) ** 2
    assert np.allclose(phi[inner], phi_true[inner], rtol=1e-9, atol=1e-9)
    assert np.abs(helicity_density(v, m, vtw, G)).max() < 1e-6
    assert np.abs(np.trace(G, axis1=-2, axis2=-1))[inner].max() < 1e-9
    assert np.all(G[~m] == 0)


def test_kinetic_energy_density():
    p = _poiseuille(GEOMETRIES["identity"]())
    ke = kinetic_energy_density(p["velocity"][0], p["mask"])
    assert np.isclose(ke.max(), 0.5 * BLOOD_DENSITY_KG_M3 * 0.8 ** 2, rtol=0.01)   # 339 J/m^3


@pytest.mark.parametrize("transform", list(WORLD_TRANSFORMS))
def test_world_frame_invariance(transform):
    T = WORLD_TRANSFORMS[transform]()
    vtw = GEOMETRIES["oblique"]()
    p = _poiseuille(vtw)
    v, m = p["velocity"][0], p["mask"]
    R = T[:3, :3]
    v2 = v @ R.T
    vtw2 = T @ vtw
    G1, G2 = velocity_gradient(v, m, vtw), velocity_gradient(v2, m, vtw2)
    assert np.allclose(G2, R @ G1 @ R.T, atol=1e-9)
    w1, w2 = vorticity(v, m, vtw, G1), vorticity(v2, m, vtw2, G2)
    # curl is a pseudovector: flips sign under reflection
    assert np.allclose(w2, np.sign(np.linalg.det(R)) * (w1 @ R.T), atol=1e-9)
    assert np.allclose(viscous_dissipation(v, m, vtw, G1), viscous_dissipation(v2, m, vtw2, G2), atol=1e-12)


def test_lumen_integrals_poiseuille():
    p = _poiseuille(GEOMETRIES["anisotropic"](), nt=3)
    out = lumen_integrals(p["velocity"], p["mask"], p["voxel_to_world"])
    assert out["kinetic_energy_mJ"].shape == (3,)
    # KE per unit length for Poiseuille: rho pi R^2 vmax^2 / 6 ; compare against the voxel sum of KE density
    assert out["kinetic_energy_mJ"][0] > 0 and out["energy_loss_mW"][0] > 0
    # waveform scaling: KE ~ w^2, dissipation ~ w^2
    w = p["waveform"]
    assert np.allclose(out["kinetic_energy_mJ"] / w ** 2, out["kinetic_energy_mJ"][0] / w[0] ** 2, rtol=1e-6)
    assert np.allclose(out["energy_loss_mW"] / w ** 2, out["energy_loss_mW"][0] / w[0] ** 2, rtol=1e-6)


@pytest.mark.cfd
def test_noise_amplification_on_cfd(cfd, metrics):
    """Gradients amplify noise: 5% venc noise on the field should raise the
    vorticity error substantially more than 5%, which is why derived maps
    need filtering. Documents the behaviour rather than a target."""
    from slicerheart_4dflow.divergence_free import divergence_free_filter
    v, m, vtw = cfd
    v = v * 1000.0                                   # m/s -> mm/s
    w_true = vorticity(v, m, vtw)
    v_noisy = add_noise(v, 0.05 * 1500.0, m, rng=np.random.default_rng(0))
    w_noisy = vorticity(v_noisy, m, vtw)
    v_dfw, _ = divergence_free_filter(v_noisy, m, vtw, method="dfw")
    w_dfw = vorticity(v_dfw, m, vtw)
    def rel(w):
        return np.linalg.norm((w - w_true)[m], axis=-1).mean() / np.linalg.norm(w_true[m], axis=-1).mean()
    metrics.record("cfd/vorticity/noisy/relative_error", rel(w_noisy), direction="exact")
    metrics.record("cfd/vorticity/dfw/relative_error", rel(w_dfw))
    assert rel(w_noisy) > 0.5          # noise dominates the raw vorticity
    assert rel(w_dfw) < 0.5 * rel(w_noisy)
