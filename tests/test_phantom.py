"""
Analytic phantoms: flow, pressure gradient and wall shear stress are
consistent with the closed-form solutions, under every image geometry.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES
from slicerheart_4dflow.phantom import poiseuille, womersley, chest, alias, add_noise, BLOOD_VISCOSITY_PA_S
from slicerheart_4dflow.qa import plane_flow
from slicerheart_4dflow.geometry import voxel_to_world_points


def _centre(shape, vtw):
    c = (np.array(shape)[::-1] - 1) / 2.0            # (i, j, k)
    return voxel_to_world_points(vtw, c[None])[0]


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_flow_matches_analytic(geometry):
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])      # cylinder along the k axis
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=10.0, v_max_mm_s=800.0, axis_world=n)
    q = plane_flow(p["velocity"], vtw, centre=_centre((32, 48, 48), vtw), normal=n, radius=14.0,
                   lumen_mask=p["mask"])["flow"][0]
    assert abs(q / (np.pi * 10.0 ** 2 * 400.0) - 1) < 0.03


def test_poiseuille_wss_and_gradient():
    p = poiseuille(radius_mm=12.0, v_max_mm_s=800.0)
    assert np.isclose(p["wss_pa"][0], 2 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012)
    assert np.isclose(p["pressure_gradient_pa_mm"][0], -4 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012 ** 2 * 1e-3)


def test_womersley_steady_limit_is_poiseuille():
    A = 4 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012 ** 2
    w = womersley(harmonics=((0, A),), nt=2)
    p = poiseuille(v_max_mm_s=800.0)
    assert np.allclose(w["velocity"][0], p["velocity"][0], atol=1e-6)
    assert np.isclose(w["wss_pa"][0], p["wss_pa"][0])


def test_womersley_flow_matches_analytic():
    w = womersley()
    shape = w["velocity"].shape[1:4]
    q = plane_flow(w["velocity"], w["voxel_to_world"], centre=_centre(shape, w["voxel_to_world"]),
                   normal=w["axis_world"], radius=14.0, lumen_mask=w["mask"])["flow"]
    # 1 mm voxels discretise the thin Womersley boundary layer; 2% tolerance
    assert np.max(np.abs(q - w["flow_mm3_s"])) < 0.02 * np.abs(w["flow_mm3_s"]).max()
    assert w["velocity"].max() > 500 and w["velocity"].min() < 0      # pulsatile with reverse flow


def test_chest_phantom_structure():
    c = chest(nt=6)
    assert c["velocity"].shape == (6, 40, 96, 96, 3)
    assert c["static"].sum() > 50000 and not (c["static"] & c["lumen"]).any()
    assert not (c["static"] & c["wrap_band"]).any()
    # measured = true + offset outside the wrap band
    out = ~c["wrap_band"]
    assert np.allclose(c["velocity"][:, out], (c["velocity_true"] + c["offset"][None])[:, out])


def test_alias_roundtrip():
    p = poiseuille()
    wrapped, k = alias(p["velocity"], 500.0)
    assert np.abs(wrapped).max() < 500.0
    assert np.allclose(wrapped + 1000.0 * k, p["velocity"])
    assert k.max() == 1


def test_add_noise_respects_mask():
    p = poiseuille()
    v = add_noise(p["velocity"], 50.0, mask=p["mask"], rng=np.random.default_rng(1))
    assert np.allclose(v[:, ~p["mask"]], 0.0)
    assert abs((v - p["velocity"])[:, p["mask"]].std() - 50.0) < 2.0
