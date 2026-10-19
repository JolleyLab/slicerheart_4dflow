"""
Wall shear stress against the Poiseuille and Womersley analytic solutions.

Poiseuille (parabolic profile) is recovered within about 10% at 1 mm voxels
under every geometry. Womersley flow at alpha 18 has a boundary layer of
about 0.65 mm, thinner than a voxel, so WSS is underestimated (slope about
0.6 at 1 mm, 0.4 at 1.5 mm) while the time course is well correlated;
this is the resolution limit reported by Petersson et al. (J Magn Reson
Imaging 2012;36:128-138) and is recorded here as a known bias, not a
target to tune.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, WORLD_TRANSFORMS
from slicerheart_4dflow.phantom import poiseuille, womersley
from slicerheart_4dflow.wss import wall_shear_stress, wall_points
from slicerheart_4dflow.geometry import voxel_to_world_matrix


def _interior(out, axis):
    z = (out["points"] - out["points"].mean(0)) @ axis
    return np.abs(z) < 0.35 * np.abs(z).max()


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_wss(geometry):
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, v_max_mm_s=800.0, axis_world=n)
    out = wall_shear_stress(p["velocity"], p["mask"], vtw)
    sel = _interior(out, n)
    med = np.median(out["wss_magnitude"][0][sel])
    assert abs(med / p["wss_pa"][0] - 1) < 0.15
    # direction: tangential to the wall and along the flow
    assert np.abs((out["wss"][0][sel] * out["normals"][sel]).sum(-1)).max() < 1e-9 * out["wss_magnitude"].max() + 1e-6
    assert ((out["wss"][0][sel] @ n) > 0).mean() > 0.95
    assert np.median(out["osi"][sel]) < 1e-6


def test_wall_points_inward_normals():
    p = poiseuille()
    pts, nrm = wall_points(p["mask"], p["voxel_to_world"])
    # inward normals point toward the cylinder axis (world z here)
    c = pts.mean(0)
    radial = pts - c; radial[:, 2] = 0
    cosang = -(radial * nrm).sum(-1) / (np.linalg.norm(radial, axis=-1) + 1e-12)
    z = (pts - c)[:, 2]; sel = np.abs(z) < 0.35 * np.abs(z).max()
    assert np.median(cosang[sel]) > 0.95


@pytest.mark.parametrize("transform", list(WORLD_TRANSFORMS))
def test_world_frame_invariance(transform):
    T = WORLD_TRANSFORMS[transform]()
    vtw = GEOMETRIES["oblique"]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(24, 40, 40), voxel_to_world=vtw, axis_world=n)
    R = T[:3, :3]
    out1 = wall_shear_stress(p["velocity"], p["mask"], vtw)
    out2 = wall_shear_stress(p["velocity"] @ R.T, p["mask"], T @ vtw)
    # points near the open cylinder ends have sample lines leaving the image
    # (an image-edge effect), so compare interior points only
    sel = _interior(out1, n)
    assert np.allclose(out1["wss_magnitude"][:, sel], out2["wss_magnitude"][:, sel], rtol=1e-6, atol=1e-9)


def test_womersley_time_course_and_resolution_bias():
    slopes = {}
    for h in (1.0, 1.5):
        vtw = voxel_to_world_matrix(spacing=(h, h, h)); nn = int(48 / h)
        w = womersley(shape=(16, nn, nn), voxel_to_world=vtw)
        out = wall_shear_stress(w["velocity"], w["mask"], vtw)
        sel = _interior(out, w["axis_world"])
        sgn = np.sign((out["wss"][:, sel] @ w["axis_world"]).mean(1))
        est = np.median(out["wss_magnitude"][:, sel], axis=1) * sgn
        assert np.corrcoef(est, w["wss_pa"])[0, 1] > 0.8
        slopes[h] = np.polyfit(w["wss_pa"], est, 1)[0]
        assert out["osi"][sel].mean() > 0.03          # oscillatory flow detected (analytic 0.21)
    # underestimation grows with voxel size (sub-voxel boundary layer)
    assert 0.4 < slopes[1.0] < 0.8 and slopes[1.5] < slopes[1.0]
