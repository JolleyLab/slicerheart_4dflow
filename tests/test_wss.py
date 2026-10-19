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
def test_poiseuille_wss(geometry, metrics):
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, v_max_mm_s=800.0, axis_world=n)
    out = wall_shear_stress(p["velocity"], p["mask"], vtw)
    sel = _interior(out, n)
    med = np.median(out["wss_magnitude"][0][sel])
    metrics.record(f"wss/poiseuille/{geometry}/abs_relative_bias", abs(med / p["wss_pa"][0] - 1), atol=0.01)
    assert abs(med / p["wss_pa"][0] - 1) < 0.15
    # direction: tangential to the wall and along the flow
    assert np.abs((out["wss"][0][sel] * out["normals"][sel]).sum(-1)).max() < 1e-9 * out["wss_magnitude"].max() + 1e-6
    assert ((out["wss"][0][sel] @ n) > 0).mean() > 0.95
    assert np.median(out["osi"][sel]) < 1e-6


CLINICAL_GRID = lambda: voxel_to_world_matrix(spacing=(1.0, 1.0, 2.5))


@pytest.mark.parametrize("geometry", list(GEOMETRIES) + ["clinical_1x1x2.5"])
def test_wall_points_inward_normals(geometry):
    """Normals are covectors: on anisotropic, oblique and sheared grids they
    must still point at the cylinder axis. (Mapping the index-space gradient
    with the Jacobian instead of its inverse transpose gave median errors of
    15-30 degrees on these grids and passed the old identity-grid-only test.)"""
    vtw = CLINICAL_GRID() if geometry.startswith("clinical") else GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, axis_world=n)
    pts, nrm = wall_points(p["mask"], vtw)
    c = pts.mean(0)
    axial = (pts - c) @ n
    radial = (pts - c) - axial[:, None] * n
    cosang = -(radial * nrm).sum(-1) / (np.linalg.norm(radial, axis=-1) + 1e-12)
    sel = np.abs(axial) < 0.35 * np.abs(axial).max()
    angle_deg = np.degrees(np.arccos(np.clip(cosang[sel], -1, 1)))
    assert np.median(angle_deg) < 5.0 and np.percentile(angle_deg, 95) < 15.0   # staircase tail


def test_swirl_wss_has_no_normal_component():
    """Axial plus azimuthal flow on a 1 x 1 x 2.5 mm grid: the WSS vector must
    lie in the wall plane, so its component along the true (radial) normal is
    small compared with its magnitude. With the Jacobian-mapped normals of the
    old code a median 30% of |WSS| pointed along the true normal."""
    vtw = CLINICAL_GRID()
    p = poiseuille(shape=(24, 48, 48), voxel_to_world=vtw, radius_mm=12.0, v_max_mm_s=800.0)
    R = 12.0
    # azimuthal profile with the same wall shear as the axial one: v_phi = vmax (r/R)(1 - r/R) * 2 ... simple choice
    from slicerheart_4dflow.geometry import voxel_grid_world_points
    xyz = voxel_grid_world_points(p["mask"].shape, vtw)
    c = xyz[p["mask"]].mean(0)
    d = xyz - c; d[..., 2] = 0
    r = np.linalg.norm(d, axis=-1)
    phi_hat = np.stack([-d[..., 1], d[..., 0], np.zeros_like(r)], -1) / (r[..., None] + 1e-12)
    v_phi = 800.0 * (r / R) * (1 - r / R)
    v = p["velocity"][0] + v_phi[..., None] * phi_hat
    out = wall_shear_stress(v[None] * p["mask"][..., None], p["mask"], vtw)
    pts = out["points"]
    radial = pts - c; radial[:, 2] = 0
    n_true = -radial / (np.linalg.norm(radial, axis=-1, keepdims=True) + 1e-12)
    axial = pts[:, 2] - c[2]
    sel = np.abs(axial) < 0.35 * np.abs(axial).max()
    normal_frac = np.abs((out["wss"][0] * n_true).sum(-1)) / (out["wss_magnitude"][0] + 1e-12)
    assert np.median(normal_frac[sel]) < 0.05
    assert np.percentile(normal_frac[sel], 90) < 0.15


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_same_field_on_different_grids_agrees(geometry):
    """The same physical cylinder sampled on the identity grid and on another
    grid must give the same WSS: a grid-only bug is invisible to the
    world-frame invariance test, which transforms both sides together."""
    ref = poiseuille(shape=(32, 48, 48), radius_mm=12.0)
    med_ref = np.median(wall_shear_stress(ref["velocity"], ref["mask"], ref["voxel_to_world"])["wss_magnitude"][0])
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0, axis_world=n)
    out = wall_shear_stress(p["velocity"], p["mask"], vtw)
    sel = _interior(out, n)
    med = np.median(out["wss_magnitude"][0][sel])
    assert abs(med / med_ref - 1) < 0.12


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


def test_womersley_time_course_and_resolution_bias(metrics):
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
        metrics.record(f"wss/womersley/{h:.1f}mm/slope", slopes[h], direction="higher")
        assert out["osi"][sel].mean() > 0.03          # oscillatory flow detected (analytic 0.21)
    # underestimation grows with voxel size (sub-voxel boundary layer)
    assert 0.4 < slopes[1.0] < 0.8 and slopes[1.5] < slopes[1.0]


def test_wall_surface_is_the_half_level_isosurface():
    """wall_surface(): vertices lie on the 0.5 isosurface of the binary mask
    (a 12 mm cylinder on a 1 x 1 x 2.5 mm grid comes back with that radius),
    the triangles close the lateral surface (area within 2% of 2 pi R L),
    and to_vtk_surface() with the faces gives a model with those triangles."""
    pytest.importorskip("vtk")
    from slicerheart_4dflow.wss import wall_surface, to_vtk_surface
    vtw = CLINICAL_GRID()
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=12.0)
    s = wall_surface(p["mask"], vtw)
    pts, faces = s["points"], s["faces"]
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.max() < len(pts)
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    c = pts.mean(0)
    axial = (pts - c) @ n
    r = np.linalg.norm((pts - c) - axial[:, None] * n, axis=1)
    assert abs(np.median(r) - 12.0) < 0.3 and r.std() < 0.4
    import vtk
    pd = to_vtk_surface(pts, s["normals"], {"tawss": np.zeros(len(pts))}, faces=faces)
    assert pd.GetNumberOfPolys() == len(faces)
    mp = vtk.vtkMassProperties(); mp.SetInputData(pd); mp.Update()
    lateral = 2 * np.pi * 12.0 * 32 * 2.5
    assert abs(mp.GetSurfaceArea() - lateral) / lateral < 0.02
    # wall_points() is a view of the same surface
    q, _ = wall_points(p["mask"], vtw, step=4)
    assert np.allclose(q, pts[::4])
