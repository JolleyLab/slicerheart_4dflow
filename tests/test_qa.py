"""
Plane flow quantification and mass-conservation QA.

The CFD test measures through-plane flow at two descending-aorta planes (no
branches between them, so inflow = outflow in truth) before and after
divergence-free filtering.

Run as a script to print the QA table:
    python tests/test_qa.py
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, aortic_waveform, oblique_voxel_to_world, rotation_matrix, world_transform
from slicerheart_4dflow.divergence_free import divergence_free_sequence
from slicerheart_4dflow.geometry import voxel_grid_world_points, voxel_to_world_points
from slicerheart_4dflow.qa import mass_conservation, net_volume, plane_flow, plane_samples, qa_report


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def _tube(shape, voxel_to_world, centre, direction, radius_mm):
    """Lumen mask of a straight tube along world `direction` through world `centre`."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    p = voxel_grid_world_points(shape, voxel_to_world) - centre
    radial = p - (p @ d)[..., None] * d
    return np.linalg.norm(radial, axis=-1) <= radius_mm


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("tilt_deg", [0.0, 35.0])
def test_plane_flow_uniform_tube(geometry, tilt_deg):
    """Uniform flow in a tube along an oblique world direction. The flow is
    the same through a perpendicular and through a tilted plane."""
    m = GEOMETRIES[geometry]()
    shape = (48, 48, 48)
    centre = voxel_to_world_points(m, [[23.5, 23.5, 23.5]])[0]
    spacing = np.linalg.norm(m[:3, :3], axis=0)
    radius = 9.0 * spacing.min()
    d = rotation_matrix((1.0, 0.5, -0.2), 50.0) @ np.array([0.0, 0.0, 1.0])
    lumen = _tube(shape, m, centre, d, radius)
    speeds = np.array([100.0, 200.0, -50.0])                     # mm/s
    v_t = speeds[:, None, None, None, None] * d * lumen[None, ..., None]
    normal = rotation_matrix(np.cross(d, (1.0, 0.0, 0.0)), tilt_deg) @ d
    r = plane_flow(v_t, m, centre, normal, radius=2.0 * radius, lumen_mask=lumen)
    true_area = np.pi * radius ** 2
    np.testing.assert_allclose(r["flow"], speeds * true_area, rtol=0.03)
    np.testing.assert_allclose(r["area"] * np.cos(np.radians(tilt_deg)), true_area, rtol=0.03)
    # reversed normal flips the sign
    r2 = plane_flow(v_t, m, centre, -normal, radius=2.0 * radius, lumen_mask=lumen)
    np.testing.assert_allclose(r2["flow"], -r["flow"])


def test_plane_flow_world_coordinate_system_invariance():
    rng = np.random.default_rng(0)
    m = oblique_voxel_to_world()
    T = world_transform()
    R = T[:3, :3]
    shape = (20, 22, 24)
    v_t = rng.normal(size=(2,) + shape + (3,))
    lumen = rng.uniform(size=shape) > 0.3
    centre = voxel_to_world_points(m, [[11.0, 10.0, 9.0]])[0]
    normal = np.array([0.2, -0.5, 1.0])
    r = plane_flow(v_t, m, centre, normal, radius=6.0, lumen_mask=lumen)
    r_t = plane_flow(v_t @ R.T, T @ m, R @ centre + T[:3, 3], R @ normal, radius=6.0, lumen_mask=lumen)
    np.testing.assert_allclose(r_t["flow"], r["flow"], rtol=1e-9)
    assert r_t["n_samples"] == r["n_samples"]


def test_plane_samples_disc():
    m = oblique_voxel_to_world()
    shape = (60, 60, 60)
    centre = voxel_to_world_points(m, [[30.0, 30.0, 30.0]])[0]
    pts, idx, n, area_per_sample = plane_samples(shape, m, centre, (1.0, 1.0, 0.0), 10.0, sampling_mm=0.25)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)
    np.testing.assert_allclose((pts - centre) @ n, 0.0, atol=1e-9)
    np.testing.assert_allclose(len(pts) * area_per_sample, np.pi * 100.0, rtol=0.01)
    assert idx.shape == (3, len(pts))


def test_mass_conservation_and_net_volume():
    t = np.linspace(0, 1, 20, endpoint=False)
    q = 100 + 50 * np.sin(2 * np.pi * t)
    assert net_volume(q) == pytest.approx(100.0)
    assert net_volume(q, heart_rate_bpm=60) == pytest.approx(100.0)
    assert net_volume(q, heart_rate_bpm=120) == pytest.approx(50.0)
    flows = {"in": q, "a": 0.6 * q, "b": 0.4 * q, "c": 0.3 * q}
    res = mass_conservation(flows, [("ok", ["in"], ["a", "b"]), ("bad", ["in"], ["a", "c"])])
    assert res[0]["percent_error"] == pytest.approx(0.0, abs=1e-9)
    assert res[1]["difference"] == pytest.approx(10.0)
    assert res[1]["percent_error"] == pytest.approx(100 * 10 / 95)


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

def cfd_planes(mask, m):
    """Planes at the lumen centroids of two axial slices in the descending aorta."""
    planes = {}
    for name, k in (("DAo_prox", 62), ("DAo_dist", 80)):
        _, jj, ii = np.nonzero(mask[k:k + 1])
        centre = voxel_to_world_points(m, [[ii.mean(), jj.mean(), k]])[0]
        # flow is along -k in this dataset; normal so that forward is positive
        planes[name] = dict(centre=centre, normal=-m[:3, 2], radius=15.0)
    return planes


def cfd_stages(cfd, nt=8, world=None):
    v_cfd, mask, m = cfd
    v_cfd = v_cfd * 1000.0                                                   # mm/s
    if world is not None:
        v_cfd = v_cfd @ world[:3, :3].T
        m = world @ m
    rng = np.random.default_rng(0)
    v_true = np.stack([v_cfd * w for w in aortic_waveform(nt)])
    sigma = 0.10 * 1500.0                                                    # 10% of 1.5 m/s venc
    v_noisy = v_true + rng.normal(0, sigma, v_true.shape) * mask[None, ..., None]
    planes = cfd_planes(mask, m)

    def flows(v):  # mL/s
        return {n: plane_flow(v, m, lumen_mask=mask, **p)["flow"] / 1000.0 for n, p in planes.items()}

    v_dfw, _ = divergence_free_sequence(v_noisy, mask, m, method="dfw")
    v_fdm, _ = divergence_free_sequence(v_noisy, mask, m, method="fdm")
    return {"truth": flows(v_true), "noisy": flows(v_noisy), "fdm": flows(v_fdm), "dfw": flows(v_dfw)}


CHECKS = [("prox vs dist", ["DAo_prox"], ["DAo_dist"])]


@pytest.mark.cfd
@pytest.mark.parametrize("world", ["aligned", "oblique"])
def test_cfd_mass_conservation(cfd, world):
    stages = cfd_stages(cfd, world=None if world == "aligned" else world_transform())
    err = {s: mass_conservation(f, CHECKS, heart_rate_bpm=100)[0]["percent_error"] for s, f in stages.items()}
    assert abs(err["truth"]) < 1.0
    assert abs(err["fdm"]) < 1.5
    assert abs(err["dfw"]) < 2.5
    assert stages["truth"]["DAo_prox"].mean() > 10.0          # forward flow, mL/s


def main():
    import conftest
    stages = cfd_stages(conftest.load_cfd(), nt=24)
    print("\nMass-conservation QA: two descending-aorta planes (truth: equal flow)")
    qa_report(stages, CHECKS, heart_rate_bpm=100)
    print(f"  mean flows (mL/s): truth prox {stages['truth']['DAo_prox'].mean():.1f}, "
          f"dist {stages['truth']['DAo_dist'].mean():.1f}")


if __name__ == "__main__":
    main()
