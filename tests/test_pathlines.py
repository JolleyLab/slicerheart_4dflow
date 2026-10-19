"""
Pathlines (numpy RK4 and VTK particle tracer) and centerline planes.

Run as a script to print a summary on the pulsatile aortic CFD series:
    python tests/test_pathlines.py
"""

import time

import numpy as np
import pytest

from conftest import GEOMETRIES, WORLD_TRANSFORMS, aortic_waveform, oblique_voxel_to_world
from slicerheart_4dflow.geometry import voxel_to_world_points, world_to_voxel_points
from slicerheart_4dflow.pathlines import advect_pathlines, centerline_planes, seeds_from_plane, to_vtk_polydata
from slicerheart_4dflow.qa import plane_flow

RR_MS = 600.0
FLOW_DIRECTION = np.array([0.3, -0.4, 0.866])      # world direction of the uniform test flow
FLOW_SPEED = 50.0                                   # mm/s


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def _uniform_flow(nt=10, shape=(30, 30, 30)):
    d = FLOW_DIRECTION / np.linalg.norm(FLOW_DIRECTION)
    v_t = np.zeros((nt,) + shape + (3,))
    v_t[...] = FLOW_SPEED * d
    return v_t, FLOW_SPEED * d


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_advect_pathlines_uniform_flow(geometry):
    m = GEOMETRIES[geometry]()
    v_t, v = _uniform_flow()
    seeds = voxel_to_world_points(m, [[15.0, 15.0, 10.0], [12.0, 16.0, 14.0]])
    traces = advect_pathlines(v_t, m, RR_MS, seeds, duration_ms=60.0)
    for seed, tr in zip(seeds, traces):
        dt_s = (tr[-1, 3] - tr[0, 3]) / 1000.0
        # integration covers the duration, rounded up to whole RK4 steps (15 ms here)
        assert 0.06 <= dt_s < 0.06 + 0.015 + 1e-9
        np.testing.assert_allclose(tr[:, :3], seed + v * (tr[:, 3:] - tr[0, 3]) / 1000.0, atol=1e-9)


def test_advect_pathlines_stops_at_mask():
    m = oblique_voxel_to_world()
    v_t, v = _uniform_flow()
    mask = np.zeros(v_t.shape[1:4], bool)
    mask[5:25, 5:25, 5:25] = True
    traces = advect_pathlines(v_t, m, RR_MS, voxel_to_world_points(m, [[15.0, 15.0, 15.0]]), mask=mask)
    ijk = world_to_voxel_points(m, traces[0][:, :3])
    # the trilinearly interpolated mask drops below 0.5 halfway past the last voxel
    assert np.all((ijk >= 4.5) & (ijk <= 24.5))
    assert len(traces[0]) > 3


def test_advect_pathlines_backward():
    m = oblique_voxel_to_world()
    v_t, v = _uniform_flow()
    seed = voxel_to_world_points(m, [[15.0, 15.0, 15.0]])
    traces = advect_pathlines(v_t, m, RR_MS, seed, duration_ms=60.0, backward=True)
    dt_s = (traces[0][0, 3] - traces[0][-1, 3]) / 1000.0
    assert dt_s >= 0.06
    np.testing.assert_allclose(traces[0][-1, :3], seed[0] - v * dt_s, atol=1e-9)


@pytest.mark.parametrize("world", list(WORLD_TRANSFORMS))
def test_advect_pathlines_world_coordinate_system_invariance(world):
    rng = np.random.default_rng(0)
    m = oblique_voxel_to_world()
    T = WORLD_TRANSFORMS[world]()
    R = T[:3, :3]
    shape = (16, 16, 16)
    v_t = 20.0 * rng.normal(size=(6,) + shape + (3,))
    seeds = voxel_to_world_points(m, rng.uniform(4, 11, (5, 3)))
    traces = advect_pathlines(v_t, m, RR_MS, seeds, duration_ms=100.0)
    traces_t = advect_pathlines(v_t @ R.T, T @ m, RR_MS, seeds @ R.T + T[:3, 3], duration_ms=100.0)
    for tr, tr_t in zip(traces, traces_t):
        np.testing.assert_allclose(tr_t[:, :3], tr[:, :3] @ R.T + T[:3, 3], atol=1e-8)
        np.testing.assert_allclose(tr_t[:, 3], tr[:, 3])


def test_seeds_from_plane():
    m = oblique_voxel_to_world()
    shape = (30, 30, 30)
    mask = np.ones(shape, bool)
    centre = voxel_to_world_points(m, [[15.0, 15.0, 15.0]])[0]
    normal = np.array([1.0, 2.0, -0.5])
    seeds = seeds_from_plane(mask, m, centre, normal, 4.0, spacing_mm=0.5)
    n = normal / np.linalg.norm(normal)
    np.testing.assert_allclose((seeds - centre) @ n, 0.0, atol=1e-9)
    assert np.all(np.linalg.norm(seeds - centre, axis=1) <= 4.0 + 1e-9)
    np.testing.assert_allclose(len(seeds) * 0.25, np.pi * 16.0, rtol=0.05)
    # seeds outside the lumen are dropped
    mask[:, :, :15] = False
    assert 0 < len(seeds_from_plane(mask, m, centre, normal, 4.0, spacing_mm=0.5)) < len(seeds)


def test_centerline_planes_straight_line():
    d = FLOW_DIRECTION / np.linalg.norm(FLOW_DIRECTION)
    cl = np.array([10.0, -5.0, 3.0]) + np.arange(30.0)[:, None] * d
    planes = centerline_planes(cl, step_mm=5.0, radius_mm=7.0)
    assert len(planes) == 6
    for p in planes:
        np.testing.assert_allclose(p["normal"], d, atol=1e-9)
        assert p["radius"] == 7.0


def test_to_vtk_polydata():
    pytest.importorskip("vtk")
    traces = [np.array([[0, 0, 0, 0], [0, 0, 1, 10], [2, 1, 0, 20.0]]), np.array([[1, 1, 1, 0.0]])]
    pd = to_vtk_polydata(traces)
    assert pd.GetNumberOfLines() == 1            # single-point traces are skipped
    assert pd.GetNumberOfPoints() == 3
    assert pd.GetPoint(2) == (2.0, 1.0, 0.0)
    assert pd.GetPointData().GetArray("time_ms").GetValue(2) == 20.0


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_pathlines_vtk_uniform_flow(geometry):
    pytest.importorskip("vtk")
    from vtk.util import numpy_support as ns
    from slicerheart_4dflow.pathlines_fast import pathlines_vtk
    m = GEOMETRIES[geometry]()
    v_t, v = _uniform_flow()
    seed = voxel_to_world_points(m, [[15.0, 15.0, 10.0]])
    pd = pathlines_vtk(v_t, m, RR_MS, seed, duration_ms=100.0)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData())
    t_ms = ns.vtk_to_numpy(pd.GetPointData().GetArray("time_ms"))
    vel = ns.vtk_to_numpy(pd.GetPointData().GetArray("velocity"))
    assert pts.shape[0] > 2
    np.testing.assert_allclose(pts, seed[0] + v * t_ms[:, None] / 1000.0, atol=1e-3)
    np.testing.assert_allclose(vel, np.broadcast_to(v, vel.shape), atol=1e-3)


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

def cfd_series(cfd, nt=24):
    v_cfd, mask, m = cfd
    v_t = np.stack([v_cfd * 1000.0 * w for w in aortic_waveform(nt)])   # mm/s
    return v_t, mask, m


def cfd_pathlines(v_t, mask, m, emit="single"):
    # emit from slice 80 of the aorta (flow toward -k)
    _, jj, ii = np.nonzero(mask[80:81])
    centre = voxel_to_world_points(m, [[ii.mean(), jj.mean(), 80]])[0]
    seeds = seeds_from_plane(mask, m, centre, m[:3, 2], 14.0, spacing_mm=1.2)
    t0 = 0.0 if emit == "single" else np.linspace(0, RR_MS * 0.5, seeds.shape[0])
    traces = advect_pathlines(v_t, m, RR_MS, seeds, t0_ms=t0, duration_ms=RR_MS, mask=mask)
    return seeds, traces


def cfd_centerline_flows(v_t, mask, m):
    # centerline of the descending aorta (slices 55..82, single tube)
    cl = []
    for k in range(55, 83):
        _, jj, ii = np.nonzero(mask[k:k + 1])
        cl.append((ii.mean(), jj.mean(), k))
    planes = centerline_planes(voxel_to_world_points(m, cl), step_mm=4.0, radius_mm=15.0)
    flows = np.array([plane_flow(v_t, m, lumen_mask=mask, **p)["flow"].mean() / 1000.0 for p in planes])
    return planes, flows


@pytest.mark.cfd
@pytest.mark.parametrize("emit", ["single", "staggered"])
def test_cfd_pathlines(cfd, emit):
    v_t, mask, m = cfd_series(cfd)
    seeds, traces = cfd_pathlines(v_t, mask, m, emit)
    assert seeds.shape[0] > 100
    lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
    assert 30.0 < lengths.mean() < 70.0
    # particles stop when they leave the lumen
    kji = np.round(world_to_voxel_points(m, np.concatenate([tr[:, :3] for tr in traces]))[:, ::-1]).astype(int)
    inside = mask[tuple(np.clip(kji, 0, np.array(mask.shape) - 1).T)]
    assert inside.mean() > 0.99


@pytest.mark.cfd
def test_cfd_centerline_planes_conserve_flow(cfd):
    v_t, mask, m = cfd_series(cfd)
    planes, flows = cfd_centerline_flows(v_t, mask, m)
    assert len(planes) >= 4
    assert flows.std() / abs(flows.mean()) < 0.02


def main():
    import conftest
    v_t, mask, m = cfd_series(conftest.load_cfd())
    for emit in ("single", "staggered"):
        tic = time.perf_counter()
        seeds, traces = cfd_pathlines(v_t, mask, m, emit)
        dt = time.perf_counter() - tic
        lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
        alive_end = np.mean([tr[-1, 3] - tr[0, 3] >= RR_MS - 1e-6 for tr in traces])
        print(f"{seeds.shape[0]} seeds, {emit:9s} emission: {dt:.1f} s for one RR, mean path length "
              f"{lengths.mean():.0f} mm (max {lengths.max():.0f}), {alive_end:.0%} still in lumen after one cycle")
    planes, flows = cfd_centerline_flows(v_t, mask, m)
    print(f"\n{len(planes)} planes every 4 mm along a {planes[-1]['arc_mm']:.0f} mm descending-aorta centerline")
    print(f"  mean flow per plane (mL/s): {np.round(flows, 1)}")
    print(f"  spread along the tube: {100 * flows.std() / abs(flows.mean()):.1f}% CV (truth: conserved)")


if __name__ == "__main__":
    main()
