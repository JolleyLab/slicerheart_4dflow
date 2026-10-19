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


def _rotating_in_time(nt=20, shape=(30, 30, 30), speed=20.0):
    """Spatially uniform velocity rotating in time: v(t) = speed (cos wt, sin wt, 0),
    w = 2 pi / RR; the exact displacement over [t0, t0 + d] is analytic."""
    rr = 1000.0
    t = np.arange(nt) * rr / nt
    v = np.zeros((nt,) + shape + (3,))
    v[..., 0] = speed * np.cos(2 * np.pi * t / rr)[:, None, None, None]
    v[..., 1] = speed * np.sin(2 * np.pi * t / rr)[:, None, None, None]

    def exact(t0, d):
        w0, w1 = 2 * np.pi * t0 / rr, 2 * np.pi * (t0 + d) / rr
        k = speed * rr / 1000.0 / (2 * np.pi)
        return np.array([k * (np.sin(w1) - np.sin(w0)), -k * (np.cos(w1) - np.cos(w0)), 0.0])
    return v, rr, exact


def test_advect_pathlines_time_varying_field_and_staggered_emission():
    """Each particle must be advected at its own time: a batch emitted at
    different phases must reproduce the individually integrated traces
    (the old median-time batch put a particle emitted at 500 ms through the
    0 ms phase, a 90 degree error). The trace also ends exactly at duration."""
    v, rr, exact = _rotating_in_time()
    seeds = np.full((3, 3), 15.0)
    t0 = np.array([0.0, 250.0, 500.0])
    batch = advect_pathlines(v, np.eye(4), rr, seeds, t0_ms=t0, duration_ms=200.0)
    for i in range(3):
        single = advect_pathlines(v, np.eye(4), rr, seeds[i:i + 1], t0_ms=t0[i], duration_ms=200.0)[0]
        np.testing.assert_allclose(batch[i], single, atol=1e-9)
        assert abs(batch[i][-1, 3] - (t0[i] + 200.0)) < 1e-9
        assert np.linalg.norm(batch[i][-1, :3] - 15.0 - exact(t0[i], 200.0)) < 0.05
    # one full cycle returns to the start
    full = advect_pathlines(v, np.eye(4), rr, seeds[:1], t0_ms=0.0, duration_ms=rr)[0]
    assert np.linalg.norm(full[-1, :3] - 15.0) < 0.05


def test_centerline_planes_straight_line():
    d = FLOW_DIRECTION / np.linalg.norm(FLOW_DIRECTION)
    cl = np.array([10.0, -5.0, 3.0]) + np.arange(30.0)[:, None] * d
    planes = centerline_planes(cl, step_mm=5.0, radius_mm=7.0)
    assert len(planes) == 6
    for p in planes:
        np.testing.assert_allclose(p["normal"], d, atol=1e-9)
        assert p["radius"] == 7.0


def test_centerline_planes_arc_normal_is_tangent_at_centre():
    """On a circular arc the plane normal must be the tangent at the plane's
    own centre, i.e. perpendicular to the radius there (the old code used
    the tangent at the nearest centreline point: 11 degrees off at 4 mm
    point spacing on a 20 mm arc)."""
    R = 20.0
    phi = np.radians(np.arange(0, 181, 12))            # 4.2 mm point spacing
    cl = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], 1)
    planes = centerline_planes(cl, step_mm=10.0, radius_mm=5.0, smooth_window=1)
    for p in planes[1:-1]:
        radial = p["centre"] / np.linalg.norm(p["centre"])
        tilt = np.degrees(np.arcsin(abs(radial @ p["normal"])))
        assert tilt < 1.5


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


def _rotating_flow(voxel_to_world, nt=4, n=40, omega=2.0):
    """Solid-body rotation (rad/s) about the world z axis through the image
    centre. Linear in space, so trilinear interpolation is exact and any
    deviation from a circle is an interpolation or geometry error."""
    from slicerheart_4dflow.geometry import voxel_grid_world_points
    shape = (n, n, n)
    x = voxel_grid_world_points(shape, voxel_to_world)
    c = voxel_to_world_points(voxel_to_world, [[(n - 1) / 2.0] * 3])[0]
    v = np.cross([0.0, 0.0, omega], x - c)
    return np.broadcast_to(v, (nt,) + v.shape).copy(), c


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_pathlines_vtk_rotating_flow(geometry):
    """VTK < 9.7 interpolates wrongly in images with a direction matrix
    (deviations of 0.1-0.4 mm in this test with VTK 9.6); tracing in the
    axis-aligned space must give the exact circle with any VTK version."""
    pytest.importorskip("vtk")
    from vtk.util import numpy_support as ns
    from slicerheart_4dflow.pathlines_fast import pathlines_vtk
    m = GEOMETRIES[geometry]()
    v_t, c = _rotating_flow(m)
    seed = c + np.array([8.0, 0.0, 0.0])
    pd = pathlines_vtk(v_t, m, 1000.0, seed[None], duration_ms=1000.0, terminal_speed_mm_s=0.1)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    t_s = ns.vtk_to_numpy(pd.GetPointData().GetArray("time_ms")) / 1000.0
    vel = ns.vtk_to_numpy(pd.GetPointData().GetArray("velocity")).astype(float)
    assert t_s.max() > 0.9                            # more than a quarter turn
    angle = 2.0 * t_s
    expected = c + 8.0 * np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], 1)
    np.testing.assert_allclose(pts, expected, atol=1e-3)
    np.testing.assert_allclose(vel, np.cross([0.0, 0.0, 2.0], pts - c), atol=1e-3)


def test_pathlines_vtk_emission_time():
    """Seeds must be injected at t0 (scalar or per seed) and integrated for
    `duration` from there; VTK's tracer has no start-time setting, so the
    served frames start at t0. The old code injected every seed at pipeline
    time 0 and, for a scalar t0, integrated to t0 + duration from there."""
    vtk = pytest.importorskip("vtk")
    from vtk.util import numpy_support as ns
    from slicerheart_4dflow.pathlines_fast import pathlines_vtk
    v, rr, exact = _rotating_in_time()
    seeds = np.full((2, 3), 15.0)
    t0 = np.array([0.0, 500.0])
    pd = pathlines_vtk(v, np.eye(4), rr, seeds, t0_ms=t0, duration_ms=200.0)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData())
    idx = ns.vtk_to_numpy(pd.GetPointData().GetArray("index"))
    tm = ns.vtk_to_numpy(pd.GetPointData().GetArray("time_ms"))
    for i in range(2):
        sel = idx == i
        assert abs(tm[sel].min() - t0[i]) < 1e-6 and 180.0 < tm[sel].max() - t0[i] <= 200.0 + 1e-6
        end = pts[sel][np.argmax(tm[sel])]
        assert np.linalg.norm(end - 15.0 - exact(t0[i], tm[sel].max() - t0[i])) < 0.15
    # scalar t0
    pd = pathlines_vtk(v, np.eye(4), rr, seeds[:1], t0_ms=500.0, duration_ms=rr)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData()); tm = ns.vtk_to_numpy(pd.GetPointData().GetArray("time_ms"))
    assert abs(tm.min() - 500.0) < 1e-6
    assert np.linalg.norm(pts[np.argmax(tm)] - 15.0) < 0.2       # one cycle: back to the start


def test_pathlines_vtk_terminal_speed_is_world_speed():
    """The terminal speed applies to the world speed (oblique, anisotropic,
    orthogonal voxel axes)."""
    pytest.importorskip("vtk")
    from slicerheart_4dflow.pathlines_fast import pathlines_vtk
    m = GEOMETRIES["anisotropic"]()
    m[:3, :3] = oblique_voxel_to_world()[:3, :3]
    v_t, v = _uniform_flow()
    seed = voxel_to_world_points(m, [[15.0, 15.0, 10.0]])
    moving = pathlines_vtk(v_t, m, RR_MS, seed, duration_ms=100.0, terminal_speed_mm_s=0.9 * FLOW_SPEED)
    stopped = pathlines_vtk(v_t, m, RR_MS, seed, duration_ms=100.0, terminal_speed_mm_s=1.1 * FLOW_SPEED)
    assert moving.GetNumberOfPoints() > 2
    assert stopped.GetNumberOfPoints() <= 1


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
    return seeds, traces, t0


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
    seeds, traces, t0 = cfd_pathlines(v_t, mask, m, emit)
    assert seeds.shape[0] > 100
    lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
    assert 30.0 < lengths.mean() < 70.0
    if emit == "staggered":
        # batch integration equals integrating each particle on its own
        for i in np.linspace(0, seeds.shape[0] - 1, 6).astype(int):
            single = advect_pathlines(v_t, m, RR_MS, seeds[i:i + 1], t0_ms=float(t0[i]), duration_ms=RR_MS,
                                      mask=mask)[0]
            assert single.shape == traces[i].shape
            np.testing.assert_allclose(traces[i], single, atol=1e-9)
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
