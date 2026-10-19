"""
Pathlines (numpy RK4 and VTK particle tracer) and centerline planes.

Run as a script to print a summary on the pulsatile aortic CFD series:
    python tests/test_pathlines.py
"""

import time

import numpy as np
import pytest

from conftest import aortic_waveform
from slicerheart_4dflow.pathlines import advect_pathlines, centerline_planes, seeds_from_plane, to_vtk_polydata
from slicerheart_4dflow.qa import plane_flow

RR_MS = 600.0


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def _uniform_flow(nt=10, shape=(20, 20, 40), velocity_mm_s=(0.0, 0.0, 50.0)):
    v_t = np.zeros((nt,) + shape + (3,))
    v_t[...] = velocity_mm_s
    return v_t


def test_advect_pathlines_uniform_flow():
    v_t = _uniform_flow()
    spacing = (1.0, 1.0, 1.0)
    seeds = np.array([[10.0, 10.0, 5.0], [8.0, 12.0, 6.0]])
    traces = advect_pathlines(v_t, spacing, RR_MS, seeds, duration_ms=200.0)
    for seed, tr in zip(seeds, traces):
        dt_s = (tr[-1, 3] - tr[0, 3]) / 1000.0
        np.testing.assert_allclose(tr[-1, :3], seed + np.array([0.0, 0.0, 50.0]) * dt_s, atol=1e-9)
        # integration covers the duration, rounded up to whole RK4 steps (15 ms here)
        assert 0.2 <= dt_s < 0.2 + 0.015 + 1e-9


def test_advect_pathlines_stops_at_mask():
    v_t = _uniform_flow()
    mask = np.zeros(v_t.shape[1:4], bool)
    mask[:, :, :20] = True
    traces = advect_pathlines(v_t, (1.0, 1.0, 1.0), RR_MS, [[10.0, 10.0, 5.0]], mask=mask)
    # the trilinearly interpolated mask drops below 0.5 halfway past the last voxel
    assert 18.0 < traces[0][:, 2].max() <= 19.5


def test_advect_pathlines_backward():
    v_t = _uniform_flow()
    traces = advect_pathlines(v_t, (1.0, 1.0, 1.0), RR_MS, [[10.0, 10.0, 30.0]], duration_ms=100.0, backward=True)
    dt_s = (traces[0][0, 3] - traces[0][-1, 3]) / 1000.0
    assert dt_s >= 0.1
    np.testing.assert_allclose(traces[0][-1, 2], 30.0 - 50.0 * dt_s, atol=1e-9)


def test_seeds_from_plane():
    shape, spacing = (20, 20, 20), (1.0, 1.0, 1.0)
    mask = np.ones(shape, bool)
    seeds = seeds_from_plane(mask, spacing, (10, 10, 10), (1, 0, 0), 4.0, spacing_mm=1.0)
    np.testing.assert_allclose(seeds[:, 0], 10.0)
    assert np.all(np.linalg.norm(seeds[:, 1:] - 10.0, axis=1) <= 4.0 + 1e-9)


def test_centerline_planes_straight_line():
    cl = np.stack([np.zeros(30), np.zeros(30), np.arange(30.0)], 1)
    planes = centerline_planes(cl, step_mm=5.0, radius_mm=7.0)
    assert len(planes) == 6
    for p in planes:
        np.testing.assert_allclose(p["normal"], (0, 0, 1), atol=1e-9)
        assert p["physical"] and p["radius"] == 7.0


def test_to_vtk_polydata():
    pytest.importorskip("vtk")
    traces = [np.array([[0, 0, 0, 0], [0, 0, 1, 10], [0, 1, 2, 20.0]]), np.array([[1, 1, 1, 0.0]])]
    pd = to_vtk_polydata(traces)
    assert pd.GetNumberOfLines() == 1            # single-point traces are skipped
    assert pd.GetNumberOfPoints() == 3
    assert pd.GetPoint(2) == (2.0, 1.0, 0.0)     # VTK order x, y, z
    assert pd.GetPointData().GetArray("time_ms").GetValue(2) == 20.0


def test_pathlines_vtk_uniform_flow():
    pytest.importorskip("vtk")
    from vtk.util import numpy_support as ns
    from slicerheart_4dflow.pathlines_fast import pathlines_vtk
    v_t = _uniform_flow()
    seeds = np.array([[10.0, 10.0, 5.0]])
    pd = pathlines_vtk(v_t, (1.0, 1.0, 1.0), RR_MS, seeds, duration_ms=300.0)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData())
    t_ms = ns.vtk_to_numpy(pd.GetPointData().GetArray("time_ms"))
    assert pts.shape[0] > 2
    # x = x0 + 50 mm/s * t
    np.testing.assert_allclose(pts[:, 0], 5.0 + 0.05 * t_ms, atol=1e-3)
    np.testing.assert_allclose(pts[:, 1:], 10.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Aortic CFD
# ---------------------------------------------------------------------------

def cfd_series(cfd, nt=24):
    v_cfd, mask, sp = cfd
    v_t = np.stack([v_cfd * 1000.0 * w for w in aortic_waveform(nt)])   # mm/s
    return v_t, mask, sp


def cfd_pathlines(v_t, mask, sp, emit="single"):
    # emit from slice 80 of the aorta (flow toward -axis 0)
    _, yy, xx = np.nonzero(mask[80:81])
    seeds = seeds_from_plane(mask, sp, (80, yy.mean(), xx.mean()), (1, 0, 0), 14.0, spacing_mm=1.2)
    t0 = 0.0 if emit == "single" else np.linspace(0, RR_MS * 0.5, seeds.shape[0])
    traces = advect_pathlines(v_t, sp, RR_MS, seeds, t0_ms=t0, duration_ms=RR_MS, mask=mask)
    return seeds, traces


def cfd_centerline_flows(v_t, mask, sp):
    # centerline of the descending aorta (slices 55..82, single tube)
    cl = []
    for k in range(55, 83):
        _, yy, xx = np.nonzero(mask[k:k + 1])
        cl.append((k * sp[0], yy.mean() * sp[1], xx.mean() * sp[2]))
    planes = centerline_planes(np.array(cl), step_mm=4.0, radius_mm=15.0)
    flows = np.array([plane_flow(v_t, sp, lumen_mask=mask, **p)["flow"].mean() / 1000.0 for p in planes])
    return planes, flows


@pytest.mark.cfd
@pytest.mark.parametrize("emit", ["single", "staggered"])
def test_cfd_pathlines(cfd, emit):
    v_t, mask, sp = cfd_series(cfd)
    seeds, traces = cfd_pathlines(v_t, mask, sp, emit)
    assert seeds.shape[0] > 100
    lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
    assert 30.0 < lengths.mean() < 70.0
    # particles stop when they leave the lumen
    idx = np.concatenate([tr[:, :3] for tr in traces]) / np.asarray(sp)
    inside = mask[tuple(np.clip(np.round(idx).astype(int), 0, np.array(mask.shape) - 1).T)]
    assert inside.mean() > 0.99


@pytest.mark.cfd
def test_cfd_centerline_planes_conserve_flow(cfd):
    v_t, mask, sp = cfd_series(cfd)
    planes, flows = cfd_centerline_flows(v_t, mask, sp)
    assert len(planes) >= 4
    assert flows.std() / abs(flows.mean()) < 0.02


def main():
    import conftest
    v_t, mask, sp = cfd_series(conftest.load_cfd())
    for emit in ("single", "staggered"):
        tic = time.perf_counter()
        seeds, traces = cfd_pathlines(v_t, mask, sp, emit)
        dt = time.perf_counter() - tic
        lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
        alive_end = np.mean([tr[-1, 3] - tr[0, 3] >= RR_MS - 1e-6 for tr in traces])
        print(f"{seeds.shape[0]} seeds, {emit:9s} emission: {dt:.1f} s for one RR, mean path length "
              f"{lengths.mean():.0f} mm (max {lengths.max():.0f}), {alive_end:.0%} still in lumen after one cycle")
    planes, flows = cfd_centerline_flows(v_t, mask, sp)
    print(f"\n{len(planes)} planes every 4 mm along a {planes[-1]['arc_mm']:.0f} mm descending-aorta centerline")
    print(f"  mean flow per plane (mL/s): {np.round(flows, 1)}")
    print(f"  spread along the tube: {100 * flows.std() / abs(flows.mean()):.1f}% CV (truth: conserved)")


if __name__ == "__main__":
    main()
