"""
Pathlines and centerline planes on the pulsatile aortic CFD series.

Usage: python test_pathlines.py /path/to/4DFlowNet/data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flowfilters"))
import time
import numpy as np
import h5py
from pathlines import advect_pathlines, seeds_from_plane, centerline_planes, to_vtk_polydata
from qa import plane_flow


def load(path):
    with h5py.File(path) as h:
        d = {k: h[k][0] for k in h.keys()}
    vel = np.stack([d["u"], d["v"], d["w"]], -1).astype(float) * 1000.0   # mm/s
    return vel, d["mask"] > 0.5, tuple(float(x) for x in d["dx"])


def main(data_dir):
    v_cfd, mask, sp = load(f"{data_dir}/example_data_HR.h5")
    nt, rr = 24, 600.0
    t = np.arange(nt) / nt
    wave = 0.1 + 0.9 * np.exp(-((t - 0.25) / 0.09) ** 2) + 0.15 * np.exp(-((t - 0.55) / 0.08) ** 2)
    v_t = np.stack([v_cfd * w for w in wave])

    # --- pathlines: emit from the ascending aorta (slice 80, flow toward -z)
    zz, yy, xx = np.nonzero(mask[80:81])
    seeds = seeds_from_plane(mask, sp, (80, yy.mean(), xx.mean()), (1, 0, 0), 14.0, spacing_mm=1.2)
    print(f"{seeds.shape[0]} seeds on the ascending-aorta plane")
    for emit in ("single", "staggered"):
        t0 = 0.0 if emit == "single" else np.linspace(0, rr * 0.5, seeds.shape[0])
        tic = time.perf_counter()
        traces = advect_pathlines(v_t, sp, rr, seeds, t0_ms=t0, duration_ms=rr, mask=mask)
        dt = time.perf_counter() - tic
        lengths = np.array([np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum() for tr in traces])
        alive_end = np.mean([tr[-1, 3] - tr[0, 3] >= rr - 1e-6 for tr in traces])
        print(f"  {emit:9s} emission: {dt:.1f} s for one RR, mean path length {lengths.mean():.0f} mm "
              f"(max {lengths.max():.0f}), {alive_end:.0%} still in lumen after one cycle")
    # all particles must stay inside the mask by construction; verify
    idx = np.concatenate([tr[:, :3] for tr in traces]) / np.asarray(sp)
    inside = mask[tuple(np.clip(np.round(idx).astype(int), 0, np.array(mask.shape) - 1).T)]
    print(f"  points inside lumen: {inside.mean():.1%}")
    try:
        pd = to_vtk_polydata(traces)
        print(f"  vtkPolyData: {pd.GetNumberOfPoints()} points, {pd.GetNumberOfLines()} lines")
    except ImportError:
        print("  (vtk not installed here; export untested)")

    # --- centerline planes along the descending aorta (slices 55..82, single tube)
    cl = []
    for k in range(55, 83):
        zz, yy, xx = np.nonzero(mask[k:k + 1])
        cl.append((k * sp[0], yy.mean() * sp[1], xx.mean() * sp[2]))
    planes = centerline_planes(np.array(cl), step_mm=4.0, radius_mm=15.0)
    print(f"\n{len(planes)} planes every 4 mm along a {planes[-1]['arc_mm']:.0f} mm descending-aorta centerline")
    flows = []
    for p in planes:
        r = plane_flow(v_t, sp, lumen_mask=mask, **p)
        flows.append(r["flow"].mean() / 1000.0)
    flows = np.array(flows)
    print(f"  mean flow per plane (mL/s): {np.round(flows, 1)}")
    print(f"  spread along the tube: {100 * flows.std() / abs(flows.mean()):.1f}% CV (truth: conserved)")


if __name__ == "__main__":
    main(sys.argv[1])
