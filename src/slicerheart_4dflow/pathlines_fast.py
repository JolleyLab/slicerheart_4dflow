"""
Pathlines with VTK's own time-varying particle tracer.

pathlines_vtk() wraps vtkParticlePathFilter, the tracer ParaView uses for
"Particle Pathlines". A small VTKPythonAlgorithmBase source serves the
velocity frames as a temporal pipeline; the frames are
interpolated in time (linearly, periodically) before they reach the tracer,
which integrates RK4 in parallel through vtkSMPTools. Pure VTK, nothing to install in Slicer. Returns vtkPolyData
with 'time_ms' and 'index' point scalars, same as pathlines.to_vtk_polydata.

Note: VTK has no GPU pathline tracer. Its VTK-m accelerator covers steady
streamlines only and is not built into Slicer's VTK. This filter is the
fastest route available inside Slicer; thousands of seeds take seconds.

Conventions as elsewhere (see geometry.py): velocity_t (nt, nk, nj, ni, 3)
mm/s with world components (vx, vy, vz), voxel_to_world 4x4 voxel (i, j, k, 1)
to world (x, y, z, 1) matrix, rr_ms cycle length, seeds (n, 3) world (x, y, z)
mm. Output points are world (x, y, z) mm.

The particle tracer of VTK < 9.7 (e.g. VTK 9.6 in 3D Slicer 5.12) computes
wrong interpolation weights in images that have a non-identity direction
matrix, so the image is never given to VTK with its world orientation.
Instead, the tracing is done in an axis-aligned space (see
geometry.world_to_axis_aligned): voxel spacing is kept, origin is zero and
directions are identity. Velocities and seeds are mapped into that space
with the same affine transform as the voxel positions, which is exact for any
affine voxel-to-world mapping, and the path points and the 'velocity' point
data are mapped back to world space. For orthogonal voxel axes the transform
is a rotation (possibly with reflection), so speeds, and therefore the
terminal speed, are not changed.
"""

from __future__ import annotations

import numpy as np

from .geometry import as_voxel_to_world, transform_points, voxel_spacing, world_to_axis_aligned


# ===========================================================================
# VTK native
# ===========================================================================

def _frames_to_vtk_images(velocity_t, spacing, mask=None):
    """List of axis-aligned vtkImageData (spacing (si, sj, sk), zero origin,
    identity directions), one per frame, point vectors 'velocity' given in the
    axis-aligned space."""
    import vtk
    from vtk.util import numpy_support as ns
    nt, nk, nj, ni, _ = velocity_t.shape
    images = []
    for t in range(nt):
        v = velocity_t[t]
        if mask is not None:
            v = v * mask[..., None]
        arr = np.ascontiguousarray(v.reshape(-1, 3).astype(np.float32))   # i fastest
        img = vtk.vtkImageData()
        img.SetDimensions(ni, nj, nk)
        img.SetSpacing(*[float(s) for s in spacing])
        img.SetOrigin(0.0, 0.0, 0.0)
        va = ns.numpy_to_vtk(arr, deep=True); va.SetName("velocity")
        img.GetPointData().SetVectors(va)
        images.append(img)
    return images


def _temporal_source(images, times_ms):
    """A VTK algorithm that serves images[i] at times_ms[i] (temporal pipeline)."""
    import vtk
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    class Source(VTKPythonAlgorithmBase):
        def __init__(self):
            super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkImageData")
            self.times = np.asarray(times_ms, float)

        def RequestInformation(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            ex = vtk.vtkStreamingDemandDrivenPipeline
            info.Set(ex.TIME_STEPS(), self.times.tolist(), len(self.times))
            info.Set(ex.TIME_RANGE(), [float(self.times[0]), float(self.times[-1])], 2)
            d = images[0].GetDimensions()
            info.Set(ex.WHOLE_EXTENT(), (0, d[0] - 1, 0, d[1] - 1, 0, d[2] - 1), 6)
            return 1

        def RequestData(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            ex = vtk.vtkStreamingDemandDrivenPipeline
            t = info.Get(ex.UPDATE_TIME_STEP()) if info.Has(ex.UPDATE_TIME_STEP()) else self.times[0]
            i = int(np.argmin(np.abs(self.times - t)))
            out = vtk.vtkImageData.GetData(outInfo)
            out.ShallowCopy(images[i])
            out.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), float(self.times[i]))
            return 1

    return Source()


def pathlines_vtk(velocity_t, voxel_to_world, rr_ms, seeds_mm, t0_ms=0.0, duration_ms=None,
                  mask=None, terminal_speed_mm_s=1.0,
                  temporal_upsample=8, max_step_frames=1.0):
    """
    Returns vtkPolyData of pathlines (point scalars 'time_ms', 'index').

    vtkParticlePathFilter advances one RK4 step per input frame and records
    one path point per frame, holding each frame's velocity over the step
    (it does not interpolate in time between the frames it is given), so
    the integration step and the temporal accuracy are both set by the
    frame spacing: at 25 ms frames and 0.8 m/s that is a 20 mm step. The
    frames are therefore linearly upsampled in time by `temporal_upsample`
    before they reach the tracer (default 8, i.e. 3 ms steps for a 24-frame
    600 ms cycle). The remaining error is first order in the step: on a
    6.4 mm-diameter rotating-flow orbit it is 1.0, 0.5, 0.25, 0.14, 0.08 mm
    for upsample 1, 2, 4, 8, 16; advect_pathlines (RK4 with linear temporal
    interpolation) gives 0.05 mm on the same frames. Frames wrap
    periodically so integration can run past one RR. Outside-mask velocity
    is zero and a particle slower than `terminal_speed_mm_s` is stopped
    (for non-orthogonal voxel axes the speed is measured in the axis-aligned
    space, so the threshold is approximate).

    t0_ms is the emission time: a scalar, or one value per seed (staggered
    emission), in which case one tracer run is made per distinct emission
    time and the results are appended (use advect_pathlines for many
    distinct emission times).
    """
    import vtk
    t0 = np.broadcast_to(np.asarray(t0_ms, float), (np.atleast_2d(seeds_mm).shape[0],))
    starts = np.unique(t0)
    if len(starts) == 1:
        return _pathlines_vtk_one(velocity_t, voxel_to_world, rr_ms, seeds_mm, float(starts[0]), duration_ms,
                                  mask, terminal_speed_mm_s, temporal_upsample, max_step_frames)
    seeds = np.atleast_2d(np.asarray(seeds_mm, float))
    app = vtk.vtkAppendPolyData()
    offset = 0
    pieces = []
    for t_start in starts:
        sel = np.flatnonzero(t0 == t_start)
        pd = _pathlines_vtk_one(velocity_t, voxel_to_world, rr_ms, seeds[sel], float(t_start), duration_ms,
                                mask, terminal_speed_mm_s, temporal_upsample, max_step_frames)
        idx = pd.GetPointData().GetArray("index")
        if idx is not None:
            from vtk.util import numpy_support as ns
            # particle ids are local to the run: map them to the seed indices
            local = ns.vtk_to_numpy(idx).astype(int)
            mapped = ns.numpy_to_vtk(sel[np.clip(local, 0, len(sel) - 1)].astype(float), deep=True)
            mapped.SetName("index")
            pd.GetPointData().RemoveArray("index"); pd.GetPointData().AddArray(mapped)
        pieces.append(pd)
        app.AddInputData(pd)
        offset += len(sel)
    app.Update()
    out = vtk.vtkPolyData(); out.DeepCopy(app.GetOutput())
    out.GetPointData().SetActiveScalars("time_ms")
    return out


def _pathlines_vtk_one(velocity_t, voxel_to_world, rr_ms, seeds_mm, t0_ms, duration_ms, mask,
                       terminal_speed_mm_s, temporal_upsample, max_step_frames):
    import vtk
    from vtk.util import numpy_support as ns
    m = as_voxel_to_world(voxel_to_world)
    spacing = voxel_spacing(m)
    to_aligned = world_to_axis_aligned(m)
    from_aligned = np.linalg.inv(to_aligned)
    # world velocity -> axis-aligned space (same units, mm/s)
    velocity_aligned = (np.asarray(velocity_t, float) @ to_aligned[:3, :3].T).astype(np.float32)
    nt0 = velocity_aligned.shape[0]
    duration = rr_ms if duration_ms is None else duration_ms
    dt_frame = rr_ms / (nt0 * int(temporal_upsample))

    # Pipeline time is in SECONDS so that mm/s velocities integrate to mm.
    # VTK 9.7's tracer has no start-time setting: it injects the seeds at
    # the first time step it is served and integrates from there. So the
    # served sequence starts exactly at t0: frames are interpolated linearly
    # and periodically from the acquired ones at t0 + k * dt_frame, for the
    # window the integration needs (float32; a 2-cycle float64 copy of a
    # large study exhausts memory).
    n_needed = int(np.ceil(duration / dt_frame)) + 2
    times_ms = t0_ms + np.arange(n_needed) * dt_frame
    frames = []
    for tm in times_ms:
        f = (tm % rr_ms) / (rr_ms / nt0)
        i0 = int(np.floor(f)) % nt0; w = np.float32(f - np.floor(f))
        frames.append((1 - w) * velocity_aligned[i0] + w * velocity_aligned[(i0 + 1) % nt0])
    images = _frames_to_vtk_images(np.stack(frames), spacing, mask)
    src = _temporal_source(images, times_ms / 1000.0)

    seeds = transform_points(to_aligned, np.atleast_2d(seeds_mm))
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(seeds), deep=True))
    seed_pd = vtk.vtkPolyData(); seed_pd.SetPoints(pts)

    tracer = vtk.vtkParticlePathFilter()
    tracer.SetInputConnection(0, src.GetOutputPort())
    tracer.SetInputData(1, seed_pd)
    tracer.SetTerminalSpeed(terminal_speed_mm_s)
    tracer.SetComputeVorticity(False)   # would be in the axis-aligned space
    tracer.SetIntegratorType(vtk.vtkParticleTracerBase.RUNGE_KUTTA4)
    tracer.SetForceReinjectionEveryNSteps(0)
    tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "velocity")
    t_start = t0_ms / 1000.0
    step = dt_frame * max_step_frames / 1000.0
    for t in np.arange(t_start, t_start + duration / 1000.0 + 1e-9, step):
        tracer.UpdateTimeStep(float(t))
    out = vtk.vtkPolyData()
    out.DeepCopy(tracer.GetOutputDataObject(0))

    # axis-aligned space -> world space
    if out.GetNumberOfPoints():
        aligned = ns.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
        world = vtk.vtkPoints()
        world.SetData(ns.numpy_to_vtk(np.ascontiguousarray(transform_points(from_aligned, aligned)), deep=True))
        out.SetPoints(world)
        vel = out.GetPointData().GetArray("velocity")
        if vel is not None:
            vw = ns.numpy_to_vtk(np.ascontiguousarray(
                ns.vtk_to_numpy(vel).astype(float) @ from_aligned[:3, :3].T), deep=True)
            vw.SetName("velocity")
            out.GetPointData().RemoveArray("velocity")
            out.GetPointData().AddArray(vw)

    # expose time_ms and index like pathlines.to_vtk_polydata
    pd = out.GetPointData()
    age = pd.GetArray("ParticleAge")
    if age is not None:
        tm = ns.numpy_to_vtk(ns.vtk_to_numpy(age) * 1000.0 + t0_ms, deep=True)
        tm.SetName("time_ms"); pd.AddArray(tm); pd.SetActiveScalars("time_ms")
    pid = pd.GetArray("ParticleId")
    if pid is not None:
        idx = ns.numpy_to_vtk(ns.vtk_to_numpy(pid).astype(float), deep=True)
        idx.SetName("index"); pd.AddArray(idx)
    return out
