"""
Pathlines with VTK's own time-varying particle tracer.

pathlines_vtk() wraps vtkParticlePathFilter, the tracer ParaView uses for
"Particle Pathlines". A small VTKPythonAlgorithmBase source serves the
velocity frames as a temporal pipeline; vtkTemporalInterpolator
interpolates between frames; the tracer integrates RK4 in parallel through
vtkSMPTools. Pure VTK, nothing to install in Slicer. Returns vtkPolyData
with 'time_ms' and 'index' point scalars, same as pathlines.to_vtk_polydata.

Note: VTK has no GPU pathline tracer. Its VTK-m accelerator covers steady
streamlines only and is not built into Slicer's VTK. This filter is the
fastest route available inside Slicer; thousands of seeds take seconds.

Conventions as elsewhere: velocity_t (nt, nz, ny, nx, 3) mm/s, components
(vz, vy, vx), spacing (dz, dy, dx) mm, rr_ms cycle length, seeds (n, 3) mm
(z, y, x). VTK coordinates are x, y, z; conversions are internal.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# VTK native
# ===========================================================================

def _frames_to_vtk_images(velocity_t, spacing, mask=None):
    """List of vtkImageData, one per frame, point vectors 'velocity' (x,y,z)."""
    import vtk
    from vtk.util import numpy_support as ns
    nt, nz, ny, nx, _ = velocity_t.shape
    images = []
    for t in range(nt):
        v = velocity_t[t]
        if mask is not None:
            v = v * mask[..., None]
        arr = np.ascontiguousarray(v[..., ::-1].reshape(-1, 3).astype(np.float32))   # (vx, vy, vz), x fastest
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(spacing[2], spacing[1], spacing[0])
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


def _upsample_time(velocity_t, factor):
    """Linear periodic temporal interpolation to factor x more frames."""
    if factor <= 1:
        return velocity_t
    nt = velocity_t.shape[0]
    out = np.empty((nt * factor,) + velocity_t.shape[1:], velocity_t.dtype)
    for i in range(nt):
        a, b = velocity_t[i], velocity_t[(i + 1) % nt]
        for k in range(factor):
            w = k / factor
            out[i * factor + k] = (1 - w) * a + w * b
    return out


def pathlines_vtk(velocity_t, spacing, rr_ms, seeds_mm, t0_ms=0.0, duration_ms=None,
                  mask=None, terminal_speed_mm_s=1.0,
                  temporal_upsample=4, max_step_frames=1.0):
    """
    Returns vtkPolyData of pathlines (point scalars 'time_ms', 'index').

    vtkParticlePathFilter advances one RK4 step per input frame and records
    one path point per frame, so the integration step is set by the frame
    spacing: at 25 ms frames and 0.8 m/s that is a 20 mm step. The frames
    are therefore linearly upsampled in time by `temporal_upsample` before
    they reach the tracer (default 4, i.e. 6 ms steps for a 24-frame
    600 ms cycle). Frames wrap periodically so integration can run past one
    RR. Outside-mask velocity is zero and a particle slower than
    `terminal_speed_mm_s` is stopped.
    """
    import vtk
    from vtk.util import numpy_support as ns
    velocity_t = _upsample_time(np.asarray(velocity_t, np.float32), int(temporal_upsample))
    nt = velocity_t.shape[0]
    duration = rr_ms if duration_ms is None else duration_ms
    dt_frame = rr_ms / nt

    # Pipeline time is in SECONDS so that mm/s velocities integrate to mm.
    # Serve only the frames the window needs, wrapping periodically, as
    # float32 (a 2-cycle float64 copy of a large study exhausts memory).
    n_needed = int(np.ceil((float(np.max(t0_ms)) + duration) / dt_frame)) + 2
    frame_idx = np.arange(n_needed) % nt
    times_s = np.arange(n_needed) * dt_frame / 1000.0
    images = _frames_to_vtk_images(velocity_t[frame_idx], spacing, mask)
    src = _temporal_source(images, times_s)
    # No vtkTemporalInterpolator: the tracer caches two frames and
    # interpolates between them itself (and the interpolator segfaults
    # downstream of a Python temporal source in VTK 9.7).

    seeds = np.asarray(seeds_mm, float)[:, ::-1]           # to x, y, z
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(seeds), deep=True))
    seed_pd = vtk.vtkPolyData(); seed_pd.SetPoints(pts)

    tracer = vtk.vtkParticlePathFilter()
    tracer.SetInputConnection(0, src.GetOutputPort())
    tracer.SetInputData(1, seed_pd)
    tracer.SetTerminalSpeed(terminal_speed_mm_s)
    tracer.SetIntegratorType(vtk.vtkParticleTracerBase.RUNGE_KUTTA4)
    tracer.SetForceReinjectionEveryNSteps(0)
    tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "velocity")
    # VTK >= 9.3: the pipeline time drives the tracer. Step through the
    # requested window; the filter injects at the first step and accumulates.
    # Path points are recorded at each requested time, so the step here is
    # the output sampling; the integrator substeps internally.
    t_start = float(np.min(t0_ms)) / 1000.0
    step = dt_frame * max_step_frames / 1000.0
    for t in np.arange(t_start, t_start + duration / 1000.0 + 1e-9, step):
        tracer.UpdateTimeStep(float(t))
    out = vtk.vtkPolyData()
    out.ShallowCopy(tracer.GetOutputDataObject(0))

    # expose time_ms and index like pathlines.to_vtk_polydata
    pd = out.GetPointData()
    age = pd.GetArray("ParticleAge")
    if age is not None:
        tm = ns.numpy_to_vtk(ns.vtk_to_numpy(age) * 1000.0 + float(np.min(t0_ms)), deep=True)
        tm.SetName("time_ms"); pd.AddArray(tm); pd.SetActiveScalars("time_ms")
    pid = pd.GetArray("ParticleId")
    if pid is not None:
        idx = ns.numpy_to_vtk(ns.vtk_to_numpy(pid).astype(float), deep=True)
        idx.SetName("index"); pd.AddArray(idx)
    return out
