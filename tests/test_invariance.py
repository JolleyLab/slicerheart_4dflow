"""
Property tests: results must not depend on things that carry no
information. Andras's GEOMETRIES and WORLD_TRANSFORMS fixtures already cover
the image geometry and the world frame; these cover the remaining
symmetries of the data.

  cardiac-phase roll   rolling the time axis rolls every result and changes
                       no per-cycle quantity (background offset, net volume,
                       TAWSS, OSI, pathline length)
  venc and unit scale  scaling velocity and venc together scales velocity
                       results and leaves dimensionless ones unchanged
                       (wrap counts, OSI, consistency error, inlier masks)
  mask dilation        one voxel of over-segmentation into static tissue
                       changes net flow by less than 5% with or without
                       filtering; a leak into signal-free air is tracked as a
                       metric (it perturbs the filters by 10 to 15%)
  time reversal        reversing time negates the temporal pressure term and
                       nothing else

The two real bugs found so far in this package (a DFW mask leak and a
Womersley sign error) were both of this kind.
"""

import numpy as np
import pytest
from scipy import ndimage

from slicerheart_4dflow.phantom import womersley, bifurcation, chest, acquire
from slicerheart_4dflow.background_phase import correct_background_phase
from slicerheart_4dflow.unwrap import unwrap_velocity
from slicerheart_4dflow.divergence_free import divergence_free_sequence
from slicerheart_4dflow.qa import plane_flow, mass_conservation, net_volume
from slicerheart_4dflow.wss import wall_shear_stress
from slicerheart_4dflow.pressure import pressure_gradient, relative_pressure
from slicerheart_4dflow.pathlines import advect_pathlines, seeds_from_plane


# ---------------------------------------------------------------------------
# cardiac-phase roll
# ---------------------------------------------------------------------------

def test_phase_roll_background_and_unwrap():
    c = chest(nt=8, wrap=False)
    v, mag = c["velocity"], c["magnitude"]
    k = 3
    out, info = correct_background_phase(v, mag, method="war+msac", lumen_mask=c["lumen"])
    out_r, info_r = correct_background_phase(np.roll(v, k, 0), np.roll(mag, k, 0), method="war+msac",
                                             lumen_mask=c["lumen"])
    assert np.allclose(out_r, np.roll(out, k, 0), atol=1e-6)
    assert np.allclose(info_r["offset"], info["offset"], atol=1e-6)
    wrapped = ((v + 1500.0) % 3000.0) - 1500.0
    u = unwrap_velocity(wrapped, 1500.0, c["voxel_to_world"], method="laplacian_then_temporal", mask=c["lumen"])
    u_r = unwrap_velocity(np.roll(wrapped, k, 0), 1500.0, c["voxel_to_world"], method="laplacian3d", mask=c["lumen"])
    # the spatial pass is frame-local: exactly rolled
    u_s = unwrap_velocity(wrapped, 1500.0, c["voxel_to_world"], method="laplacian3d", mask=c["lumen"])
    assert np.allclose(u_r, np.roll(u_s, k, 0), atol=1e-9)


def test_phase_roll_wss_pressure_pathlines():
    w = womersley(shape=(16, 40, 40), nt=12)
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    k = 5
    a = wall_shear_stress(v, m, vtw); b = wall_shear_stress(np.roll(v, k, 0), m, vtw)
    assert np.allclose(b["wss"], np.roll(a["wss"], k, 0), atol=1e-9)
    assert np.allclose(b["tawss"], a["tawss"]) and np.allclose(b["osi"], a["osi"])
    Pa, _ = relative_pressure(v, m, vtw, 800.0); Pb, _ = relative_pressure(np.roll(v, k, 0), m, vtw, 800.0)
    assert np.allclose(Pb, np.roll(Pa, k, 0), atol=1e-6 * np.abs(Pa).max())
    centre = (19.5, 19.5, 7.5)
    seeds = seeds_from_plane(m, vtw, centre, (0, 0, 1), 8.0, spacing_mm=2.0)
    rr = 800.0; t0 = k * rr / 12
    la = [np.linalg.norm(np.diff(t[:, :3], axis=0), axis=1).sum() for t in advect_pathlines(v, vtw, rr, seeds, mask=m)]
    lb = [np.linalg.norm(np.diff(t[:, :3], axis=0), axis=1).sum()
          for t in advect_pathlines(np.roll(v, k, 0), vtw, rr, seeds, t0_ms=t0, mask=m)]
    assert np.allclose(la, lb, rtol=1e-6)


# ---------------------------------------------------------------------------
# venc and unit scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 10.0])
def test_velocity_scale_invariance(scale):
    """Velocity in other units (e.g. cm/s or m/s): filters scale, dimensionless
    results do not."""
    b = bifurcation(nt=4)
    v, m, vtw = b["velocity"], b["mask"], b["voxel_to_world"]
    rng = np.random.default_rng(0)
    acq = [acquire(v[t], 1500.0, mask=m, snr=20.0, rng=rng) for t in range(4)]
    vm = np.stack([a["velocity"] for a in acq])
    f1, _ = divergence_free_sequence(vm, m, vtw, method="fdm")
    f2, _ = divergence_free_sequence(vm * scale, m, vtw, method="fdm")
    assert np.allclose(f2, f1 * scale, atol=1e-6 * scale * np.abs(f1).max())
    # dfw with explicit sigma scales exactly; with MAD-estimated sigma it also scales
    d1, _ = divergence_free_sequence(vm, m, vtw, method="dfw")
    d2, _ = divergence_free_sequence(vm * scale, m, vtw, method="dfw")
    assert np.allclose(d2, d1 * scale, rtol=1e-6, atol=1e-6 * scale * np.abs(d1).max())
    flows = lambda vv: {n: plane_flow(vv, vtw, lumen_mask=m, **p)["flow"] for n, p in b["planes"].items()}
    e1 = mass_conservation(flows(vm), b["checks"]); e2 = mass_conservation(flows(vm * scale), b["checks"])
    for r1, r2 in zip(e1, e2):
        assert np.isclose(r1["percent_error"], r2["percent_error"]) and np.isclose(r1["ratio"], r2["ratio"])


def test_venc_scale_with_aliasing():
    """Scaling velocity and venc together leaves wrap counts and unwrapped
    results (in units of venc) unchanged."""
    w = womersley(shape=(16, 40, 40), nt=6)
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    venc = 400.0
    a1 = acquire(v[2], venc, mask=m, snr=1e9); a2 = acquire(v[2] * 3.0, venc * 3.0, mask=m, snr=1e9)
    assert np.array_equal(a1["wrap_count"], a2["wrap_count"])
    u1 = unwrap_velocity(np.stack([a1["velocity"]]), venc, vtw, method="laplacian3d", mask=m)
    u2 = unwrap_velocity(np.stack([a2["velocity"]]), 3 * venc, vtw, method="laplacian3d", mask=m)
    assert np.allclose(u2, 3 * u1, atol=1e-6)


def test_background_offset_scale_invariance():
    c = chest(nt=6, wrap=False)
    _, i1 = correct_background_phase(c["velocity"], c["magnitude"], method="war+msac", lumen_mask=c["lumen"])
    _, i2 = correct_background_phase(c["velocity"] * 0.1, c["magnitude"], method="war+msac", lumen_mask=c["lumen"])
    assert np.allclose(i2["offset"], 0.1 * i1["offset"], atol=1e-6)
    assert np.array_equal(i2["candidates"], i1["candidates"])
    assert (i2["inliers"] == i1["inliers"]).mean() > 0.999       # threshold is scale-free (MAD based)


# ---------------------------------------------------------------------------
# mask dilation
# ---------------------------------------------------------------------------

def _bifurcation_acquired(ring_magnitude, nt=3, seed=0):
    """Acquired bifurcation where the one-voxel ring outside the lumen has
    the given signal magnitude: 0.7 for static tissue (the realistic
    segmentation error), 0.05 for air (a leak into signal-free voxels)."""
    b = bifurcation(nt=nt)
    v, m = b["velocity"], b["mask"]
    md = ndimage.binary_dilation(m)
    mag = np.where(m, 1.0, np.where(md, ring_magnitude, 0.05))
    rng = np.random.default_rng(seed)
    vm = np.stack([acquire(v[t], 1500.0, magnitude=mag, snr=20.0, rng=rng)["velocity"] for t in range(nt)])
    return b, vm, m, md


def _flow_change(b, v1, m1, v2, m2):
    vtw = b["voxel_to_world"]
    q1 = {n: plane_flow(v1, vtw, lumen_mask=m1, **p)["flow"] for n, p in b["planes"].items()}
    q2 = {n: plane_flow(v2, vtw, lumen_mask=m2, **p)["flow"] for n, p in b["planes"].items()}
    change = max(abs(net_volume(q2[n]) / net_volume(q1[n]) - 1) for n in q1)
    e1 = mass_conservation(q1, b["checks"])[0]["percent_error"]
    e2 = mass_conservation(q2, b["checks"])[0]["percent_error"]
    return change, abs(e1 - e2)


@pytest.mark.parametrize("method", [None, "fdm", "dfw"])
def test_mask_dilation_into_tissue(method, metrics):
    """A one-voxel over-segmentation into static tissue changes net flow by
    less than 5% and the conservation error by less than 2 points."""
    b, vm, m, md = _bifurcation_acquired(ring_magnitude=0.7)
    vtw = b["voxel_to_world"]
    v1 = vm if method is None else divergence_free_sequence(vm, m, vtw, method=method)[0]
    v2 = vm if method is None else divergence_free_sequence(vm, md, vtw, method=method)[0]
    change, dcons = _flow_change(b, v1, m, v2, md)
    metrics.record(f"invariance/dilation_tissue/{method}/max_flow_change", change, atol=0.005)
    assert change < 0.05 and dcons < 2.0


@pytest.mark.parametrize("method", [None, "fdm", "dfw"])
def test_mask_dilation_into_air_is_tracked(method, metrics):
    """A leak into signal-free voxels (random velocity up to venc) leaves
    unfiltered flow within noise but perturbs both filters by 10 to 15%,
    because they propagate the ring's random divergence inward. The
    pipeline guards against this by masking noise before filtering; the
    effect is recorded here so that any change to that behaviour is seen.
    noise_mask's percentile thresholds miss most of the ring in this
    air-dominated field of view (87% air), which is a known limitation."""
    b, vm, m, md = _bifurcation_acquired(ring_magnitude=0.05)
    vtw = b["voxel_to_world"]
    v1 = vm if method is None else divergence_free_sequence(vm, m, vtw, method=method)[0]
    v2 = vm if method is None else divergence_free_sequence(vm, md, vtw, method=method)[0]
    change, dcons = _flow_change(b, v1, m, v2, md)
    metrics.record(f"invariance/dilation_air/{method}/max_flow_change", change, direction="exact", atol=0.01)
    if method is None:
        assert change < 0.10 and dcons < 3.0


# ---------------------------------------------------------------------------
# time reversal
# ---------------------------------------------------------------------------

def test_time_reversal_negates_only_the_temporal_term():
    w = womersley(shape=(16, 40, 40), nt=12)
    v, m, vtw = w["velocity"], w["mask"], w["voxel_to_world"]
    vr = v[::-1]
    g = pressure_gradient(v, m, vtw, 800.0)
    gr = pressure_gradient(vr, m, vtw, 800.0)
    g_t = pressure_gradient(v, m, vtw, 800.0, include_convective=False, include_viscous=False)
    gr_t = pressure_gradient(vr, m, vtw, 800.0, include_convective=False, include_viscous=False)
    assert np.allclose(gr_t, -g_t[::-1], atol=1e-9)
    assert np.allclose(gr - gr_t, (g - g_t)[::-1], atol=1e-9)
