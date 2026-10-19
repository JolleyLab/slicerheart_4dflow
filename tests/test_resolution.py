"""
Resolution sweep: how WSS, relative pressure, flow and the divergence-free
filters degrade with voxel size.

Phantoms are generated at 0.5 mm and acquired at 1.0, 1.5, 2.0 and 2.5 mm by
complex block averaging (partial volume as a scanner sees it), noise-free, so
that the curves isolate resolution. The CFD field (0.59 mm) is cropped and
averaged to 1.2, 1.8 and 2.4 mm with 10% venc noise added at the target
resolution. Every point of every curve is a regression metric; the
assertions only pin the shape of the curves (degradation with voxel size)
and generous absolute bounds, since the point is to track the numbers.

Run as a script to print the tables:
    python tests/test_resolution.py
"""

import numpy as np
import pytest

from slicerheart_4dflow.phantom import poiseuille, womersley, acquire, block_average, add_noise
from slicerheart_4dflow.geometry import voxel_to_world_matrix
from slicerheart_4dflow.wss import wall_shear_stress
from slicerheart_4dflow.pressure import relative_pressure
from slicerheart_4dflow.qa import plane_flow
from slicerheart_4dflow.divergence_free import divergence_free_filter

FINE_MM = 0.5
FINE_SHAPE = (60, 120, 120)          # divisible by 2, 3, 4, 5 -> 1.0, 1.5, 2.0, 2.5 mm
FACTORS = {1.0: 2, 1.5: 3, 2.0: 4, 2.5: 5}
RADIUS_MM = 12.0


def downsample_phantom(ph, factor):
    """Acquire a fine phantom at `factor` x coarser voxels, noise-free.
    Returns velocity (nt, ..., 3), lumen mask (fraction > 0.5), voxel_to_world."""
    v_t = ph["velocity"]
    out = [acquire(v_t[t], 1e6, mask=ph["mask"], snr=1e12, downsample=factor)["velocity"]
           for t in range(v_t.shape[0])]
    mask = block_average(ph["mask"].astype(float), factor) > 0.5
    vtw = ph["voxel_to_world"].copy()
    vtw[:3, :3] *= factor
    vtw[:3, 3] += (factor - 1) / 2.0 * np.diag(ph["voxel_to_world"])[:3]      # centre of the block
    return np.stack(out), mask, vtw


def _interior(points, axis):
    z = (points - points.mean(0)) @ axis
    return np.abs(z) < 0.35 * np.abs(z).max()


@pytest.fixture(scope="module")
def fine_poiseuille():
    return poiseuille(shape=FINE_SHAPE, voxel_to_world=voxel_to_world_matrix(spacing=(FINE_MM,) * 3),
                      radius_mm=RADIUS_MM, v_max_mm_s=800.0)


@pytest.fixture(scope="module")
def fine_womersley():
    return womersley(shape=FINE_SHAPE, voxel_to_world=voxel_to_world_matrix(spacing=(FINE_MM,) * 3),
                     radius_mm=RADIUS_MM, nt=16)


def sweep_poiseuille(ph):
    rows = {}
    for mm, f in FACTORS.items():
        v, m, vtw = downsample_phantom(ph, f)
        n = ph["axis_world"]
        out = wall_shear_stress(v, m, vtw)
        sel = _interior(out["points"], n)
        wss_ratio = np.median(out["wss_magnitude"][0][sel]) / ph["wss_pa"][0]
        P, _ = relative_pressure(v, m, vtw, 800.0)
        # axial slope on interior voxels: use world z and radial distance recomputed from the coarse grid
        from slicerheart_4dflow.geometry import voxel_grid_world_points
        pts = voxel_grid_world_points(m.shape, vtw); c = pts.reshape(-1, 3).mean(0)
        d = pts - c; z = d @ n; rad = np.linalg.norm(d - z[..., None] * n, axis=-1)
        inner = m & (rad < 0.7 * RADIUS_MM)
        slope = np.polyfit(z[inner], P[0][inner], 1)[0]
        p_ratio = slope / ph["pressure_gradient_pa_mm"][0]
        q = plane_flow(v, vtw, centre=tuple(c), normal=tuple(n), radius=RADIUS_MM + 3, lumen_mask=m)["flow"][0]
        q_ratio = q / (np.pi * RADIUS_MM ** 2 * 800.0 / 2)
        rows[mm] = dict(wss_ratio=wss_ratio, pressure_ratio=p_ratio, flow_ratio=q_ratio, lumen_voxels=int(m.sum()))
    return rows


def sweep_womersley(ph):
    rows = {}
    for mm, f in FACTORS.items():
        v, m, vtw = downsample_phantom(ph, f)
        out = wall_shear_stress(v, m, vtw)
        sel = _interior(out["points"], ph["axis_world"])
        sgn = np.sign((out["wss"][:, sel] @ ph["axis_world"]).mean(1))
        est = np.median(out["wss_magnitude"][:, sel], axis=1) * sgn
        rows[mm] = dict(wss_slope=np.polyfit(ph["wss_pa"], est, 1)[0],
                        wss_corr=np.corrcoef(est, ph["wss_pa"])[0, 1],
                        osi=float(np.median(out["osi"][sel])))
    return rows


def test_poiseuille_resolution_curve(fine_poiseuille, metrics):
    rows = sweep_poiseuille(fine_poiseuille)
    for mm, r in rows.items():
        metrics.record(f"resolution/poiseuille/{mm:.1f}mm/wss_ratio", r["wss_ratio"], direction="exact", atol=0.02)
        metrics.record(f"resolution/poiseuille/{mm:.1f}mm/pressure_ratio", r["pressure_ratio"], direction="exact", atol=0.02)
        metrics.record(f"resolution/poiseuille/{mm:.1f}mm/flow_ratio", r["flow_ratio"], direction="exact", atol=0.01)
        assert abs(r["flow_ratio"] - 1) < 0.05                 # flow is robust to resolution
        assert 0.6 < r["wss_ratio"] < 1.3
        assert 0.6 < r["pressure_ratio"] < 1.1


def test_womersley_resolution_curve(fine_womersley, metrics):
    rows = sweep_womersley(fine_womersley)
    slopes = [rows[mm]["wss_slope"] for mm in sorted(rows)]
    for mm, r in rows.items():
        metrics.record(f"resolution/womersley/{mm:.1f}mm/wss_slope", r["wss_slope"], direction="exact", atol=0.02)
        metrics.record(f"resolution/womersley/{mm:.1f}mm/osi", r["osi"], direction="exact", atol=0.01)
        assert r["wss_corr"] > 0.8                             # time course survives at every resolution
    assert slopes[0] > slopes[-1]                              # underestimation grows with voxel size
    assert slopes[0] > 0.4 and slopes[-1] > 0.1


# ---------------------------------------------------------------------------
# CFD
# ---------------------------------------------------------------------------

CFD_FACTORS = (2, 3, 4)      # 0.59 mm -> 1.19, 1.78, 2.38 mm


def cfd_at(cfd, factor, sigma, rng):
    v, m, vtw = cfd
    v = v[:84, :72, :72] * 1000.0; m = m[:84, :72, :72]                # mm/s, crop divisible by 2, 3, 4
    S = np.stack([block_average(v[..., c], factor) for c in range(3)], -1)
    mk = block_average(m.astype(float), factor) > 0.5
    vt = vtw.copy(); vt[:3, :3] *= factor
    noisy = add_noise(S, sigma, mk, rng)
    return S, noisy, mk, vt


@pytest.mark.cfd
def test_cfd_filter_resolution_curve(cfd, metrics):
    rng = np.random.default_rng(0)
    sigma = 0.10 * 1500.0
    for f in CFD_FACTORS:
        v_true, v_noisy, m, vtw = cfd_at(cfd, f, sigma, rng)
        mm = 0.59375 * f
        rmse = lambda a: float(np.sqrt(np.mean((a - v_true)[m] ** 2))) / 10   # cm/s
        row = {"noisy": rmse(v_noisy)}
        for method in ("fdm", "dfw"):
            out, _ = divergence_free_filter(v_noisy, m, vtw, method=method)
            row[method] = rmse(out)
            metrics.record(f"resolution/cfd/{mm:.2f}mm/{method}/rmse_cm_s", row[method])
        assert row["fdm"] < row["noisy"] and row["dfw"] < row["fdm"]
        assert row["dfw"] < 8.0                                    # dfw keeps working at 2.4 mm


def main():
    import conftest
    fp = poiseuille(shape=FINE_SHAPE, voxel_to_world=voxel_to_world_matrix(spacing=(FINE_MM,) * 3), radius_mm=RADIUS_MM)
    fw = womersley(shape=FINE_SHAPE, voxel_to_world=voxel_to_world_matrix(spacing=(FINE_MM,) * 3), radius_mm=RADIUS_MM, nt=16)
    print(f"{'voxel mm':>9s} {'WSS/true':>9s} {'dp/true':>8s} {'Q/true':>7s} {'Wom slope':>10s} {'Wom OSI':>8s}")
    rp, rw = sweep_poiseuille(fp), sweep_womersley(fw)
    for mm in FACTORS:
        print(f"{mm:9.1f} {rp[mm]['wss_ratio']:9.2f} {rp[mm]['pressure_ratio']:8.2f} {rp[mm]['flow_ratio']:7.3f} "
              f"{rw[mm]['wss_slope']:10.2f} {rw[mm]['osi']:8.3f}")
    try:
        cfd = conftest.load_cfd()
    except Exception as exc:
        print("CFD unavailable:", exc); return
    rng = np.random.default_rng(0)
    print(f"\n{'voxel mm':>9s} {'noisy':>7s} {'fdm':>7s} {'dfw':>7s}   RMSE cm/s at 10% venc noise")
    for f in CFD_FACTORS:
        v_true, v_noisy, m, vtw = cfd_at(cfd, f, 150.0, rng)
        r = lambda a: float(np.sqrt(np.mean((a - v_true)[m] ** 2))) / 10
        print(f"{0.59375*f:9.2f} {r(v_noisy):7.2f} {r(divergence_free_filter(v_noisy, m, vtw, method='fdm')[0]):7.2f} "
              f"{r(divergence_free_filter(v_noisy, m, vtw, method='dfw')[0]):7.2f}")


if __name__ == "__main__":
    main()
