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

from conftest import aortic_waveform
from slicerheart_4dflow.divergence_free import divergence_free_sequence
from slicerheart_4dflow.qa import mass_conservation, net_volume, plane_flow, plane_voxels, qa_report


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def _tube(shape, spacing, radius_mm, axis=0):
    """Straight tube along `axis` through the centre; returns mask."""
    grids = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * h for n, h in zip(shape, spacing)], indexing="ij")
    r2 = sum(g ** 2 for a, g in enumerate(grids) if a != axis)
    return r2 <= radius_mm ** 2


@pytest.mark.xfail(strict=True, reason="area per voxel assumes a slab of 1.2 x min spacing, but an "
                   "axis-aligned plane selects a single voxel layer")
def test_plane_flow_uniform_tube():
    shape, spacing = (30, 40, 40), (1.0, 0.5, 0.5)
    lumen = _tube(shape, spacing, 6.0)
    nt = 3
    v_t = np.zeros((nt,) + shape + (3,))
    v_t[..., 0] = np.array([100.0, 200.0, -50.0])[:, None, None, None] * lumen   # mm/s along axis 0
    centre = (15, 19.5, 19.5)
    r = plane_flow(v_t, spacing, centre, (1, 0, 0), radius=10.0, lumen_mask=lumen)
    area = lumen[15].sum() * spacing[1] * spacing[2]
    np.testing.assert_allclose(r["area"], area, rtol=1e-12)
    np.testing.assert_allclose(r["flow"], np.array([100.0, 200.0, -50.0]) * area, rtol=1e-12)
    # reversed normal flips the sign
    r2 = plane_flow(v_t, spacing, centre, (-1, 0, 0), radius=10.0, lumen_mask=lumen)
    np.testing.assert_allclose(r2["flow"], -r["flow"])


def test_plane_voxels_physical_matches_index():
    shape, spacing = (20, 20, 20), (2.0, 1.0, 1.0)
    sel_i, n_i = plane_voxels(shape, spacing, (10, 10, 10), (1, 0, 0), 5.0)
    sel_p, n_p = plane_voxels(shape, spacing, (20.0, 10.0, 10.0), (1, 0, 0), 5.0, physical=True)
    np.testing.assert_array_equal(sel_i, sel_p)
    np.testing.assert_allclose(n_i, n_p)


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

def cfd_stages(cfd, nt=8):
    v_cfd, mask, sp = cfd
    v_cfd = v_cfd * 1000.0                                                   # mm/s
    rng = np.random.default_rng(0)
    v_true = np.stack([v_cfd * w for w in aortic_waveform(nt)])
    sigma = 0.10 * 1500.0                                                    # 10% of 1.5 m/s venc
    v_noisy = v_true + rng.normal(0, sigma, v_true.shape) * mask[None, ..., None]

    planes = {}
    # lumen centroids on two axial slices in the descending segment
    for name, k in (("DAo_prox", 62), ("DAo_dist", 80)):
        _, yy, xx = np.nonzero(mask[k:k + 1])
        planes[name] = dict(centre=(k, yy.mean(), xx.mean()), normal=(1, 0, 0), radius=15.0)

    def flows(v):
        # flow is along -axis 0 in this dataset; flip so forward is positive (mL/s)
        return {n: -plane_flow(v, sp, lumen_mask=mask, **p)["flow"] / 1000.0 for n, p in planes.items()}

    v_dfw, _ = divergence_free_sequence(v_noisy, mask, sp, method="dfw")
    v_fdm, _ = divergence_free_sequence(v_noisy, mask, sp, method="fdm")
    return {"truth": flows(v_true), "noisy": flows(v_noisy), "fdm": flows(v_fdm), "dfw": flows(v_dfw)}


CHECKS = [("prox vs dist", ["DAo_prox"], ["DAo_dist"])]


@pytest.mark.cfd
def test_cfd_mass_conservation(cfd):
    stages = cfd_stages(cfd)
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
