"""
Analytic phantoms: flow, pressure gradient and wall shear stress are
consistent with the closed-form solutions, under every image geometry.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES
from slicerheart_4dflow.phantom import poiseuille, womersley, chest, alias, add_noise, BLOOD_VISCOSITY_PA_S
from slicerheart_4dflow.qa import plane_flow
from slicerheart_4dflow.geometry import voxel_to_world_points


def _centre(shape, vtw):
    c = (np.array(shape)[::-1] - 1) / 2.0            # (i, j, k)
    return voxel_to_world_points(vtw, c[None])[0]


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_poiseuille_flow_matches_analytic(geometry):
    vtw = GEOMETRIES[geometry]()
    n = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])      # cylinder along the k axis
    p = poiseuille(shape=(32, 48, 48), voxel_to_world=vtw, radius_mm=10.0, v_max_mm_s=800.0, axis_world=n)
    q = plane_flow(p["velocity"], vtw, centre=_centre((32, 48, 48), vtw), normal=n, radius=14.0,
                   lumen_mask=p["mask"])["flow"][0]
    assert abs(q / (np.pi * 10.0 ** 2 * 400.0) - 1) < 0.03


def test_poiseuille_wss_and_gradient():
    p = poiseuille(radius_mm=12.0, v_max_mm_s=800.0)
    assert np.isclose(p["wss_pa"][0], 2 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012)
    assert np.isclose(p["pressure_gradient_pa_mm"][0], -4 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012 ** 2 * 1e-3)


def test_womersley_steady_limit_is_poiseuille():
    A = 4 * BLOOD_VISCOSITY_PA_S * 0.8 / 0.012 ** 2
    w = womersley(harmonics=((0, A),), nt=2)
    p = poiseuille(v_max_mm_s=800.0)
    assert np.allclose(w["velocity"][0], p["velocity"][0], atol=1e-6)
    assert np.isclose(w["wss_pa"][0], p["wss_pa"][0])


def test_womersley_flow_matches_analytic():
    w = womersley()
    shape = w["velocity"].shape[1:4]
    q = plane_flow(w["velocity"], w["voxel_to_world"], centre=_centre(shape, w["voxel_to_world"]),
                   normal=w["axis_world"], radius=14.0, lumen_mask=w["mask"])["flow"]
    # 1 mm voxels discretise the thin Womersley boundary layer; 2% tolerance
    assert np.max(np.abs(q - w["flow_mm3_s"])) < 0.02 * np.abs(w["flow_mm3_s"]).max()
    assert w["velocity"].max() > 500 and w["velocity"].min() < 0      # pulsatile with reverse flow


def test_chest_phantom_structure():
    c = chest(nt=6)
    assert c["velocity"].shape == (6, 40, 96, 96, 3)
    assert c["static"].sum() > 50000 and not (c["static"] & c["lumen"]).any()
    assert not (c["static"] & c["wrap_band"]).any()
    # measured = true + offset outside the wrap band
    out = ~c["wrap_band"]
    assert np.allclose(c["velocity"][:, out], (c["velocity_true"] + c["offset"][None])[:, out])


def test_alias_roundtrip():
    p = poiseuille()
    wrapped, k = alias(p["velocity"], 500.0)
    assert np.abs(wrapped).max() < 500.0
    assert np.allclose(wrapped + 1000.0 * k, p["velocity"])
    assert k.max() == 1


def test_add_noise_respects_mask():
    p = poiseuille()
    v = add_noise(p["velocity"], 50.0, mask=p["mask"], rng=np.random.default_rng(1))
    assert np.allclose(v[:, ~p["mask"]], 0.0)
    assert abs((v - p["velocity"])[:, p["mask"]].std() - 50.0) < 2.0


# ---------------------------------------------------------------------------
# Acquisition model
# ---------------------------------------------------------------------------

from slicerheart_4dflow.phantom import acquire, encode, decode, block_average   # noqa: E402


@pytest.mark.parametrize("snr", [10.0, 20.0, 50.0])
def test_acquire_velocity_noise_follows_pelc(snr):
    """sigma_v = sqrt(2) venc / (pi SNR) in high-SNR voxels (Pelc 1991)."""
    venc = 1500.0
    a = acquire(np.zeros((40, 40, 40, 3)), venc, snr=snr, rng=np.random.default_rng(1))
    expected = np.sqrt(2) * venc / (np.pi * snr)
    assert abs(a["velocity"].std() / expected - 1) < 0.03
    assert np.allclose(a["velocity_noise_mm_s"], expected)
    assert abs(a["magnitude"].mean() - 1.0) < 0.02


def test_acquire_air_has_random_phase():
    m = np.zeros((40, 40, 40), bool); m[10:30, 10:30, 10:30] = True
    a = acquire(np.zeros(m.shape + (3,)), 1500.0, mask=m, snr=20.0, rng=np.random.default_rng(2))
    # air magnitude 0.05 at SNR 20 gives SNR 1 there: velocity spread approaches uniform on +-venc
    assert a["velocity"][~m].std() > 0.5 * 1500.0 / np.sqrt(3)
    assert a["velocity"][m].std() < 40.0
    assert a["magnitude"][~m].mean() < 0.3 * a["magnitude"][m].mean()


def test_acquire_aliasing_round_trip():
    p = poiseuille(v_max_mm_s=800.0)
    a = acquire(p["velocity"][0], 500.0, mask=p["mask"], snr=1e9)
    assert np.abs(a["velocity"]).max() < 500.0
    assert a["wrap_count"].max() == 1
    assert np.allclose(a["velocity"] + 1000.0 * a["wrap_count"], p["velocity"][0], atol=1e-3)


def test_encode_decode_identity_and_block_average():
    p = poiseuille(shape=(32, 96, 96), voxel_to_world=np.diag([0.5, 0.5, 0.5, 1.0]), radius_mm=12.0)
    S = encode(p["velocity"][0], 1500.0, mask=p["mask"])
    v, mag = decode(S, 1500.0)
    assert np.allclose(v, p["velocity"][0], atol=1e-9)
    assert S.shape == (4, 32, 96, 96) and block_average(S, 2).shape == (4, 16, 48, 48)
    with pytest.raises(ValueError):
        block_average(S, 5)


def test_acquire_partial_volume_is_complex_averaging():
    """At a wall voxel half inside the lumen, complex averaging gives the
    lumen velocity weighted by signal (not diluted by the zero-velocity,
    low-signal outside), which is how a scanner voxel behaves."""
    p = poiseuille(shape=(32, 96, 96), voxel_to_world=np.diag([0.5, 0.5, 0.5, 1.0]), radius_mm=12.0)
    a = acquire(p["velocity"][0], 1500.0, mask=p["mask"], snr=1e9, downsample=2)
    v_avg = np.stack([block_average(p["velocity"][0][..., c], 2) for c in range(3)], -1)
    frac = block_average(p["mask"].astype(float), 2)
    edge = (frac > 0.2) & (frac < 0.8)
    assert edge.sum() > 100
    assert np.allclose(a["velocity"][edge, 2], v_avg[edge, 2] / frac[edge], rtol=0.15)   # signal-weighted
    assert a["velocity"][edge, 2].mean() > 1.3 * v_avg[edge, 2].mean()                 # not diluted
    assert np.allclose(a["magnitude"][edge], 0.05 + 0.95 * frac[edge], atol=0.05)


def test_noise_mask_and_static_tissue_on_acquired_data():
    """The downstream selectors work on the acquisition model's noise."""
    from slicerheart_4dflow.unwrap import noise_mask
    from slicerheart_4dflow.background_phase import static_tissue_candidates
    c = chest(nt=6, wrap=False)
    rng = np.random.default_rng(3)
    nt = c["velocity_true"].shape[0]
    mag = np.where(c["body"], 1.0, 0.05); mag[c["lungs"]] = 0.1; mag[c["lumen"]] = 1.5
    acq = [acquire(c["velocity_true"][t], 1500.0, magnitude=mag, snr=15.0, rng=rng) for t in range(nt)]
    v_t = np.stack([a["velocity"] for a in acq]); m_t = np.stack([a["magnitude"] for a in acq])
    noise = noise_mask(m_t, v_t)
    assert noise[~c["body"]].mean() > 0.9 and noise[c["lumen"]].mean() < 0.05
    cand = static_tissue_candidates(m_t, v_t, lumen_mask=c["lumen"])
    assert (cand & c["lumen"]).sum() == 0 and cand[c["static"]].mean() > 0.2
