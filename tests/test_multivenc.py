"""
Multi-venc (dual- and triple-venc) unwrapping. DRAFT: phantom tests only.

Cases
-----
exact        noise-free uniform random velocities: sdv exact within the high
             venc, odv exact within the effective venc, including a venc
             ratio (150/100) whose effective range (300) exceeds the high
             venc, and three encodings.
stenosis     the stenosis phantom (jet above venc in every frame) acquired
             with the phase-contrast forward model at several vencs with a
             shared reference (Schnell 2017 / Ma 2019 / Weiss, Roy 2024):
             - Löcke 2023's case: venc_low = 30% of the peak velocity, where
               single-venc unwrapping fails; triple venc (120/60/30%) must
               recover it and dual venc (120/30%) too;
             - Franco 2022's comparison: beta = 1/2, 1/3, 2/3 across noise
               levels; beta = 1/2 is the most robust ratio;
             - velocity noise of the combined image relative to the high
               venc (Weiss 2024: 85% VNR increase with beta = 1/2, i.e. a
               noise ratio near 0.54).
chest        preprocess() with extra_encodings on the thorax phantom: the
             multi-venc step runs before background correction, the vein
             keeps the low-venc noise, the report carries the section.

Run as a script for the full comparison table:
    python tests/test_multivenc.py
"""

import numpy as np
import pytest

from slicerheart_4dflow import phantom
from slicerheart_4dflow.multivenc import (effective_venc, ivenc, ivenc_image, multivenc_unwrap, noise_ratio,
                                           wrap_velocity)
from slicerheart_4dflow.unwrap import preprocess, unwrap_velocity


def failure_fraction(v, v_true, mask, venc_low):
    """Fraction of lumen voxel-frame-components still off by a wrap or more."""
    return float((np.abs(v - v_true)[:, mask] > venc_low).mean())


# ---------------------------------------------------------------------------
# Exact cases
# ---------------------------------------------------------------------------

def test_effective_venc_and_ivenc():
    np.testing.assert_allclose(effective_venc([150.0, 75.0]), 150.0)
    np.testing.assert_allclose(effective_venc([150.0, 50.0]), 150.0)
    np.testing.assert_allclose(effective_venc([150.0, 100.0]), 300.0)        # 3:2 extends the range
    np.testing.assert_allclose(effective_venc([120.0, 60.0, 30.0]), 120.0)
    np.testing.assert_allclose(effective_venc([[150, 150, 150], [75, 75, 100]]), [150, 150, 300])
    np.testing.assert_allclose(ivenc(75.0, 150.0), 150.0)                   # beta = 1/2: ivenc = venc_high
    np.testing.assert_allclose(ivenc(100.0, 150.0), 300.0)
    with pytest.raises(ValueError):
        ivenc(150.0, 100.0)


def test_sdv_and_odv_exact_without_noise():
    rng = np.random.default_rng(0)
    u = rng.uniform(-290.0, 290.0, (4, 6, 6, 6, 3))
    # beta = 1/2: both methods exact within the high venc
    vh, vl = 150.0, 75.0
    ul, uh = wrap_velocity(u, vl), wrap_velocity(u, vh)
    inside = np.abs(u) < vh
    v, k, info = multivenc_unwrap([ul, uh], [vl, vh], method="sdv")
    np.testing.assert_allclose(v[inside], u[inside], atol=1e-9)
    assert info["reference"] == ["high"] * 3
    v, k, info = multivenc_unwrap([ul, uh], [vl, vh], method="odv", correction=False)
    np.testing.assert_allclose(v[inside], u[inside], atol=1e-9)
    assert k.dtype.kind == "i" and info["effective_venc"] == [150.0] * 3
    # ivenc image reproduces the velocity at venc_i
    vi = ivenc(100.0, 150.0)
    np.testing.assert_allclose(ivenc_image(wrap_velocity(u, 100.0), 100.0, uh, vh), wrap_velocity(u, vi), atol=1e-9)
    # 150/100: odv unambiguous up to 300, beyond the aliased high venc
    vl = 100.0
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), uh], [vl, vh], method="odv", correction=False,
                                  return_cost=True)
    inside = np.abs(u) < 299.0
    np.testing.assert_allclose(v[inside], u[inside], atol=1e-9)
    assert info["margin"].min() > 0.3                                       # candidates well separated
    # sdv with ivenc (beta = 2/3 > 1/2): the ivenc image (venc 300) is the reference
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), uh], [vl, vh], method="sdv")
    assert info["reference"] == ["ivenc"] * 3
    np.testing.assert_allclose(v[inside], u[inside], atol=1e-9)
    # three encodings 120/60/30 (Löcke 2023's set): exact within 120
    vencs = [120.0, 60.0, 30.0]
    u3 = rng.uniform(-119.0, 119.0, (3, 5, 5, 5, 3))
    v, k, info = multivenc_unwrap([wrap_velocity(u3, c) for c in vencs], vencs, method="odv", correction=False)
    np.testing.assert_allclose(v, u3, atol=1e-9)
    assert info["venc_low"] == [30.0] * 3


def test_input_checks():
    a = np.zeros((2, 4, 4, 4, 3)); b = np.zeros((2, 4, 4, 5, 3))
    with pytest.raises(ValueError):
        multivenc_unwrap([a, b], [50.0, 100.0])
    with pytest.raises(ValueError):
        multivenc_unwrap([a], [50.0])
    with pytest.raises(ValueError):
        multivenc_unwrap([a, a], [100.0, 50.0], method="sdv")                # low venc must come first
    with pytest.raises(ValueError):
        multivenc_unwrap([a, a], [50.0, 100.0], method="nope")


# ---------------------------------------------------------------------------
# Stenosis phantom through the forward model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jet():
    ph = phantom.stenosis(nt=4, shape=(48, 32, 32), v_max_inlet_mm_s=800.0, severity=0.5)
    return ph["velocity"], ph["mask"], ph["voxel_to_world"], float(np.abs(ph["velocity"]).max())


def test_loecher_low_venc_case(jet, metrics):
    """venc_low = 30% of the peak, Löcke 2023's case where single-venc
    unwrapping degrades (double wraps); dual venc 120/30 and triple venc
    120/60/30 recover the field with the low-venc noise."""
    v_true, mask, vtw, vmax = jet
    vencs = [1.2 * vmax, 0.6 * vmax, 0.3 * vmax]
    acq = phantom.acquire_multivenc(v_true, vencs, mask=mask, snr=12.0, rng=np.random.default_rng(3))
    vl = vencs[2]
    single = unwrap_velocity(acq[2]["velocity"], vl, vtw, method="laplacian_then_temporal", mask=mask)
    f_single = failure_fraction(single, v_true, mask, vl)
    v_dual, k, info = multivenc_unwrap([acq[2]["velocity"], acq[0]["velocity"]], [vl, vencs[0]], method="odv",
                                       mask=mask)
    f_dual = failure_fraction(v_dual, v_true, mask, vl)
    v_tri, _, _ = multivenc_unwrap([a["velocity"] for a in acq], vencs, method="odv", mask=mask)
    f_tri = failure_fraction(v_tri, v_true, mask, vl)
    v_sdv, _, _ = multivenc_unwrap([acq[2]["velocity"], acq[0]["velocity"]], [vl, vencs[0]], method="sdv")
    f_sdv = failure_fraction(v_sdv, v_true, mask, vl)
    metrics.record("multivenc/loecher30/fail_single_laplacian", f_single, direction="record")
    metrics.record("multivenc/loecher30/fail_dual_odv", f_dual)
    metrics.record("multivenc/loecher30/fail_triple_odv", f_tri)
    metrics.record("multivenc/loecher30/fail_dual_sdv", f_sdv)
    # On this smooth phantom the Laplacian pass recovers most double wraps
    # (about 1% residual); the multi-venc estimators recover them without a
    # spatial model, which is what matters on real jets with partial volume.
    assert f_dual <= f_single and f_tri <= f_single
    assert f_dual < 0.005 and f_tri < 0.005 and f_sdv < 0.005
    assert np.abs(k).max() >= 2                                              # double wraps were recovered
    # the result keeps the low-venc noise: RMSE in the lumen close to that of
    # the noise-free-aliased low venc image, well below the high venc image
    rmse = lambda x: float(np.sqrt(((x - v_true)[:, mask] ** 2).mean()))
    assert rmse(v_tri) < 0.5 * rmse(acq[0]["velocity"])


def test_franco_ratio_comparison(jet, metrics):
    """Franco 2022: beta = 1/2 is the most robust venc ratio, and sdv, odv and
    corrected odv behave alike at moderate noise."""
    v_true, mask, vtw, vmax = jet
    vh = 1.2 * vmax
    fails = {}
    for beta in (0.5, 1 / 3, 2 / 3):
        vl = beta * vh
        acq = phantom.acquire_multivenc(v_true, [vl, vh], mask=mask, snr=2.5, rng=np.random.default_rng(4))
        series = [acq[0]["velocity"], acq[1]["velocity"]]
        for method, kw in (("sdv", {}), ("odv", {"correction": False}), ("odv_corrected", {"correction": True})):
            v, _, _ = multivenc_unwrap(series, [vl, vh], method=method.split("_")[0], mask=mask, **kw)
            fails[(beta, method)] = failure_fraction(v, v_true, mask, vl)
            metrics.record(f"multivenc/franco/beta{beta:.2f}/fail_{method}", fails[(beta, method)], atol=0.002)
    for method in ("sdv", "odv", "odv_corrected"):
        assert fails[(0.5, method)] <= fails[(1 / 3, method)] + 1e-6
        assert fails[(0.5, method)] <= fails[(2 / 3, method)] + 1e-6
    assert fails[(0.5, "odv_corrected")] <= fails[(0.5, "sdv")] + 1e-6


def test_noise_gain_and_shared_reference(jet, metrics):
    """Static-tissue velocity noise of the combined image relative to the high
    venc is close to beta (Weiss 2024 report an 85% VNR increase in vivo for
    beta = 1/2, a ratio of about 0.54), and the shared reference correlates
    the noise of the two encodings."""
    v_true, mask, vtw, vmax = jet
    vh, vl = 1.2 * vmax, 0.6 * vmax
    static = ~mask
    for shared in (True, False):
        # unit magnitude everywhere: tissue with signal and zero velocity
        # outside the lumen, where the noise is measured
        acq = phantom.acquire_multivenc(v_true, [vl, vh], magnitude=np.ones(mask.shape), snr=10.0,
                                        shared_reference=shared, rng=np.random.default_rng(5))
        v, _, _ = multivenc_unwrap([acq[0]["velocity"], acq[1]["velocity"]], [vl, vh], method="odv", mask=mask)
        ratio = noise_ratio(v, acq[1]["velocity"], static)
        metrics.record(f"multivenc/noise_ratio_shared{int(shared)}", ratio, direction="exact", tol=0.1)
        assert 0.4 < ratio < 0.65
        n_low = (acq[0]["velocity"] - v_true)[:, static, 0].ravel()
        n_high = (acq[1]["velocity"] - v_true)[:, static, 0].ravel()
        corr = float(np.corrcoef(n_low, n_high)[0, 1])
        if shared:
            assert corr > 0.3                                                # common reference noise
        else:
            assert abs(corr) < 0.1


# ---------------------------------------------------------------------------
# preprocess() with extra encodings
# ---------------------------------------------------------------------------

def test_preprocess_extra_encodings(metrics):
    ph = phantom.chest(shape=(24, 64, 64), nt=8, rng=np.random.default_rng(6), wrap=False)
    v_true = ph["velocity_true"]; vtw = ph["voxel_to_world"]
    peak = float(np.abs(v_true).max())
    vh, vl = 1.1 * peak, 0.55 * peak
    rng = np.random.default_rng(7)
    acq = phantom.acquire_multivenc(v_true + ph["offset"][None], [vl, vh], magnitude=ph["magnitude"][0],
                                    snr=15.0, rng=rng)
    mag_t = np.repeat(acq[0]["magnitude"][None], v_true.shape[0], 0) if acq[0]["magnitude"].ndim == 3 \
        else acq[0]["magnitude"]
    out = preprocess(acq[0]["velocity"], mag_t, vl, vtw, lumen_mask=ph["lumen"], background_method="ols",
                     extra_encodings=[(acq[1]["velocity"], vh)], unwrap_kwargs={"method": "odv"}, verbose=False)
    rep = out["report"]
    assert "multivenc" in rep and rep["multivenc"]["method"] == "odv"
    assert rep["unwrap"]["method"] is None and "wrap_count" in out
    assert "multivenc" in rep["timing_s"]
    err = out["velocity"] - v_true
    body = ph["body"] & ~ph["lungs"]
    rmse_body = float(np.sqrt((err[:, body] ** 2).mean()))
    # the combined image has the low-venc noise: compare with the high-venc
    # series put through the same pipeline
    ref = preprocess(acq[1]["velocity"], mag_t, vh, vtw, lumen_mask=ph["lumen"], background_method="ols",
                     unwrap_method="temporal", verbose=False)
    rmse_ref = float(np.sqrt(((ref["velocity"] - v_true)[:, body] ** 2).mean()))
    metrics.record("multivenc/chest/rmse_body_mm_s", rmse_body)
    metrics.record("multivenc/chest/rmse_ratio_to_high_venc", rmse_body / rmse_ref)
    assert rmse_body < 0.75 * rmse_ref
    # no wraps left in the artery
    assert failure_fraction(out["velocity"], v_true, ph["lumen"], vl) < 0.01
    # the quality report takes the section
    from slicerheart_4dflow.report import quality_report, to_markdown
    q = quality_report(acq[0]["velocity"], out["velocity"], mag_t, vl, vtw, lumen_mask=ph["lumen"],
                       wrap_count=out["wrap_count"], timing=rep["timing_s"],
                       multivenc=dict(rep["multivenc"], velocity_high_t=acq[1]["velocity"]))
    assert "noise_ratio_to_high_venc" in q["multivenc"] and 0.3 < q["multivenc"]["noise_ratio_to_high_venc"] < 0.8
    assert "Multi-venc" in to_markdown(q)


if __name__ == "__main__":
    ph = phantom.stenosis(nt=4, shape=(48, 32, 32), v_max_inlet_mm_s=800.0, severity=0.5)
    v_true, mask, vmax = ph["velocity"], ph["mask"], float(np.abs(ph["velocity"]).max())
    print(f"peak velocity {vmax:.0f} mm/s")
    print("beta   snr   fail_sdv  fail_odv  fail_odv_corrected  noise_ratio")
    for beta in (0.5, 1 / 3, 2 / 3):
        vh = 1.2 * vmax; vl = beta * vh
        for snr in (10.0, 5.0, 3.5, 2.5):
            acq = phantom.acquire_multivenc(v_true, [vl, vh], magnitude=np.ones(mask.shape), snr=snr,
                                            rng=np.random.default_rng(4))
            s = [acq[0]["velocity"], acq[1]["velocity"]]
            row = []
            for method, kw in (("sdv", {}), ("odv", {"correction": False}), ("odv", {"correction": True})):
                v, _, _ = multivenc_unwrap(s, [vl, vh], method=method, mask=mask, **kw)
                row.append(failure_fraction(v, v_true, mask, vl))
            print(f"{beta:.2f}  {snr:5.1f}  {row[0]:.4f}    {row[1]:.4f}    {row[2]:.4f}              "
                  f"{noise_ratio(v, acq[1]['velocity'], ~mask):.2f}")
