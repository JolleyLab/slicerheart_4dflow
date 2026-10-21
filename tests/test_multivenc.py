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
from slicerheart_4dflow.multivenc import (effective_venc, error_summary, ivenc, ivenc_image, multivenc_unwrap,
                                           noise_ratio, wrap_velocity)
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
    # 150/100: odv unambiguous up to 300, beyond the aliased high venc, when
    # the search is widened to the effective venc (default: capped at 150)
    vl = 100.0
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), uh], [vl, vh], method="odv", correction=False,
                                  return_cost=True, max_velocity=300.0)
    inside = np.abs(u) < 299.0
    np.testing.assert_allclose(v[inside], u[inside], atol=1e-9)
    assert info["margin"].min() > 0.3                                       # candidates well separated
    assert info["unambiguous_venc"] == [300.0] * 3
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), uh], [vl, vh], method="odv", correction=False)
    assert info["effective_venc"] == [150.0] * 3 and np.abs(v).max() <= 150.0 + 1e-9
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
    """Franco 2022: with the full unambiguous range searched, beta = 1/2 is
    the most robust venc ratio, and sdv, odv and corrected odv behave alike
    at moderate noise. With the default cap at the high venc, odv at beta =
    2/3 has only two candidates in range and fewer failures, at the cost of
    the range beyond the high venc that the iVENC reference gives sdv."""
    v_true, mask, vtw, vmax = jet
    vh = 1.2 * vmax
    fails = {}
    variants = (("sdv", "sdv", {}),
                ("odv_full", "odv", {"correction": False, "max_velocity": "effective"}),
                ("odv_corrected_full", "odv", {"correction": True, "max_velocity": "effective"}),
                ("odv_capped", "odv", {"correction": True}),
                ("schnell", "schnell", {}))
    for beta in (0.5, 1 / 3, 2 / 3):
        vl = beta * vh
        acq = phantom.acquire_multivenc(v_true, [vl, vh], mask=mask, snr=2.5, rng=np.random.default_rng(4))
        series = [acq[0]["velocity"], acq[1]["velocity"]]
        for label, method, kw in variants:
            kw = dict(kw)
            if kw.get("max_velocity") == "effective":
                kw["max_velocity"] = effective_venc([vl, vh])
            v, _, _ = multivenc_unwrap(series, [vl, vh], method=method, mask=mask, **kw)
            fails[(beta, label)] = failure_fraction(v, v_true, mask, vl)
            metrics.record(f"multivenc/franco/beta{beta:.2f}/fail_{label}", fails[(beta, label)], atol=0.002)
    for label in ("sdv", "odv_full", "odv_corrected_full"):
        assert fails[(0.5, label)] <= fails[(1 / 3, label)] + 1e-6
        assert fails[(0.5, label)] <= fails[(2 / 3, label)] + 1e-6
    assert fails[(0.5, "odv_corrected_full")] <= fails[(0.5, "sdv")] + 1e-6
    assert fails[(0.5, "odv_capped")] <= fails[(0.5, "sdv")] + 1e-6
    # Schnell's dead bands trade missed wraps in the jet for fewer false
    # unwraps elsewhere; the two error types are recorded separately
    vl = 0.5 * vh
    acq = phantom.acquire_multivenc(v_true, [vl, vh], mask=mask, snr=2.5, rng=np.random.default_rng(4))
    series = [acq[0]["velocity"], acq[1]["velocity"]]
    wrapped = (acq[0]["wrap_count"] != 0)[:, mask]
    for label, method in (("sdv", "sdv"), ("schnell", "schnell"), ("odv_capped", "odv")):
        v, _, _ = multivenc_unwrap(series, [vl, vh], method=method, mask=mask)
        err = (np.abs(v - v_true)[:, mask] > vl)
        missed = float(err[wrapped].mean()); false = float(err[~wrapped].mean())
        metrics.record(f"multivenc/franco/beta0.50/missed_wraps_{label}", missed, atol=0.01)
        metrics.record(f"multivenc/franco/beta0.50/false_unwraps_{label}", false, atol=0.002)
        if label == "schnell":
            assert false <= fails[(0.5, "sdv")] + 1e-6                     # conservative where nothing wraps


def test_schnell_exact_and_dead_band():
    rng = np.random.default_rng(13)
    u = rng.uniform(-140.0, 140.0, (3, 6, 6, 6, 3))
    vh, vl = 150.0, 75.0
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), wrap_velocity(u, vh)], [vl, vh], method="schnell")
    np.testing.assert_allclose(v, u, atol=1e-9)
    # a difference in Schnell's dead band (between venc_low and 1.6 venc_low) is left alone
    low = np.full((2, 2, 2, 3), 70.0); high = low + 1.3 * vl
    v, k, info = multivenc_unwrap([low, high], [vl, vh], method="schnell")
    assert (k == 0).all()
    assert np.rint(1.3 * vl / (2 * vl)) == 1.0                              # sdv would unwrap it
    # Ma 2019's biconditional rule: with velocity images the iVENC image is
    # a function of the other two, so the rule equals sdv with the iVENC
    # reference (see the module docstring); checked here for 50/75 (iVENC 150)
    vh, vl = 75.0, 50.0
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), wrap_velocity(u, vh)], [vl, vh], method="sdv")
    assert info["reference"] == ["ivenc"] * 3
    np.testing.assert_allclose(v, u, atol=1e-9)


def test_preprocess_schnell_order(metrics):
    """Schnell 2017's order: background per encoding, single-venc unwrapping
    of the high-venc series, then the threshold combination."""
    ph = phantom.chest(shape=(16, 48, 48), nt=6, rng=np.random.default_rng(14), wrap=False)
    v_true = ph["velocity_true"]; vtw = ph["voxel_to_world"]
    peak = float(np.abs(v_true).max()); vh, vl = 1.1 * peak, 0.55 * peak
    acq = phantom.acquire_multivenc(v_true + ph["offset"][None], [vl, vh], magnitude=ph["magnitude"][0],
                                    snr=15.0, rng=np.random.default_rng(15))
    out = preprocess(acq[0]["velocity"], acq[0]["magnitude"], vl, vtw, lumen_mask=ph["lumen"],
                     background_method="ols", extra_encodings=[(acq[1]["velocity"], vh)], background_first=True,
                     unwrap_kwargs={"method": "schnell", "reference_unwrap": "temporal"}, verbose=False)
    mv = out["report"]["multivenc"]
    assert mv["method"] == "schnell" and mv["reference_unwrap"] == "temporal" and mv["background_first"]
    body = ph["body"] & ~ph["lungs"]
    rmse = float(np.sqrt(((out["velocity"] - v_true)[:, body] ** 2).mean()))
    metrics.record("multivenc/chest/schnell_order/rmse_body_mm_s", rmse)
    assert failure_fraction(out["velocity"], v_true, ph["lumen"], vl) < 0.01


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


def test_respiratory_axis_and_ambiguity():
    """5D input (respiratory states x frames): the functions act on the
    trailing component axis, and the ambiguity fraction is reported."""
    rng = np.random.default_rng(8)
    u = rng.uniform(-140.0, 140.0, (3, 4, 6, 6, 6, 3))                      # (nr, nt, nk, nj, ni, 3)
    vh, vl = 150.0, 75.0
    v, k, info = multivenc_unwrap([wrap_velocity(u, vl), wrap_velocity(u, vh)], [vl, vh], method="odv",
                                  include_time=True, return_cost=True)
    np.testing.assert_allclose(v, u, atol=1e-9)
    assert info["margin"].shape == u.shape and info["fraction_ambiguous"] == 0.0
    # noisy voxels near the decision boundary are flagged as ambiguous
    noisy = [wrap_velocity(u + rng.normal(0, 0.25 * vl, u.shape), vl), wrap_velocity(u + rng.normal(0, 0.25 * vh, u.shape), vh)]
    v, k, info = multivenc_unwrap(noisy, [vl, vh], method="odv")
    assert info["fraction_ambiguous"] > 0.0


def test_preprocess_background_first(metrics):
    """Per-encoding background correction before combining gives the same
    result as correcting the combined series when the offsets are equal in
    velocity units (the eddy-current case)."""
    ph = phantom.chest(shape=(16, 48, 48), nt=6, rng=np.random.default_rng(9), wrap=False)
    v_true = ph["velocity_true"]; vtw = ph["voxel_to_world"]
    peak = float(np.abs(v_true).max()); vh, vl = 1.1 * peak, 0.55 * peak
    acq = phantom.acquire_multivenc(v_true + ph["offset"][None], [vl, vh], magnitude=ph["magnitude"][0],
                                    snr=15.0, rng=np.random.default_rng(10))
    mag_t = acq[0]["magnitude"]                                              # (nt, ...) from the model
    outs = []
    for first in (False, True):
        out = preprocess(acq[0]["velocity"], mag_t, vl, vtw, lumen_mask=ph["lumen"], background_method="ols",
                         extra_encodings=[(acq[1]["velocity"], vh)], background_first=first, verbose=False)
        assert out["report"]["multivenc"]["background_first"] is first
        outs.append(out["velocity"])
    body = ph["body"] & ~ph["lungs"]
    for i, out in enumerate(outs):
        metrics.record(f"multivenc/chest/background_first{i}/rmse_body_mm_s",
                       float(np.sqrt(((out - v_true)[:, body] ** 2).mean())))
    diff = float(np.sqrt(((outs[0] - outs[1])[:, body] ** 2).mean()))
    assert diff < 0.05 * float(np.sqrt(((outs[0] - v_true)[:, body] ** 2).mean())) + 1.0


# ---------------------------------------------------------------------------
# Aortic CFD with partial volume (the realistic case for dual-venc 5D flow)
# ---------------------------------------------------------------------------

@pytest.mark.cfd
def test_cfd_dual_venc_partial_volume(cfd, metrics):
    """The aortic CFD field acquired at two vencs through the forward model,
    block-averaged to 1.19 mm (partial volume at the wall) with a shared
    reference. Partial-volume voxels have different mean phase at different
    vencs (the signal-weighted average is not linear in venc), which is
    where dual-venc reconstructions show residual sub-2pi wraps (Loecher
    2016, Fig. 8). Compared: single-venc Laplacian on the low venc, sdv,
    odv, corrected odv; failures split into wall and core voxels."""
    from scipy import ndimage
    v_cfd, mask, m = cfd
    nt = 6
    scale = 0.75 + 0.25 * np.clip(np.sin(np.linspace(0, np.pi, nt)), 0, 1)
    v_true = np.stack([v_cfd * s for s in scale]) * 1000.0                   # mm/s
    peak = float(np.abs(v_true).max())
    vh, vl = 1.2 * peak, 0.3 * peak                                          # Löcke 2023: venc 30% of peak
    acq = phantom.acquire_multivenc(v_true, [vl, vh], mask=mask, snr=10.0, downsample=2,
                                    rng=np.random.default_rng(11))
    vt = acq[0]["velocity_true"]                                            # partial-volume reference
    mask_lr = phantom.block_average(mask.astype(float), 2) > 0.5
    core = ndimage.binary_erosion(mask_lr)
    wall = mask_lr & ~core
    m_lr = m.copy(); m_lr[:3, :3] *= 2.0
    low, high = acq[0]["velocity"], acq[1]["velocity"]
    results = {}
    results["laplacian"] = unwrap_velocity(low, vl, m_lr, method="laplacian_then_temporal", mask=mask_lr)
    results["sdv"] = multivenc_unwrap([low, high], [vl, vh], method="sdv")[0]
    results["odv"] = multivenc_unwrap([low, high], [vl, vh], method="odv", mask=mask_lr, correction=False)[0]
    results["odv_corrected"] = multivenc_unwrap([low, high], [vl, vh], method="odv", mask=mask_lr)[0]
    fails = {}
    for name, v in results.items():
        err = np.abs(v - vt)
        fails[name] = (float((err[:, core] > vl).mean()), float((err[:, wall] > vl).mean()))
        metrics.record(f"multivenc/cfd_pv/fail_core_{name}", fails[name][0], atol=0.001)
        metrics.record(f"multivenc/cfd_pv/fail_wall_{name}", fails[name][1], atol=0.002)
        metrics.record(f"multivenc/cfd_pv/rmse_core_{name}", float(np.sqrt((err[:, core] ** 2).mean())))
    rmse_high = float(np.sqrt(((high - vt)[:, core] ** 2).mean()))
    metrics.record("multivenc/cfd_pv/rmse_core_high_venc", rmse_high, direction="record")
    # Löcke 2023's finding: single-venc unwrapping fails on the double wraps
    # (about a third of the lumen here), dual venc does not and keeps the
    # low-venc noise
    assert fails["laplacian"][0] > 0.1
    for name in ("sdv", "odv", "odv_corrected"):
        assert fails[name][0] < 0.002 and fails[name][1] < 0.005
        rmse = float(np.sqrt(((results[name] - vt)[:, core] ** 2).mean()))
        assert rmse < 0.5 * rmse_high
    # an odd venc ratio with the search widened to the full unambiguous range
    # (6 x peak here) lets noise pick far candidates: recorded, and the
    # default cap at the high venc must avoid it
    vl2 = 0.25 * peak
    acq2 = phantom.acquire_multivenc(v_true, [vl2, vh], mask=mask, snr=6.0, downsample=2,
                                     rng=np.random.default_rng(12))
    vt2, low2, high2 = acq2[0]["velocity_true"], acq2[0]["velocity"], acq2[1]["velocity"]
    wide = multivenc_unwrap([low2, high2], [vl2, vh], method="odv", mask=mask_lr, max_velocity=6.0 * vh)[0]
    capped = multivenc_unwrap([low2, high2], [vl2, vh], method="odv", mask=mask_lr)[0]
    f_wide = float((np.abs(wide - vt2)[:, mask_lr] > vl2).mean())
    f_capped = float((np.abs(capped - vt2)[:, mask_lr] > vl2).mean())
    metrics.record("multivenc/cfd_pv/odd_ratio/fail_wide_search", f_wide, direction="record")
    metrics.record("multivenc/cfd_pv/odd_ratio/fail_capped", f_capped, atol=0.001)
    assert f_capped < 0.002 and f_wide > f_capped


# ---------------------------------------------------------------------------
# Valve jet phantom: dephasing, timing offsets, three vencs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jet_phantom():
    return phantom.valve_jet(shape=(32, 32, 48), nt=12, turbulence_intensity=0.2)


def test_valve_jet_phantom_geometry(jet_phantom):
    ph = jet_phantom
    v, w = ph["velocity"], ph["waveform"]
    assert v.shape == (12, 32, 32, 48, 3)
    assert not (ph["jet_core"] & ph["jet_edge"]).any() and not (ph["jet_core"] & ph["chamber_only"]).any()
    assert (ph["jet_core"] | ph["jet_edge"] | ph["chamber_only"]).sum() == ph["mask"].sum()
    # the jet is present only in its frames, the swirl always
    jet_frames = [t for t in range(12) if w[t] > 0]
    assert jet_frames == [4, 5, 6, 7, 8]
    assert np.abs(v[0][ph["jet_core"]]).max() < 0.2 * ph["jet_peak_mm_s"]
    assert np.abs(v[6][ph["jet_core"]]).max() > 0.9 * ph["jet_peak_mm_s"]
    assert 0 < np.abs(v[0][ph["chamber_only"]]).max() <= ph["swirl_peak_mm_s"] + 1e-6
    # orifice flow and regurgitant volume follow the Gaussian profile integral
    np.testing.assert_allclose(ph["flow_mm3_s"], np.pi * 4.0 ** 2 * ph["jet_peak_mm_s"] * w)
    np.testing.assert_allclose(ph["regurgitant_volume_mm3"], ph["flow_mm3_s"].sum() / 12)
    # intravoxel spread only where the jet is
    assert ph["intravoxel_std"][0].max() == 0.0 and ph["intravoxel_std"][6].max() > 0.0
    assert (ph["intravoxel_std"][6][ph["chamber_only"]] == 0.0).all()


def test_dephasing_and_time_offset_models(jet_phantom):
    ph = jet_phantom
    np.testing.assert_allclose(phantom.dephasing_factor(100.0, 100.0), np.exp(-0.5 * np.pi ** 2))
    np.testing.assert_allclose(phantom.dephasing_factor(np.zeros((2, 2)), 50.0), 1.0)
    v = ph["velocity"]
    acq = phantom.acquire_multivenc(v, [550.0, 4400.0], magnitude=np.ones(v.shape[1:4]), snr=1e6,
                                    intravoxel_std=ph["intravoxel_std"], rng=np.random.default_rng(16))
    core6 = ph["jet_core"]
    m_low, m_high = acq[0]["magnitude"][6][core6].mean(), acq[1]["magnitude"][6][core6].mean()
    assert m_low < 0.6 and m_high > 0.9                                      # dephasing stronger at low venc
    assert acq[0]["magnitude"][0][core6].mean() > 0.99                        # no jet, no dephasing
    # timing offset: the high venc samples the field half a frame later
    acq = phantom.acquire_multivenc(v, [2200.0, 4400.0], magnitude=np.ones(v.shape[1:4]), snr=1e6,
                                    time_offset_frames=[0.0, 0.5], rng=np.random.default_rng(17))
    expected = 0.5 * (v[4] + v[5])
    np.testing.assert_allclose(acq[1]["velocity"][4][core6], expected[core6], atol=0.05)
    np.testing.assert_allclose(acq[0]["velocity"][4][core6], wrap_velocity(v[4], 2200.0)[core6], atol=0.05)


def test_valve_jet_multivenc(jet_phantom, metrics):
    """Three vencs for a valve: the lowest gives the chamber its noise level,
    the highest the jet its range; dephasing in the jet core defeats the
    low venc and the magnitude-guarded fallback recovers it."""
    ph = jet_phantom
    v = ph["velocity"]; mask = ph["mask"]
    regions = {"jet_core": ph["jet_core"], "jet_edge": ph["jet_edge"], "chamber": ph["chamber_only"]}
    vencs = [550.0, 2200.0, 4400.0]
    acq = phantom.acquire_multivenc(v, vencs, magnitude=np.ones(v.shape[1:4]), snr=12.0,
                                    intravoxel_std=ph["intravoxel_std"], rng=np.random.default_rng(18))
    series = [a["velocity"] for a in acq]; mags = [a["magnitude"] for a in acq]
    vt = acq[0]["velocity_true"]; wrapped = acq[0]["wrap_count"] != 0
    plain, _, info = multivenc_unwrap(series, vencs, method="odv", mask=mask)
    guarded, _, info_g = multivenc_unwrap(series, vencs, method="odv", mask=mask, magnitudes=mags,
                                          fallback_ratio=0.5)
    dual, _, _ = multivenc_unwrap(series[1:], vencs[1:], method="odv", mask=mask)
    e_plain = error_summary(plain, vt, vencs[0], regions, wrapped)
    e_guard = error_summary(guarded, vt, vencs[0], regions, wrapped)
    e_dual = error_summary(dual, vt, vencs[1], regions)
    for name, e in (("odv", e_plain), ("odv_fallback", e_guard)):
        for r in regions:
            metrics.record(f"multivenc/valve_jet/{name}/fail_{r}", e[r]["fail"], atol=0.002)
            metrics.record(f"multivenc/valve_jet/{name}/rmse_{r}", e[r]["rmse"])
    metrics.record("multivenc/valve_jet/dual_2200_4400/rmse_chamber", e_dual["chamber"]["rmse"], direction="record")
    metrics.record("multivenc/valve_jet/odv_fallback/fraction_fallback", info_g["fraction_fallback"], direction="record")
    # chamber: the 550 venc gives a quarter of the dual-venc noise
    assert e_plain["chamber"]["rmse"] < 0.4 * e_dual["chamber"]["rmse"]
    assert e_plain["chamber"]["fail"] == 0.0
    # jet core: dephasing hurts the plain estimate, the fallback repairs most of it
    assert e_plain["jet_core"]["fail"] > e_guard["jet_core"]["fail"]
    assert e_guard["jet_core"]["rmse"] < 0.5 * e_plain["jet_core"]["rmse"]
    assert 0.0 < info_g["fraction_fallback"] < 0.1
    # nothing falls back where there is no jet
    assert not info_g["fallback_mask"][0].any()


if __name__ == "__main__":
    ph = phantom.stenosis(nt=4, shape=(48, 32, 32), v_max_inlet_mm_s=800.0, severity=0.5)
    v_true, mask, vmax = ph["velocity"], ph["mask"], float(np.abs(ph["velocity"]).max())
    print(f"peak velocity {vmax:.0f} mm/s")
    print("beta   snr   fail_sdv  fail_odv  fail_odv_corrected  fail_schnell  noise_ratio")
    for beta in (0.5, 1 / 3, 2 / 3):
        vh = 1.2 * vmax; vl = beta * vh
        for snr in (10.0, 5.0, 3.5, 2.5):
            acq = phantom.acquire_multivenc(v_true, [vl, vh], magnitude=np.ones(mask.shape), snr=snr,
                                            rng=np.random.default_rng(4))
            s = [acq[0]["velocity"], acq[1]["velocity"]]
            row = []
            for method, kw in (("sdv", {}), ("odv", {"correction": False}), ("odv", {"correction": True}),
                               ("schnell", {})):
                v, _, _ = multivenc_unwrap(s, [vl, vh], method=method, mask=mask, **kw)
                row.append(failure_fraction(v, v_true, mask, vl))
            print(f"{beta:.2f}  {snr:5.1f}  {row[0]:.4f}    {row[1]:.4f}    {row[2]:.4f}     {row[3]:.4f}         "
                  f"{noise_ratio(v, acq[1]['velocity'], ~mask):.2f}")
