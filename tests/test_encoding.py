"""
Encoding-direction self-check (encoding.py).

The check ranks the 48 signed permutations of the velocity components by
the divergence of a low-pass version of the field. It must

  * accept the declared mapping of a correctly imported field (ratio to the
    minimum ~1) on the aortic CFD, the stenosis and the bifurcation phantoms,
    for every voxel geometry, clean and with heavy velocity noise;
  * flag a permuted or sign-flipped import (ratio well above 1.5);
  * keep its discrimination (next distinct mapping / minimum) above the
    flag threshold under noise, so that a wrong mapping cannot hide.

Known blind spots are asserted as such: a global sign flip ties with the
identity (settled by expected_flow_sign on a labelled plane), and a planar
flow cannot have the sign of its out-of-plane component checked.

Metrics recorded: ratio of the identity and of a wrong mapping, and the
discrimination, for the CFD at 0/10/30% noise and the phantoms at SNR 10.
"""

import numpy as np
import pytest

from conftest import GEOMETRIES, aortic_waveform
from slicerheart_4dflow.encoding import (
    CONSISTENT_RATIO, apply_mapping, encoding_check, expected_flow_sign, PERMUTATIONS, SIGNS)
from slicerheart_4dflow.phantom import acquire, add_noise, bifurcation, stenosis

WRONG = [((1, 0, 2), (1, 1, 1)), ((0, 2, 1), (1, 1, 1)), ((2, 1, 0), (1, 1, 1)),
         ((0, 1, 2), (-1, 1, 1)), ((0, 1, 2), (1, -1, 1)), ((0, 1, 2), (1, 1, -1)),
         ((1, 2, 0), (1, -1, 1))]


def _acquired(phantom, venc, snr, seed):
    rng = np.random.default_rng(seed)
    vt = np.stack([acquire(phantom["velocity"][t], venc, mask=phantom["mask"], snr=snr, rng=rng)["velocity"]
                   for t in range(phantom["velocity"].shape[0])])
    return vt, phantom["mask"], phantom["voxel_to_world"]


def test_apply_mapping_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 4, 5, 6, 3))
    assert len(PERMUTATIONS) == 6 and len(SIGNS) == 8
    out = apply_mapping(v, (2, 0, 1), (1, -1, 1))
    np.testing.assert_array_equal(out[..., 0], v[..., 2])
    np.testing.assert_array_equal(out[..., 1], -v[..., 0])
    np.testing.assert_array_equal(out[..., 2], v[..., 1])


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_stenosis_identity_consistent_and_wrong_mappings_flagged(geometry, metrics):
    s = stenosis(nt=6, voxel_to_world=GEOMETRIES[geometry]())
    vt, mask, vtw = _acquired(s, venc=4000.0, snr=10.0, seed=2)     # venc above the 3.1 m/s jet
    r = encoding_check(vt, mask, vtw)
    assert r["consistent"] and r["ratio"] < 1.1
    assert r["sign_ambiguous"]
    assert r["discrimination"] > CONSISTENT_RATIO
    if geometry == "identity":
        metrics.record("encoding/stenosis_snr10/discrimination", r["discrimination"], direction="higher")
    # an axisymmetric tube along z cannot tell x from y (symmetry), but every
    # mapping that moves the axial component, or flips its sign, is flagged
    for perm, signs in WRONG:
        if perm[2] == 2 and signs[2] == 1:
            continue
        rw = encoding_check(apply_mapping(vt, perm, signs), mask, vtw)
        assert not rw["consistent"], (perm, signs, rw["ratio"])


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_bifurcation_identity_consistent_and_wrong_mappings_flagged(geometry, metrics):
    b = bifurcation(nt=6, voxel_to_world=GEOMETRIES[geometry]())
    vt, mask, vtw = _acquired(b, venc=1500.0, snr=10.0, seed=3)
    r = encoding_check(vt, mask, vtw)
    assert r["consistent"] and r["ratio"] < 1.1
    assert r["discrimination"] > CONSISTENT_RATIO
    if geometry == "identity":
        metrics.record("encoding/bifurcation_snr10/discrimination", r["discrimination"], direction="higher")
    # the bifurcation lies in the x-z plane: the y component is ~0 so its
    # sign is untestable (documented limitation); everything else is flagged
    for perm, signs in WRONG:
        if perm == (0, 1, 2) and signs == (1, -1, 1):
            rw = encoding_check(apply_mapping(vt, perm, signs), mask, vtw)
            assert rw["consistent"]                       # blind spot, asserted as such
            continue
        rw = encoding_check(apply_mapping(vt, perm, signs), mask, vtw)
        assert not rw["consistent"], (perm, signs, rw["ratio"])


def test_global_sign_flip_ties_and_plane_settles_it():
    s = stenosis(nt=4)
    vt, mask, vtw = s["velocity"], s["mask"], s["voxel_to_world"]
    r = encoding_check(vt, mask, vtw)
    r_neg = encoding_check(-vt, mask, vtw)
    assert r_neg["consistent"] and abs(r_neg["declared_div"] - r["declared_div"]) < 1e-9 * r["declared_div"]
    # flow is +z through the inlet plane
    plane = dict(centre=(19.5, 19.5, 8.0), normal=(0.0, 0.0, 1.0), radius=12.0)
    assert expected_flow_sign(vt, vtw, lumen_mask=mask, **plane)["sign_ok"]
    assert not expected_flow_sign(-vt, vtw, lumen_mask=mask, **plane)["sign_ok"]


def test_aliased_field_is_flagged():
    """Running the check on wrapped data (venc below the jet) breaks the
    consistency; it is meant to run after unwrapping."""
    s = stenosis(nt=8)                                     # 8 frames reach the 3.1 m/s peak
    vt, mask, vtw = _acquired(s, venc=1500.0, snr=10.0, seed=2)
    r = encoding_check(vt, mask, vtw)
    assert not r["consistent"]


@pytest.mark.cfd
@pytest.mark.parametrize("noise_frac", [0.0, 0.10, 0.30])
def test_cfd_encoding_check(cfd, noise_frac, metrics):
    v, mask, vtw = cfd
    nt = 12
    vt = np.stack([v * 1000.0 * s for s in aortic_waveform(nt)])
    if noise_frac:
        vt = add_noise(vt, noise_frac * 1500.0, mask, np.random.default_rng(0))
    r = encoding_check(vt, mask, vtw)
    tag = f"cfd/encoding/noise{int(100 * noise_frac)}"
    metrics.record(f"{tag}/identity_ratio", r["ratio"], direction="lower")
    metrics.record(f"{tag}/discrimination", r["discrimination"], direction="higher")
    assert r["consistent"] and r["ratio"] < 1.05
    assert r["discrimination"] > CONSISTENT_RATIO
    worst = min(encoding_check(apply_mapping(vt, p, s), mask, vtw)["ratio"] for p, s in WRONG)
    metrics.record(f"{tag}/worst_wrong_ratio", worst, direction="higher")
    assert worst > 2.0
    # the best mapping of a permuted field is the inverse permutation (up to global sign)
    rw = encoding_check(apply_mapping(vt, (1, 2, 0), (1, -1, 1)), mask, vtw)
    assert not rw["consistent"]
    best_perm, best_signs = rw["best"]
    recovered = apply_mapping(apply_mapping(vt, (1, 2, 0), (1, -1, 1)), best_perm, best_signs)
    assert np.allclose(recovered, vt) or np.allclose(recovered, -vt)


def test_report_includes_encoding_section():
    from slicerheart_4dflow.report import quality_report, to_markdown
    s = stenosis(nt=4)
    vt, mask, vtw = s["velocity"], s["mask"], s["voxel_to_world"]
    mag = np.broadcast_to(np.where(mask, 1.0, 0.7), vt.shape[:4])
    static = ~mask
    rep = quality_report(vt, vt, mag, 4000.0, vtw, lumen_mask=mask, static_mask=static)
    assert rep["encoding"]["consistent"]
    assert "Encoding directions  (PASS)" in to_markdown(rep)
    rep = quality_report(vt, apply_mapping(vt, (0, 2, 1), (1, 1, 1)), mag, 4000.0, vtw, lumen_mask=mask,
                         static_mask=static)
    assert not rep["encoding"]["consistent"]
    assert rep["encoding"]["best_permutation"] == [0, 2, 1]


@pytest.mark.parametrize("cut", [(16, 48), (24, 40)])
def test_segmentation_ending_inside_fov(cut, metrics):
    """A mask cut inside the field of view has open inlet/outlet faces whose
    source/sink swamped the comparison (discrimination 3.75 -> 1.2 for the
    middle third of the stenosis, and an axial sign flip passed). The band
    around open faces, found on the best mapping's field so that a wrong
    import cannot hide its walls as 'open faces', is left out."""
    s = stenosis(nt=6)
    vt, mask, vtw = _acquired(s, venc=4000.0, snr=10.0, seed=2)
    k = np.arange(mask.shape[0])[:, None, None]
    mk = mask & (k >= cut[0]) & (k < cut[1])
    r = encoding_check(vt, mk, vtw)
    assert r["consistent"] and r["conclusive"] and r["excluded_voxels"] > 0
    assert r["discrimination"] > CONSISTENT_RATIO
    metrics.record(f"encoding/stenosis_cut_{cut[0]}_{cut[1]}/discrimination", r["discrimination"], direction="higher")
    for perm, signs in (((0, 1, 2), (1, 1, -1)), ((0, 2, 1), (1, 1, 1)), ((1, 2, 0), (1, -1, 1))):
        rw = encoding_check(apply_mapping(vt, perm, signs), mk, vtw)
        assert not rw["consistent"], (perm, signs, rw["ratio"])
    # without the exclusion the axial sign flip is not caught on the middle third
    if cut == (24, 40):
        rz = encoding_check(apply_mapping(vt, (0, 1, 2), (1, 1, -1)), mk, vtw, exclude_open_faces=False)
        assert rz["consistent"]


def test_inconclusive_when_every_mapping_ties():
    """A short straight segment of uniform axial flow constrains nothing:
    the result must say so instead of reporting a pass."""
    shape = (12, 20, 20)
    Y, X = np.meshgrid(np.arange(20) - 9.5, np.arange(20) - 9.5, indexing="ij")
    tube = np.broadcast_to((X ** 2 + Y ** 2) < 36.0, shape).copy()
    v = np.zeros(shape + (3,)); v[tube, 2] = 500.0
    r = encoding_check(v, tube, np.eye(4))
    assert r["consistent"] and not r["conclusive"]
    assert "INCONCLUSIVE" in r["note"]
