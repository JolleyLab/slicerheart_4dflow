"""
Self-check of the velocity-encoding direction mapping.

The most common way a 4D flow import goes wrong is a permuted or sign-flipped
velocity component: vendors store the three encoded velocities in different
orders and with different sign conventions relative to the image axes, and
the error is silent (the field looks plausible and every downstream number,
including Qp:Qs, is wrong). Blood is incompressible, so the correct mapping
is the one that makes the lumen divergence smallest; a wrong permutation or
sign mixes in a component that is not a velocity along that axis and the
divergence rises sharply. The 4DFlowNet CFD example used in this package's
tests was itself assigned its component order this way.

encoding_check() evaluates the divergence RMS for all 48 signed permutations
of the components (24 permutations times 8 sign patterns) and reports the
ranking. To make the test robust to measurement noise it works on a
low-pass version of the field: the time-mean velocity, zero outside the
lumen, smoothed with a Gaussian, with the divergence evaluated on the lumen
dilated by a couple of voxels. By Gauss's theorem the smoothed field of an
incompressible flow confined to a tube has near-zero divergence everywhere
(the flux entering a blurred wall region equals the flux leaving it), while
a wrong mapping turns the through-plane velocity into a wall-normal one and
the blurred wall becomes a strong source/sink layer. Noise is suppressed by
the time average and the smoothing, so the discrimination stays at 2-3x for
velocity noise of 30% of peak (test_encoding.py).

Two limitations are intrinsic:

  * A global sign flip (all three components negated) leaves the divergence
    unchanged, so the 48 fall into 24 pairs; the global sign is settled by
    the known direction of flow through a labelled plane (for example the
    ascending aorta is cranial), which expected_flow_sign() tests.
  * Mappings related by a symmetry of the flow tie. A component that is
    everywhere near zero (purely axial flow along one axis) cannot have its
    sign checked, and an axisymmetric tube cannot distinguish a swap of its
    two in-plane components. Real anatomy (curved vessels, branches, helical
    flow) breaks these symmetries; symmetric phantoms do not. For this reason
    the consistency decision uses the ratio of the declared mapping's
    divergence to the minimum, not the uniqueness of the minimum.

References
----------
    Dyverfeldt P, Bissell M, Barker AJ, et al. 4D flow cardiovascular
    magnetic resonance consensus statement. J Cardiovasc Magn Reson
    2015;17:72. PMID 26257141. doi:10.1186/s12968-015-0174-5 (encoding
    conventions, data checking).
    Song SM, Napel S, Glover GH, Pelc NJ. Noise reduction in
    three-dimensional phase-contrast MR velocity measurements. J Magn
    Reson Imaging 1993;3(4):587-596. PMID 8347951 (divergence of measured
    flow as a consistency criterion).
    Pelc NJ, Bernstein MA, Shimakawa A, Glover GH. Encoding strategies for
    three-direction phase-contrast MR imaging of flow. J Magn Reson
    Imaging 1991;1(4):405-413. PMID 1790362.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage

from .divergence_free import build_divergence_operator, _stack

PERMUTATIONS = list(itertools.permutations(range(3)))
SIGNS = [s for s in itertools.product((1, -1), repeat=3)]
IDENTITY = ((0, 1, 2), (1, 1, 1))

CONSISTENT_RATIO = 1.5      # declared / minimum divergence above which the mapping is flagged


def apply_mapping(velocity, permutation, signs):
    """v_out[..., a] = signs[a] * v_in[..., permutation[a]]."""
    v = np.asarray(velocity, float)
    return np.stack([signs[a] * v[..., permutation[a]] for a in range(3)], -1)


def smoothed_mean_field(velocity_t, mask, sigma_vox=3.0, frames=None):
    """
    Time-mean velocity over `frames`, zeroed outside `mask` and Gaussian
    smoothed (zero padded) with `sigma_vox` voxels; the low-pass field the
    encoding check is evaluated on. Returns (nk, nj, ni, 3).
    """
    v = np.asarray(velocity_t, float)
    if v.ndim == 4:
        v = v[None]
    if frames is not None:
        v = v[np.asarray(frames)]
    m = np.asarray(mask, bool)
    vmean = v.mean(0) * m[..., None]
    if sigma_vox <= 0:
        return vmean
    return np.stack([ndimage.gaussian_filter(vmean[..., c], sigma_vox) for c in range(3)], -1)


def encoding_check(velocity_t, mask, voxel_to_world, frames=None, sigma_vox=3.0, dilate=2,
                   ratio_threshold=CONSISTENT_RATIO):
    """
    Rank all signed permutations of the velocity components by the
    divergence of the low-pass field (see module docstring).

    velocity_t : (nt, nk, nj, ni, 3) or (nk, nj, ni, 3), world components as
                 imported (the mapping under test is the identity).
    mask       : (nk, nj, ni) bool lumen.
    frames     : frame indices to average (default all; the time mean is
                 what suppresses noise, so use the whole cycle when you can).
    sigma_vox  : Gaussian smoothing in voxels (0 disables); 2-3 is the
                 plateau on the CFD and phantom tests.
    dilate     : voxels of mask dilation for the evaluation region, so the
                 smeared wall flux is inside it; ~sigma_vox.
    Returns dict:
      ranking        list of (div_rms, permutation, signs) ascending
      best           (permutation, signs) of the minimum
      declared_div   divergence RMS (1/s) of the identity mapping
      best_div       minimum divergence RMS
      ratio          declared_div / best_div (1 means the import is consistent)
      consistent     ratio < ratio_threshold
      discrimination divergence of the best mapping outside the identity's
                     tie class (ratio > 1.05 to the minimum) over the
                     minimum; how strongly the data constrain the mapping
                     (>= ratio_threshold means a wrong mapping would be flagged)
      sign_ambiguous True (always: global sign is not testable by divergence)
      note           text
    """
    m = np.asarray(mask, bool)
    field = smoothed_mean_field(velocity_t, m, sigma_vox, frames)
    region = ndimage.binary_dilation(m, iterations=int(dilate)) if dilate else m
    D = build_divergence_operator(region, voxel_to_world)
    s = _stack(field, region)
    n = int(region.sum())
    results = []
    for perm in PERMUTATIONS:
        for signs in SIGNS:
            comp = [signs[a] * s[perm[a] * n:(perm[a] + 1) * n] for a in range(3)]
            d = D @ np.concatenate(comp)
            results.append((float(np.sqrt(np.mean(d ** 2))), perm, signs))
    results.sort(key=lambda r: r[0])
    declared = next(r[0] for r in results if (r[1], r[2]) == IDENTITY)
    best_div, best_perm, best_signs = results[0]
    ratio = declared / best_div if best_div > 0 else 1.0
    consistent = bool(ratio < ratio_threshold)
    outside_tie = [r[0] for r in results if r[0] > 1.05 * best_div]
    discrimination = (outside_tie[0] / best_div) if (outside_tie and best_div > 0) else float("inf")
    if consistent:
        note = (f"declared mapping is within {ratio:.2f}x of the minimum divergence (next distinct mapping "
                f"{discrimination:.1f}x); global sign is not testable by divergence, confirm with "
                "expected_flow_sign() on a labelled plane")
    else:
        neg_best = tuple(-x for x in best_signs)
        note = (f"declared mapping is {ratio:.1f}x the minimum divergence; best is permutation {best_perm} "
                f"with signs {best_signs} (or its global negation {neg_best}); check the import's encoding axes")
    return {"ranking": results, "best": (best_perm, best_signs), "declared_div": declared, "best_div": best_div,
            "ratio": ratio, "consistent": consistent, "discrimination": discrimination,
            "sign_ambiguous": True, "note": note}


def expected_flow_sign(velocity_t, voxel_to_world, centre, normal, radius, lumen_mask=None,
                       expected="positive"):
    """
    Settle the global sign with a plane whose forward direction is known
    (normal points the way blood flows, e.g. cranially in the ascending
    aorta). Returns dict with net_flow, sign_ok, note.
    """
    from .qa import plane_flow, net_volume
    q = plane_flow(velocity_t, voxel_to_world, centre, normal, radius, lumen_mask=lumen_mask)["flow"]
    net = float(net_volume(q))
    ok = (net > 0) if expected == "positive" else (net < 0)
    return {"net_flow": net, "sign_ok": bool(ok),
            "note": "net flow has the expected sign" if ok else
                    "net flow has the wrong sign: negate all three velocity components"}
