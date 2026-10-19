"""
Mass-conservation QA on the aortic CFD field: through-plane flow at two
descending-aorta planes (no branches between them, so inflow = outflow in
truth) at each processing stage.

Usage: python test_qa.py /path/to/4DFlowNet/data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flowfilters"))
import numpy as np
import h5py
from qa import plane_flow, qa_report
from divergence_free import divergence_free_sequence


def load(path):
    with h5py.File(path) as h:
        d = {k: h[k][0] for k in h.keys()}
    vel = np.stack([d["u"], d["v"], d["w"]], -1).astype(float) * 1000.0   # mm/s
    return vel, d["mask"] > 0.5, tuple(float(x) for x in d["dx"])


def main(data_dir):
    v_cfd, mask, sp = load(f"{data_dir}/example_data_HR.h5")
    rng = np.random.default_rng(0)
    nt = 24
    t = np.arange(nt) / nt
    # a physiological-looking aortic waveform: systolic pulse + small diastolic flow
    wave = 0.1 + 0.9 * np.exp(-((t - 0.25) / 0.09) ** 2) + 0.15 * np.exp(-((t - 0.55) / 0.08) ** 2)
    v_true = np.stack([v_cfd * w for w in wave])
    sigma = 0.10 * 1500.0                                                   # 10% of 1.5 m/s venc
    v_noisy = v_true + rng.normal(0, sigma, v_true.shape) * mask[None, ..., None]

    print("\nMass-conservation QA: two descending-aorta planes (truth: equal flow)")
    planes = {"DAo_prox": dict(centre=None, normal=(1, 0, 0), radius=15.0),
              "DAo_dist": dict(centre=None, normal=(1, 0, 0), radius=15.0)}
    # locate lumen centroids on two axial slices in the descending segment
    for name, k in (("DAo_prox", 62), ("DAo_dist", 80)):
        zz, yy, xx = np.nonzero(mask[k:k + 1])
        planes[name]["centre"] = (k, yy.mean(), xx.mean())

    def flows(v):
        # sign: flow is along -z in this dataset; flip so forward is positive
        return {n: -plane_flow(v, sp, lumen_mask=mask, **p)["flow"] / 1000.0 for n, p in planes.items()}  # mL/s

    v_dfw, _ = divergence_free_sequence(v_noisy, mask, sp, method="dfw")
    v_fdm, _ = divergence_free_sequence(v_noisy, mask, sp, method="fdm")
    stages = {"truth": flows(v_true), "noisy": flows(v_noisy), "fdm": flows(v_fdm), "dfw": flows(v_dfw)}
    checks = [("prox vs dist", ["DAo_prox"], ["DAo_dist"])]
    qa_report(stages, checks, heart_rate_bpm=100)
    print(f"  mean flows (mL/s): truth prox {stages['truth']['DAo_prox'].mean():.1f}, "
          f"dist {stages['truth']['DAo_dist'].mean():.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
