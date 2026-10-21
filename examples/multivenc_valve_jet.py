"""
Multi-venc test bed on the synthetic valve-jet phantom, and the same
evaluation on exported scanner data. DRAFT.

Phantom mode (default): an orifice jet (regurgitant or stenotic) entering a
chamber of slow swirling blood, acquired at two or three vencs through the
phase-contrast forward model with optional intravoxel dephasing (turbulence
intensity), timing offsets between encodings, partial volume and noise;
every multi-venc method is run and scored per region (jet core, jet edge,
chamber), with missed wraps and false unwraps reported separately.

    python examples/multivenc_valve_jet.py
    python examples/multivenc_valve_jet.py --vencs 550 2200 4400 --snr 10 --ti 0.3 --offset 0.3
    python examples/multivenc_valve_jet.py --vencs 2200 4400 --downsample 2 --save jet.npz

Data mode: an .npz with the arrays of a dual- or triple-venc acquisition,
exported from 3D Slicer with slicer.util.arrayFromVolume (arrays
(nt, nk, nj, ni, 3) for velocity, (nt, nk, nj, ni) for magnitude, mm/s):

    np.savez("case.npz", velocity_0=..., venc_0=550.0, magnitude_0=...,
             velocity_1=..., venc_1=2200.0, magnitude_1=..., mask=lumen)
    python examples/multivenc_valve_jet.py --npz case.npz [--truth velocity_ref]

Without a truth array the data mode reports what can be known without
one: fraction unwrapped, fraction ambiguous, fraction fallen back, and the
static-tissue noise ratio to the highest venc.
"""

import argparse
import time

import numpy as np

from slicerheart_4dflow import phantom
from slicerheart_4dflow.multivenc import error_summary, multivenc_unwrap, noise_ratio

METHODS = (("sdv", {}), ("schnell", {}), ("odv", {}), ("odv+fallback", {"fallback_ratio": 0.5}))


def run(series, vencs, magnitudes, mask, truth=None, wrapped=None, regions=None, static=None):
    rows = []
    for label, kw in METHODS:
        method = label.split("+")[0]
        if method != "odv" and len(series) != 2:
            continue
        kw = dict(kw)
        if "fallback_ratio" in kw:
            kw["magnitudes"] = magnitudes
        t0 = time.perf_counter()
        v, k, info = multivenc_unwrap(series, vencs, method=method, mask=mask, **kw)
        row = {"method": label, "seconds": time.perf_counter() - t0,
               "changed": info["fraction_changed"], "ambiguous": info.get("fraction_ambiguous"),
               "fallback": info.get("fraction_fallback")}
        if static is not None and len(series) > 1:
            row["noise_ratio_to_high"] = noise_ratio(v, series[-1], static)
        if truth is not None:
            row["errors"] = error_summary(v, truth, vencs[0], regions, wrapped)
        rows.append(row)
    return rows


def print_rows(rows, regions):
    for r in rows:
        head = (f"{r['method']:13s} {r['seconds']:5.1f} s  changed {100 * r['changed']:5.2f}%"
                + (f"  ambiguous {100 * r['ambiguous']:5.2f}%" if r.get("ambiguous") is not None else "")
                + (f"  fallback {100 * r['fallback']:5.2f}%" if r.get("fallback") is not None else "")
                + (f"  noise/high {r['noise_ratio_to_high']:.2f}" if "noise_ratio_to_high" in r else ""))
        print(head)
        if "errors" in r:
            for name in regions:
                e = r["errors"][name]
                extra = f"  missed {100 * e['missed']:5.1f}%  false {100 * e['false']:5.2f}%" if "missed" in e else ""
                print(f"    {name:10s} fail {100 * e['fail']:6.3f}%  rmse {e['rmse']:6.1f} mm/s{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vencs", type=float, nargs="+", default=[550.0, 2200.0, 4400.0], help="mm/s, low to high")
    ap.add_argument("--snr", type=float, default=12.0)
    ap.add_argument("--ti", type=float, default=0.2, help="turbulence intensity in the jet (0 = no dephasing)")
    ap.add_argument("--offset", type=float, default=0.0, help="timing offset of the higher vencs, in frames")
    ap.add_argument("--downsample", type=int, default=1, help="block-average factor (partial volume)")
    ap.add_argument("--jet-peak", type=float, default=4000.0)
    ap.add_argument("--nt", type=int, default=20)
    ap.add_argument("--shape", type=int, nargs=3, default=[48, 48, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="write the phantom acquisition to this .npz (data-mode format plus truth)")
    ap.add_argument("--npz", help="data mode: evaluate this exported acquisition instead of the phantom")
    ap.add_argument("--truth", default="velocity_true", help="data mode: name of the reference array, if any")
    a = ap.parse_args()

    if a.npz:
        d = np.load(a.npz)
        n = sum(1 for key in d.files if key.startswith("velocity_") and key[9:].isdigit())
        series = [d[f"velocity_{i}"] for i in range(n)]
        vencs = [float(d[f"venc_{i}"]) for i in range(n)]
        mags = [d[f"magnitude_{i}"] for i in range(n)] if f"magnitude_{n - 1}" in d.files else None
        mask = d["mask"].astype(bool) if "mask" in d.files else None
        truth = d[a.truth] if a.truth in d.files else None
        regions = {"mask": mask} if mask is not None else {"all": np.ones(series[0].shape[1:4], bool)}
        static = (~mask) if mask is not None else None
        print(f"{a.npz}: {n} encodings, vencs {vencs}, shape {series[0].shape}")
        rows = run(series, vencs, mags, mask, truth, None, regions, static)
        print_rows(rows, regions)
        return

    shape = tuple(a.shape)
    ph = phantom.valve_jet(shape=shape, nt=a.nt, jet_peak_mm_s=a.jet_peak, turbulence_intensity=a.ti)
    v = ph["velocity"]
    offsets = [0.0] + [a.offset] * (len(a.vencs) - 1)
    # unit magnitude everywhere: the chamber wall is myocardium (signal, no
    # flow), which is where the noise ratio is measured
    acq = phantom.acquire_multivenc(v, a.vencs, magnitude=np.ones(shape), snr=a.snr, downsample=a.downsample,
                                    intravoxel_std=ph["intravoxel_std"] if a.ti > 0 else None,
                                    time_offset_frames=offsets, rng=np.random.default_rng(a.seed))
    series = [x["velocity"] for x in acq]; mags = [x["magnitude"] for x in acq]
    truth = acq[0]["velocity_true"]; wrapped = acq[0]["wrap_count"] != 0
    f = a.downsample
    if f > 1:
        ds = lambda m: phantom.block_average(m.astype(float), f) > 0.5
        mask = ds(ph["mask"]); regions = {k: ds(ph[k]) for k in ("jet_core", "jet_edge", "chamber_only")}
    else:
        mask = ph["mask"]; regions = {k: ph[k] for k in ("jet_core", "jet_edge", "chamber_only")}
    print(f"valve jet: peak {ph['jet_peak_mm_s']:.0f} mm/s, swirl peak {ph['swirl_peak_mm_s']:.0f} mm/s, "
          f"regurgitant volume {ph['regurgitant_volume_mm3'] / 1000:.1f} mL per cycle (1 s), "
          f"vencs {a.vencs}, snr {a.snr}, TI {a.ti}, offset {a.offset} frames, downsample {a.downsample}")
    print("jet-core magnitude per venc at peak frame:",
          [round(float(m[int(np.argmax(ph['waveform']))][regions['jet_core']].mean()), 3) for m in mags])
    rows = run(series, a.vencs, mags, mask, truth, wrapped, regions, ~mask)
    print_rows(rows, regions)
    if a.save:
        out = {f"velocity_{i}": s for i, s in enumerate(series)}
        out.update({f"venc_{i}": np.float64(c) for i, c in enumerate(a.vencs)})
        out.update({f"magnitude_{i}": m for i, m in enumerate(mags)})
        out.update(mask=mask, velocity_true=truth, wrap_count=acq[0]["wrap_count"],
                   voxel_to_world=ph["voxel_to_world"])
        np.savez_compressed(a.save, **out)
        print(f"saved {a.save}")


if __name__ == "__main__":
    main()
