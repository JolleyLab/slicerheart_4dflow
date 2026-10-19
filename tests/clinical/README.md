# Clinical 4D flow test cases

Real scanner data for end-to-end tests: DICOM import, encoding-direction
handling, the actual noise and background-offset characteristics of each
vendor, and flow quantification against a commercial reference. Phantoms
and the CFD field cannot test those. One folder per vendor; each holds a
`README.md` with vendor notes and, once data exist, a `manifest.json`.

Data are **not** stored in git. Each case is published on Zenodo under
CC BY 4.0 with a DOI and downloaded on first use into the folder named by
`SLICERHEART_4DFLOW_CLINICAL_DATA` (default `tests/data/clinical/<vendor>/`),
following the pattern used for the CFD data in `conftest.py`. Tests that
need a case are marked `clinical` and skip when its manifest is absent:

    python -m pytest -m clinical            # only the clinical tier
    python -m pytest -m "not clinical"      # everything else (default CI)

The overall plan for the synthetic, semi-synthetic and clinical tiers and
the transfer audit between them is in `docs/VALIDATION_STRATEGY.md`; its
section 3a records what the SlicerHeart Import4DFlow and Analyze4DFlow
modules actually write (flow field as a grid-transform sequence in m/s and
RAS, TriggerTime index in ms, venc and heart rate as node attributes,
anonymized `.mrb` export, plane markups and flow tables).

**Draft layout.** The directory layout below was written before the first
case and will be replaced by the anonymized `.mrb` that Import4DFlow
exports, plus Analyze4DFlow's plane markups and tables, once a real export
has been inspected. The mock case keeps the directory layout so the checks
can run in CI meanwhile.

## Status

| vendor | case | status |
|---|---|---|
| Siemens | CHOP, 1.5 T Avanto Fit, compressed-sensing 4D flow, pediatric CHD | planned: one case with a jet above venc, one without |
| Philips | none yet | placeholder |
| GE | none yet | placeholder |

## What a case contains

```
<vendor>/<case_id>/
  manifest.json          see schema below
  dicom/                 as exported from the scanner after de-identification:
                         magnitude series and the three velocity-encoded series
                         (not needed by the tests; kept for the Import module's own tests)
  derived/
    velocity.seq.nrrd    5D Slicer sequence of the velocity vector volume,
                         written by the SlicerHeart Import 4D Flow module:
                         kinds "vector domain domain domain list", sizes
                         3 ni nj nk nt, space left-posterior-superior, frame
                         times in "axis 4 index values" (seconds). See
                         slicerheart_4dflow/io.py for the convention and
                         https://discourse.slicer.org/t/45588 for its origin.
    magnitude.seq.nrrd   4D Slicer sequence of the magnitude image, same geometry
    lumen.seg.nrrd       segmentation used for the reference analysis, same geometry
                         (resample in Slicer if it was drawn on another grid)
    planes.json          {"coordinate_system": "LPS" | "RAS",
                          "planes": {name: {"centre": [x, y, z], "normal": [nx, ny, nz], "radius": r}}}
                         in world mm; RAS if copied from Slicer markups
  reference/
    flows.json           {"planes": {name: {"net_volume_mL": ..., "forward_volume_mL": ...,
                                            "peak_velocity_mm_s": ...}},
                          "conservation_checks": [[label, [inflow names], [outflow names]]],
                          "ratio_checks": [[label, numerator, denominator, expected]],
                          "pc2d": {name: {"net_volume_mL": ...}},          # optional, 2D phase contrast
                          "source": "cvi42 6.0.4"}
    truth.npz            mock cases only: velocity_true, offset, tissue
  audit.md               written by the transfer audit test
```

`slicerheart_4dflow.io.load_case(manifest)` loads all of it into the
package's arrays (mm/s, LPS). The velocity unit and the frame the vector
components are expressed in are NOT fixed by the Slicer convention, so the
manifest states them (`velocity_units`, `velocity_component_frame`), and
the loader converts to mm/s and LPS. Open point to confirm with the Import
module: whether it writes the components in LPS (the NRRD space) or RAS
(Slicer's internal frame); the mock cases write LPS and say so in the
header (`velocity_component_frame:=LPS`).

A mock case in exactly this layout is generated from the bifurcation phantom
by `tests/clinical/casegen.py` and run through every clinical check in the
default (synthetic) test tier, so the machinery is tested before any real
case exists.

## manifest.json (version 2)

```json
{
  "manifest_version": 2,
  "case_id": "chop_siemens_001",
  "vendor": "siemens",
  "scanner": "MAGNETOM Avanto Fit 1.5 T",
  "software_version": "syngo MR E11",
  "sequence": "compressed-sensing 4D flow, retrospectively gated, navigator",
  "gating": "retrospective",
  "voxel_size_mm": [1.6, 1.6, 1.6],
  "n_phases": 25,
  "rr_ms": 650.0,
  "venc_mm_s": [1500.0, 1500.0, 1500.0],
  "velocity_units": "mm/s",
  "velocity_component_frame": "LPS",
  "aliasing_present": true,
  "on_scanner_corrections": {"maxwell": true, "eddy_current": false, "gradient_nonlinearity": true},
  "contrast": "ferumoxytol",
  "age_group": "child",
  "diagnosis_category": "coarctation",
  "deidentification": {"tool": "...", "date": "YYYY-MM-DD", "private_tags": "removed", "dates": "shifted"},
  "reference_software": "cvi42 4D Flow 6.0.4",
  "license": "CC BY 4.0",
  "doi": "10.5281/zenodo.XXXXXXX",
  "files": {"velocity": "derived/velocity.seq.nrrd", "magnitude": "derived/magnitude.seq.nrrd",
            "lumen": "derived/lumen.seg.nrrd", "planes": "derived/planes.json",
            "reference_flows": "reference/flows.json"}
}
```

`gating` matters: with prospective gating the frames cover part of the RR
interval and the pressure temporal term and per-beat volumes need the
acquired frame duration. `on_scanner_corrections` says what the
background correction has to remove. No field identifies a patient.
`age_group` is a category, not an age; `diagnosis_category` is one of a
short fixed list. Dates in the DICOM header are shifted, private tags
removed, and the magnitude images reviewed for burned-in text before
release.

## What the clinical tests check (`tests/test_clinical.py`, `tests/clinical/checks.py`)

The same check functions run on the mock case (synthetic tier) and on
every real case (clinical tier, separate baseline
`tests/metrics/clinical_baseline.json`):

1. manifest and arrays: fields, shapes, units, frames, geometry of the
   segmentation
2. consistency (no truth needed): residual static velocity after `war+msac`
   under 0.6 cm/s, venc adequacy agrees with `aliasing_present`, encoding
   check consistent, velocity noise in range, inflow = outflow within 5%,
   ratio checks within 10%
3. reference: net volume per beat at each plane within the manuscript's
   limits of agreement (7 mL)
4. stability: plane flows move less than 5% under a one-voxel dilation of
   the segmentation and not at all under a rigid change of world frame
5. semi-synthetic: a divergence-free reference built from the case,
   re-acquired at its venc and noise, must be recovered by the pipeline to
   within the noise with no residual wraps
6. transfer audit: paired quantities against a synthetic case built at the
   case's settings, recorded and written to `audit.md` (a flag is a finding
   to act on, not a test failure)

## De-identification checklist before uploading a case

- DICOM: remove patient name, ID, birth date, all private tags unless
  needed for velocity decoding (list those kept in the manifest), shift
  acquisition dates, remove institution and operator names, clear
  `ImageComments` and `StudyDescription` if they carry identifiers.
- Check magnitude images for burned-in annotations.
- Segmentation and plane files carry no header metadata beyond geometry.
- Reference flows are numbers only.
- IRB approval for data sharing on file; note the protocol number in the
  Zenodo record, not in the repository.
