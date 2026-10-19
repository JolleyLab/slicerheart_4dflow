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
  derived/
    velocity.npz         velocity_t (nt, nk, nj, ni, 3) float32 mm/s, world
                         components; magnitude_t (nt, nk, nj, ni); voxel_to_world
                         (4, 4); venc_mm_s (3,); rr_ms. Produced by the SlicerHeart
                         Import 4D Flow module from dicom/, so that this package's
                         tests do not depend on a DICOM reader, and so that the
                         Import module's own tests can compare against it.
    lumen.seg.nrrd       segmentation used for the reference analysis
    planes.json          measurement planes: world centre, normal, radius, label
  reference/
    flows.json           {"planes": {name: {"net_volume_mL": ..., "forward_volume_mL": ...,
                                            "peak_velocity_mm_s": ...}},
                          "conservation_checks": [[label, [inflow names], [outflow names]]],
                          "ratio_checks": [[label, numerator, denominator, expected]]}
                         from the reference software (cvi42 version noted), plus
                         2D phase-contrast values where available
```

## manifest.json

```json
{
  "case_id": "chop_siemens_001",
  "vendor": "siemens",
  "scanner": "MAGNETOM Avanto Fit 1.5 T",
  "software_version": "syngo MR E11",
  "sequence": "compressed-sensing 4D flow, retrospectively gated, navigator",
  "voxel_size_mm": [1.6, 1.6, 1.6],
  "n_phases": 25,
  "rr_ms": 650.0,
  "venc_mm_s": [1500.0, 1500.0, 1500.0],
  "aliasing_present": true,
  "contrast": "ferumoxytol",
  "age_group": "child",
  "diagnosis_category": "coarctation",
  "deidentification": {"tool": "...", "date": "YYYY-MM-DD", "private_tags": "removed", "dates": "shifted"},
  "reference_software": "cvi42 4D Flow 6.0.4",
  "license": "CC BY 4.0",
  "doi": "10.5281/zenodo.XXXXXXX",
  "files": {"velocity": "derived/velocity.npz", "lumen": "derived/lumen.seg.nrrd",
            "planes": "derived/planes.json", "reference_flows": "reference/flows.json"}
}
```

No field identifies a patient. `age_group` is a category, not an age;
`diagnosis_category` is one of a short fixed list. Dates in the DICOM header
are shifted, private tags removed, and the magnitude images reviewed for
burned-in text before release.

## What the clinical tests check (`tests/test_clinical.py`)

1. the manifest validates and the derived arrays load with consistent shapes
2. `quality_report` on the raw data: venc adequacy agrees with
   `aliasing_present`, residual static velocity after `war+msac` is under
   0.6 cm/s, lumen divergence falls after `fdm`
3. net volume per beat at each reference plane, after background correction
   and unwrapping, agrees with the reference software within the limits of
   agreement reported in the manuscript (forward volume bias within 2 mL,
   limits 7 mL), and inflow/outflow consistency is within 5%
4. every recorded number goes to the regression baseline

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
