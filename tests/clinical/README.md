# Clinical 4D flow test cases

Real scanner data for end-to-end tests: DICOM import, encoding-direction
handling, the actual noise and background-offset characteristics of each
vendor, and flow quantification against a commercial reference. Phantoms
and the CFD field cannot test those. One folder per vendor; each holds a
`README.md` with vendor notes and, once data exist, a `manifest.json`.

Data are **not** stored in git. The cases are planned to be published on
Zenodo (de-identified, CC BY 4.0, with a DOI) and then downloaded on first
use, following the pattern used for the CFD data in `conftest.py`, but they
are not released publicly yet. Until then, the cases are kept locally, in the
folder named by `SLICERHEART_4DFLOW_CLINICAL_DATA` (default
`tests/data/clinical/<vendor>/`), and are available only to those who have
access to the data. Tests that need a case are marked `clinical` and skip
when its manifest is absent:

    python -m pytest -m clinical            # only the clinical tier
    python -m pytest -m synthetic           # everything else (phantoms and CFD example)

The overall plan for the synthetic, semi-synthetic and clinical tiers and
the transfer audit between them is in `docs/VALIDATION_STRATEGY.md`.

## Status

| vendor | case | status |
|---|---|---|
| Siemens | CHOP, 1.5 T Avanto Fit, compressed-sensing 4D flow, pediatric CHD | 15 de-identified cases (Patient0001 to Patient0015), local only (`"license": "internal"`); publication on Zenodo is planned |
| Philips | none yet | placeholder |
| GE | none yet | placeholder |

## What a case contains

Cases are written in 3D Slicer's native file formats by the SlicerHeart
Import 4D Flow module (export format "Test case", or its batch script
`Import4DFlow/Testing/Python/Export4DFlowTestCases.py`, which imports a
folder of de-identified DICOM patients). They are read without Slicer and
without a DICOM reader by `slicerheart_4dflow.slicer_io.load_case`, and loaded
back into Slicer, without DICOM import, by `Import4DFlowLogic.loadTestCase`
("Load test case..." in Import 4D Flow).

```
<vendor>/<case_id>/
  manifest.json                format_version (3), case_id, license, doi
  inputs/                      data converted from DICOM, without any processing
    acquisition.json           description of the acquisition (see below)
    magnitude.seq.nrrd         magnitude images, one frame per trigger time
    velocity_raw.seq.nrrd      velocity as acquired, aliased in [-venc, venc), mm/s
  outputs/                     results of the SlicerHeart processing
    processing.json            {"software": {"slicer", "slicerheart_4dflow"},
                                "processing": {noise mask, unwrapping, background, ... options}}
    velocity.seq.nrrd          after the SlicerHeart corrections, mm/s
    wrap_count.seq.nrrd        wraps applied by SlicerHeart, int16 per encoded velocity
                               component (not a spatial vector): velocity =
                               encoding_to_world (raw_encoded + 2 venc wraps) - background
    pcmra.nrrd                 velocity-weighted PC-MRA
    flows.json                 flow through each plane by the SlicerHeart Analyze 4D Flow plane probe:
                               {"software", "sampling_distance_mm", "time_ms", "rr_ms",
                                "planes": {name: {"net_volume_mL", "forward_volume_mL",
                                 "backward_volume_mL", "peak_velocity_mm_s",
                                 "flow_rate_ml_s": [per frame], "area_mm2": [per frame]}},
                                "conservation_checks": [[label, [inflow names], [outflow names]]],
                                "ratio_checks": [[label, numerator, denominator, expected]]}
    quality_report.json        quality_report() of the corrected velocity
  annotations/                 generated automatically or placed/edited by a person
    annotations.json           how the lumen and planes were created (see below)
    lumen.seg.nrrd             lumen segment (ID "lumen")
    <name>.mrk.json            measurement planes (disc radius = half the plane size)
  references/<software>/       optional: results of independent software, for example references/cvi42/
    flows.json                 same structure as outputs/flows.json
```

The velocity volumes are spatial vectors: the files store LPS components (as
every Slicer volume file, with a measurement frame), and `load_case(world="RAS")`
returns RAS. The raw velocity is stored in world components too; `load_case` also
returns it as encoded components (`velocity_raw_encoded_t`, the velocity along each
column of `encoding_to_world`, each aliased at its own venc), which is what
background correction and unwrapping operate on. The wrap counts are per encoded
component; they are not spatial vectors (no measurement frame) and are read as
stored, in any world frame. Velocity is in mm/s; the velocity files also store
it in the NRRD header (`units` and `labels: "velocity"` of the component axis)
when they are written by a Slicer version that supports it. Each image file holds
only image data and geometry; node names, DICOM fields, dates and UIDs are not
written, and `test_files_carry_no_identifiers` checks this.

Only the current format version is supported: cases of an earlier version must
be exported again.

### Plane flows: consistency and validation

`outputs/flows.json` is computed by SlicerHeart from `outputs/velocity.seq.nrrd`
and the annotations. The plane probe samples the lumen inside the plane's disc,
which is the same region that `plane_flow` samples, so comparing with it tests
the implementations against each other (consistency), and plane pairs give
mass-conservation checks. Results of independent software, such as a commercial
analysis of the same planes, go to `references/<software>/flows.json`; agreement
with those is validation. A case has a `references/` folder only if such results
are available.

### Lumen and planes

For cases exported from local data:

- **Lumen.** `lumen.seg.nrrd` is estimated from the PC-MRA (Otsu threshold,
  connected components that are not image-border artifacts).
- **Planes.** Pairs of planes 4 mm apart along tubular vessels with the highest
  net flow, perpendicular to the vessel axis (`Analyze4DFlowLogic.findMeasurementPlanes`).

`annotations/annotations.json` describes how they were created:

```json
{
  "lumen": {
    "file": "lumen.seg.nrrd", "segment": "lumen",
    "method": "automatic", "reviewed": false,
    "software": {"name": "SlicerHeart Import 4D Flow", "slicer": "5.12.4", "slicerheart_4dflow": "0.1.1"},
    "algorithm": {"name": "Import4DFlowLogic.estimateLumenMask", "description": "...",
                  "parameters": {"minimumComponentVolumeMl": 1.0}, "inputs": ["outputs/pcmra.nrrd"]}
  },
  "planes": {
    "files": ["site1_a.mrk.json", "site1_b.mrk.json"],
    "method": "automatic", "reviewed": false,
    "software": {"name": "SlicerHeart Analyze 4D Flow", "slicer": "5.12.4", "slicerheart_4dflow": "0.1.1"},
    "algorithm": {"name": "Analyze4DFlowLogic.findMeasurementPlanes", "description": "...",
                  "parameters": {"numberOfSites": 3, "pairDistanceMm": 4.0, "minimumSiteDistanceMm": 30.0},
                  "inputs": ["outputs/velocity.seq.nrrd", "annotations/lumen.seg.nrrd"]}
  }
}
```

`method` is `automatic`, `manual` (specified when the case was exported), or
`automatic, manually edited`; `reviewed` is set by a person. File names are
relative to `annotations/`, algorithm inputs to the case folder. Exporting a case
again keeps the existing lumen and planes (unless overwriting them is requested).
To review or edit them:

1. Load the case in Slicer ("Load test case...") and edit the lumen segment and
   planes.
2. Save them with File / Save data. The case files are overwritten.
3. In `annotations.json`, set `method` to `automatic, manually edited` and
   `reviewed` to `true` (or only `reviewed` if the automatic result is acceptable).
4. Run the batch script with `--regenerate-flows` (no DICOM import) to
   compute `outputs/flows.json` for the edited annotations.

Mass conservation is enforced only for planes that were placed or reviewed by a
person: automatic planes in small pediatric vessels (a few voxels across) are
only recorded in the metrics.

### Running the clinical tests

Set `SLICERHEART_4DFLOW_CLINICAL_DATA` to the folder that holds the `<vendor>/<case>/` folders, then run pytest with the
`clinical` marker.

Linux and macOS:

    export SLICERHEART_4DFLOW_CLINICAL_DATA=/data/4Dflow/_test_cases_anonymized_
    python -m pytest -m clinical                                   # all cases
    python -m pytest -m clinical -k Patient0006                    # one case
    python -m pytest -m clinical --update-baseline                 # accept the current clinical metrics
    python -m pytest -m clinical -k Patient0006 --update-baseline  # accept them for one case

Windows (command prompt):

    set SLICERHEART_4DFLOW_CLINICAL_DATA=D:\Data\4Dflow\_test_cases_anonymized_
    python -m pytest -m clinical
    python -m pytest -m clinical -k Patient0006
    python -m pytest -m clinical --update-baseline
    python -m pytest -m clinical -k Patient0006 --update-baseline

`-k` selects tests by name: the tests of a case have the case ID in their names, so `-k Patient0006` runs all tests
of that case, and `-k "Patient0006 and unwrap"` only its unwrapping tests.

The baseline of the clinical metrics is stored with the data, in one file for
all cases: `<clinical data folder>/tests/slicerheart_4dflow_baseline.json`
(for example `D:/Data/4Dflow/_test_cases_anonymized_/tests/`). It has a section
per case (key: the case folder, for example `siemens/Patient0006`) with the
metric values, the package version, the date, and the sha256 hash of every case
file the values were computed from (`inputs/`, `outputs/`, `annotations/`,
`references/`). The file is sorted and indented, so it can also be edited by
hand. If any file of a case changes, for example after a re-export or an edited
lumen or plane, the baseline of that case is stale: its metrics are recorded but
not compared, and a warning lists the changed files. `--update-baseline` writes
the sections of the tested cases and keeps the others, so a single case can be
updated with, for example,
`python -m pytest -m clinical -k Patient0006 --update-baseline`. The values of
the last run are in `.test-reports/clinical_latest.json` at the repository root
(not committed; `SLICERHEART_4DFLOW_TEST_REPORT_DIR` sets another folder).

## manifest.json and inputs/acquisition.json

```json
{
  "format_version": 3,
  "case_id": "chop_siemens_001",
  "license": "CC BY 4.0",
  "doi": "10.5281/zenodo.XXXXXXX"
}
```

```json
{
  "vendor": "siemens",
  "scanner": "MAGNETOM Avanto Fit 1.5 T",
  "software_version": "syngo MR E11",
  "field_strength_t": 1.5,
  "sequence": "compressed-sensing 4D flow, retrospectively gated, navigator",
  "contrast": "ferumoxytol",
  "rr_ms": 650.0,
  "venc_encoded_mm_s": [1500.0, 1500.0, 1500.0],
  "encoding_to_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "component_encoding": [{"direction_code": "through", "max_value": 4096.0, "invert_to_ras": true}, "..."],
  "subject": {"age_group": "child", "diagnosis_category": "coarctation"},
  "deidentification": {"tool": "...", "date": "YYYY-MM-DD", "private_tags": "removed", "dates": "shifted"}
}
```

`venc_encoded_mm_s` is per encoded component; `encoding_to_world` has the LPS
directions of the encoded components as columns (`load_case` also returns the
venc per world component, `venc_mm_s`). `component_encoding` describes how the
phase images were decoded (used to load the case back into Slicer). Local cases
that are not published use `"license": "internal"` and `"doi": null`.

No field identifies a patient. `age_group` is a category, not an age;
`diagnosis_category` is one of a short fixed list. Dates in the DICOM header
are shifted, private tags removed, and the magnitude images reviewed for
burned-in text before release.

## What the clinical tests check (`tests/test_clinical.py`)

1. **Arrays.** The manifest and the acquisition description validate, and the
   arrays load with consistent shapes, geometry and frame counts.
2. **No identifiers.** Files carry no identifiers: NRRD headers hold only
   geometry and Slicer metadata, and the description files hold no names, dates or UIDs.
3. **One pipeline.** The raw velocity is within ±venc, and the corrected
   velocity equals raw + 2·venc·wraps minus a smooth background, so the files
   describe one pipeline.
4. **Encoding.** `encoding_check` finds the import's component mapping (the
   SlicerHeart `axisForDirectionCodes` and sign flags) consistent.
5. **Background.** Every background method (`ols`, `msac`, `wrls`, `war+msac`)
   leaves less than 0.6 cm/s in static tissue. The difference to the SlicerHeart
   fit is recorded.
6. **Unwrapping.** Every unwrapping method agrees with the reviewed SlicerHeart
   wraps in more than 99% of the lumen voxel-frames, and the unwrapped peak
   speed is physiological.
7. **Quality report.** The venc adequacy is consistent with the wraps that SlicerHeart applied.
8. **Plane flow, consistency.** `plane_flow` agrees with the SlicerHeart plane probe
   (`outputs/flows.json`): net volume within max(0.5 mL, 8%) with the same sampling,
   and the flow curve within 15% of its peak.
9. **Plane flow, validation.** `plane_flow` agrees with independent software
   (`references/<software>/flows.json`, if the case has any): net volume within
   max(0.5 mL, 15%).
10. **Mass conservation.** Plane groups conserve mass within 10% (enforced for
   reviewed planes).
11. **Divergence filtering.** `fdm`, `relaxed` and `leray` reduce the lumen
    divergence.
12. **World frame.** Reading the case in RAS instead of LPS changes no result.
13. **Pathlines.** Pathlines from each plane pair stay in the lumen.
14. **Hemodynamics.** Kinetic energy, TAWSS, OSI and the relative pressure drop
    are physiological.
15. **Metrics.** Every recorded number goes to the clinical regression baseline.

## De-identification checklist before uploading a case

- DICOM: remove patient name, ID, birth date, all private tags unless
  needed for velocity decoding (list those kept in `inputs/acquisition.json`), shift
  acquisition dates, remove institution and operator names, clear
  `ImageComments` and `StudyDescription` if they carry identifiers.
- Check magnitude images for burned-in annotations.
- Segmentation and plane files carry no header metadata beyond geometry.
- Plane flows are numbers only.
- IRB approval for data sharing on file; note the protocol number in the
  Zenodo record, not in the repository.
