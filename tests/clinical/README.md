# Clinical 4D flow test cases

Real scanner data for end-to-end tests: DICOM import, encoding-direction
handling, the actual noise and background-offset characteristics of each
vendor, and flow quantification against a commercial reference. Phantoms
and the CFD field cannot test those. The folder of each vendor here holds a
`README.md` with vendor notes; the cases of all vendors are in one data set.

Data are **not** stored in git. The data set (de-identified, CC BY 4.0) is published in the
[TestingData release](https://github.com/JolleyLab/slicerheart_4dflow/releases/tag/TestingData) as the
`clinical_*_<revision>` files (one zip per folder of the data set) and downloaded on first use into the testing data
cache (`slicerheart_4dflow.testing_data`, about 4 GB). The manifest of the revision, `clinical_manifest_<revision>.json`
(kept in `tests/` too, and the `manifest.json` of the data set), identifies the data set: its metadata (format version,
name, license, DOI), the revision, and the path, size and sha256 of every other file with the zip that contains it. A
case is downloaded when a test first needs it; later runs check the cache by the size and modification time of each
file (see "Testing data cache" below). Tests that need a case are marked `clinical` and run on each case of the data
set:

    python -m pytest -m clinical            # only the clinical tier
    python -m pytest -m synthetic           # everything else (phantoms and CFD example)

The clinical tests also run on a synthetic case in this layout
(`tests/clinical_case.py`), as synthetic tests, so that the test code is checked
in CI without clinical data.

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

A data set holds the inputs of each case (what the processing starts from:
replaced only by a new export or a review) separately from the outputs
(everything computed from the inputs: can be deleted and regenerated). The case
ID is the name of its folder in `inputs/`; the cases of the data set are those
folders.

```
<data set>/
  manifest.json                format_version (4), name, license, doi of the data set
  inputs/<case_id>/
    acquisition/               data converted from DICOM, without any processing
      acquisition.json         description of the acquisition (see below)
      magnitude.seq.nrrd       magnitude images, one frame per trigger time
      velocity_raw.seq.nrrd    velocity as acquired, aliased in [-venc, venc), mm/s
    annotations/               generated automatically or placed/edited by a person
      annotations.json         how the lumen and planes were created (see below)
      Segmentation.seg.nrrd    lumen segment (ID "lumen"); may be on a grid finer than the image (2x for the
                               automatic lumen), load_case returns the fraction of each voxel in it; optionally the
                               branch segments of the planes (see "Planes placed by hand"); annotations.json names the
                               file and segment of the lumen and of the branches (older data sets: lumen.seg.nrrd)
      <name>.mrk.json          measurement planes (disc radius = half the plane size)
    references/<source>/       optional: reference measurements of the planes, for example references/cvi42/
      flows.json               same structure as outputs/<case_id>/flows.json; per plane any of net_volume_mL,
                               forward_volume_mL, backward_volume_mL, peak_velocity_mm_s, flow_rate_ml_s (with
                               time_ms, the frames of the curve, if it lacks frames), plane_normal_sign (-1 if the
                               positive flow direction of the reference is opposite to the plane normal)
  outputs/<case_id>/           results of the SlicerHeart processing of the inputs
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
  tests/
    slicerheart_4dflow_baseline.json   baseline of the clinical metrics (see below)
```

`load_case(data_set, case_id)` reads a case; without its outputs (deleted to be
regenerated) it reads the inputs only (`has_outputs` is false).

The velocity volumes are spatial vectors: the files store LPS components (as
every Slicer volume file, with a measurement frame), and `load_case(world="RAS")`
returns RAS. The raw velocity is stored in world components too; `load_case` also
returns it as encoded components (`velocity_raw_encoded_t`, the velocity along each
column of `encoding_to_world`, each aliased at its own venc), which is what
background correction and unwrapping operate on. The wrap counts are per encoded
component; they are not spatial vectors (no measurement frame) and are read as
stored, in any world frame. Velocity is in mm/s; the velocity files also store
it in the NRRD header (`units: mm/s` and `labels: velocity` of the component axis). Each image file holds
only image data and geometry; node names, DICOM fields, dates and UIDs are not
written, and `test_files_carry_no_identifiers` checks this.

Only the current format version is supported: data sets of an earlier version
must be converted or exported again.

### Plane flows: consistency and validation

`outputs/<case_id>/flows.json` is computed by SlicerHeart from the processed
velocity and the annotations. The plane probe samples the lumen inside the plane's disc,
which is the same region that `plane_flow` samples, so comparing with it tests
the implementations against each other (consistency), and plane pairs give
mass-conservation checks. Reference measurements go to
`inputs/<case_id>/references/<source>/flows.json`; agreement with those is
validation. A case has a `references/` folder only if such measurements are
available. Sources:

- `pc2d`: 2D phase contrast of the same exam at the plane (net volume);
- `cine`: ventricular stroke volume from the cine volumetry of the same exam,
  minus the regurgitant volume of the inflow valve, as the forward volume of the
  outflow plane (aorta for the left, pulmonary artery for the right ventricle);
- `cvi42`: values recorded when the case was analyzed in cvi42 (there is no live
  access to the software);
- any other name: results of independent software.

The net volume of `plane_flow` is the mean flow times the R-R interval, while
Analyze4DFlow integrates over the frame times with the trapezoid rule; they
differ by the handling of the last frame, so compare them on the same plane
before interpreting a tolerance. The R-R interval should come from the DICOM
nominal interval (heart rate), not from the span of the frame times, which is
one frame short.

### Lumen and planes

For cases exported from local data:

- **Lumen.** The `lumen` segment of `Segmentation.seg.nrrd` is estimated by `slicerheart_4dflow.estimate_lumen`
  on a grid twice as fine as the image: Otsu threshold of the PC-MRA, connected
  components that are not image-border artifacts, then grown toward the vessel
  wall into voxels with flow (the PC-MRA threshold alone cuts inside the wall
  and missed about 10% of the net flow). `load_case` returns `lumen_fraction`
  (fraction of each image voxel in the segment, for `plane_flow`), `lumen`
  (voxels at least half in it) and the segmentation on its own grid
  (`lumen_labelmap`).
- **Planes.** Pairs of planes 4 mm apart along tubular vessels with the highest
  net flow, perpendicular to the vessel axis (`Analyze4DFlowLogic.findMeasurementPlanes`).

`inputs/<case_id>/annotations/annotations.json` describes how they were created:

```json
{
  "lumen": {
    "file": "Segmentation.seg.nrrd", "segment": "lumen",
    "method": "automatic", "reviewed": false,
    "software": {"name": "SlicerHeart Import 4D Flow", "slicer": "5.12.4", "slicerheart_4dflow": "0.1.1"},
    "algorithm": {"name": "slicerheart_4dflow.estimate_lumen", "description": "...",
                  "parameters": {"minimumComponentVolumeMl": 1.0, "oversampling": 2, "growthVoxels": 1.5, "growthNoiseFactor": 2.0},
                  "inputs": ["outputs/Patient0001/pcmra.nrrd", "outputs/Patient0001/velocity.seq.nrrd",
                             "inputs/Patient0001/acquisition/magnitude.seq.nrrd"]}
  },
  "planes": {
    "files": ["site1_a.mrk.json", "site1_b.mrk.json"],
    "method": "automatic", "reviewed": false,
    "software": {"name": "SlicerHeart Analyze 4D Flow", "slicer": "5.12.4", "slicerheart_4dflow": "0.1.1"},
    "algorithm": {"name": "Analyze4DFlowLogic.findMeasurementPlanes", "description": "...",
                  "parameters": {"numberOfSites": 3, "pairDistanceMm": 4.0, "minimumSiteDistanceMm": 30.0},
                  "inputs": ["outputs/Patient0001/velocity.seq.nrrd", "inputs/Patient0001/annotations/Segmentation.seg.nrrd"]}
  }
}
```

`method` is `automatic`, `manual` (specified when the case was exported), or
`automatic, manually edited`; `reviewed` is set by a person. File names are
relative to the annotations folder, algorithm inputs to the data set folder. The
automatic lumen and planes are computed from outputs (the PC-MRA and the processed
velocity) when a case is first exported, and are then inputs: they are kept when
the outputs are regenerated or the case is exported again (unless overwriting them
is requested), because they may have been reviewed or edited; their algorithm
inputs name the files they were computed from at that time. To review or edit
them:

1. Load the case in Slicer ("Load test case...") and edit the lumen segment and
   planes.
2. Save them with File / Save data. The case files are overwritten.
3. In `annotations.json`, set `method` to `automatic, manually edited` and
   `reviewed` to `true` (or only `reviewed` if the automatic result is acceptable).
4. Run the batch script with `--regenerate-outputs` (no DICOM import) to
   compute the outputs of the case for the edited annotations.

### Planes placed by hand

Planes can also be placed by hand, for example at the planes of an analysis in other software whose results are
the references of the case (`references/cvi42/`). The disc of a plane placed by hand is often large (Slicer plane
markups are sized for viewing) and cuts other vessels near the plane, so the lumen within the disc can be twice the
cross-section of the vessel. A branch segmentation, drawn approximately around each vessel, gives the measurement
region of each plane: the lumen within the branch segment of the plane (`plane_lumen(case, name)`, from
`plane_lumen_fraction`, `plane_lumen` and `plane_lumen_labelmap` of `load_case`; the SlicerHeart plane probe measures
in the same region). For checking the flow, secondary planes can be placed along the vessel on both sides of each plane
(suffix `_prox` upstream and `_dist` downstream, a few mm away), so that the flow through the three planes must be the
same: their differences are listed as conservation checks, and the spread of the three (`placement_relative_spread` in
the metrics) shows how much the flow depends on the exact position of the plane:

```json
"planes": {
  "files": ["Ao.mrk.json", "Ao_prox.mrk.json", "Ao_dist.mrk.json"],
  "method": "manual", "reviewed": true,
  "secondary": {"files": ["Ao_prox.mrk.json", "Ao_dist.mrk.json"], "method": "automatic", "reviewed": false,
                "planes": {"Ao_prox": "Ao", "Ao_dist": "Ao"},
                "not_placed": {"MPA_dist": "cross-section not similar to the one at the plane (branch)"},
                "algorithm": {"name": "..."}},
  "conservation_checks": [["Ao proximal", ["Ao_prox"], ["Ao"]], ["Ao distal", ["Ao"], ["Ao_dist"]]],
  "segmentation": {"file": "Segmentation.seg.nrrd", "method": "manual", "reviewed": true,
                   "segments": {"Ao": "Ao", "Ao_prox": "Ao", "Ao_dist": "Ao"}}
}
```

A conservation check is `[label, [inflow planes], [outflow planes]]`: the upstream plane first. Checks that include secondary planes are enforced only after the secondary planes were reviewed (`"reviewed": true`); until then they are recorded. The normal of a plane
placed by hand may point upstream; a reference along the opposite direction has `"plane_normal_sign": -1` for the
plane, and its values are compared along the plane normal (net volume and flow rate times -1, forward volume minus the
backward volume of the reference).

Mass conservation is enforced only for planes that were placed or reviewed by a
person: automatic planes in small pediatric vessels (a few voxels across) are
only recorded in the metrics.

### Running the clinical tests

Run pytest with the `clinical` marker; the cases are downloaded into the testing data cache on first use:

    python -m pytest -m clinical                                   # all cases
    python -m pytest -m clinical -k Patient0006                    # one case (downloads only that case)

To use a local data set folder instead (for example to work on a new revision, or to record a new baseline, which is
written into the data set), set `SLICERHEART_4DFLOW_CLINICAL_DATA` to the folder of the data set (with `manifest.json`).
Nothing is downloaded then.

Linux and macOS:

    export SLICERHEART_4DFLOW_CLINICAL_DATA=/data/4Dflow/_test_cases_anonymized_rev04
    python -m pytest -m clinical                                   # all cases
    python -m pytest -m clinical -k Patient0006                    # one case
    python -m pytest -m clinical --update-baseline                 # accept the current clinical metrics
    python -m pytest -m clinical -k Patient0006 --update-baseline  # accept them for one case

Windows (command prompt):

    set SLICERHEART_4DFLOW_CLINICAL_DATA=D:\Data\4Dflow\_test_cases_anonymized_rev04
    python -m pytest -m clinical
    python -m pytest -m clinical -k Patient0006
    python -m pytest -m clinical --update-baseline
    python -m pytest -m clinical -k Patient0006 --update-baseline

`-k` selects tests by name: the tests of a case have the case ID in their names, so `-k Patient0006` runs all tests
of that case, and `-k "Patient0006 and unwrap"` only its unwrapping tests.

### Testing data cache

The data set is extracted into `<base folder>/clinical_<revision>/` (several revisions can be kept in one base
folder). The base folder is the folder named by `SLICERHEART_4DFLOW_TESTDATA`, or else
`.SlicerHeart/slicerheart_4dflow_testingdata` in the profile folder of the user, so that the tests of every clone of
the repository, every Python environment and the SlicerHeart modules in every 3D Slicer version share one copy. Each zip is
downloaded when a test first needs a file in it, verified by its sha256, extracted and deleted. `cache_state.json` records the
size, modification time and sha256 of every extracted file: a file with the recorded size and modification time is up
to date without being read; otherwise its sha256 is computed, and its zip is downloaded again if the content changed.
Files edited in the cache are therefore restored to the release; edit a local data set folder instead. To download the
whole data set at once (for example before going offline):

    python -m slicerheart_4dflow.testing_data download rev001

The tests download through `slicerheart_4dflow.clinical_test_data`, which can be used outside the tests too: it
downloads a revision into the base folder (or `base_dir`) and returns the full path of the data set root:

    import slicerheart_4dflow as f4d
    root = f4d.clinical_test_data("rev001")
    case = f4d.load_case(root, f4d.dataset_cases(root)[0])

In CI (`.github/workflows/ci.yml`) the `clinical` job keeps the extracted data set in the GitHub Actions cache,
keyed by the hash of the manifest, so it is downloaded and extracted only once per revision.

### Publishing a new revision of the data set

1. Write the release assets of the data set folder with a new revision:
   `python -m slicerheart_4dflow.testing_data package <data set folder> <output folder> rev002`
   (it prints the sha256 of the manifest).
2. Upload the files of the output folder to the TestingData release (the earlier revisions stay, so older versions
   of the tests and of the package keep working).
3. Add the revision and the sha256 of its manifest to `MANIFEST_SHA256` (and `DEFAULT_REVISION`) in
   `src/slicerheart_4dflow/testing_data.py`; copy `clinical_manifest_rev002.json` to `tests/`, set
   `CLINICAL_MANIFEST` in `tests/conftest.py` to it and remove the previous manifest from `tests/`; copy it to the
   data set folder as its `manifest.json`.

The baseline of the clinical metrics is stored with the data, in one file for
all cases: `<clinical data folder>/tests/slicerheart_4dflow_baseline.json`
(for example `D:/Data/4Dflow/_test_cases_anonymized_/tests/`). It has a section
per case (key: the case ID, for example `Patient0006`) with the metric values,
the package version, the date, and the sha256 hash of every file of the case the
values were computed from (`inputs/<case_id>/` and `outputs/<case_id>/`). The
file is sorted and indented, so it can also be edited by hand. If any file of a
case changes, for example after a re-export, regenerated outputs or an edited
lumen or plane, the baseline of that case is stale: its metrics are recorded but
not compared, and a warning lists the changed files. `--update-baseline` writes
the sections of the tested cases and keeps the others, so a single case can be
updated with, for example,
`python -m pytest -m clinical -k Patient0006 --update-baseline`. The values of
the last run are in `.test-reports/clinical_latest.json` at the repository root
(not committed; `SLICERHEART_4DFLOW_TEST_REPORT_DIR` sets another folder).

## manifest.json and acquisition.json

The manifest describes the data set (one for all cases):

```json
{
  "format_version": 4,
  "name": "4D flow test cases (anonymized)",
  "license": "CC BY 4.0",
  "doi": "10.5281/zenodo.XXXXXXX"
}
```

`inputs/<case_id>/acquisition/acquisition.json` describes the acquisition of a
case and, optionally, the subject (`subject`) and the de-identification:

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
phase images were decoded (used to load the case back into Slicer). Local data
sets that are not published use `"license": "internal"` and `"doi": null`.

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
9. **Plane flow, validation.** `plane_flow` (in the measurement region of each plane) agrees with the references of the case
   (`references/<source>/flows.json`, if the case has any): net and forward volume
   within max(0.5 mL, 15%) (2D phase contrast, cine stroke volumes, independent
   software) or 7 mL (recorded cvi42 values).
10. **Mass conservation.** Plane groups conserve mass within 10% (enforced for
   reviewed planes).
11. **Divergence filtering.** `fdm`, `relaxed` and `leray` reduce the lumen
    divergence.
12. **World frame.** Reading the case in RAS instead of LPS, or moving it to an
    oblique world frame (rotation, reflection, translation), changes no result.
13. **Plane flow, stability.** Net flow through each plane changes by less than
    max(0.5 mL, 5%) when the lumen is dilated by one voxel.
14. **Semi-synthetic reference.** A clean field built from the case
    (`reference_from_measurement`), re-acquired at its venc, noise, magnitude and
    background offset (`reacquire`), is recovered by `preprocess` within 1.5 times
    the noise that the re-acquisition put into each voxel (`acquisition_noise`;
    low-signal voxels are noisier) where it unwraps correctly, with wrong wraps in
    less than 1% of the lumen voxel-frames.
15. **Transfer audit.** Quantities that need no ground truth are compared with a
    synthetic case matched to the case (`tests/transfer_audit.py`); reports go to
    `.test-reports/transfer_audit/`. Fails only if the synthetic case cannot be
    matched.
16. **Pathlines.** Pathlines from each plane pair stay in the lumen.
17. **Hemodynamics.** Kinetic energy, TAWSS, OSI and the relative pressure drop
    are physiological.
18. **Metrics.** Every recorded number goes to the clinical regression baseline.

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
