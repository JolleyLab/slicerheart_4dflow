# Validation strategy: synthetic tier, clinical tier, and how to tell when they disagree

Everything in `tests/` today runs on data with a known answer: analytic
phantoms, the 4DFlowNet aortic CFD field, and both passed through the
phase-contrast acquisition model (`phantom.acquire`). That is the right first
tier: it is fast (about five minutes, runs in CI on every pull request), it
has ground truth, and it catches the class of error the review pass found
(wrong transforms, wrong thresholds, wrong units). It cannot tell us how the
filters behave on a scanner's data, because the synthetic data only contain
the effects we put into them. This document sets out a second tier on real
CMR data, a bridge between the two so that a difference in behaviour is
detected rather than discovered in the clinic, and what we need from the
first clinical cases.

## 1. What real data have that the synthetic data do not

Listed so that each item is either put into the synthetic tier, measured in
the clinical tier, or knowingly left out.

| effect | in synthetic tier today | how it shows up on the scanner |
|---|---|---|
| velocity noise from complex Gaussian noise, sigma_v = sqrt(2) venc / (pi SNR) | yes (`acquire`) | yes, but SNR varies across the field of view with coil sensitivity, and the noise is spatially correlated by the reconstruction |
| partial volume at the wall | yes (complex block averaging) | yes, with the wall not at a voxel boundary and the segmentation itself approximate |
| aliasing | yes | yes, plus intravoxel dephasing in jets: the magnitude drops and the phase becomes noise rather than merely wrapped |
| background phase offset | yes, second-order polynomial | mostly polynomial (eddy currents) but with Maxwell and gradient-nonlinearity terms that the vendor may or may not have removed; spatial wrap-around of tissue |
| temporal structure | per-frame snapshots of a waveform | each frame is an average over many heartbeats (segmented acquisition), temporal resolution 30 to 40 ms, and with k-t or compressed-sensing reconstructions the frames are smoothed and the residual noise is structured, not white |
| respiratory and bulk motion | none | blurring, misregistration between the velocity-encoded acquisitions, displacement artefacts |
| static tissue | clean, uniform | moving chest wall, flowing CSF and venous blood inside the "static" candidates, lungs with near-zero magnitude |
| resolution | 0.6 mm CFD, 1 mm phantoms, downsampled to 1.5 to 2.5 mm | 1.5 to 3 mm, anisotropic, often with slice thickness larger than in-plane |
| metadata | exact | venc per axis, component sign conventions, frame times, prospective vs retrospective gating, units (cm/s or mm/s) all come from headers and are where imports go wrong |
| ground truth | exact | none; only internal consistency and reference measurements |

The last row is the important one. On real data we cannot measure error
against truth, so the clinical tier is built on quantities that can be
measured without truth and that the synthetic tier can also report.

## 2. The three tiers

### Tier 1: synthetic (exists)

Analytic phantoms, CFD, acquisition model. Ground truth. Runs on every pull
request through `tests/metrics/baseline.json`. Nothing changes here except
that its acquisition settings should be driven by what tier 2 measures (see
section 3).

### Tier 1.5: semi-synthetic from real cases (`semisynthetic.py`, `test_clinical.test_semisynthetic`)

Take a real case, run the full pipeline on it (background correction,
unwrapping, noise mask, FDM projection) and keep the result as a smooth,
divergence-free, physically plausible reference field with real anatomy,
real waveform shape and a real segmentation. Then re-acquire that reference
with `phantom.acquire` at the case's own venc, SNR and resolution, and run
the pipeline on the re-acquired data. Now there is a known answer, on an
anatomy and flow pattern the phantoms do not have (tortuous vessels,
branches, helical flow, jets at the real venc). This is the standard
"in-vivo-derived digital phantom" construction and it is where the
transposition question is answered most directly: same anatomy, controlled
perturbations, measured error.

`reference_from_measurement` builds the reference and `reacquire` re-acquires
it with the case's own magnitude image and background offset. The clinical
test runs `preprocess` on the re-acquired data and scores it against the
reference (`tests/test_semisynthetic.py` checks the construction on a phantom
with known truth). It runs in the clinical tier, not the default CI, because
it needs the case data; about a minute per case. Scoring WSS and pressure
against the reference is still to do.

### Tier 2: clinical (`tests/test_clinical.py`)

Real cases, in a data set that keeps the inputs of each case separate from the outputs of the processing (schema
in `tests/clinical/README.md`).
No ground truth, so the checks are of four kinds:

1. Internal consistency: residual static-tissue velocity after correction
   (below 0.6 cm/s, Gatehouse 2010), mass conservation at branch points
   (inflow versus sum of outflows within 5%), Qp:Qs near 1 in cases without
   a shunt, flow continuity along a vessel between planes, net volume of
   a regurgitation-free valve equal to the stroke volume from the cine.
2. Agreement with reference measurements (`references/<source>/flows.json`
   of a case): 2D phase-contrast flow at the same planes where acquired (the
   clinical standard), cine stroke volumes as the forward volume of the
   outflow planes, and the numbers that reference software (cvi42, Arterys)
   reported for the same segmentation and planes, within the limits of
   agreement in the manuscript (bias within 2 mL, limits 7 mL). There is no
   live access to commercial software, so its values are recorded when a
   case is analyzed with it.
3. Stability: the invariance tests of tier 1 (world transform, velocity
   scale, mask dilation by a voxel) applied to the real case; a filter whose
   output moves more than its stated tolerance under a one-voxel change of
   the segmentation is not ready.
4. Regression: every number goes into a clinical baseline with the same 5%
   rule, so a change in the code that moves a clinical number is caught even
   when it stays inside the hard limits. The baseline is stored with the data,
   in one file with a section per case
   (`<clinical data folder>/tests/slicerheart_4dflow_baseline.json`), with the
   hashes of the case files each section was computed from: a re-exported or
   re-annotated case makes its section stale instead of reporting false
   regressions.

Plus a human step: the Markdown quality report and a pathline rendering per
case, reviewed once when the case is added and whenever its baseline is
updated.

Where it runs: the Siemens cases are planned for Zenodo (de-identified,
CC BY 4.0), so the clinical tier can run on GitHub Actions nightly or on
demand with the data downloaded on first use, the same way the CFD file is.
Cases that cannot be published run on a lab machine with
`SLICERHEART_4DFLOW_CLINICAL_DATA` pointing at them; their baselines stay with
the data and are not committed. Either way it is not on the pull-request path:
it is slower and it needs the data.

## 3. The bridge: a transfer audit

The question "does the synthetic behaviour transfer" has to be asked per
filter, with a quantity that both tiers can report without ground truth.
For each filter there is one:

| filter | paired quantity (no truth needed) | synthetic value today |
|---|---|---|
| background correction | residual static velocity after correction, mean and RMS per component; fraction of candidates rejected by MSAC / WaR | 0.05 to 0.14 cm/s on the chest phantom |
| unwrapping | aliased voxel-frames detected (wrap count), residual jumps larger than venc along time after unwrapping, fraction of lumen voxels whose unwrapped component exceeds venc | 0 residual jumps with laplacian_then_temporal |
| noise mask | fraction of the field of view flagged; lumen voxels flagged (should be ~0); air-class ratio from `air_threshold` | 100% of air, 0% of lumen |
| divergence-free | lumen divergence RMS before and after, and the fraction of the velocity energy removed (should be near 1/3 of the noise energy for a projection, i.e. small relative to the signal) | sqrt(2/3) noise floor on white noise |
| encoding check | discrimination and the ratio of the declared mapping | 2 to 6 |
| velocity noise | sigma_v from magnitude-selected tissue; venc / sigma_v | matches the model to 2% |
| WSS, pressure | these need truth; only tier 1.5 can score them, tier 2 can only check stability and plausibility (WSS in the 0.5 to 3 Pa range, pressure drop against simplified Bernoulli) | within 10% at 1 mm |

The audit is implemented in `tests/transfer_audit.py`. For a given case it:

1. measures its properties: venc, voxel size, number of frames, R-R
   interval, sigma_v from magnitude-selected tissue, lumen, tissue and air
   magnitude, the vessel radius and peak mean velocity in the plane of the
   largest flow, the flow waveform there, and the fitted background offset;
2. builds a matched synthetic case: a bifurcation phantom on a grid of the
   same voxel size and frames, with that radius, peak velocity and waveform,
   in static tissue of that magnitude, with the clinical background offset
   (resampled and scaled to its RMS), acquired at that venc and noise
   (`phantom.acquire`);
3. runs the fixed pipeline of `preprocess()` on both (war+msac background,
   noise mask, laplacian_then_temporal unwrapping, FDM filter) and compares
   normalized paired quantities (for example the static residual over the
   noise of a time average, so that the comparison does not depend on the
   size of the field of view); a pair outside a factor of two is flagged.

The matched properties are measured on the synthetic case the same way as on
the clinical case and must agree within 25%, otherwise the comparison is not
meaningful. A self-test (`tests/test_transfer_audit.py`) audits a synthetic
case and must flag nothing. The clinical test `test_transfer_audit` writes a
report per case and a summary of all cases to
`.test-reports/transfer_audit/`, and records the values as record-only
metrics; a single case can be audited with
`python tests/transfer_audit.py <case folder>`.

A flagged pair is the signal that the synthetic model is missing something
for that filter. The response is not to tune the filter on the clinical
case (that is how synthetic and real diverge quietly) but to find the
missing effect and put it into the acquisition model or a phantom, then
check that tier 1 now reproduces the clinical number. Examples of what we
expect to find, from what is known about 4D flow data:

- The static-tissue residual will be larger on real data than on the
  phantom because the candidate set contains moving chest wall and venous
  flow. Expected fix: tighter candidate selection, or a higher-order
  polynomial where the vendor has not removed Maxwell terms; the phantom
  gains a moving-tissue component.
- The Laplacian unwrapping will see intravoxel-dephased jets, where the
  phase is noise rather than wrapped. The acquisition model gains a
  dephasing term (magnitude loss proportional to the intravoxel velocity
  spread).
- The noise mask's velocity-std rule will flag CSF and slow venous blood.
  Expected fix: raise the percentile or restrict the rule to low-magnitude
  voxels.
- The divergence after FDM will not be zero on real data at the mask
  boundary because the segmentation is not at the wall; the fraction of
  energy removed will be larger than 1/3 of the noise. The phantoms gain a
  segmentation-error model (mask dilated or eroded by a fraction of a
  voxel, already partly in `test_invariance`).

Each of these becomes a tier 1 test once observed, which is how the two
tiers converge.

## 4. The order of work for a new case

1. Export it from DICOM with the SlicerHeart Import 4D Flow module
   (`Export4DFlowTestCases.py`, see `tests/clinical/README.md`), review the
   automatically generated lumen and planes, run the encoding check and the
   quality report, and look at the report and the pathlines. This tells
   whether the component mapping, venc and units are right, which is where
   real data fail first.
2. Run the clinical tier on it. Fix what breaks.
3. Run the transfer audit (`test_transfer_audit`) and look at the flagged
   quantities; each flag is a candidate for a new effect in the phantoms.
4. Look at the semi-synthetic scores (`test_semisynthetic`).
5. Update the clinical baseline. The clinical tier runs before a release,
   the synthetic tier on every pull request.

## 4a. Multi-venc data (DRAFT)

`multivenc.py` is tested on the stenosis and thorax phantoms through
`phantom.acquire_multivenc`. The first in vivo target is the interleaved
dual-venc 5D flow acquisition of Weiss, Roy et al. (ISMRM 2024,
doi:10.58530/2024/4767): two velocity series on the same grid and frames,
venc ratio 1/2, shared reference scan. A case from that group, or two
consecutive single-venc 4D flow scans at venc and venc/2 with otherwise
identical parameters (Franco 2022's setup, `shared_reference=False`), is what
the clinical tier needs to add for this method. Comparisons to make: failure
fraction against the high-venc image, static-tissue noise ratio (expected near
the venc ratio; Weiss 2024 report an 85% VNR increase in vivo), and flow at
planes before and after.

## 5. What we need from the clinic

Three to five cases to start, chosen to cover the failure modes rather
than to be typical:

- a normal or near-normal study (consistency checks should pass cleanly;
  this is the one that calibrates the audit bands);
- a case with a jet above venc (aliasing persistent through the cycle:
  coarctation, pulmonary stenosis, a baffle) with the venc recorded;
- a shunt or a Fontan (Qp:Qs or a branch split with a known reference);
- if available, one case acquired twice (test-retest) or with both a
  compressed-sensing and a conventional reconstruction;
- if the MR physics group has a flow phantom with a pump, one scan of it:
  the only tier 2 data with a true flow rate, and the cleanest test of
  background correction and partial volume on a real scanner.

Per case, in addition to what the manifest already lists: 2D phase-contrast
flows at the same planes when acquired, the reference software's numbers
with the software version, the planes and segmentation used for those
numbers, the gating mode (prospective coverage fraction matters for the
pressure temporal term and net volumes), and whether the vendor applied
Maxwell and eddy-current corrections on the scanner (it changes what the
background correction has to remove). De-identification as in
`tests/clinical/README.md`.

## 6. Where each piece lives, and what has been validated on what

| piece | location | tier |
|---|---|---|
| phantoms, CFD, acquisition model | `phantom.py`, `tests/conftest.py` | synthetic |
| case reader (format of the SlicerHeart export) | `slicer_io.py` | clinical |
| clinical checks | `tests/test_clinical.py` | clinical |
| semi-synthetic reference | `semisynthetic.py`, `tests/test_semisynthetic.py` | both |
| transfer audit | `tests/transfer_audit.py`, `tests/test_transfer_audit.py` | both |
| baselines | `tests/metrics/baseline.json`; `<clinical data folder>/tests/slicerheart_4dflow_baseline.json` | one per tier |
| markers | `synthetic` (automatic), `clinical`, `cfd` | |

Validation status per method (update when a case or a check is added):

| method | phantoms | CFD | semi-synthetic | clinical |
|---|---|---|---|---|
| background correction | yes | no static tissue in the CFD | yes | consistency (static residual), audit |
| unwrapping | yes | yes | yes | against SlicerHeart, audit |
| noise mask / PC-MRA | yes | yes | indirectly | audit |
| divergence-free | yes | yes | | divergence reduction, audit |
| encoding check | yes | yes | | yes (Siemens), audit |
| plane flow, conservation | yes | yes | | against SlicerHeart, conservation, stability, audit |
| pathlines | yes | yes | | plausibility |
| WSS | yes | | not yet | plausibility |
| pressure | yes | | not yet | plausibility |
| quality report | yes | yes | | yes |

No case has reference measurements yet (2D phase contrast, cine or
recorded cvi42 values), so the clinical tier has no validation against an
independent measurement.

## 7. What this does not solve

Reference software numbers are not truth either; two packages disagree by
amounts comparable to our tolerances. Agreement with cvi42 shows we are in
the same place as the field, not that the field is right. The flow phantom
and the tier 1.5 construction are the only places in this plan with a true
answer on scanner-like data, which is why both are in it.
