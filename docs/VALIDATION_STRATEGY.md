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

### Tier 1.5: semi-synthetic from real cases (to add once a case is in hand)

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

Cost: one function (`tests/semisynthetic.py`) and a few minutes per case.
It should be added to the clinical tier, not the default CI, because it
needs the case data.

### Tier 2: clinical (skeleton exists in `tests/test_clinical.py`)

Real cases, one manifest per case (schema in `tests/clinical/README.md`).
No ground truth, so the checks are of four kinds:

1. Internal consistency: residual static-tissue velocity after correction
   (below 0.6 cm/s, Gatehouse 2010), mass conservation at branch points
   (inflow versus sum of outflows within 5%), Qp:Qs near 1 in cases without
   a shunt, flow continuity along a vessel between planes, net volume of
   a regurgitation-free valve equal to the stroke volume from the cine.
2. Agreement with reference measurements: 2D phase-contrast flow at the
   same planes where acquired (the clinical standard), and the numbers the
   reference software (cvi42, Arterys) reports for the same segmentation
   and planes, within the limits of agreement in the manuscript (bias
   within 2 mL, limits 7 mL).
3. Stability: the invariance tests of tier 1 (world transform, velocity
   scale, mask dilation by a voxel) applied to the real case; a filter whose
   output moves more than its stated tolerance under a one-voxel change of
   the segmentation is not ready.
4. Regression: every number goes into a clinical baseline
   (`tests/metrics/clinical_baseline.json`, separate from the synthetic one)
   with the same 5% rule, so a change in the code that moves a clinical
   number is caught even when it stays inside the hard limits.

Plus a human step: the Markdown quality report and a pathline rendering per
case, reviewed once when the case is added and whenever its baseline is
updated.

Where it runs: the Siemens cases are planned for Zenodo (de-identified,
CC BY 4.0), so the clinical tier can run on GitHub Actions nightly or on
demand with the data downloaded on first use, the same way the CFD file is.
Cases that cannot be published run on a lab machine with
`SLICERHEART_4DFLOW_CLINICAL_DATA` pointing at them, and the resulting
`clinical_baseline.json` is what gets committed. Either way it is not on the
pull-request path: it is slower and it needs the data.

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

The audit is a script (`tests/transfer_audit.py`, to write once the first
case exists) that, for a given clinical case:

1. reads its measured properties: venc, voxel size, number of frames,
   sigma_v from static tissue, aliased fraction, air fraction, lumen size in
   voxels;
2. runs the synthetic tier at those settings: the CFD field (or the tier
   1.5 reference of the same case) downsampled to the same voxel size and
   acquired at the same venc and SNR;
3. prints the paired quantities side by side, synthetic versus clinical,
   and flags any pair outside a band (a factor of two to start; tighten as
   we learn what normal is).

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

## 4. The order of work when the first case arrives

1. Import it with the SlicerHeart Import 4D Flow module to
   `derived/velocity.npz` and fill the manifest; run the encoding check and
   the quality report; look at the report and the pathlines. This alone will
   tell us whether the component mapping, venc and units are right, which
   is where real data fail first.
2. Run the clinical tier as it stands (manifest, arrays, pipeline against
   the reference flows). Fix what breaks.
3. Run the transfer audit. Record the paired table in the case's folder.
4. Build the tier 1.5 reference from the case and score WSS, pressure,
   unwrapping and the divergence filters against it.
5. Commit the clinical baseline. From then on the clinical tier runs
   nightly or before a release and the synthetic tier on every pull request.

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

## 6. What this does not solve

Reference software numbers are not truth either; two packages disagree by
amounts comparable to our tolerances. Agreement with cvi42 shows we are in
the same place as the field, not that the field is right. The flow phantom
and the tier 1.5 construction are the only places in this plan with a true
answer on scanner-like data, which is why both are in it.
