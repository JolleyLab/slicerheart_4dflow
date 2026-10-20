# Background notes: 4D flow preprocessing filters for SlicerHeart

Working notes from the October 2026 review that produced this code. They record
why each method was chosen, what was ruled out, and the test results, so the
decisions can be revisited without redoing the survey.

## 1. Starting point

Alison Marsden asked about adding divergence-free filtering and eddy current
correction to the SlicerHeart 4D flow module. The module (Iacovella et al. 2026;
the JCMR framework manuscript) already had temporal phase unwrapping (Xiang 1995),
polynomial background phase correction on static tissue (Walker 1993 / Pruitt 2019
style), registration-propagated segmentation, measurement-plane quantification,
and 4D/5D NRRD support in ITK.

## 2. Library survey

There is no maintained open-source library for either capability.

- BART (Berkeley Advanced Reconstruction Toolbox) once shipped divergence-free
  wavelet thresholding (`bart threshold -D`). Development is active (moved from
  GitHub to Codeberg March 2026; commits September 2026), but the current source
  stubs `-D` to `error("divergence free wavelet not supported anymore")` and the
  DFW code is gone from the tree. Not usable.
- Ong and Lustig's reference DFW code (MRM 2015) is on Lustig's software page as a
  MATLAB zip with no stated license. Pirrello (Politecnico di Torino thesis, 2024)
  reimplemented it in C/CUDA/MATLAB. No Python port exists.
- Eddy current / background phase correction: every open tool carries its own copy
  of the Walker polynomial fit (FourFlow, Amsterdam UMC FlowProcessing, Sotelo's
  4D Flow MATLAB Toolbox). MSAC (Fischer 2022) has no public code. The one open
  Python 4D flow pipeline (Xtaltec left-atrium pipeline) assumes correction was done
  in commercial software.
- Adjacent: ETH's 4Dflow-unwrap (Dirix, Jacobs, Kozerke, ISMRM 2024) is a small
  Python collection of Laplacian, NPRS and PUMA unwrapping on GitLab.
- Learning-based methods (DAF-FlowNet, Bisbal/Sotelo/Uribe arXiv 2604.00205;
  Hsiao's deep-learning background correction, Radiology 2021 and UCSD patent) have
  no released code and no CHD validation.

Conclusion: implement in-house in numpy/scipy with selectable methods, which is
also the "common inspectable substrate" argument the manuscript makes.

## 3. What the commercial packages ship

From Burkhardt et al. 2023 (four packages in 47 CHD patients) and Oechtering et al.
2023 (multi-software, multi-vendor), plus vendor sources.

| Package | Background offset | Anti-aliasing | Noise mask | Static tissue | Divergence-free |
|---|---|---|---|---|---|
| Arterys Cardio AI (Tempus) | polynomial + smoothing | none (move planes) | n/d | AI semi-auto | no |
| cvi42 4D Flow | polynomial (Lankhaar) | yes, 4D volume | yes | semi-auto, threshold | no |
| CAAS MR 4D Flow | 1st-order per frame | none | n/d | automatic | no |
| Medis QFlow 4D | 2nd-order | none | n/d | lowest-25% velocity-std | no |
| GTFlow | static-tissue fit | yes | yes | manual/semi | no |
| MevisFlow | static-tissue fit | yes | yes | semi | no |
| Siemens prototype (Gulsun 2012) | automatic | yes | yes | auto | no |
| iTFlow (FDA 2023) | yes | yes | yes | n/d | no |

Observations: anti-aliasing splits the market; nobody ships divergence-free
filtering (clean differentiation for the paper); Arterys's differentiator was
AI static-tissue selection and cloud speed, not a better filter. Every package
leads with pathlines (particle traces), which the manuscript lacked.

Two findings that shape defaults: processing order matters (Keller/Schnell ISMRM
2016, Lurie Children's: background correction must precede noise masking and
anti-aliasing); inter-software variability exceeds inter-scanner variability and
peak-velocity agreement between packages is poor (Oechtering 2023).

## 4. Background phase correction (background_phase.py)

Five fitters behind one call, shared candidate front end (magnitude threshold,
temporal-velocity-std threshold, body-edge erosion, PC-MRA vessel exclusion,
optional lumen mask):

- `ols`: Walker 1993 ordinary least squares.
- `msac`: Fischer et al. 2022 M-estimate SAmple Consensus.
- `wrls`: this package's adaptation of Pruitt et al. 2019: inverse-temporal-variance
  weights (Pruitt: 1/sigma) with ridge penalty (Pruitt: l1), z-score outlier rejection
  with morphological cleanup for steady venous flow (not in Pruitt), and wrap-around
  rejection (central 65/50/50% FOV fit, two-component Gaussian mixture on weighted
  residuals (Pruitt: three), keep voxels within 2 SD of the dominant component
  (Pruitt: 3 SD), refit once (Pruitt: two iterations)).
- `war+msac` (default): MSAC, then the `wrls` wrap rejection on the survivors, then
  MSAC refit.
- `arto`: Pruitt et al. 2019 as published (WRLS + ARTO): 1/sigma weights, l1 penalty
  solved by FISTA, first-order fit of the central FOV, three-component Gaussian
  mixture of the weighted residuals with the outer means held 2 gamma_c away, 3 SD
  cut, two iterations, final fit on the survivors. Steady flow and folded-in tissue
  are both caught by the residual test. Pruitt's lambda = 50 is in the units of their
  normalised phase problem; here the default is 1e-3 of the lasso null penalty.

Phantom (body spanning FOV, 2nd-order offset +/-3 cm/s, lungs, pulsatile artery,
5 cm/s steady vein, phase-encode wrap band carrying folded-in tissue with the
offset of its true out-of-FOV location). Max offset error in cm/s:

| scenario | OLS | MSAC | WRLS | MSAC+WaR | ARTO |
|---|---|---|---|---|---|
| clean | 0.08 | 0.08 | 0.06 | 0.06 | 0.13 |
| wrap band | 0.51 | 0.46 | 0.10 | 0.10 | 0.27 |
| steady vein in candidates | 0.27 | 0.01 | 0.10 | 0.02 | 0.03 |
| both | 0.96 | 0.58 | 0.99 | 0.06 | 0.36 |

(The "steady vein in candidates" and "both" rows are the scenarios without
PC-MRA exclusion; the first two rows use the default candidate front end.)
MSAC cannot see wrap (folded tissue has a smooth offset within its threshold);
the two-component mixture of `wrls` fails when a second outlier population is
present. MSAC first removes the vein, which lets the mixture model find the
wrap. The published ARTO handles both populations with its three-component
mixture, but its 3 SD cut keeps the folded voxels whose offset lies close to
the fitted surface, so it trails the 2 SD hybrid on the wrap scenarios. Both
the hybrid and ARTO stay under the 0.6 cm/s Gatehouse 2010 acceptance
threshold in every case.

## 5. Divergence-free filtering (divergence_free.py)

Comparisons in the literature: Loecher 2012 (FDM vs RBF: FDM better streamlines,
2 min vs 60 min); Sereno/Koehler/Preim 2018 (FDM or DFW recommended on
practicality); Pirrello 2024 (DFW best and least sensitive to segmentation error;
FDM distorts when the mask is wrong; RBF weak near the wall). RBF deliberately
not implemented.

Four methods:

- `fdm`: Song 1993 / Loecher 2012 masked finite-difference hard projection
  (sparse masked divergence operator, central differences inside, one-sided at the
  wall; minimum-norm correction via CG on D D^T).
- `relaxed`: Mura 2016 penalised form, lam ~ spacing^2 leaves residual divergence.
- `leray`: FFT Helmholtz-Hodge projection with re-masking. Building block only.
- `dfw`: Ong 2015 / Bostan 2015 spirit. Leray split, undecimated isotropic (starlet)
  wavelet transform of each part with mask-aware normalized convolution (Knutsson
  1993), soft-threshold divergence-free details lightly and curl-free details
  harder, recombine. Dependency-free (no PyWavelets in Slicer).

Aortic CFD ground truth (4DFlowNet example data, Ferdian 2020), MRI resolution
1.19 mm, 6,869 lumen voxels, 10% venc noise (15 cm/s):

| method | RMSE cm/s | angular err deg | div RMS /s | flow err mL/s | time s |
|---|---|---|---|---|---|
| noisy | 15.0 | 25.2 | 0.19 | 2.3 | |
| fdm | 12.3 | 20.7 | 2e-9 | 1.9 | 0.01 |
| relaxed | 13.4 | 22.5 | 0.10 | 2.2 | 0.01 |
| leray | 13.5 | 26.6 | 0.09 | 16.2 | 0.20 |
| dfw | 3.1 | 4.1 | 0.012 | 1.8 | 0.16 |
| wavelet only, no split | 3.4 | 4.5 | 0.023 | 1.3 | 0.08 |
| dfw then fdm | 3.0 | 3.8 | 1e-10 | 1.8 | 0.16 |

Key facts: projection alone removes exactly the curl-free third of isotropic white
noise (RMSE ratio sqrt(2/3) at every noise level), so FDM has a hard floor. DFW
beats it because it also shrinks the solenoidal noise; the ablation shows most of
the gain is mask-aware wavelet denoising, the divergence split adds roughly 10%.
An early DFW version leaked mass across the wall (flow bias 6 to 13 mL/s) until the
wavelet smoothing was made mask-aware. DFW is more mask-sensitive than FDM
(dilated mask capturing a ring of noise raised its flow error to 7.8 mL/s). Scaling:
309k lumen voxels, FDM 1.3 s per frame. Default DFW thresholds are k_df 1.5, k_cf 3.0 (soft), the 2:1 ratio Ong reported;
2.0/3.0 performed the same on this dataset. Re-check on real cases.

Recommended use: quantification with no filter or fdm; visualization and derived
quantities (vorticity, WSS, CFD inlet profiles) with dfw, or dfw then fdm when a
strictly divergence-free field is required.

## 6. Unwrapping, noise mask, PC-MRA (unwrap.py)

Laplacian unwrapping (Loecher 2016; Schofield and Zhu 2003 estimator; DCT Poisson
solve with Neumann boundaries). Bug found: the Laplacian stencil and the DCT must
use the same half-sample mirror or the 4D solve fails silently.

Aortic CFD aliased at venc 0.5 (peak 0.85), residual wrapped lumen voxel-frames of
1.12 million:

| scenario | temporal | laplacian3d | laplacian4d | laplacian then temporal |
|---|---|---|---|---|
| pulsatile, 5% noise | 0 | 386 | 117 | 0 |
| pulsatile, 15% | 136 | 1,144 | 283 | 136 |
| persistent jet, 5% | 722 | 1,374 | 1,024 | 502 |
| persistent jet, 15% | 22,878 | 7,290 | 5,945 | 4,638 |

Temporal alone fails where the jet exceeds venc all cycle (coarctation, baffle).
Default: Laplacian then temporal. 4D is not worth double runtime.

Noise mask (Bock 2007) keeps 99.4 to 100% of lumen on the test data. PC-MRA per
Bock 2010.

## 7. Temporal harmonic filter (removed) and QA (qa.py)

Temporal harmonic low-pass was prototyped: weak at 24 phases (12 harmonics; keeping
6 gets RMSE 15.0 to 11.1 cm/s with jet-core peak within 2%). Removed from the
package on 2026-10-16 as not worth its risk to peak velocity.

Mass-conservation QA on two descending-aorta planes: truth 0.2%, noisy 1.6%,
FDM 0.4%, DFW 0.4%, temporal filter 1.6%.

## 8. Pathlines and centerline planes (pathlines.py, pathlines_fast.py)

numpy RK4 reference: 202 seeds, one 600 ms cycle, 0.1 s; 99.7% of points inside
the lumen. VTK version uses vtkParticlePathFilter (ParaView's tracer). VTK has no
GPU pathline tracer (VTK-m accelerator is steady streamlines only, not in Slicer).
Lessons: pipeline time must be in seconds; vtkTemporalInterpolator segfaults after
a Python temporal source in VTK 9.7 and is unnecessary; `SetIntegratorType(2)` is
RK45, use the `RUNGE_KUTTA4` constant; the tracer takes one RK4 step per input
frame, so frames are temporally upsampled x4 before the tracer (median deviation
from a fine reference 2.0 mm at x1, 0.49 mm at x4, 0.25 mm at x8); float32 frames
only for the integration window (float64 two-cycle tiling was OOM-killed).

Centerline planes every 4 mm along a descending-aorta centerline: flows 27.1 to
27.7 mL/s, 0.7% CV on a conserved field.
