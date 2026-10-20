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

We did not find a maintained open-source Python library providing these
filters in a form usable inside 3D Slicer, so they were implemented in
numpy/scipy with selectable methods, which is also the "common inspectable
substrate" argument the manuscript makes. (An earlier version of this section
listed specific tools and their status; those statements were removed
because they could not be re-verified against a source.)

## 3. What the commercial packages ship

From Burkhardt et al. 2023 (four packages in 47 patients with congenital
heart disease; PMID 37620674), as stated in its Methods:

| Package | Background offset | Static tissue | Anti-aliasing |
|---|---|---|---|
| Arterys | piecewise linear polynomial model with Gaussian smoothing | semi-automatic, AI-based | not available |
| cvi42 | polynomial fit to static tissue, after Lankhaar et al. 2005 | semi-automatic with manual threshold | phase unwrapping of the 4D volume |
| CAAS | first-order surface through the time-averaged velocities | stationary tissue (selection not described) | not available |
| Medis QFlow 4D | second-order fit | 25% of the volume with the lowest velocity standard deviation | not available |

Burkhardt et al. used conservation of mass (ascending aorta against main
pulmonary artery; main against branch pulmonary arteries) as an internal
control. Oechtering et al. 2023 (PMID 36978131; four packages, two vendors)
found peak velocity agreed poorly between most packages (ICC <= 0.4) and that
inter-scanner differences tended to be smaller than inter-software
differences.

Processing order: background phase correction should be applied before any
other filter (Keller et al., ISMRM 2016, abstract 1939; see also Keller et
al. 2017, PMID 27940231).

## 4. Background phase correction (background_phase.py)

Four fitters behind one call, shared candidate front end (magnitude threshold,
temporal-velocity-std threshold, body-edge erosion, PC-MRA vessel exclusion,
optional lumen mask):

- `ols`: ordinary least squares (Walker 1993 fitted a plane; Lankhaar 2005
  compared polynomial orders 0 to 5).
- `msac`: Fischer et al. 2022 M-estimate SAmple Consensus.
- `wrls`: after Pruitt et al. 2019 (second-order WRLS with automatic rejection
  of temporally invariant outliers by a Gaussian mixture model of the residuals).
  Implementation choices here: inverse-temporal-variance weights with a ridge
  penalty, z-score outlier rejection with morphological cleanup for steady venous
  flow, and wrap-around rejection (central 65/50/50% FOV fit, two-component
  Gaussian mixture on weighted residuals, keep the dominant component, refit).
- `war+msac` (default): MSAC, then wrap rejection on the survivors, then
  MSAC refit.

Phantom (body spanning FOV, 2nd-order offset +/-3 cm/s, lungs, pulsatile artery,
5 cm/s steady vein, phase-encode wrap band carrying folded-in tissue with the
offset of its true out-of-FOV location). Max offset error in cm/s:

| scenario | OLS | MSAC | WRLS | MSAC+WaR |
|---|---|---|---|---|
| clean | 0.08 | 0.08 | 0.06 | 0.06 |
| wrap band | 0.51 | 0.46 | 0.10 | 0.10 |
| steady vein in candidates | 0.27 | 0.01 | 0.10 | 0.02 |
| both | 0.96 | 0.58 | 0.99 | 0.06 |

MSAC cannot see wrap (folded tissue has a smooth offset within its threshold);
The mixture model as implemented here fails when a second outlier population is present. MSAC
first removes the vein, which lets the mixture model find the wrap. Only the
hybrid stays under the 0.6 cm/s Gatehouse 2010 acceptance threshold in every case.

## 5. Divergence-free filtering (divergence_free.py)

Comparisons in the literature: Loecher 2012 (FDM vs divergence-free RBF:
FDM gave longer streamlines and ran in about 2 min against 60 min; RBF changed
flow values less); Pirrello 2024 (MSc thesis, Politecnico di Torino: DFW best
and least sensitive to segmentation error). RBF deliberately not implemented.

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
