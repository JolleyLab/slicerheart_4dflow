# SlicerHeart 4D flow filters

Python (numpy, scipy; VTK for pathlines_fast) for the SlicerHeart 4D flow module.
Arrays are indexed `[k, j, i]` (as `slicer.util.arrayFromVolume`): `(nt, nk, nj, ni, 3)`
velocity with world components `(vx, vy, vz)`, `(nt, nk, nj, ni)` magnitude,
`(nk, nj, ni)` bool lumen masks. Image geometry is a 4x4 `voxel_to_world` matrix
mapping voxel `(i, j, k, 1)` to world `(x, y, z, 1)` in mm, in any world coordinate
system (e.g. LPS or RAS); any affine mapping (anisotropic, oblique, flipped) is supported. Points and plane normals
are world coordinates. See `slicerheart_4dflow/geometry.py`. Ordered from most to
least critical.

## 1. Divergence-free filtering: `slicerheart_4dflow/divergence_free.py`

Blood is incompressible, so the divergent part of a measured velocity field is
noise. These filters remove it inside the lumen segmentation. Projection methods
(`fdm`, `relaxed`) remove exactly the curl-free third of white noise and preserve
flow quantification; `dfw` also attenuates the remaining noise with wavelet
shrinkage, cutting velocity error by about four-fold on the aortic CFD at
10% venc noise, and is intended for visualization and derived quantities
rather than flow measurement. `dfw` needs structure larger than its wavelet
scales: on a vessel about 10 voxels across at 10% noise it is worse than
`fdm` (soft-threshold bias), and it is a Leray split followed by
per-component shrinkage, not Ong's divergence-free wavelet basis, so its
output is only approximately divergence-free (run `fdm` after it for a
divergence-free field). `relaxed` defaults to `lam = spacing^2`, which
removes about 70% of white-noise divergence; `10 x spacing^2` is in effect a
hard projection. Non-convergence of the conjugate-gradient solves is
reported with a warning.


`divergence_free_filter(velocity, mask, voxel_to_world, method=...)`, one frame;
`divergence_free_sequence(...)` for all frames.

| Option | Method | Citation |
|---|---|---|
| `fdm` (default) | masked finite-difference hard projection | Song et al. 1993, PMID 8347951; Loecher et al. 2012, doi:10.1186/1532-429X-14-S1-W64 |
| `relaxed` | penalised divergence reduction | Mura et al. 2016, doi:10.1109/TMI.2016.2570010 |
| `dfw` | Leray split + mask-aware undecimated wavelet shrinkage (in the spirit of, not an implementation of, Ong 2015) | Ong et al. 2015, PMID 24549830; Bostan et al. 2015, doi:10.1109/LSP.2015.2388794 |
| `leray` | FFT Helmholtz-Hodge projection, building block only | Bostan et al. 2015 |

## 2. Background phase (eddy current) correction: `slicerheart_4dflow/background_phase.py`

Eddy currents add a smooth, spatially varying velocity offset that biases every
flow measurement. Static tissue is identified automatically, a low-order polynomial
is fitted to its time-averaged velocity and subtracted from every cardiac phase.
The robust options reject vessels, slow steady venous flow and folded-in wrap-around
tissue that would otherwise corrupt the fit.


`correct_background_phase(velocity_t, magnitude_t, method=...)`.

| Option | Method | Citation |
|---|---|---|
| `ols` | polynomial fit to static tissue, least squares | Walker et al. 1993, PMID 8324312 (first-order surface); Lankhaar et al. 2005, PMID 15971181 (orders 0 to 5) |
| `msac` | M-estimate sample consensus robust fit; inlier threshold 2.5 x robust sigma of the residuals of a fit of the requested order | Fischer et al. 2022, doi:10.1002/mrm.29363 |
| `wrls` | weighted regularised LS + steady-flow outlier rejection + wrap-around rejection; this package's adaptation (inverse-variance weights, ridge penalty, two-component mixture, 2 SD cut, one pass) | adapted from Pruitt et al. 2019, PMID 30431176 |
| `war+msac` (default) | MSAC, then the wrap-around rejection of `wrls` on the survivors, then MSAC refit | Fischer 2022; adapted from Pruitt 2019 |
| `arto` | WRLS + ARTO as published: 1/sigma weights, l1 penalty (FISTA), first-order fit of the central FOV, three-component Gaussian mixture of the weighted residuals, 3 SD cut, two iterations; steady flow and wrap-around rejected by the same residual test. Validated by Pruitt on 2D phase contrast; the 4D use follows their ISMRM 2018 abstract 2940 (WRLS with wrap-around rejection applied to 4D flow) | Pruitt et al. 2019, PMID 30431176 |

QA: `residual_static_velocity` against the 0.6 cm/s threshold (Gatehouse et al. 2010, PMID 20074359).

Static-tissue candidates (`static_tissue_candidates`, shared by all methods):
body = `body_mask()` (DRAFT, chosen on phantoms): the signal to
phase-noise score, log(time-averaged magnitude) minus log(temporal velocity
std), split into two classes by the Kittler-Illingworth minimum-error
criterion (Kittler 1986, doi:10.1016/0031-3203(86)90030-0; the calculator
3D Slicer exposes as "Kittler-Illingworth", of which Otsu is the
equal-variance special case), with the threshold placed 4 standard
deviations above the background class mean, capped at the midpoint of the
class means. It adapts to the air fraction of the FOV and, unlike a
magnitude-only threshold, is not fooled by coil shading or fat-suppressed
tissue (on phantoms with 70% coil shading at SNR 5 or a dark subcutaneous
rim, Dice against the true body 1.00 and 0.96 versus 0.74 and 0.80 for
Otsu). On a field of view without a background class the whole FOV is body
with a warning: a cropped ROI where the split does not lower the criterion
or the classes are closer than 2 pooled standard deviations (Ashman 1994,
doi:10.1086/117248), or tissue around a pulsatile lumen, where the
low-score class is the brighter one (the background must be darker than
half the body's median magnitude). On noise-free input (synthetic phantoms,
where the velocity of static tissue is exactly constant over time, so the
score is infinite) bright voxels with constant velocity are tissue, not zero
filling or scanner-masked air; when they are at least 5% of the
nonzero-magnitude voxels the score is not used and the body is the
magnitude above Otsu's threshold (`air_threshold`, the whole FOV without an
air class). `magnitude_threshold="otsu"` restores Otsu's threshold
of the magnitude (Otsu 1979, doi:10.1109/TSMC.1979.4310076), a fixed value
or `magnitude_percentile=` are accepted), outer rim eroded by
3 voxels (this package's heuristic for spatial wrap; internal cavities such as
the lungs are not eroded around), the quietest 25% of body voxels by temporal
velocity std (the fraction used by Lankhaar et al. 2005, PMID 15971181, and by
Medis QFlow 4D, Burkhardt et al. 2023, PMID 37620674), minus the top 10% of the
velocity-weighted PC-MRA inside the body (after Bock et al. 2010,
PMID 20024953; the same definition as `pcmra()`) and the dilated lumen
segmentation.

## 3. Velocity anti-aliasing: `slicerheart_4dflow/unwrap.py`

Velocities above the venc wrap around. Laplacian unwrapping resolves wraps within a
single frame, so it works where a jet exceeds venc throughout the cycle; temporal
unwrapping detects jumps between frames and cleans up what the spatial solve leaves.


`unwrap_velocity(velocity_t, venc, voxel_to_world, method=...)`, applied to the velocity components as encoded (each aliases at its own venc).

| Option | Method | Citation |
|---|---|---|
| `laplacian3d` | single-step Laplacian (DCT Poisson solve), per frame | Loecher et al. 2016, doi:10.1002/jmri.25045; Schofield and Zhu 2003, PMID 12885018 |
| `laplacian4d` | same with time as fourth dimension; `time_boundary="neumann"` (default) or `"periodic"` (Loecher: the retrospectively gated cycle is periodic, so a peak in the first or last frame is unwrapped against its neighbour across the cycle boundary) | Loecher et al. 2016 |
| `temporal` | jump detection along time | Xiang 1995, PMID 8574036 |
| `laplacian_then_temporal` (default) | both passes | Loecher 2016; Xiang 1995 |

### 3a. Multi-venc unwrapping (DRAFT): `slicerheart_4dflow/multivenc.py`

With the same field acquired at two or more vencs (interleaved dual-venc with a
shared reference scan, as in Schnell 2017, Ma 2019 and the dual-venc 5D flow of
Weiss, Roy et al. 2024, or two separate single-venc scans), the higher-venc
image(s) select the wrap of the low-venc image, so the result has the low-venc
noise with the high-venc range. Phantom-tested only; no dual-venc scanner data
processed yet. `multivenc_unwrap(velocities, vencs, method=...)` returns the
unwrapped low-venc image, the wrap counts and the effective venc; `preprocess(...,
extra_encodings=[(velocity_high_t, venc_high)])` runs it before background
correction. `phantom.acquire_multivenc` simulates the acquisition with a shared or
independent reference.

| Option | Method | Citation |
|---|---|---|
| `sdv` | standard dual-venc: nearest wrap of the low-venc image to the high-venc (or, when venc_low > venc_high/2, the iVENC phase-difference) image; the reference must be alias-free | Lee et al. 1995, doi:10.1002/mrm.1910330119; Franco et al. 2022, PMID 35716109 |
| `schnell` | the threshold rule of the Northwestern dual-venc reconstruction: 1.6 venc_low < \|v_high - v_low\| < 2.4 venc_low adds two venc_low, 3.2 to 4.8 adds four, dead bands in between; Schnell correct the background of each venc separately and unwrap the high-venc series with a single-venc method first (`preprocess(background_first=True, unwrap_kwargs={"reference_unwrap": ...})`) | Schnell et al. 2017, PMID 28152256 |
| `stepwise` | highest venc down: each lower-venc image unwrapped against the image one venc above it by the nearest-wrap rule, as scanner-side pipelines with three or more vencs do; two encodings: `sdv` without the iVENC reference | Ha et al. 2016, doi:10.1002/mrm.25715; Ma et al. 2019 |
| `odv` (default) | optimal dual-venc / multiple motion encoding: per voxel, the candidate low-venc wrap minimising sum_j [1 - cos(pi (u_j - u)/venc_j)]; any number of encodings; the search is capped at the highest venc by default (`max_velocity`) and can be widened to the unambiguous range (lcm of the vencs) when the high venc itself aliases; with Franco's sign-consistency correction (`correction=True`); reports the fraction of ambiguous voxels (small cost margin) | Carrillo et al. 2019, PMID 30475716; Herthum et al. 2022, doi:10.1016/j.media.2022.102416; Franco et al. 2022 |

Jets (valvar regurgitation and stenosis) are the main use: slow chamber or venous
flow needs the low venc, the jet needs the high venc, and the regurgitant volume is
measured from the slow flow. Two tools for that case: `multivenc_unwrap(...,
magnitudes=[...], fallback_ratio=0.5)` replaces voxels whose low-venc magnitude has
collapsed relative to the high venc (intravoxel dephasing in the jet core, which
no unwrapping can repair) by the high-venc velocity (`dephasing_fallback`, a choice
of this library); and `phantom.valve_jet` with `acquire_multivenc(...,
intravoxel_std=..., time_offset_frames=...)` is a test bed with an orifice jet
entering a chamber of slow swirl, venc-dependent dephasing from the intravoxel
velocity spread (magnitude exp(-pi^2 sigma_v^2 / 2 venc^2), Dyverfeldt et al. 2006,
doi:10.1002/mrm.21022), timing offsets between encodings (Ma 2019 attribute systolic
errors to them) and partial volume. `examples/multivenc_valve_jet.py` runs every
method on it, or on an exported acquisition, with per-region (jet core, jet edge,
chamber) failure, RMSE, missed-wrap and false-unwrap rates (`error_summary`). On the
phantom (jet 4 m/s, chamber swirl 0.1 m/s, SNR 12, turbulence intensity 0.2): three
vencs 550/2200/4400 give the chamber a quarter of the noise of 2200/4400 alone, and
the fallback cuts the jet-core RMSE from 232 to 85 mm/s with 2% of voxels replaced.

Alignment: all encodings must share the grid and the frames. Interleaved
acquisitions do by construction. For separate scans, `encoding_shift` (phase
correlation of the magnitude images, sub-voxel) reports the residual translation
and `preprocess` warns above half a voxel; registration itself is left to the host
application (3D Slicer: Elastix, ANTs, greedy on the magnitude images), and the
velocity components must be rotated by the rotational part of the transform as
well as resampled. `resample_frames` interpolates a series onto other frame times
(periodic over the cycle) when the vencs were binned to different cardiac phases.

`preprocess(..., extra_encodings=..., background_first=False)`: by default the
series are combined first and the background phase corrected once; with
`background_first=True` each series is corrected before combining (the two
orders agree on the thorax phantom, where the offsets are equal in velocity
units; which is right on scanner data, where concomitant-field terms differ
between encodings, is to be tested).

Ma 2019's bi- and triconditional rules (7-point triple venc) compare the low- and
high-venc images with the iVENC image; given velocity images the iVENC image is a
function of the other two, so the biconditional rule equals `sdv` with the iVENC
reference, and the triconditional rule's third condition could not be established
from the paper's text; neither is a separate method here.

Findings used and reproduced in `tests/test_multivenc.py`: for unwrapping success
within the high venc, venc_low/venc_high = 1/2 is the most robust ratio (Franco
2022, Herthum 2022); Carrillo 2019 recommend 0.66 when the aim is a range beyond
the high venc with both images aliased, which the capped default does not attempt;
Schnell's dead bands trade missed wraps in the jet (23% at SNR 2.5 against 5% for
sdv) for seven times fewer false unwraps where nothing wraps; on the aortic CFD
field acquired
through the forward model with partial volume at 1.19 mm, venc_low at 30% of the
peak velocity defeats the single-venc Laplacian (a third of the lumen left
wrapped) while dual venc leaves none (Löcke et al. 2023, PMID 37345719); a wide
search at an odd venc ratio lets noise pick far candidates (5.6% failures at
0.25/1.2 with the search opened to 6 x peak, none with the default cap), which
is why the cap is the default. Not implemented: Ma 2019's bi/tri-conditional
threshold rules (to be added from the paper's tables), PRoM (Zhao et al. 2022,
PMID 35862337; needs multi-coil data).

## 4. Noise masking and PC-MRA: `slicerheart_4dflow/unwrap.py`

Noise voxels (lung, air) are flagged from low magnitude and random velocity over
time so they do not seed streamlines or enter the static-tissue fit. The PC-MRA is a
velocity-weighted angiogram used as the input for vessel segmentation. `preprocess`
runs the steps above in a fixed order, background correction first (Keller
et al. 2016, ISMRM 24:1939; Keller et al. 2017, PMID 27940231). The background
class is found by `body_mask()` by default (DRAFT: Kittler-Illingworth
split of log magnitude / velocity std, see static-tissue candidates above;
`"otsu"` restores the magnitude-only Otsu threshold; the velocity-std
percentile rule is skipped when no background class is found, since the
highest velocity std is then the pulsatile lumen) because it adapts to the air
fraction of the field of view (a fixed 30th percentile masks all of the air
in a thoracic FOV but only a third of it around an isolated vessel or phantom). With a tight
noise mask the Laplacian unwrapping extends the wrapped phase a few voxels
beyond the mask by nearest-neighbour fill before solving
(`edge_extension_voxels=3`), so the step to zero at the wall is not a
discontinuity the solve has to absorb; on the aortic CFD this cut the
single-frame wraps left by the spatial pass from 353 to 53.


| Function | Method | Citation |
|---|---|---|
| `noise_mask` | background by `body_mask()` (Kittler-Illingworth split of log magnitude / velocity std, DRAFT; Otsu threshold, a percentile or fixed value optional) or high temporal velocity std (wrap-invariant frame-to-frame differences when venc is given) | Bock et al. 2007, ISMRM 15:3138; Kittler and Illingworth 1986, doi:10.1016/0031-3203(86)90030-0; Otsu 1979, doi:10.1109/TSMC.1979.4310076; Ashman 1994, doi:10.1086/117248 |
| `pcmra` | velocity-weighted angiogram | Bock et al. 2010, PMID 20024953 |
| `preprocess` | fixed order: background, unwrap, noise mask, PC-MRA, optional divergence filter | Keller et al. 2016, ISMRM 24:1939; Keller et al. 2017, PMID 27940231 |

## 4a. Automatic lumen segmentation: `slicerheart_4dflow/segmentation.py`

`estimate_lumen` thresholds the PC-MRA (Otsu threshold of its log within the
bright half of the image) on a grid twice as fine as the image, opens and
closes it by one voxel, keeps connected components that are not image-border
artifacts, and grows the result toward the vessel wall, by at most 1.5 image
voxels, into voxels whose RMS speed over the cycle is above twice that of the
velocity noise. The growth is needed because the PC-MRA is magnitude times
speed and the speed falls to zero at the wall, so any threshold cuts inside
it: on 15 pediatric clinical cases the threshold alone missed about 10% of the
net flow through the measurement planes (median), after the growth a further
one-voxel dilation adds 1.3%. Distances are in mm (spacing-aware distance
maps) and resampling is done with VTK in physical space (`vtk_image`). The
result is the fine mask with its `voxel_to_world`, and the fraction of each
image voxel in it, which `plane_flow` accepts as a partial-volume mask. It is
a starting point for review; the mask is static (vessels that move are covered
over the volume they sweep). The velocity noise (`velocity_noise_in_tissue`) is that of
the quality report (RMS over the components): the same tissue (`noise_tissue_mask`) and
estimator (`velocity_noise`); when the dilated lumen leaves no static
tissue it is taken in all nonzero-magnitude voxels outside the lumen, and on
noise-free input it is 0, so the growth then follows every voxel with nonzero
speed (above 1e-6 of the largest), which needs background-corrected velocity.

## 5. Pathlines: `slicerheart_4dflow/pathlines.py`, `slicerheart_4dflow/pathlines_fast.py`

Particles emitted from a plane are advected through the time-varying velocity field
over the cardiac cycle, producing the animated particle traces that every commercial
4D flow package uses as its primary display. The VTK version is the tracer ParaView
uses; the numpy version is a dependency-free reference. Both zero the
velocity outside the lumen mask; the numpy tracer stops a particle where the
interpolated mask falls below 0.5, the VTK tracer where the speed falls below
`terminal_speed_mm_s`. Emission time `t0_ms` may be given per seed
(staggered emission); each particle is advected at its own time. VTK's
tracer holds each served frame's velocity over the step (first order in
time), so frames are interpolated to `temporal_upsample` x finer spacing
before tracing (default 8: 0.13 mm error over one cycle on a 6 mm orbit,
0.25 at 4, 0.07 at 16; the numpy RK4 tracer gives 0.05 mm).


| Function | Method | Citation |
|---|---|---|
| `pathlines_vtk` | vtkParticlePathFilter (ParaView's tracer), frames interpolated x8 in time | Wigstrom et al. 1999, PMID 10332856; Schroeder et al. 2006 (VTK) |
| `advect_pathlines` | numpy RK4 reference implementation | Wigstrom et al. 1999 |
| `seeds_from_plane`, `to_vtk_polydata` | seeding and export helpers | |

Output: vtkPolyData with `time_ms` and `index` point scalars.

## 6. Quantification and QA: `slicerheart_4dflow/qa.py`, `slicerheart_4dflow/pathlines.py`

Through-plane flow at planes placed by hand or automatically along a vessel
centerline, and a consistency check comparing inflow with outflow at user-defined
plane sets before and after each processing step. This is the acceptance test for
all of the filters above.


| Function | Method | Citation |
|---|---|---|
| `plane_flow` | through-plane flow per frame, plane disc resampled on a regular grid | |
| `centerline_planes` | planes at fixed arc length perpendicular to a VMTK centerline | Antiga et al. 2008, PMID 19002516 |
| `mass_conservation`, `qa_report` | inflow vs outflow, Qp:Qs, per processing stage | Bissell et al. 2023, PMID 37474977 |

## Default parameters and how certain they are

"Literature" means the value is taken from a published method or a commercial
package's documented behaviour. "Reasonable" means it is a sensible engineering
choice that the tests support but that no paper fixes; verify on CHOP data with the
consistency table before release.

| Parameter | Default | Basis |
|---|---|---|
| Background polynomial order | 2 | literature: Medis QFlow 4D, Pruitt 2019 in vivo |
| Static tissue: quietest fraction of body by velocity std | 25% | literature: Lankhaar 2005; Medis rule (Burkhardt 2023) |
| Static tissue: body magnitude percentile | 35th | reasonable |
| Static tissue: body-rim erosion | 3 voxels | reasonable |
| MSAC inlier threshold | 2.5 x robust sigma | reasonable, standard outlier cut |
| WaR central FOV fractions (array axes k, j, i) | 0.65, 0.5, 0.5 | package choice |
| WaR mixture cut | 2 SD of dominant component | package choice |
| Outlier rejection z-score | 1.0 | package choice |
| Unwrap default | Laplacian then temporal | reasonable, supported by test table in BACKGROUND |
| Temporal max wraps | 3 | matches the existing module |
| Noise mask magnitude percentile / velocity-std percentile | 30th / 97th | reasonable |
| Divergence-free default for quantification | fdm (hard projection) | literature: Loecher 2012 |
| DFW threshold ratio k_cf : k_df | 2 : 1 (3.0 : 1.5 sigma) | package choice |
| DFW wavelet levels / padding | 3 / 8 voxels | reasonable |
| Relaxed lam | 10 x min(spacing)^2 | package choice |
| Pathline seed spacing | 1.5 mm (one per voxel) | reasonable |
| Pathline temporal upsampling (VTK) | x4 | reasonable, verified sub-voxel accuracy in tests |
| Centerline plane spacing / radius | 10 mm / 15 mm | reasonable |
| Mass-conservation acceptance | 0.6 cm/s residual static velocity | literature: Gatehouse 2010 |

Not implemented: a temporal harmonic low-pass was prototyped and removed; at
typical 20 to 30 cardiac phases it removes little noise and risks peak velocity.

## Other files

- `tests/`: one script per module (`test_qa.py` covers plane flow and consistency); `tests/README.md` explains the 4DFlowNet example data.
- `examples/pathlines_vtk_example.vtp`: loads in Slicer's Models module.
- `docs/BACKGROUND.md`: survey, method choices, result tables, industry comparison.
- `docs/METHODS_DRAFT.md` / `.docx`: Methods text with full references.

## 7. Derived hemodynamics: `derivatives.py`

Vorticity, helicity density, kinetic energy and viscous energy loss from the
masked velocity gradient (central differences inside the lumen, one-sided at
the wall, exact for affine voxel geometry). Gradients amplify noise: at 5% venc
noise the raw vorticity error is 11x the signal, FDM projection does not help,
and DFW brings it to 1.2x, so compute these from `dfw`-filtered velocity.

| Function | Method | Citation |
|---|---|---|
| `vorticity`, `helicity_density` | curl v, v . curl v (relative helicity option) | Lorenz et al. 2014, PMID 23716466; Garcia et al. 2019, PMID 30732721 |
| `viscous_dissipation`, `lumen_integrals` | 2 mu S:S, integrated over the lumen | Barker et al. 2014, PMID 24122967 |
| `kinetic_energy_density` | 1/2 rho v^2 | Carlsson et al. 2012, PMID 22180653 |
| validation reference | finite-element formulation | Sotelo et al. 2016, PMID 26780787; 2018, PMID 28980342 |

## 8. Wall shear stress: `wss.py`

Wall points on the marching-cubes surface of the binary lumen mask (VTK
flying edges, the surface 3D Slicer builds for a segmentation; `wall_surface`
also returns the triangles for export as a model), inward
normals from the gradient of a smoothed mask (transformed to world as
covectors, which matters on anisotropic, oblique or sheared voxels: the
earlier Jacobian mapping tilted normals by up to 45 degrees on a
1 x 1 x 2.5 mm grid); velocity sampled 1, 2, 3 voxels inward; tangential
component fitted with a quadratic through zero at the wall; WSS = mu x wall
shear rate; TAWSS and OSI. On Poiseuille flow it is within 10% at 1 mm
voxels on every geometry. On Womersley flow (boundary
layer 0.65 mm) it underestimates by about 40% at 1 mm and 60% at 1.5 mm with
the time course well correlated; this is the resolution limit of 4D flow WSS
(Petersson 2012) and is recorded by the test, not tuned away.

| Function | Method | Citation |
|---|---|---|
| `wall_shear_stress` | wall-normal velocity fit | Stalder et al. 2008, PMID 18956416; Potters et al. 2015, PMID 24436246 |
| `osi` in output | oscillatory shear index | Ku et al. 1985, PMID 3994585 |
| `wall_points`, `wall_surface` | marching cubes surface by VTK flying edges | Lorensen and Cline 1987, doi:10.1145/37402.37422; Schroeder et al. 2015, doi:10.1109/LDAV.2015.7348069 |
| accuracy limit | resolution dependence | Petersson et al. 2012, PMID 22336966 |

## 9. Relative pressure: `pressure.py`

Navier-Stokes pressure gradient (temporal, convective and viscous terms, each
switchable; the temporal term takes `frame_duration_ms` and `periodic=False`
for prospectively triggered data covering part of the RR interval) and
least-squares integration to a relative pressure field over
the lumen (equivalent to the Poisson problem with Neumann boundaries).
Poiseuille gradient within 8%; Womersley pressure gradient time course
correlation 0.9999, slope 0.93. `wall_weight=0.1` down-weights the wall layer
and brings the viscous-flow error to 3%, but biases a convection-dominated
jet by 30 to 40%, so uniform weights are the default. On the stenosis phantom
the volumetric pressure drop is one third of the centreline Bernoulli drop,
because the parabolic-profile convective term averages to one third over the
section; the simplified Bernoulli estimate from peak velocity matches the
centreline value within 10%.

| Function | Method | Citation |
|---|---|---|
| `pressure_gradient` | Navier-Stokes from velocity | Ebbers et al. 2001, PMID 11323814; Tyszka et al. 2000, PMID 10931596 |
| `relative_pressure` | pressure Poisson equation on the lumen | Ebbers and Farneback 2009, PMID 19557846; Krittian et al. 2012, PMID 22626833 (finite-element alternative) |
| in vivo reference | aortic 4D pressure mapping | Bock et al. 2011, PMID 21437978 |

## 10. Quality report: `report.py`

`quality_report()` returns one JSON-serialisable dict per study: venc adequacy
(peak component along the image axes, i.e. the encoding directions, over
venc; aliased fraction from wrap counts), velocity noise (median temporal
variance of magnitude-selected tissue, rescaled to be unbiased; the
static-tissue candidates are selected for low variance and would bias it
low; `segmentation.noise_tissue_mask` and `segmentation.velocity_noise`,
also used by the lumen estimate: the tissue is above the Otsu threshold of
the magnitude, as the noise scales with 1 / magnitude), residual offset against the 0.6 cm/s threshold with pass/fail,
lumen divergence before and after, noise-mask fraction, plane flows and
mass-conservation error at each stage. `to_markdown()` renders a one-page
summary. Citations: Bissell et al. 2023, PMID 37474977; Gatehouse et al. 2010,
PMID 20074359; Burkhardt et al. 2023, PMID 37620674. The velocity noise
relation sigma_v = sqrt(2) venc / (pi SNR) is derived from first principles
(phase noise 1/SNR per acquisition, sqrt(2)/SNR for the difference of two).

When a lumen mask is given the report also runs the encoding-direction
self-check below and adds an `encoding` section (PASS/FAIL). `preprocess()`
times each step (`report['timing_s']`: background, unwrap, noise_mask,
pcmra, divergence, total) and `quality_report(..., timing=)` carries those
into a `processing_time_s` section together with the report's own time, so
the cost of a configuration is visible next to its effect. In the test suite
timings are recorded in the metrics baseline for reference but never
compared (they are machine dependent).

## 10a. Encoding-direction self-check: `encoding.py`

A permuted or sign-flipped velocity component is the most common silent
import error (vendors order and sign the three encoded velocities
differently). `encoding_check()` ranks all 48 signed permutations of the
components by the divergence of a low-pass version of the field (time-mean,
zero outside the lumen, Gaussian-smoothed, evaluated on the lumen dilated
by two voxels): blood is incompressible, so the declared mapping should sit
at the minimum, and a wrong mapping turns through-plane flow into a
wall-normal source/sink layer. It returns the ratio of the declared mapping
to the minimum (`consistent` when < 1.5), the `discrimination` (how far the
next distinct mapping is above the minimum: 2-5x on the CFD and phantoms,
still 2x at 30% velocity noise), and the best mapping to apply. A
segmentation that ends inside the field of view has open inlet/outlet faces;
the band around them (found on the best mapping's field, so a wrong import
cannot hide its walls as open faces) is left out, otherwise their
source/sink swamps the comparison. When no distinct mapping is at least
1.5x the minimum the result is reported as inconclusive rather than as a
pass. Run it after unwrapping; wrapped data fail it. Two blind spots: a
global sign flip ties with the identity (settle it with
`expected_flow_sign()` on a plane whose flow direction is known, e.g. the
ascending aorta), and a component that is near zero everywhere (planar
flow) cannot have its sign tested.
Citations: Dyverfeldt et al. 2015, PMID 26257141; Song et al. 1993, PMID
8347951; Pelc et al. 1991, PMID 1790362.

## 11. Analytic phantoms: `phantom.py`

`poiseuille` (exact gradient and WSS), `womersley` (Womersley 1955,
PMID 14368548; exact velocity, flow, gradient and WSS per frame), `stenosis`
(cosine constriction after Young and Tsai 1973, PMID 4732939; lubrication
velocity field that is exactly divergence-free and flow-conserving, with
throat velocities, centreline pressure, simplified-Bernoulli gradient and WSS
contrast; the jet exceeds a clinical venc all cycle, which temporal
unwrapping cannot recover and Laplacian unwrapping does), `bifurcation`
(parent and two daughters with Murray's-law radii, Murray 1926,
PMID 16576980, and a set flow split; returns measurement planes and the
consistency checks so mass conservation and Qp:Qs-type ratios have a known
answer; exact away from the junction, which is flagged by a mask that
scales with the branch angle, as does the plane distance), `chest`
(thorax with lungs, parabolic artery, steady vein, second-order offset,
wrap-around band), plus `add_noise`, `alias` and the acquisition model
`acquire` (four-point encoding, complex noise, complex block averaging for
partial volume; the reference velocity and wrap counts on a downsampled
grid are the signal-weighted ones the measurement actually sees). The
test suite runs on these without downloading data; the 4DFlowNet CFD tests
are marked `cfd`.

A note on temporal unwrapping found while testing: Xiang's method requires the
true velocity to change by less than venc between frames. With too few cardiac
phases for the peak velocity it can misread a recovered systolic rise as a wrap,
so `laplacian3d` alone is preferable in that case.

## 11a. Semi-synthetic references: `semisynthetic.py`

A real case has no ground truth, but a plausible one with its real anatomy,
waveform and segmentation can be made from it (an "in-vivo-derived digital
phantom", as in Ferdian 2020, doi:10.3389/fphy.2020.00138).
`reference_from_measurement` runs background correction and unwrapping on the
encoded components, zeroes the field outside the lumen and projects it to
divergence-free; `reacquire` encodes that reference again with `acquire` at the
case's venc, with the noise measured in its tissue (`acquisition_snr`), its own
magnitude image and its background offset. The pipeline is then scored on the
re-acquired data against the reference: errors there say how well it recovers
a known field under the acquisition conditions of the case, which the
phantoms cannot. The reference is not the truth of the original case.

## 12. Resolution dependence

`tests/test_resolution.py` acquires the Poiseuille and Womersley phantoms
(generated at 0.5 mm) at 1.0, 1.5, 2.0 and 2.5 mm by complex block averaging,
and the CFD field at 1.2, 1.8 and 2.4 mm with 10% venc noise. Every point is a
regression metric. The curves, noise-free unless stated:

| voxel mm | Poiseuille WSS / true | Poiseuille dp / true | flow / true | Womersley WSS slope | Womersley WSS time-course r | Womersley OSI (true 0.21) |
|---|---|---|---|---|---|---|
| 1.0 | 1.08 | 0.88 | 0.994 | 0.56 | 0.86 | 0.09 |
| 1.5 | 0.89 | 0.74 | 0.996 | 0.36 | 0.81 | 0.05 |
| 2.0 | 1.06 | 0.77 | 0.980 | 0.26 | 0.75 | 0.00 |
| 2.5 | 0.91 | 0.69 | 0.990 | 0.21 | 0.74 | 0.00 |

| CFD voxel mm | noisy RMSE cm/s | fdm | dfw |
|---|---|---|---|
| 1.19 | 15.0 | 12.2 | 3.1 |
| 1.78 | 15.1 | 12.4 | 4.1 |
| 2.38 | 15.1 | 12.3 | 4.9 |

Flow is robust to resolution; WSS magnitude is within 10% for a parabolic
profile but the oscillatory component of a Womersley boundary layer is lost
above 2 mm (OSI reads zero), which is the resolution limit Petersson et al.
2012 describe; the viscous pressure term loses 30% at 2.5 mm; DFW keeps most
of its benefit at clinical resolution.
