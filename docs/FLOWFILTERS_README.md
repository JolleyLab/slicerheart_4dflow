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
shrinkage, cutting velocity error by about four-fold on test data, and is intended
for visualization and derived quantities rather than flow measurement.


`divergence_free_filter(velocity, mask, voxel_to_world, method=...)`, one frame;
`divergence_free_sequence(...)` for all frames.

| Option | Method | Citation |
|---|---|---|
| `fdm` (default) | masked finite-difference hard projection | Song et al. 1993, PMID 8347951; Loecher et al. 2012, doi:10.1186/1532-429X-14-S1-W64 |
| `relaxed` | penalised divergence reduction | Mura et al. 2016, doi:10.1109/TMI.2016.2570010 |
| `dfw` | Leray split + mask-aware undecimated wavelet shrinkage | Ong et al. 2015, PMID 24549830; Bostan et al. 2015, doi:10.1109/LSP.2014.2388794 |
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
| `ols` | polynomial fit to static tissue, least squares | Walker et al. 1993, PMID 8324312 |
| `msac` | M-estimate sample consensus robust fit | Fischer et al. 2022, doi:10.1002/mrm.29363 |
| `wrls` | weighted regularised LS + steady-flow outlier rejection + wrap-around rejection | Pruitt et al. 2019, PMID 30431176 |
| `war+msac` (default) | MSAC, then Pruitt wrap-around rejection, then MSAC refit | Fischer 2022; Pruitt 2019 |

QA: `residual_static_velocity` against the 0.6 cm/s threshold (Gatehouse et al. 2010, PMID 20074359).

Static-tissue candidates (`static_tissue_candidates`, shared by all methods):
body = time-averaged magnitude above Otsu's threshold (Otsu 1979,
doi:10.1109/TSMC.1979.4310076; adapts to the air fraction of the FOV, a
fixed value or `magnitude_percentile=` are accepted), outer rim eroded by
3 voxels (Pruitt 2019; internal cavities such as the lungs are not eroded
around), the quietest 25% of body voxels by temporal velocity std
(Burkhardt et al. 2023, PMID 37620674), minus the top 10% of PC-MRA inside
the body and the dilated lumen segmentation.

## 3. Velocity anti-aliasing: `slicerheart_4dflow/unwrap.py`

Velocities above the venc wrap around. Laplacian unwrapping resolves wraps within a
single frame, so it works where a jet exceeds venc throughout the cycle; temporal
unwrapping detects jumps between frames and cleans up what the spatial solve leaves.


`unwrap_velocity(velocity_t, venc, voxel_to_world, method=...)`, applied to the velocity components as encoded (each aliases at its own venc).

| Option | Method | Citation |
|---|---|---|
| `laplacian3d` | single-step Laplacian (DCT Poisson solve), per frame | Loecher et al. 2016, doi:10.1002/jmri.25045; Schofield and Zhu 2003, PMID 12885018 |
| `laplacian4d` | same with time as fourth dimension | Loecher et al. 2016 |
| `temporal` | jump detection along time | Xiang 1995, PMID 8574036 |
| `laplacian_then_temporal` (default) | both passes | Loecher 2016; Xiang 1995 |

## 4. Noise masking and PC-MRA: `slicerheart_4dflow/unwrap.py`

Noise voxels (lung, air) are flagged from low magnitude and random velocity over
time so they do not seed streamlines or enter the static-tissue fit. The PC-MRA is a
velocity-weighted angiogram used as the input for vessel segmentation. `preprocess`
runs the steps above in the order the consensus requires. The magnitude
threshold is Otsu's by default because it adapts to the air fraction of the
field of view (a fixed 30th percentile masks all of the air in a thoracic FOV
but only a third of it around an isolated vessel or phantom). With a tight
noise mask the Laplacian unwrapping extends the wrapped phase a few voxels
beyond the mask by nearest-neighbour fill before solving
(`edge_extension_voxels=3`), so the step to zero at the wall is not a
discontinuity the solve has to absorb; on the aortic CFD this cut the
single-frame wraps left by the spatial pass from 353 to 53.


| Function | Method | Citation |
|---|---|---|
| `noise_mask` | low magnitude (Otsu threshold by default; a percentile or fixed value optional) or high temporal velocity std | Bock et al. 2007, ISMRM 15:3138; Otsu 1979, doi:10.1109/TSMC.1979.4310076 |
| `pcmra` | velocity-weighted angiogram | Bock et al. 2010, PMID 20099326 |
| `preprocess` | fixed order: background, unwrap, noise mask, PC-MRA, optional divergence filter | Keller et al. 2016, ISMRM 24:1939 |

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
| Static tissue: quietest fraction of body by velocity std | 25% | literature: Medis rule (Burkhardt 2023) |
| Static tissue: body magnitude percentile | 35th | reasonable |
| Static tissue: body-rim erosion | 3 voxels | reasonable, motivated by Pruitt 2019 |
| MSAC inlier threshold | 2.5 x robust sigma | reasonable, standard outlier cut |
| WaR central FOV fractions (array axes k, j, i) | 0.65, 0.5, 0.5 | literature: Pruitt 2019 (65% slice, 50% phase-encode) |
| WaR mixture cut | 2 SD of dominant component | reasonable; Pruitt's exact cut not published |
| Outlier rejection z-score | 1.0 | literature: Pruitt 2019 |
| Unwrap default | Laplacian then temporal | reasonable, supported by test table in BACKGROUND |
| Temporal max wraps | 3 | matches the existing module |
| Noise mask magnitude percentile / velocity-std percentile | 30th / 97th | reasonable; Bock 2007 used absolute thresholds set per scan |
| Divergence-free default for quantification | fdm (hard projection) | literature: Loecher 2012, Sereno 2018 |
| DFW threshold ratio k_cf : k_df | 2 : 1 (3.0 : 1.5 sigma) | literature: Ong 2015 ratio; absolute level reasonable |
| DFW wavelet levels / padding | 3 / 8 voxels | reasonable |
| Relaxed lam | 10 x min(spacing)^2 | reasonable; Mura 2016 does not give a universal value |
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
| `vorticity`, `helicity_density` | curl v, v . curl v (relative helicity option) | Lorenz et al. 2014, PMID 23716466; Garcia et al. 2019, PMID 30732719 |
| `viscous_dissipation`, `lumen_integrals` | 2 mu S:S, integrated over the lumen | Barker et al. 2014, PMID 24122967 |
| `kinetic_energy_density` | 1/2 rho v^2 | Carlsson et al. 2012, PMID 22180653 |
| validation reference | finite-element formulation | Sotelo et al. 2016, PMID 26780788; 2018, PMID 28980342 |

## 8. Wall shear stress: `wss.py`

Wall points on the marching-cubes surface of the binary lumen mask, inward
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
| `wall_points` | marching cubes surface | Lorensen and Cline 1987, doi:10.1145/37402.37422 |
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
| `relative_pressure` | least-squares / Poisson integration, reliability weighting | Ebbers and Farneback 2009, PMID 19557846; Krittian et al. 2012, PMID 22626833 |
| in vivo reference | aortic 4D pressure mapping | Bock et al. 2011, PMID 21437978 |

## 10. Quality report: `report.py`

`quality_report()` returns one JSON-serialisable dict per study: venc adequacy
(peak component over venc, aliased fraction from wrap counts), static-tissue
velocity noise, residual offset against the 0.6 cm/s threshold with pass/fail,
lumen divergence before and after, noise-mask fraction, plane flows and
mass-conservation error at each stage. `to_markdown()` renders a one-page
summary. Citations: Bissell et al. 2023, PMID 37474977; Gatehouse et al. 2010,
PMID 20074359; Pelc et al. 1991, PMID 1790111; Burkhardt et al. 2023, PMID 37620674.

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
still 2x at 30% velocity noise), and the best mapping to apply. Run it
after unwrapping; wrapped data fail it. Two blind spots: a global sign flip
ties with the identity (settle it with `expected_flow_sign()` on a plane
whose flow direction is known, e.g. the ascending aorta), and a component
that is near zero everywhere (planar flow) cannot have its sign tested.
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
answer; exact away from the junction, which is flagged by a mask), `chest`
(thorax with lungs, parabolic artery, steady vein, second-order offset,
wrap-around band), plus `add_noise` and `alias`. The test suite runs on
these without downloading data; the 4DFlowNet CFD tests are marked `cfd`.

A note on temporal unwrapping found while testing: Xiang's method requires the
true velocity to change by less than venc between frames. With too few cardiac
phases for the peak velocity it can misread a recovered systolic rise as a wrap,
so `laplacian3d` alone is preferable in that case.

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
