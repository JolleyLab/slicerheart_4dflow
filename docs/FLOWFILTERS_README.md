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
runs the steps above in the order the consensus requires.


| Function | Method | Citation |
|---|---|---|
| `noise_mask` | low magnitude or high temporal velocity std | Bock et al. 2007, ISMRM 15:3138 |
| `pcmra` | velocity-weighted angiogram | Bock et al. 2010, PMID 20099326 |
| `preprocess` | fixed order: background, unwrap, noise mask, PC-MRA, optional divergence filter | Keller et al. 2016, ISMRM 24:1939 |

## 5. Pathlines: `slicerheart_4dflow/pathlines.py`, `slicerheart_4dflow/pathlines_fast.py`

Particles emitted from a plane are advected through the time-varying velocity field
over the cardiac cycle, producing the animated particle traces that every commercial
4D flow package uses as its primary display. The VTK version is the tracer ParaView
uses; the numpy version is a dependency-free reference.


| Function | Method | Citation |
|---|---|---|
| `pathlines_vtk` | vtkParticlePathFilter (ParaView's tracer), frames upsampled x4 in time | Wigstrom et al. 1999, PMID 10332856; Schroeder et al. 2006 (VTK) |
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

Inward normals from the marching-cubes surface of the lumen mask; velocity
sampled 1, 2, 3 voxels inward; tangential component fitted with a quadratic
through zero at the wall; WSS = mu x wall shear rate; TAWSS and OSI. On
Poiseuille flow it is within 10% at 1 mm voxels. On Womersley flow (boundary
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
switchable) and least-squares integration to a relative pressure field over
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
