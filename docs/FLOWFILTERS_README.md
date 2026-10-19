# SlicerHeart 4D flow filters

Python (numpy, scipy; VTK for pathlines_fast) for the SlicerHeart 4D flow module.
Arrays are indexed `[k, j, i]` (as `slicer.util.arrayFromVolume`): `(nt, nk, nj, ni, 3)`
velocity with world components `(vx, vy, vz)`, `(nt, nk, nj, ni)` magnitude,
`(nk, nj, ni)` bool lumen masks. Image geometry is a 4x4 `voxel_to_world` matrix
mapping voxel `(i, j, k, 1)` to world `(x, y, z, 1)` in mm (IJKToRAS in Slicer); any
affine mapping (anisotropic, oblique, flipped) is supported. Points and plane normals
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
