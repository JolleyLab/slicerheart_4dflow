# slicerheart_4dflow

WARNING: This implementation is a draft, under active refinement.

Draft Implementation of Filters for 4D flow MRI processing by the [SlicerHeart](https://github.com/SlicerHeart/SlicerHeart) project:

- background phase (eddy current) correction: `ols`, `msac`, `wrls`, `war+msac`, `arto`
- velocity anti-aliasing (phase unwrapping): temporal, Laplacian 3D/4D, Laplacian then temporal
- multi-venc (dual-/triple-venc) unwrapping (DRAFT, phantom-tested): `sdv`, `odv`, `stepwise`, `schnell`
- noise masking and PC-MRA
- divergence-free filtering and denoising: `fdm`, `relaxed`, `leray`, `dfw`
- plane flow quantification and mass-conservation QA
- pathlines (numpy RK4 or VTK particle tracer) and centerline-based plane placement
- derived hemodynamics: vorticity, helicity, kinetic energy, viscous energy loss
- wall shear stress (TAWSS, OSI) from the lumen segmentation
- relative pressure by the pressure Poisson equation
- per-study quality report (venc adequacy, noise, residual offset, mass conservation) as JSON and Markdown
- analytic phantoms (Poiseuille, Womersley, stenosis, bifurcation, thorax, valve jet) and a phase-contrast forward model (noise, aliasing, partial volume, dephasing, multi-venc) for tests without data download

Only numpy and scipy are required; VTK is needed for pathline export and the VTK particle tracer.
Methods, defaults and literature references are described in
[docs/FLOWFILTERS_README.md](docs/FLOWFILTERS_README.md), [docs/BACKGROUND.md](docs/BACKGROUND.md)
and [docs/METHODS_DRAFT.md](docs/METHODS_DRAFT.md).


## Installation

```
pip install slicerheart_4dflow
```

In 3D Slicer: `slicer.util.pip_install("slicerheart_4dflow")`.

## Usage

```python
import slicerheart_4dflow as f4d

# encoded velocity components, each aliased at its venc
corrected, info = f4d.correct_background_phase(velocity_t, magnitude_t, method="war+msac")
unwrapped = f4d.unwrap_velocity(corrected, venc, voxel_to_world, method="laplacian_then_temporal")
# world velocity components
velocity_world_t = unwrapped @ encoding_to_world.T
filtered, infos = f4d.divergence_free_sequence(velocity_world_t, lumen_mask, voxel_to_world, method="fdm")
flow = f4d.plane_flow(filtered, voxel_to_world, centre=(10.0, -32.0, 54.0), normal=(0.0, 0.3, 1.0),
                      radius=15.0, lumen_mask=lumen_mask)["flow"]
```

Conventions (see [geometry.py](src/slicerheart_4dflow/geometry.py)):

- Image arrays are indexed `[k, j, i]` (i varies fastest), as returned by
  ITK `GetArrayFromImage` or `slicer.util.arrayFromVolume`: velocity `(nt, nk, nj, ni, 3)`,
  magnitude `(nt, nk, nj, ni)`, lumen mask `(nk, nj, ni)` bool.
- `voxel_to_world` is a 4x4 homogeneous matrix that maps voxel coordinates `(i, j, k, 1)`
  to world coordinates `(x, y, z, 1)` in mm. Any affine mapping is supported:
  anisotropic, oblique, flipped or even sheared voxel axes.
  `f4d.voxel_to_world_matrix(spacing, origin, directions)` builds one.
- The world coordinate system can be any (e.g. LPS or RAS); the package makes no
  assumption about it, the inputs only need to be consistent.
- Velocity vectors have world components `(vx, vy, vz)` in the coordinate system of
  `voxel_to_world`. Unwrapping is the exception: it must be applied to the components
  as encoded, because each encoding direction aliases at its own venc.
- Points (plane centres, seeds, centerlines, pathline points) are world `(x, y, z)` mm;
  plane normals are world vectors.

### Reading data prepared in 3D Slicer

[slicer_io.py](src/slicerheart_4dflow/slicer_io.py) reads the files that 3D Slicer writes, so data can be
processed without Slicer and without a DICOM reader. It needs the `clinical` extra
(`pip install slicerheart_4dflow[clinical]`, which installs pynrrd).

- **Formats:**
  - volumes and volume sequences (`.nrrd`, `.seq.nrrd`), with vector components converted by the
    measurement frame
  - segmentations (`.seg.nrrd`), including several layers and cropped extents
  - plane markups (`.mrk.json`)
  - whole 4D flow test cases exported by SlicerHeart Import 4D Flow (see
    [tests/clinical/README.md](tests/clinical/README.md))
- **World frame:** arrays follow the conventions above. The world frame is the file's (LPS), or RAS
  with `world="RAS"`.

```python
f4d.dataset_cases("D:/Data/4Dflow/_test_cases_anonymized_")      # ["Patient0001", ...]
case = f4d.load_case("D:/Data/4Dflow/_test_cases_anonymized_", "Patient0004")
case["velocity_raw_t"], case["magnitude_t"], case["voxel_to_world"], case["venc_mm_s"], case["lumen"], case["planes"]
lumen, voxel_to_world = f4d.read_segmentation("lumen.seg.nrrd", segment="lumen")
```

## Development

```
pip install -e ".[test]"
python -m pytest
```

Tests that use an aortic CFD example download it into `tests/data` on first use, from the
[TestingData release](https://github.com/JolleyLab/slicerheart_4dflow/releases/tag/TestingData)
(see [tests/README.md](tests/README.md)).

Test results (the regression metrics of the last run) are written to `.test-reports/` at the
repository root, which is not committed; set `SLICERHEART_4DFLOW_TEST_REPORT_DIR` to use another
folder. The clinical tests and the baselines kept with the clinical data are described in
[tests/clinical/README.md](tests/clinical/README.md).

## Releasing

Versions are derived from git tags. Publishing a GitHub release with a tag such as `v0.1.0`
runs the tests, builds the sdist and wheel, and uploads them to PyPI
([.github/workflows/publish.yml](.github/workflows/publish.yml)).
This requires a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/) set up for
project `slicerheart_4dflow`, repository `JolleyLab/slicerheart_4dflow`, workflow `publish.yml`,
environment `pypi`.

## License

MIT
