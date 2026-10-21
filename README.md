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

## Development

```
pip install -e ".[test]"
python -m pytest
```

Tests that use the aortic CFD example of [4DFlowNet](https://github.com/EdwardFerdian/4DFlowNet)
download it into `tests/data` on first use (see [tests/README.md](tests/README.md)).

## Releasing

Versions are derived from git tags. Publishing a GitHub release with a tag such as `v0.1.0`
runs the tests, builds the sdist and wheel, and uploads them to PyPI
([.github/workflows/publish.yml](.github/workflows/publish.yml)).
This requires a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/) set up for
project `slicerheart_4dflow`, repository `JolleyLab/slicerheart_4dflow`, workflow `publish.yml`,
environment `pypi`.

## License

MIT
