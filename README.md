# slicerheart_4dflow

Filters for 4D flow MRI processing by the [SlicerHeart](https://github.com/SlicerHeart/SlicerHeart) project:

- background phase (eddy current) correction: `ols`, `msac`, `wrls`, `war+msac`
- velocity anti-aliasing (phase unwrapping): temporal, Laplacian 3D/4D, Laplacian then temporal
- noise masking and PC-MRA
- divergence-free filtering and denoising: `fdm`, `relaxed`, `leray`, `dfw`
- plane flow quantification and mass-conservation QA
- pathlines (numpy RK4 or VTK particle tracer) and centerline-based plane placement

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

corrected, info = f4d.correct_background_phase(velocity_t, magnitude_t, method="war+msac")
unwrapped = f4d.unwrap_velocity(corrected, venc, spacing, method="laplacian_then_temporal")
filtered, infos = f4d.divergence_free_sequence(unwrapped, lumen_mask, spacing, method="fdm")
```

Array conventions: velocity `(nt, nz, ny, nx, 3)` with components `(vz, vy, vx)`,
magnitude `(nt, nz, ny, nx)`, spacing `(dz, dy, dx)` in mm, lumen mask `(nz, ny, nx)` bool.

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
project `slicerheart_4dflow`, repository `lassoan/slicerheart_4dflow`, workflow `publish.yml`,
environment `pypi`.

## License

MIT
