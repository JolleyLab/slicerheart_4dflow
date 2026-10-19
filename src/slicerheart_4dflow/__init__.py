"""
slicerheart_4dflow: preprocessing, filtering, QA and pathline tools for the
SlicerHeart 4D flow module. numpy + scipy only (VTK for pathlines_fast).

Recommended order: correct_background_phase -> unwrap_velocity ->
noise_mask / pcmra -> divergence_free_filter (optional) -> quantify / QA.
Derived hemodynamics (vorticity, energy loss, wall shear stress, relative
pressure) should be computed from dfw-filtered velocity: they depend on
velocity gradients, which amplify noise.

Image geometry is given by a 4x4 voxel_to_world matrix, in any world
coordinate system (e.g. LPS or RAS); see the geometry module for the array
and coordinate conventions.
"""
from .geometry import (voxel_to_world_matrix, voxel_spacing, voxel_to_world_points, world_to_voxel_points,
                       voxel_grid_world_points)
from .background_phase import correct_background_phase, static_tissue_candidates, residual_static_velocity
from .unwrap import unwrap_velocity, laplacian_unwrap_phase, temporal_unwrap, noise_mask, pcmra, preprocess
from .divergence_free import (divergence_free_filter, divergence_free_sequence,
                              build_divergence_operator, divergence_stats)
from .qa import plane_flow, plane_samples, mass_conservation, qa_report, net_volume
from .pathlines import advect_pathlines, seeds_from_plane, to_vtk_polydata, centerline_planes
from .derivatives import (velocity_gradient, vorticity, helicity_density, kinetic_energy_density,
                          strain_rate, viscous_dissipation, lumen_integrals)
from .wss import wall_shear_stress, wall_points
from .pressure import pressure_gradient, relative_pressure, pressure_drop
from .report import quality_report, to_json, to_markdown
from .phantom import poiseuille, womersley, chest, add_noise, alias

try:
    from importlib.metadata import version as _version
    __version__ = _version("slicerheart_4dflow")
except Exception:  # not installed, e.g. source tree on sys.path
    __version__ = "unknown"

__all__ = [n for n in dir() if not n.startswith("_")]
