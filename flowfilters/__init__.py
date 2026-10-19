"""
flowfilters: preprocessing, filtering, QA and pathline tools for the
SlicerHeart 4D flow module. numpy + scipy only (VTK for pathlines_fast).

Recommended order: correct_background_phase -> unwrap_velocity ->
noise_mask / pcmra -> divergence_free_filter (optional) -> quantify / QA.
"""
from .background_phase import correct_background_phase, static_tissue_candidates, residual_static_velocity
from .unwrap import unwrap_velocity, laplacian_unwrap_phase, temporal_unwrap, noise_mask, pcmra, preprocess
from .divergence_free import (divergence_free_filter, divergence_free_sequence,
                              build_divergence_operator, divergence_stats)
from .qa import plane_flow, mass_conservation, qa_report, net_volume
from .pathlines import advect_pathlines, seeds_from_plane, to_vtk_polydata, centerline_planes

__all__ = [n for n in dir() if not n.startswith("_")]
