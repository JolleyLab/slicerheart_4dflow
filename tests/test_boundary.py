"""
Vessel boundary in each frame (boundary.refine_vessel_boundary) on a straight vessel whose radius pulsates over the
cardiac cycle, with partial volumes at the wall: from a static approximate segmentation (the mean radius, on a grid
twice as fine), the boundary follows the radius in each frame within a fraction of a voxel, the flow through a plane is
closer to the true flow than with the static segmentation, the boundary moves by at most the maximum offset, and a
bright vessel next to it is not added.
"""

import numpy as np

from slicerheart_4dflow import plane_flow, refine_vessel_boundary
from slicerheart_4dflow.boundary import signed_distance_mm

VOXEL_MM = 1.5
SHAPE = (24, 36, 44)                     # (k, j, i); the vessel runs along k
CENTRE_JI = (17.5, 17.5)                 # vessel axis, voxels
NEIGHBOUR_JI = (17.5, 30.5)              # axis of a second vessel, voxels
NEIGHBOUR_RADIUS_MM = 3.0
RADII_MM = 5.0 + 1.0 * np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))
PEAK_VELOCITY_MM_S = 600.0
BLOOD, TISSUE = 1000.0, 400.0


def voxel_to_world(scale=1.0):
    m = np.eye(4)
    m[:3, :3] *= VOXEL_MM / scale
    return m


def phantom(seed=0):
    """Magnitude and through-plane velocity (supersampled 6 x 6 in the plane: partial volumes, the velocity of a
    voxel is the magnitude-weighted mean of its parts), the true flow (ml/s) and the true radius of each frame."""
    rng = np.random.default_rng(seed)
    sub = (np.arange(6) + 0.5) / 6 - 0.5
    j = np.arange(SHAPE[1])[:, None, None, None] + sub[None, None, :, None]
    i = np.arange(SHAPE[2])[None, :, None, None] + sub[None, None, None, :]
    rho = np.hypot(j - CENTRE_JI[0], i - CENTRE_JI[1]) * VOXEL_MM
    rho_neighbour = np.hypot(j - NEIGHBOUR_JI[0], i - NEIGHBOUR_JI[1]) * VOXEL_MM
    magnitude, velocity, flow = [], [], []
    for radius in RADII_MM:
        blood = (rho < radius) | (rho_neighbour < NEIGHBOUR_RADIUS_MM)
        m = np.where(blood, BLOOD, TISSUE)
        v = np.where(rho < radius, PEAK_VELOCITY_MM_S * (1 - (rho / radius) ** 2), 0.0)
        v = np.where(rho_neighbour < NEIGHBOUR_RADIUS_MM, 300.0, v)
        m_voxel = m.mean(axis=(2, 3))
        v_voxel = (m * v).sum(axis=(2, 3)) / m.sum(axis=(2, 3))
        magnitude.append(np.broadcast_to(m_voxel, SHAPE) + rng.normal(0, 15.0, SHAPE))
        velocity.append(np.broadcast_to(v_voxel, SHAPE))
        flow.append(PEAK_VELOCITY_MM_S * np.pi * radius ** 2 / 2 / 1000.0)
    velocity_t = np.zeros(np.shape(velocity) + (3,))
    velocity_t[..., 2] = velocity                # along k (world z)
    return np.array(magnitude), velocity_t, np.array(flow)


def approximate_region(scale=2):
    """Static region: the mean radius, on a grid twice as fine as the image (the voxel centres of the fine grid are
    those of an image grid with spacing VOXEL_MM / scale, starting at the same corner)."""
    shape = tuple(n * scale for n in SHAPE)
    k, j, i = np.indices(shape)
    vtw = voxel_to_world(scale)
    vtw[:3, 3] = -(scale - 1) / (2 * scale) * VOXEL_MM       # fine voxels subdivide the image voxels
    jw, iw = j * VOXEL_MM / scale + vtw[1, 3], i * VOXEL_MM / scale + vtw[0, 3]
    rho = np.hypot(jw - CENTRE_JI[0] * VOXEL_MM, iw - CENTRE_JI[1] * VOXEL_MM)
    return rho < RADII_MM.mean(), vtw


def equivalent_radius(fraction_slice):
    return np.sqrt(fraction_slice.sum() * VOXEL_MM ** 2 / np.pi)


def test_signed_distance_of_a_mask():
    mask = np.zeros((5, 5, 5), bool)
    mask[:, :, :2] = True
    distance = signed_distance_mm(mask, voxel_to_world())
    np.testing.assert_allclose(distance[2, 2, :4], [-1.5 * VOXEL_MM, -0.5 * VOXEL_MM, 0.5 * VOXEL_MM, 1.5 * VOXEL_MM])


def test_boundary_follows_the_pulsating_radius(metrics):
    magnitude, velocity, true_flow = phantom()
    region, region_vtw = approximate_region()
    result = refine_vessel_boundary(region, magnitude, voxel_to_world(), region_voxel_to_world=region_vtw)
    fraction = result["fraction"]
    middle = SHAPE[0] // 2
    # the vessel cross-section only (the neighbour is far from the plane disc used below)
    j, i = np.indices(SHAPE[1:])
    own = np.hypot(j - CENTRE_JI[0], i - CENTRE_JI[1]) * VOXEL_MM < 8.5
    radii = np.array([equivalent_radius(f[middle][own]) for f in fraction])
    error = np.abs(radii - RADII_MM)
    metrics.record("boundary/radius_max_error_mm", error.max(), atol=0.05)
    assert error.max() < 0.3 * VOXEL_MM, f"radius {radii.round(2)} mm, true {RADII_MM.round(2)} mm"
    assert np.all(np.abs(result["offset_mm"]) <= 2.0 * VOXEL_MM + 1e-4)

    # flow through a plane across the vessel: closer to the truth than with the static region
    plane = {"centre": [CENTRE_JI[1] * VOXEL_MM, CENTRE_JI[0] * VOXEL_MM, middle * VOXEL_MM], "normal": [0, 0, 1],
             "radius": 8.0}
    refined = plane_flow(velocity, voxel_to_world(), lumen_mask=fraction, **plane)["flow"] / 1000.0
    static_mask = result["signed_distance_mm"] <= 0
    static = plane_flow(velocity, voxel_to_world(), lumen_mask=static_mask, **plane)["flow"] / 1000.0
    refined_error = np.abs(refined - true_flow) / true_flow
    static_error = np.abs(static - true_flow) / true_flow
    metrics.record("boundary/flow_max_relative_error", refined_error.max(), atol=0.01)
    assert refined_error.max() < 0.06, f"refined flow error {refined_error.round(3)}"
    assert refined_error.mean() < 0.5 * static_error.mean(), f"refined {refined_error.mean():.3f}, static {static_error.mean():.3f}"


def test_region_cropped_to_its_extent():
    """The labelmap of a segment is cropped to the segment: the boundary still moves outward beyond the crop box (the
    result is the same as with the full grid)."""
    magnitude, _, _ = phantom()
    region, region_vtw = approximate_region()
    full = refine_vessel_boundary(region, magnitude, voxel_to_world(), region_voxel_to_world=region_vtw)["fraction"]
    nz = np.nonzero(region)
    lower = np.array([a.min() for a in nz])                               # (k, j, i)
    cropped = region[tuple(slice(int(a.min()), int(a.max()) + 1) for a in nz)]
    cropped_vtw = region_vtw.copy()
    cropped_vtw[:3, 3] += region_vtw[:3, :3] @ lower[::-1]
    fraction = refine_vessel_boundary(cropped, magnitude, voxel_to_world(), region_voxel_to_world=cropped_vtw)["fraction"]
    np.testing.assert_allclose(fraction, full, atol=1e-5)


def test_neighbouring_vessel_is_not_added():
    """A bright vessel next to the region (its wall 2.5 voxels from the largest radius) is not added: the boundary
    moves by a smooth offset of at most the maximum, so no region that is not connected to it can be added."""
    magnitude, _, _ = phantom()
    region, region_vtw = approximate_region()
    fraction = refine_vessel_boundary(region, magnitude, voxel_to_world(), region_voxel_to_world=region_vtw)["fraction"]
    j, i = np.indices(SHAPE[1:])
    neighbour = np.hypot(j - NEIGHBOUR_JI[0], i - NEIGHBOUR_JI[1]) * VOXEL_MM < NEIGHBOUR_RADIUS_MM - 0.5 * VOXEL_MM
    assert fraction[:, :, neighbour].max() == 0.0


def test_excluded_voxels_are_not_added():
    magnitude, _, _ = phantom()
    region, region_vtw = approximate_region()
    exclude = np.zeros(SHAPE, bool)
    exclude[:, :, int(CENTRE_JI[1]) + 3:] = True              # the right side of the vessel belongs to something else
    fraction = refine_vessel_boundary(region, magnitude, voxel_to_world(), region_voxel_to_world=region_vtw,
                                      exclude=exclude)["fraction"]
    assert fraction[:, exclude & (refine_vessel_boundary(region, magnitude[:1], voxel_to_world(),
                                                         region_voxel_to_world=region_vtw)["signed_distance_mm"] > 0)].max() == 0.0


def test_region_is_kept_without_contrast():
    """Uniform magnitude (no edge to find): the boundary is not moved."""
    magnitude = np.full((3,) + SHAPE, 500.0)
    region, region_vtw = approximate_region()
    offset = refine_vessel_boundary(region, magnitude, voxel_to_world(), region_voxel_to_world=region_vtw)["offset_mm"]
    assert np.abs(offset).max() < 1e-6
