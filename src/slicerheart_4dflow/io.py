"""
DRAFT: written before any clinical case was available. The case layout,
units and component frame follow the Slicer 5D convention from the
Discourse post; they must be aligned with what the SlicerHeart Import4DFlow
and Analyze4DFlow modules actually write before this is relied on.

Reading and writing 4D flow data in the 3D Slicer 5D image convention.

Slicer stores a time series of vector volumes as a single 5D NRRD file
(https://discourse.slicer.org/t/new-feature-support-of-5d-images-and-transforms/45588):

    dimension: 5
    space: left-posterior-superior
    sizes: 3 ni nj nk nt
    space directions: none (di) (dj) (dk) none
    kinds: vector domain domain domain list
    labels: "" "" "" "" "time"
    units: "" "" "" "" "s"
    axis 4 index type: numeric
    axis 4 index values: t0 t1 ... (seconds)
    space origin: (ox, oy, oz)

and a time series of scalar volumes (magnitude) as the 4D form without the
vector axis. Files conventionally end in `.seq.nrrd`; segmentations
exported from Slicer are `.seg.nrrd` labelmaps (3D, or 4D with one segment
per leading axis).

This module converts between that layout and the package's arrays:
velocity_t (nt, nk, nj, ni, 3) in mm/s with world components, magnitude_t
(nt, nk, nj, ni), voxel_to_world 4x4 mapping voxel (i, j, k, 1) to world
(x, y, z, 1), and frame times in ms.

Coordinate frames. NRRD geometry is written in LPS; Slicer works in RAS
internally and converts the geometry on read and write. The vector
components of a velocity volume are voxel VALUES, and whether they are
expressed in LPS or RAS depends on the writer. This package is agnostic to
the world frame (every function takes voxel_to_world and velocity in the
same frame), so the reader returns both in the frame of the file (LPS) by
default and `component_frame` says what frame the stored components are
in; pass "RAS" if the writer stored RAS components and they are flipped to
LPS. `to_ras()` converts a whole data set to RAS when Slicer-side
quantities (markup planes in RAS) are to be combined with it.

Units. The convention does not fix the velocity unit; `velocity_units`
("mm/s", "cm/s" or "m/s") converts to mm/s on read.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .geometry import as_voxel_to_world

_UNIT_TO_MM_S = {"mm/s": 1.0, "cm/s": 10.0, "m/s": 1000.0}
_FLIP_LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _nrrd():
    try:
        import nrrd
    except ImportError as exc:
        raise ImportError("reading Slicer NRRD files needs pynrrd: pip install pynrrd") from exc
    return nrrd


def _parse_header_geometry(header):
    """voxel_to_world (4x4) and the indices of the domain axes from an NRRD header."""
    kinds = header.get("kinds")
    if kinds is None:
        kinds = ["domain"] * int(header["dimension"])
    domain_axes = [i for i, k in enumerate(kinds) if k in ("domain", "space")]
    if len(domain_axes) != 3:
        raise ValueError(f"expected three domain axes, got kinds {kinds}")
    dirs = header.get("space directions")
    if dirs is None:
        raise ValueError("NRRD header has no 'space directions'")
    M = np.eye(4)
    cols = [np.asarray(dirs[a], float) for a in domain_axes]
    if any(np.isnan(c).any() for c in cols):
        raise ValueError("a domain axis has no space direction")
    M[:3, :3] = np.stack(cols, 1)                    # columns: world direction of i, j, k
    origin = header.get("space origin")
    M[:3, 3] = np.asarray(origin, float) if origin is not None else 0.0
    return M, domain_axes, kinds


def _frame_times_ms(header, axis, nt):
    """Frame times from 'axis N index values' (seconds), or None."""
    key = f"axis {axis} index values"
    if key in header:
        vals = header[key]
        if isinstance(vals, str):
            vals = [float(x) for x in re.split(r"[\s,]+", vals.strip()) if x]
        vals = np.asarray(vals, float)
        if vals.size == nt:
            unit = (header.get("units") or [""] * (axis + 1))[axis] if "units" in header else "s"
            scale = {"s": 1000.0, "ms": 1.0, "": 1000.0}.get(unit, 1000.0)
            return vals * scale
    return None


def read_flow_sequence(velocity_path, magnitude_path=None, velocity_units="mm/s", component_frame="LPS"):
    """
    Read a 5D velocity sequence (and optionally a 4D magnitude sequence)
    written in the Slicer convention.

    Returns dict with velocity_t (nt, nk, nj, ni, 3) mm/s, magnitude_t
    (nt, nk, nj, ni) or None, voxel_to_world (4, 4) in the file's space
    (LPS), times_ms (nt,) or None, space (str), and the raw headers.
    """
    nrrd = _nrrd()
    data, header = nrrd.read(str(velocity_path))
    M, domain_axes, kinds = _parse_header_geometry(header)
    vector_axes = [i for i, k in enumerate(kinds) if k in ("vector", "3-vector", "covariant-vector")]
    list_axes = [i for i, k in enumerate(kinds) if k in ("list", "time")]
    if len(vector_axes) != 1 or data.shape[vector_axes[0]] != 3:
        raise ValueError(f"expected one vector axis of size 3, got kinds {kinds} and shape {data.shape}")
    if len(list_axes) > 1:
        raise ValueError(f"expected at most one time (list) axis, got kinds {kinds}")
    # pynrrd returns the axes in header order (Fortran index order): reorder to (t, k, j, i, component)
    order = (list_axes + domain_axes[::-1] + vector_axes) if list_axes else (domain_axes[::-1] + vector_axes)
    v = np.transpose(np.asarray(data), order).astype(np.float32)
    if not list_axes:
        v = v[None]
    v = v * _UNIT_TO_MM_S[velocity_units]
    if component_frame.upper() == "RAS":
        v = v * np.array([-1.0, -1.0, 1.0], np.float32)       # RAS components -> LPS
    elif component_frame.upper() != "LPS":
        raise ValueError("component_frame must be 'LPS' or 'RAS'")
    times = _frame_times_ms(header, list_axes[0], v.shape[0]) if list_axes else None
    out = {"velocity_t": v, "magnitude_t": None, "voxel_to_world": M, "times_ms": times,
           "space": header.get("space", "left-posterior-superior"), "header": header}
    if magnitude_path is not None:
        mag, mh = nrrd.read(str(magnitude_path))
        Mm, dom_m, kinds_m = _parse_header_geometry(mh)
        list_m = [i for i, k in enumerate(kinds_m) if k in ("list", "time")]
        order_m = (list_m + dom_m[::-1]) if list_m else dom_m[::-1]
        m = np.transpose(np.asarray(mag), order_m).astype(np.float32)
        if not list_m:
            m = m[None]
        if not np.allclose(Mm, M, atol=1e-3):
            raise ValueError("magnitude and velocity files have different geometry")
        if m.shape != v.shape[:4]:
            raise ValueError(f"magnitude shape {m.shape} does not match velocity {v.shape[:4]}")
        out["magnitude_t"] = m
        out["magnitude_header"] = mh
    return out


def write_flow_sequence(path, velocity_t, voxel_to_world, times_ms=None, velocity_units="mm/s",
                        magnitude_t=None, magnitude_path=None, component_frame="LPS"):
    """
    Write velocity_t (nt, nk, nj, ni, 3) mm/s as a 5D Slicer sequence, and
    optionally magnitude_t as a 4D one. voxel_to_world is taken to be in
    LPS (the NRRD space). Components are written in `component_frame`.
    """
    nrrd = _nrrd()
    M = as_voxel_to_world(voxel_to_world)
    v = np.asarray(velocity_t, np.float32) / _UNIT_TO_MM_S[velocity_units]
    if component_frame.upper() == "RAS":
        v = v * np.array([-1.0, -1.0, 1.0], np.float32)
    nt = v.shape[0]
    # package (t, k, j, i, c) -> file axis order (c, i, j, k, t)
    data = np.ascontiguousarray(np.transpose(v, (4, 3, 2, 1, 0)))
    # pynrrd writes a "none" direction for a row of NaN
    none = [np.nan, np.nan, np.nan]
    dirs = np.array([none] + [M[:3, a].tolist() for a in range(3)] + [none], float)
    times = np.arange(nt, dtype=float) if times_ms is None else np.asarray(times_ms, float) / 1000.0
    header = {
        "space": "left-posterior-superior",
        "space directions": dirs,
        "space origin": M[:3, 3].tolist(),
        "kinds": ["vector", "domain", "domain", "domain", "list"],
        "labels": ["", "", "", "", "time"],
        "units": ["", "", "", "", "s"],
        "axis 4 index type": "numeric",
        "axis 4 index values": " ".join(f"{t:g}" for t in times),
        "encoding": "gzip",
        "DataNodeClassName": "vtkMRMLVectorVolumeNode",
        "velocity_units": velocity_units,
        "velocity_component_frame": component_frame.upper(),
    }
    nrrd.write(str(path), data, header)
    if magnitude_t is not None:
        mpath = magnitude_path or str(path).replace("velocity", "magnitude")
        m = np.ascontiguousarray(np.transpose(np.asarray(magnitude_t, np.float32), (3, 2, 1, 0)))
        mheader = {
            "space": "left-posterior-superior",
            "space directions": dirs[1:-1],
            "space origin": M[:3, 3].tolist(),
            "kinds": ["domain", "domain", "domain", "list"],
            "labels": ["", "", "", "time"],
            "units": ["", "", "", "s"],
            "axis 3 index type": "numeric",
            "axis 3 index values": " ".join(f"{t:g}" for t in times),
            "encoding": "gzip",
            "DataNodeClassName": "vtkMRMLScalarVolumeNode",
        }
        nrrd.write(str(mpath), m, mheader)
        return path, mpath
    return path


def read_segmentation(path, voxel_to_world=None, shape=None):
    """
    Boolean mask (nk, nj, ni) from a Slicer `.seg.nrrd` or labelmap NRRD
    (any non-zero voxel of any segment). With voxel_to_world and shape of
    the velocity data given, the geometry must match (the segmentation is
    not resampled here).
    """
    nrrd = _nrrd()
    data, header = nrrd.read(str(path))
    M, domain_axes, kinds = _parse_header_geometry(header)
    others = [i for i in range(data.ndim) if i not in domain_axes]
    arr = np.asarray(data)
    if others:
        arr = np.transpose(arr, others + domain_axes[::-1])
        mask = (arr != 0).any(axis=tuple(range(len(others))))
    else:
        mask = np.transpose(arr, domain_axes[::-1]) != 0
    if voxel_to_world is not None and not np.allclose(M, as_voxel_to_world(voxel_to_world), atol=1e-3):
        raise ValueError("segmentation geometry differs from the velocity geometry; resample it in Slicer first")
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError(f"segmentation shape {mask.shape} differs from the velocity shape {tuple(shape)}")
    return mask


def to_ras(velocity_t, voxel_to_world, points=None):
    """
    Convert an LPS data set to RAS: voxel_to_world, the velocity components
    and optionally (n, 3) points. Use when combining with Slicer markups
    (which are RAS) without converting those.
    """
    F = _FLIP_LPS_RAS
    v = np.asarray(velocity_t, float) * np.array([-1.0, -1.0, 1.0])
    M = F @ as_voxel_to_world(voxel_to_world)
    if points is None:
        return v, M
    p = np.asarray(points, float) * np.array([-1.0, -1.0, 1.0])
    return v, M, p


def planes_to_frame(planes, coordinate_system, target="LPS"):
    """
    Planes {name: dict(centre, normal, radius)} given in `coordinate_system`
    ("LPS" or "RAS", e.g. from Slicer markups) converted to `target`.
    """
    if coordinate_system.upper() == target.upper():
        return {k: dict(v) for k, v in planes.items()}
    flip = np.array([-1.0, -1.0, 1.0])
    out = {}
    for name, p in planes.items():
        q = dict(p)
        q["centre"] = tuple((np.asarray(p["centre"], float) * flip).tolist())
        q["normal"] = tuple((np.asarray(p["normal"], float) * flip).tolist())
        out[name] = q
    return out


def load_case(manifest_path):
    """
    Load a clinical (or mock) case from its manifest (schema in
    tests/clinical/README.md). Returns dict with manifest, velocity_t (mm/s,
    LPS components), magnitude_t, voxel_to_world (LPS), times_ms, venc_mm_s,
    rr_ms, lumen (bool or None), planes ({name: dict(centre, normal,
    radius)} in LPS), reference (the reference flows dict) and truth (dict
    of arrays for mock cases, else None).
    """
    import json
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    files = man["files"]
    data = read_flow_sequence(base / files["velocity"], base / files["magnitude"] if "magnitude" in files else None,
                              velocity_units=man.get("velocity_units", "mm/s"),
                              component_frame=man.get("velocity_component_frame", "LPS"))
    v, M = data["velocity_t"], data["voxel_to_world"]
    lumen = read_segmentation(base / files["lumen"], M, v.shape[1:4]) if "lumen" in files else None
    planes = None
    if "planes" in files:
        pj = json.loads((base / files["planes"]).read_text())
        planes = planes_to_frame(pj["planes"], pj.get("coordinate_system", "LPS"), "LPS")
    reference = json.loads((base / files["reference_flows"]).read_text()) if "reference_flows" in files else None
    truth = None
    if "truth" in files and (base / files["truth"]).exists():
        with np.load(base / files["truth"]) as z:
            truth = {k: z[k] for k in z.files}
    times = data["times_ms"]
    if times is None and "rr_ms" in man:
        times = np.arange(v.shape[0]) * man["rr_ms"] / v.shape[0]
    return {"manifest": man, "velocity_t": v, "magnitude_t": data["magnitude_t"], "voxel_to_world": M,
            "times_ms": times, "venc_mm_s": np.broadcast_to(np.asarray(man["venc_mm_s"], float), (3,)).copy(),
            "rr_ms": float(man["rr_ms"]), "lumen": lumen, "planes": planes, "reference": reference,
            "truth": truth, "path": base}
