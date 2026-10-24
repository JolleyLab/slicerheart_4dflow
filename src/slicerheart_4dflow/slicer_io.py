"""
Readers for the file formats that 3D Slicer writes, so that data prepared in
Slicer (for example by the SlicerHeart Import 4D Flow module) can be
processed with this package without Slicer and without a DICOM reader.

- volume ``.nrrd``: scalar or vector image
- volume sequence ``.seq.nrrd``: 4D (scalar frames) or 5D (vector frames)
  NRRD with a ``list`` axis and the sequence index values in the header
- segmentation ``.seg.nrrd``: 3D (one layer) or 4D (several layers) labelmap
  with ``SegmentN_*`` header fields; the extent may differ from the image
- markups ``.mrk.json``: plane markups (centre, normal, size)

Arrays are returned in the conventions of this package (see geometry):
indexed [k, j, i] (time first, vector components last) with a 4x4
voxel_to_world matrix. The world frame is the one of the file, LPS for
files written by Slicer; pass world="RAS" to get RAS instead. Spatial
vectors (Slicer writes them with a measurement frame) are converted with the
measurement frame into the same world frame; other vector volumes (no
measurement frame, for example values per velocity encoding direction) are
returned as stored. Plane markups are converted from their coordinate system
to the same world frame, so everything read together is consistent.

The NRRD reader is pynrrd, segmentation files are read with slicerio (optional
dependencies, ``pip install slicerheart_4dflow[clinical]``).

A data set of test cases (see tests/clinical/README.md, format version
CASE_FORMAT_VERSION) holds the inputs of each case in inputs/<case>/ (INPUT_FILES:
the acquisition, the annotations and optional references) and the outputs of the
processing in outputs/<case>/ (OUTPUT_FILES), with manifest.json at its root;
load_case() reads one case.
"""
import json
import re
from pathlib import Path

import numpy as np

_SPACE_TO_LPS = {
    "left-posterior-superior": np.diag([1.0, 1.0, 1.0]),
    "right-anterior-superior": np.diag([-1.0, -1.0, 1.0]),
    "LPS": np.diag([1.0, 1.0, 1.0]),
    "RAS": np.diag([-1.0, -1.0, 1.0]),
}


def _nrrd():
    try:
        import nrrd
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading Slicer files needs pynrrd: pip install slicerheart_4dflow[clinical]") from e
    return nrrd


def _slicerio():
    try:
        import slicerio
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading Slicer segmentations needs slicerio: pip install slicerheart_4dflow[clinical]") from e
    return slicerio


def _lps_to_world(world):
    """3x3 matrix that maps LPS coordinates to the requested world frame (its own inverse)."""
    if world not in ("LPS", "RAS"):
        raise ValueError(f"world must be 'LPS' or 'RAS', not {world!r}")
    return _SPACE_TO_LPS[world]


def _space_to_lps(header):
    space = header.get("space", "left-posterior-superior")
    if space not in _SPACE_TO_LPS:
        raise ValueError(f"unsupported NRRD space {space!r}")
    return _SPACE_TO_LPS[space]


def _parse(path, world):
    """Read an NRRD file. Returns (data with axes (time, k, j, i, component) where present,
    voxel_to_world, header, info) with info listing which axes were present."""
    data, header = _nrrd().read(str(path))
    kinds = list(header.get("kinds", ["domain"] * data.ndim))
    domain_axes = [a for a, k in enumerate(kinds) if k in ("domain", "space")]
    if len(domain_axes) != 3:
        raise ValueError(f"{path}: expected 3 spatial axes, kinds are {kinds}")
    list_axes = [a for a, k in enumerate(kinds) if k == "list"]
    vector_axes = [a for a, k in enumerate(kinds) if a not in domain_axes and a not in list_axes]
    if len(list_axes) > 1 or len(vector_axes) > 1:
        raise ValueError(f"{path}: unsupported axis kinds {kinds}")

    # NRRD (and pynrrd) order is fastest axis first: domain axes are (i, j, k).
    # Target order: (list, k, j, i, vector).
    order = list_axes + domain_axes[::-1] + vector_axes
    data = np.transpose(data, order)

    to_lps = _space_to_lps(header)
    to_world = _lps_to_world(world) @ to_lps
    directions = np.asarray(header["space directions"], float)[domain_axes]   # rows: i, j, k axis vectors
    origin = np.asarray(header.get("space origin", np.zeros(3)), float)
    voxel_to_world = np.eye(4)
    voxel_to_world[:3, :3] = to_world @ directions.T
    voxel_to_world[:3, 3] = to_world @ origin

    spatial = (bool(vector_axes) and data.shape[-1] == 3 and "measurement frame" in header
               and kinds[vector_axes[0]] in ("vector", "covariant-vector", "3-vector"))
    if spatial:
        # Measurement frame: each listed vector is a measurement-frame basis vector in space coordinates.
        mf = np.asarray(header["measurement frame"], float)
        data = data.astype(np.float32, copy=False) @ (mf @ to_world.T).astype(np.float32)

    info = {"has_list": bool(list_axes), "has_vector": bool(vector_axes), "spatial": spatial}
    return data, voxel_to_world, header, info


def _index_values(header, n):
    for key, value in header.items():
        if re.fullmatch(r"axis \d+ index values", key):
            values = value.split()
            try:
                return np.asarray([float(v) for v in values])
            except ValueError:
                return np.asarray(values)
    return np.arange(n, dtype=float)


def read_volume(path, world="LPS"):
    """Read a Slicer volume (.nrrd). Returns (array, voxel_to_world).

    array : (nk, nj, ni) for a scalar volume, (nk, nj, ni, c) for a vector
            volume (spatial vectors in world components).
    """
    data, voxel_to_world, _, info = _parse(path, world)
    if info["has_list"]:
        raise ValueError(f"{path} is a sequence; use read_sequence")
    return data, voxel_to_world


def read_sequence(path, world="LPS"):
    """Read a Slicer volume sequence (.seq.nrrd), as written for a sequence of
    scalar or vector volumes that share one geometry.

    Returns (array_t, voxel_to_world, index_values): array_t is
    (nt, nk, nj, ni) or (nt, nk, nj, ni, c); index_values (nt,) are the
    sequence index values (for 4D flow, the trigger times in ms).
    """
    data, voxel_to_world, header, info = _parse(path, world)
    if not info["has_list"]:
        data = data[np.newaxis]
    return data, voxel_to_world, _index_values(header, data.shape[0])


def segment_names(path):
    """{segment_id: segment_name} of a .seg.nrrd file (header only)."""
    segmentation = _slicerio().read_segmentation(str(path), skip_voxels=True)
    return {s["id"]: s.get("name", s["id"]) for s in segmentation["segments"]}


def read_segmentation(path, segment=None, reference=None, world="LPS", fraction=False):
    """Read one segment of a Slicer segmentation (.seg.nrrd) as a bool mask.

    segment   : segment ID or name; default the only segment, or the one
                with ID or name "lumen".
    reference : optional (shape (nk, nj, ni), voxel_to_world) of the image
                grid to return the mask on. A segmentation on the image grid
                or a crop of it (what Slicer writes) is copied. Otherwise each
                image voxel is sampled at F x F x F points, F the number of
                segmentation voxels per image voxel along each axis (for
                example 2 for a segmentation on a grid twice as fine), with
                nearest neighbour; this is exact for a finer grid aligned
                with the image voxels.
    fraction  : with reference, return the fraction of each image voxel in
                the segment (float32, 0 to 1; a partial-volume mask that
                plane_flow accepts) instead of the bool mask, which is the
                fraction >= 0.5.
    Returns (mask, voxel_to_world) with the reference voxel_to_world if given.
    """
    slicerio = _slicerio()
    segmentation = slicerio.read_segmentation(str(path))
    segments = segmentation["segments"]
    if not segments:
        raise ValueError(f"{path}: no segments")
    if segment is None:
        segment = segments[0] if len(segments) == 1 else "lumen"
    try:
        mask = slicerio.segment_mask(segmentation, segment).transpose(2, 1, 0)      # (i, j, k) to (k, j, i)
    except KeyError as e:
        raise KeyError(f"{path}: no segment {segment!r}; segments are {[s.get('id') for s in segments]}") from e
    seg_to_world = np.eye(4)
    seg_to_world[:3] = _lps_to_world(world) @ np.asarray(segmentation["ijkToLPS"], float)[:3]
    if reference is None:
        return mask, seg_to_world

    shape, ref_to_world = reference
    ref_to_world = np.asarray(ref_to_world, float)
    ref_to_seg = np.linalg.inv(seg_to_world) @ ref_to_world
    out = np.zeros(tuple(shape), bool)
    if np.allclose(ref_to_seg[:3, :3], np.eye(3), atol=1e-4):
        # Same grid up to an integer offset (a crop): copy the overlapping block.
        offset = np.round(ref_to_seg[:3, 3]).astype(int)          # seg (i, j, k) of ref voxel (0, 0, 0)
        ref_lo = np.maximum(0, -offset)
        ref_hi = np.minimum(np.asarray(shape)[::-1], np.asarray(mask.shape)[::-1] - offset)
        if np.all(ref_hi > ref_lo):
            si, sj, sk = (slice(lo + o, hi + o) for lo, hi, o in zip(ref_lo, ref_hi, offset))
            ri, rj, rk = (slice(lo, hi) for lo, hi in zip(ref_lo, ref_hi))
            out[rk, rj, ri] = mask[sk, sj, si]
        return (out.astype(np.float32), ref_to_world) if fraction else (out, ref_to_world)
    # Resampled in world space with VTK; exact for a finer grid aligned with the image voxels
    from .vtk_image import mask_fraction
    out = mask_fraction(mask, seg_to_world, shape, ref_to_world)
    return (out, ref_to_world) if fraction else (out >= 0.5, ref_to_world)


def read_planes(paths, world="LPS"):
    """Read plane markups (.mrk.json) into {name: dict(centre, normal, radius)}.

    paths : a file, a folder (all *.mrk.json in it) or a list of files.
    The name is the markup's "name" if the file stores it, else the file
    stem (Slicer names the node after the file), with an index appended
    when one file holds several planes. radius is half the smaller plane
    size, or None when the plane has no size.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(p.glob("*.mrk.json")) if p.is_dir() else [p]
    planes = {}
    lps_to_world = _lps_to_world(world)
    for path in paths:
        path = Path(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        markups = [m for m in doc.get("markups", []) if m.get("type") == "Plane"]
        stem = path.name[:-len(".mrk.json")] if path.name.endswith(".mrk.json") else path.stem
        for n, m in enumerate(markups):
            name = m.get("name") or (stem if len(markups) == 1 else f"{stem}_{n + 1}")
            to_lps = _SPACE_TO_LPS[m.get("coordinateSystem", "LPS")]
            centre = np.asarray(m["center"], float)
            normal = np.asarray(m["normal"], float)
            size = m.get("size", [0, 0, 0])
            radius = 0.5 * min(size[0], size[1]) if min(size[0], size[1]) > 0 else None
            centre = lps_to_world @ to_lps @ centre
            normal = lps_to_world @ to_lps @ normal
            planes[name] = dict(centre=centre, normal=normal / np.linalg.norm(normal), radius=radius)
    return planes


CASE_FORMAT_VERSION = 4

# A data set of test cases:
#   manifest.json            format_version, name, license, doi of the data set
#   inputs/<case>/           what the processing starts from (INPUT_FILES): the acquisition (images converted from
#                            DICOM without any processing, and its description), the annotations (lumen segmentation and
#                            measurement planes; annotations.json describes them and lists the plane files) and
#                            optional references/<source>/flows.json (plane flows measured independently)
#   outputs/<case>/          results of the processing and its description (OUTPUT_FILES), computed from the inputs
# The case ID is the name of the case folder; the cases of the data set are the folders in inputs/.
DATASET_MANIFEST = "manifest.json"
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
INPUT_FILES = {
    "acquisition": "acquisition/acquisition.json",
    "magnitude": "acquisition/magnitude.seq.nrrd",
    "velocity_raw": "acquisition/velocity_raw.seq.nrrd",
    "annotations": "annotations/annotations.json",
}
OUTPUT_FILES = {
    "processing": "processing.json",
    "velocity": "velocity.seq.nrrd",
    "wrap_count": "wrap_count.seq.nrrd",
    "pcmra": "pcmra.nrrd",
    "flows": "flows.json",
    "quality_report": "quality_report.json",
}
ANNOTATIONS_DIR = "annotations"
REFERENCES_DIR = "references"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_dataset_manifest(dataset_dir):
    """manifest.json of a data set of test cases; ValueError if its format version is not CASE_FORMAT_VERSION."""
    path = Path(dataset_dir) / DATASET_MANIFEST
    manifest = _read_json(path)
    if manifest.get("format_version") != CASE_FORMAT_VERSION:
        raise ValueError(f"{path}: test case format version {manifest.get('format_version')} is not supported"
                         f" (supported: {CASE_FORMAT_VERSION}), convert or export the data set again")
    return manifest


def dataset_cases(dataset_dir):
    """IDs of the cases of a data set: the folders in inputs/ that have an acquisition (sorted)."""
    inputs = Path(dataset_dir) / INPUTS_DIR
    if not inputs.is_dir():
        return []
    return sorted(p.name for p in inputs.iterdir() if (p / INPUT_FILES["acquisition"]).is_file())


def case_directories(dataset_dir, case_id):
    """(inputs folder, outputs folder) of a case."""
    return Path(dataset_dir) / INPUTS_DIR / case_id, Path(dataset_dir) / OUTPUTS_DIR / case_id


def load_case(dataset_dir, case_id, world="LPS"):
    """Read a test case of a data set (layout: INPUT_FILES and OUTPUT_FILES, format version CASE_FORMAT_VERSION).

    Returns a dict with:

    - case_id; manifest (of the data set, with case_id added); input_dir, output_dir
    - acquisition (acquisition.json), annotations (annotations.json), processing (processing.json)
    - magnitude_t (nt, nk, nj, ni)
    - velocity_raw_t (nt, nk, nj, ni, 3) mm/s world components before any
      correction (aliased), and velocity_raw_encoded_t, the same as encoded
      components (velocity along each column of encoding_to_world, each
      aliased in [-venc, venc)), which is what unwrapping operates on
    - voxel_to_world; time_ms; rr_ms
    - venc_encoded_mm_s (per encoded component), venc_mm_s (per world
      component of velocity_raw_t), encoding_to_world (3x3, columns are the
      world directions of the encoded components)
    - has_outputs: whether the outputs of the processing are present; if so:
      velocity_t, world components after the corrections of the software that
      computed the outputs, and wrap_count_t (nt, nk, nj, ni, 3) int16 wraps that
      it applied, per encoded component: velocity_t = (velocity_raw_encoded_t +
      2 venc wrap_count_t) @ encoding_to_world.T - background
    and whichever of these the case has:
    - pcmra; lumen (bool), lumen_fraction (float32, fraction of each voxel in the lumen), lumen_labelmap and
      lumen_labelmap_voxel_to_world (the segmentation on its own grid, which may be finer than the image); planes
    - flows (plane flows computed by the software that computed the outputs), references ({source: flows} of
      independent measurements)
    """
    manifest = read_dataset_manifest(dataset_dir)
    input_dir, output_dir = case_directories(dataset_dir, case_id)
    in_path = lambda key: input_dir / INPUT_FILES[key]
    out_path = lambda key: output_dir / OUTPUT_FILES[key]
    if not in_path("acquisition").is_file():
        raise FileNotFoundError(f"{input_dir}: no case {case_id!r} in the data set (no {INPUT_FILES['acquisition']})")
    acquisition = _read_json(in_path("acquisition"))
    case = {"case_id": case_id, "manifest": dict(manifest, case_id=case_id), "dataset_dir": Path(dataset_dir),
            "input_dir": input_dir, "output_dir": output_dir, "acquisition": acquisition,
            "annotations": _read_json(in_path("annotations")) if in_path("annotations").exists() else {},
            "processing": _read_json(out_path("processing")) if out_path("processing").exists() else {}}
    lps_to_world = _lps_to_world(world)

    sequences = [(in_path("magnitude"), "magnitude_t"), (in_path("velocity_raw"), "velocity_raw_t")]
    case["has_outputs"] = out_path("velocity").exists() and out_path("wrap_count").exists()
    if case["has_outputs"]:
        sequences += [(out_path("velocity"), "velocity_t"), (out_path("wrap_count"), "wrap_count_t")]
    for path, name in sequences:
        arr, vtw, index_values = read_sequence(path, world=world)
        if "voxel_to_world" in case and not np.allclose(vtw, case["voxel_to_world"], atol=1e-4):
            raise ValueError(f"{path}: geometry differs from the other volumes")
        case[name], case["voxel_to_world"] = arr, vtw
        case.setdefault("time_ms", index_values)
    if case["has_outputs"]:
        case["wrap_count_t"] = np.rint(case["wrap_count_t"]).astype(np.int16)
    case["rr_ms"] = acquisition.get("rr_ms")

    # Stored in LPS: columns are world directions, so the matrix maps like a direction.
    encoding_to_world = lps_to_world @ np.asarray(acquisition["encoding_to_world"], float)
    case["encoding_to_world"] = encoding_to_world
    case["venc_encoded_mm_s"] = np.asarray(acquisition["venc_encoded_mm_s"], float)
    case["venc_mm_s"] = np.abs(encoding_to_world) @ case["venc_encoded_mm_s"]
    # Orthonormal encoding directions: encoded components are the projections onto the columns
    case["velocity_raw_encoded_t"] = case["velocity_raw_t"] @ encoding_to_world.astype(np.float32)

    if out_path("pcmra").exists():
        case["pcmra"], _ = read_volume(out_path("pcmra"), world=world)
    reference_grid = (case["magnitude_t"].shape[1:4], case["voxel_to_world"])
    annotations_dir = input_dir / ANNOTATIONS_DIR
    lumen = case["annotations"].get("lumen")
    if lumen:
        # The lumen may be segmented on a grid finer than the image (Import4DFlow uses 2x): lumen_fraction is the
        # fraction of each image voxel in the lumen (for plane_flow), lumen the voxels at least half in it
        lumen_path = annotations_dir / lumen["file"]
        case["lumen_fraction"], _ = read_segmentation(lumen_path, segment=lumen.get("segment"), reference=reference_grid,
                                                      world=world, fraction=True)
        case["lumen"] = case["lumen_fraction"] >= 0.5
        # the segmentation on its own grid (for comparisons with software that uses the segment surface)
        case["lumen_labelmap"], case["lumen_labelmap_voxel_to_world"] = read_segmentation(lumen_path, segment=lumen.get("segment"),
                                                                                          world=world)
    planes = case["annotations"].get("planes")
    if planes and planes.get("files"):
        case["planes"] = read_planes([annotations_dir / name for name in planes["files"]], world=world)
    if out_path("flows").exists():
        case["flows"] = _read_json(out_path("flows"))
    references_dir = input_dir / REFERENCES_DIR
    if references_dir.is_dir():
        case["references"] = {p.parent.name: _read_json(p) for p in sorted(references_dir.glob("*/flows.json"))}
    return case
