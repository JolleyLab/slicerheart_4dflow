"""
Readers of the 3D Slicer file formats. tests/fixtures/slicer holds small
files written by 3D Slicer 5.12 (an oblique 2 x 3 x 4 grid, IJK to RAS:
translate (10, -20, 30), rotate z 30 deg, rotate x 20 deg, scale
(1.5, 2, 2.5)); the synthetic tests cover what those files do not: a
cropped or resampled segmentation, several layers, a measurement frame,
a RAS file space and a whole test case.
"""
import json
from pathlib import Path

import numpy as np
import pytest

nrrd = pytest.importorskip("nrrd")

from slicerheart_4dflow import slicer_io  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "slicer"
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def slicer_ijk_to_ras():
    """IJK to RAS of the fixture files, rebuilt as in the generating script."""
    def rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        m = np.eye(4)
        a, b = [(1, 2), (2, 0), (0, 1)][axis]
        m[a, a], m[a, b], m[b, a], m[b, b] = c, -s, s, c
        return m
    t = np.eye(4)
    t[:3, 3] = (10, -20, 30)
    return t @ rot(2, 30) @ rot(0, 20) @ np.diag([1.5, 2.0, 2.5, 1.0])


# --- files written by Slicer -------------------------------------------------

def test_vector_sequence_from_slicer():
    v, vtw, t = slicer_io.read_sequence(FIXTURES / "vel.seq.nrrd")
    assert v.shape == (2, 2, 3, 4, 3)
    np.testing.assert_allclose(t, [5, 45])
    np.testing.assert_allclose(LPS_TO_RAS @ vtw, slicer_ijk_to_ras(), atol=1e-9)
    # Written from RAS components (100 + frame, 0, 0), voxel (k, j, i) = (0, 0, 1) also A = 7
    np.testing.assert_allclose(v[0, 0, 0, 0], [-100, 0, 0])
    np.testing.assert_allclose(v[1, 0, 0, 0], [-101, 0, 0])
    np.testing.assert_allclose(v[0, 0, 0, 1], [-100, -7, 0])
    v_ras, vtw_ras, _ = slicer_io.read_sequence(FIXTURES / "vel.seq.nrrd", world="RAS")
    np.testing.assert_allclose(v_ras[0, 0, 0, 1], [100, 7, 0])
    np.testing.assert_allclose(vtw_ras, slicer_ijk_to_ras(), atol=1e-9)


def test_scalar_sequence_from_slicer():
    m, vtw, t = slicer_io.read_sequence(FIXTURES / "mag.seq.nrrd")
    assert m.shape == (2, 2, 3, 4)
    expected = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(m[0], expected)
    np.testing.assert_array_equal(m[1], expected + 100)
    np.testing.assert_allclose(LPS_TO_RAS @ vtw, slicer_ijk_to_ras(), atol=1e-9)


def test_segmentation_from_slicer():
    _, vtw, _ = slicer_io.read_sequence(FIXTURES / "mag.seq.nrrd")
    assert slicer_io.segment_names(FIXTURES / "lumen.seg.nrrd") == {"lumen": "Segment_1", "other": "Segment_2"}
    lumen, _ = slicer_io.read_segmentation(FIXTURES / "lumen.seg.nrrd", reference=((2, 3, 4), vtw))
    np.testing.assert_array_equal(np.argwhere(lumen), [[1, 1, 2], [1, 2, 2]])
    other, _ = slicer_io.read_segmentation(FIXTURES / "lumen.seg.nrrd", segment="Segment_2")
    np.testing.assert_array_equal(np.argwhere(other), [[0, 0, 0]])
    with pytest.raises(KeyError):
        slicer_io.read_segmentation(FIXTURES / "lumen.seg.nrrd", segment="aorta")


def test_non_spatial_vector_sequence_from_slicer():
    """A vector volume that is not a spatial vector (Slicer writes no measurement frame), here int16 values per velocity
    encoding direction, is returned as stored, in any world frame."""
    for world in ("LPS", "RAS"):
        v, _, t = slicer_io.read_sequence(FIXTURES / "encoded_int16.seq.nrrd", world=world)
        assert v.shape == (2, 2, 3, 4, 3) and v.dtype == np.int16
        np.testing.assert_array_equal(v[0, 0, 0, 1], [1, -2, 3])
        np.testing.assert_array_equal(v[1, 0, 0, 1], [1, -2, 3])
        assert np.count_nonzero(v) == 6


def test_planes_from_slicer():
    (name, plane), = slicer_io.read_planes(FIXTURES / "P1.mrk.json").items()
    assert name == "P1"
    # Placed in Slicer at RAS (1, 2, 3), normal (0, 0, 1), size 20 x 30
    np.testing.assert_allclose(plane["centre"], [-1, -2, 3])
    np.testing.assert_allclose(plane["normal"], [0, 0, 1])
    assert plane["radius"] == 10
    ras = slicer_io.read_planes(FIXTURES, world="RAS")["P1"]
    np.testing.assert_allclose(ras["centre"], [1, 2, 3])


# --- synthetic files ---------------------------------------------------------

def oblique_vtw():
    a = np.radians(25)
    vtw = np.eye(4)
    vtw[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ np.diag([1.2, 1.5, 2.0])
    vtw[:3, 3] = (-5, 7, 11)
    return vtw


def nrrd_header(vtw, space="left-posterior-superior", **extra):
    return {"space": space, "space directions": vtw[:3, :3].T, "space origin": vtw[:3, 3],
            "kinds": ["domain"] * 3, **extra}


def test_cropped_segmentation_and_resampled_segmentation(tmp_path):
    vtw = oblique_vtw()
    shape = (6, 7, 8)                                     # k, j, i
    truth = np.zeros(shape, bool)
    truth[2:4, 3:6, 1:5] = True
    # Slicer writes the labelmap cropped to the segment's extent: offset (i, j, k) = (1, 3, 2)
    crop = truth[2:4, 3:6, 1:5].astype(np.uint8)
    crop_vtw = vtw.copy()
    crop_vtw[:3, 3] = (vtw @ [1, 3, 2, 1])[:3]
    hdr = nrrd_header(crop_vtw, Segment0_ID="lumen", Segment0_Name="Lumen", Segment0_LabelValue="1",
                      Segment0_Layer="0")
    nrrd.write(str(tmp_path / "crop.seg.nrrd"), crop.T, hdr)
    mask, out_vtw = slicer_io.read_segmentation(tmp_path / "crop.seg.nrrd", reference=(shape, vtw))
    np.testing.assert_array_equal(mask, truth)
    np.testing.assert_allclose(out_vtw, vtw)

    # A segmentation on a 2x finer grid, same origin corner: nearest-neighbour resampling
    fine_vtw = vtw.copy()
    fine_vtw[:3, :3] = vtw[:3, :3] / 2
    fine_vtw[:3, 3] = (vtw @ [-0.25, -0.25, -0.25, 1])[:3]
    fine = np.repeat(np.repeat(np.repeat(truth, 2, 0), 2, 1), 2, 2).astype(np.uint8)
    nrrd.write(str(tmp_path / "fine.seg.nrrd"), fine.T, nrrd_header(fine_vtw, Segment0_ID="lumen", Segment0_LabelValue="1"))
    mask, _ = slicer_io.read_segmentation(tmp_path / "fine.seg.nrrd", reference=(shape, vtw))
    np.testing.assert_array_equal(mask, truth)


def test_layered_segmentation(tmp_path):
    vtw = oblique_vtw()
    layers = np.zeros((2, 3, 4, 5), np.uint8)             # layer, k, j, i
    layers[0, 1, 1, 1] = 1
    layers[1, 1, 1, 1] = 3                                # overlapping segment on a second layer
    layers[1, 2, 2, 2] = 3
    hdr = nrrd_header(vtw, Segment0_ID="lumen", Segment0_LabelValue="1", Segment0_Layer="0",
                      Segment1_ID="wall", Segment1_LabelValue="3", Segment1_Layer="1")
    hdr["kinds"] = ["list", "domain", "domain", "domain"]
    hdr["space directions"] = np.vstack([np.full(3, np.nan), vtw[:3, :3].T])
    nrrd.write(str(tmp_path / "layers.seg.nrrd"), np.ascontiguousarray(layers.transpose(0, 3, 2, 1)), hdr)
    lumen, _ = slicer_io.read_segmentation(tmp_path / "layers.seg.nrrd")
    wall, _ = slicer_io.read_segmentation(tmp_path / "layers.seg.nrrd", segment="wall")
    np.testing.assert_array_equal(np.argwhere(lumen), [[1, 1, 1]])
    np.testing.assert_array_equal(np.argwhere(wall), [[1, 1, 1], [2, 2, 2]])


def test_measurement_frame_and_ras_space(tmp_path):
    vtw_ras = oblique_vtw()
    v_world_ras = np.array([3.0, -4.0, 5.0])
    # Components stored in a measurement frame rotated 90 deg about z: frame basis (0,1,0), (-1,0,0), (0,0,1)
    mf_rows = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    stored = np.linalg.solve(mf_rows.T, v_world_ras)
    data = np.broadcast_to(stored, (2, 3, 4, 3)).astype(np.float32)      # k, j, i, c
    hdr = nrrd_header(vtw_ras, space="right-anterior-superior")
    hdr["kinds"] = ["vector", "domain", "domain", "domain"]
    hdr["space directions"] = np.vstack([np.full(3, np.nan), vtw_ras[:3, :3].T])
    hdr["measurement frame"] = mf_rows
    nrrd.write(str(tmp_path / "v.nrrd"), np.ascontiguousarray(data.transpose(3, 2, 1, 0)), hdr)
    v, vtw = slicer_io.read_volume(tmp_path / "v.nrrd")
    np.testing.assert_allclose(v[1, 2, 3], LPS_TO_RAS[:3, :3] @ v_world_ras, atol=1e-5)
    np.testing.assert_allclose(vtw, LPS_TO_RAS @ vtw_ras, atol=1e-12)
    v, vtw = slicer_io.read_volume(tmp_path / "v.nrrd", world="RAS")
    np.testing.assert_allclose(v[0, 0, 0], v_world_ras, atol=1e-5)
    np.testing.assert_allclose(vtw, vtw_ras, atol=1e-12)


def write_sequence(path, array_t, vtw, times, vector=False, spatial=True):
    """Write like Slicer: (component, i, j, k, time) for vectors, LPS space; spatial vectors have a measurement frame."""
    nd = array_t.ndim
    data = array_t.transpose(*(([4] if vector else []) + [3, 2, 1, 0]))
    hdr = {"space": "left-posterior-superior", "space origin": vtw[:3, 3],
           "kinds": (["vector"] if vector else []) + ["domain"] * 3 + ["list"],
           "space directions": np.vstack(([np.full(3, np.nan)] if vector else []) + [vtw[:3, :3].T, np.full((1, 3), np.nan)]),
           f"axis {3} index values": " ".join(str(t) for t in times)}
    if vector and spatial:
        hdr["measurement frame"] = np.eye(3)
    assert data.ndim == nd
    nrrd.write(str(path), np.ascontiguousarray(data), hdr)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def write_case(tmp_path, encoding_to_world, venc_encoded, magnitude, raw_world, velocity_world, wraps_encoded, vtw, times):
    """Write a test case in the layout of slicer_io.CASE_FILES."""
    for folder in ("inputs", "outputs", "annotations"):
        (tmp_path / folder).mkdir()
    write_json(tmp_path / "manifest.json", {"format_version": slicer_io.CASE_FORMAT_VERSION, "case_id": "c1", "license": "internal", "doi": None})
    write_json(tmp_path / "inputs" / "acquisition.json", {"vendor": "siemens", "rr_ms": 600.0, "venc_encoded_mm_s": list(venc_encoded),
                                                          "encoding_to_world": np.asarray(encoding_to_world).tolist()})
    write_sequence(tmp_path / "inputs" / "magnitude.seq.nrrd", magnitude, vtw, times)
    write_sequence(tmp_path / "inputs" / "velocity_raw.seq.nrrd", raw_world, vtw, times, vector=True)
    write_sequence(tmp_path / "outputs" / "velocity.seq.nrrd", velocity_world, vtw, times, vector=True)
    write_sequence(tmp_path / "outputs" / "wrap_count.seq.nrrd", wraps_encoded.astype(np.int16), vtw, times, vector=True, spatial=False)


def test_load_case(tmp_path):
    vtw = oblique_vtw()
    nt, shape = 3, (4, 5, 6)
    rng = np.random.default_rng(0)
    mag = rng.random((nt, *shape)).astype(np.float32)
    vel = rng.normal(size=(nt, *shape, 3)).astype(np.float32)
    encoding_to_world = np.eye(3)[:, [1, 0, 2]]
    write_case(tmp_path, encoding_to_world, [1500.0, 1500.0, 1200.0], mag, vel, vel, np.zeros(vel.shape, np.int16), vtw, [0, 30, 60])
    lumen = np.zeros(shape, np.uint8)
    lumen[1:3, 2:4, 2:5] = 1
    nrrd.write(str(tmp_path / "annotations" / "lumen.seg.nrrd"), lumen.T,
               nrrd_header(vtw, Segment0_ID="lumen", Segment0_LabelValue="1"))
    plane = {"markups": [{"type": "Plane", "coordinateSystem": "LPS", "center": [1, 2, 3], "normal": [0, 0, 2],
                          "size": [10, 12, 0]}]}
    (tmp_path / "annotations" / "AAo.mrk.json").write_text(json.dumps(plane))
    write_json(tmp_path / "annotations" / "annotations.json", {
        "lumen": {"file": "lumen.seg.nrrd", "segment": "lumen", "method": "manual", "reviewed": True},
        "planes": {"files": ["AAo.mrk.json"], "method": "manual", "reviewed": True}})
    write_json(tmp_path / "outputs" / "flows.json", {"planes": {"AAo": {"net_volume_mL": 50}}})
    write_json(tmp_path / "references" / "cvi42" / "flows.json", {"planes": {"AAo": {"net_volume_mL": 48}}})

    case = slicer_io.load_case(tmp_path)
    np.testing.assert_allclose(case["magnitude_t"], mag)
    np.testing.assert_allclose(case["velocity_t"], vel)
    np.testing.assert_allclose(case["voxel_to_world"], vtw, atol=1e-12)
    np.testing.assert_allclose(case["time_ms"], [0, 30, 60])
    np.testing.assert_array_equal(case["lumen"], lumen > 0)
    np.testing.assert_allclose(case["planes"]["AAo"]["normal"], [0, 0, 1])
    assert case["flows"]["planes"]["AAo"]["net_volume_mL"] == 50
    assert case["references"]["cvi42"]["planes"]["AAo"]["net_volume_mL"] == 48
    assert case["rr_ms"] == 600.0
    assert case["annotations"]["planes"]["reviewed"]
    np.testing.assert_allclose(case["venc_encoded_mm_s"], [1500, 1500, 1200])
    np.testing.assert_allclose(case["venc_mm_s"], [1500, 1500, 1200])

    ras = slicer_io.load_case(tmp_path, world="RAS")
    flip = LPS_TO_RAS[:3, :3]
    np.testing.assert_allclose(ras["velocity_t"], vel @ flip, atol=1e-6)
    np.testing.assert_allclose(ras["voxel_to_world"], LPS_TO_RAS @ vtw, atol=1e-12)
    np.testing.assert_allclose(ras["planes"]["AAo"]["centre"], [-1, -2, 3])
    np.testing.assert_allclose(ras["encoding_to_world"], flip @ encoding_to_world)
    np.testing.assert_array_equal(ras["lumen"], case["lumen"])


def test_load_case_rejects_other_format_versions(tmp_path):
    write_json(tmp_path / "manifest.json", {"format_version": 2, "case_id": "old"})
    with pytest.raises(ValueError, match="format version 2"):
        slicer_io.load_case(tmp_path)


def write_encoded_case(tmp_path):
    """Case with an encoding that is a signed permutation of the world (LPS) axes, as Siemens through-plane / in-plane encoding."""
    vtw = oblique_vtw()
    nt, shape = 2, (3, 4, 5)
    rng = np.random.default_rng(1)
    encoding_to_world = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, -1]])          # columns: LPS directions of encoded components
    venc_encoded = np.array([1500.0, 1200.0, 1000.0])
    raw_encoded = (rng.random((nt, *shape, 3)) * 2 - 1) * venc_encoded            # aliased, as acquired
    wraps_encoded = rng.integers(-2, 3, (nt, *shape, 3))
    raw_world = (raw_encoded @ encoding_to_world.T).astype(np.float32)
    velocity_world = ((raw_encoded + 2 * venc_encoded * wraps_encoded) @ encoding_to_world.T).astype(np.float32)
    write_case(tmp_path, encoding_to_world, venc_encoded, rng.random((nt, *shape)).astype(np.float32), raw_world, velocity_world,
               wraps_encoded, vtw, [0, 30])
    return raw_encoded, wraps_encoded, venc_encoded


@pytest.mark.parametrize("world", ["LPS", "RAS"])
def test_load_case_encoded_components(tmp_path, world):
    """Raw velocity and wrap counts per encoded component: the corrected velocity is reconstructed from them in any world frame."""
    raw_encoded, wraps_encoded, venc_encoded = write_encoded_case(tmp_path)
    case = slicer_io.load_case(tmp_path, world=world)
    np.testing.assert_allclose(case["velocity_raw_encoded_t"], raw_encoded, atol=1e-3)
    np.testing.assert_array_equal(case["wrap_count_t"], wraps_encoded)
    np.testing.assert_allclose(case["venc_encoded_mm_s"], venc_encoded)
    reconstructed = (case["velocity_raw_encoded_t"] + 2 * case["venc_encoded_mm_s"] * case["wrap_count_t"]) @ case["encoding_to_world"].T
    np.testing.assert_allclose(reconstructed, case["velocity_t"], atol=1e-2)
    assert np.all(np.abs(case["velocity_raw_encoded_t"]) <= case["venc_encoded_mm_s"] + 1e-3)
