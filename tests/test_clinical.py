"""
Clinical tier: real scanner data (see tests/clinical/README.md). The tests of a clinical case are marked clinical and
run when the case is present. The same tests also run on a synthetic case in the same layout (clinical_case.py: a
bifurcation phantom written once per test session), as synthetic tests: so the test code runs in CI without clinical
data, and its metrics are in the synthetic baseline (clinical_synthetic/...).

The cases are a data set written by 3D Slicer (SlicerHeart Import 4D Flow, "Test
case" export, or its batch script Export4DFlowTestCases.py), read by
slicer_io.load_case without Slicer and without a DICOM reader:

    manifest.json                               format version, name, license, DOI of the data set
    inputs/<case>/acquisition/acquisition.json  scanner, R-R interval, velocity encoding
    inputs/<case>/acquisition/magnitude.seq.nrrd, velocity_raw.seq.nrrd (as acquired, aliased, mm/s)
    inputs/<case>/annotations/annotations.json  how the lumen and planes were created, plane file names
    inputs/<case>/annotations/lumen.seg.nrrd, *.mrk.json
    inputs/<case>/references/<source>/flows.json  flow through each plane measured independently (optional)
    outputs/<case>/processing.json              software and processing options
    outputs/<case>/velocity.seq.nrrd            after SlicerHeart corrections
    outputs/<case>/wrap_count.seq.nrrd          wraps that SlicerHeart applied, per encoded velocity component
    outputs/<case>/pcmra.nrrd                   velocity-weighted PC-MRA
    outputs/<case>/flows.json                   flow through each plane (SlicerHeart plane probe)

The tests check this package on the real data: velocity encoding, background
correction, phase unwrapping against SlicerHeart, plane flow against
SlicerHeart (consistency) and against independent software (validation), mass
conservation, divergence filtering, invariance to the world frame, pathlines
and the derived hemodynamics. Every number goes to
the regression metrics; the baseline is stored with the data, in one file for all
cases: <clinical data folder>/tests/slicerheart_4dflow_baseline.json (see
conftest.CaseMetricRecorder).

The data set is downloaded from the TestingData release into the testing data cache when a test first needs a case
(slicerheart_4dflow.testing_data; about 4 GB):

    python -m pytest -m clinical

or SLICERHEART_4DFLOW_CLINICAL_DATA names a local data set folder (with manifest.json), used as it is:

    SLICERHEART_4DFLOW_CLINICAL_DATA=D:/Data/4Dflow/_test_cases_anonymized_rev04 python -m pytest -m clinical
"""

import functools
import json
import re
from pathlib import Path

import numpy as np
import pytest

MANIFEST_FIELDS = ("format_version", "name", "license", "doi")
ACQUISITION_FIELDS = ("vendor", "rr_ms", "venc_encoded_mm_s", "encoding_to_world", "component_encoding")
VENDORS = ("siemens", "philips", "ge")
from conftest import CLINICAL_DATA_DIR, clinical_case_ids, ensure_clinical_case  # noqa: E402


@functools.lru_cache(maxsize=1)
def load(case_ref, world="LPS"):
    """Load a case; case_ref is (data set folder, case ID). Only the last case is kept (the tests run case by case, see
    conftest.pytest_collection_modifyitems), so that a clinical case (several GB) is in memory once."""
    pytest.importorskip("nrrd")
    from slicerheart_4dflow.slicer_io import load_case
    prepare(case_ref)
    return load_case(case_ref[0], case_ref[1], world=world)


def prepare(case_ref, prefix=None):
    """Make the files of a clinical case available (downloaded into the testing data cache on first use)."""
    if Path(case_ref[0]) == CLINICAL_DATA_DIR:
        ensure_clinical_case(case_ref[1], prefix)


def case_property(case, key):
    """A field of the subject of a case (for example diagnosis_category), from "subject" in its acquisition.json;
    see tests/clinical/README.md and issue #36."""
    return case["acquisition"].get("subject", {}).get(key)


CLINICAL_CASES = [(str(CLINICAL_DATA_DIR), case_id) for case_id in clinical_case_ids()]

# A synthetic case in the same layout (clinical_case.py, a bifurcation phantom), written once per test session into the
# test report folder: the same tests run on it without clinical data (in CI), as synthetic tests (not marked clinical).
from conftest import REPORT_DIR  # noqa: E402
import clinical_case  # noqa: E402
SYNTHETIC_ROOT = REPORT_DIR / "synthetic_clinical_case"
SYNTHETIC_CASE = (str(SYNTHETIC_ROOT), clinical_case.CASE_ID)

CASES = [pytest.param(ref, id=ref[1], marks=pytest.mark.clinical) for ref in CLINICAL_CASES] + \
        [pytest.param(SYNTHETIC_CASE, id=clinical_case.CASE_ID)]


@pytest.fixture(scope="module", autouse=True)
def synthetic_clinical_case():
    """Write the synthetic case before any test of this module runs (they may read its files directly)."""
    pytest.importorskip("nrrd")
    pytest.importorskip("vtk")
    import shutil
    shutil.rmtree(SYNTHETIC_ROOT, ignore_errors=True)
    return clinical_case.write_synthetic_case(SYNTHETIC_ROOT)


class _CaseMetrics:
    """Routes the metrics of the clinical cases to the clinical baseline (conftest.CaseMetricRecorder, kept with the
    data) and those of the synthetic case to the synthetic baseline (tests/metrics/baseline.json, as
    clinical_synthetic/<metric>)."""

    def __init__(self, clinical, synthetic):
        self.clinical, self.synthetic = clinical, synthetic

    def record(self, name, value, direction="lower", tol=None, atol=0.0):
        prefix, case_id, metric = name.split("/", 2)
        if case_id == clinical_case.CASE_ID:
            return self.synthetic.record(f"clinical_synthetic/{metric}", value, direction=direction, tol=tol, atol=atol)
        return self.clinical.record(name, value, direction=direction, tol=tol, atol=atol)


@pytest.fixture(scope="module")
def metrics(clinical_metrics, metrics):
    """Clinical metrics are compared with the baseline in <clinical data folder>/tests/slicerheart_4dflow_baseline.json
    (kept with the data, one section per case, see conftest.CaseMetricRecorder); the metrics of the synthetic case with
    tests/metrics/baseline.json."""
    for _, case_id in CLINICAL_CASES:
        clinical_metrics.register_case(case_id)
    return _CaseMetrics(clinical_metrics, metrics)


def case_metric(case, name):
    return f"clinical/{case['case_id']}/{name}"


def signal_mask(case):
    """Voxels with signal: time-averaged magnitude above the Otsu threshold (body)."""
    from slicerheart_4dflow import otsu_threshold
    mean_magnitude = case["magnitude_t"].astype(float).mean(axis=0)
    return mean_magnitude > otsu_threshold(mean_magnitude)


def slicer_wraps(case):
    """Wraps applied by SlicerHeart, per encoded velocity component."""
    return case["wrap_count_t"].astype(np.int8)


def unwrapped_raw_encoded(case):
    """Raw velocity (encoded components) with the wraps found by SlicerHeart applied (no background correction)."""
    return case["velocity_raw_encoded_t"] + 2.0 * case["venc_encoded_mm_s"] * slicer_wraps(case)


def unwrapped_raw(case):
    """Raw velocity (world components) with the wraps found by SlicerHeart applied (no background correction)."""
    return unwrapped_raw_encoded(case) @ case["encoding_to_world"].T


def planes_reviewed(case):
    """Plane sets whose physics checks (mass conservation) are enforced: placed or reviewed by a person.
    Automatically placed planes in small pediatric vessels (a few voxels across) are only recorded."""
    planes = case["annotations"].get("planes", {})
    return planes.get("method") != "automatic" or bool(planes.get("reviewed"))


def lumen_reviewed(case):
    """The lumen was segmented or reviewed by a person. An automatic lumen (PC-MRA threshold) can lie inside the vessel
    wall, which checks of the plane flow against the lumen boundary only record."""
    lumen = case["annotations"].get("lumen", {})
    return lumen.get("method") != "automatic" or bool(lumen.get("reviewed"))


@pytest.mark.clinical
@pytest.mark.parametrize("vendor", VENDORS)
def test_vendor_case_present(vendor):
    """The data set has cases of each vendor (skipped for a vendor without cases, see tests/clinical/<vendor>/README.md)."""
    if not CLINICAL_CASES:
        pytest.skip(f"no clinical data set in {CLINICAL_DATA_DIR} (set SLICERHEART_4DFLOW_CLINICAL_DATA)")
    import json as _json
    from slicerheart_4dflow.slicer_io import INPUT_FILES, case_directories
    for ref in CLINICAL_CASES:
        prepare(ref, f"inputs/{ref[1]}/acquisition/")
    vendors = {_json.loads((case_directories(*ref)[0] / INPUT_FILES["acquisition"]).read_text())["vendor"] for ref in CLINICAL_CASES}
    if vendor not in vendors:
        pytest.skip(f"no {vendor} clinical case; see tests/clinical/{vendor}/README.md")


@pytest.mark.parametrize("case_ref", CASES)
def test_manifest_and_arrays(case_ref):
    case = load(case_ref)
    man, acquisition = case["manifest"], case["acquisition"]
    for f in MANIFEST_FIELDS:
        assert f in man, f
    for f in ACQUISITION_FIELDS:
        assert f in acquisition, f
    assert acquisition["vendor"] in VENDORS + ("synthetic",), acquisition["vendor"]
    assert case["has_outputs"], "the case has no outputs: regenerate them"
    assert man["license"].startswith("CC BY") or man["license"] == "internal"
    v, mag, vtw = case["velocity_t"], case["magnitude_t"], case["voxel_to_world"]
    assert v.ndim == 5 and v.shape[-1] == 3 and mag.shape == v.shape[:4] and vtw.shape == (4, 4)
    assert np.all(np.linalg.norm(vtw[:3, :3], axis=0) > 0)
    for key in ("velocity_raw_t", "wrap_count_t"):
        if key in case:
            assert case[key].shape == v.shape, key
    if "lumen" in case:
        assert case["lumen"].shape == v.shape[1:4] and case["lumen"].any()
    for flows in [case.get("flows", {"planes": {}})] + list(case.get("references", {}).values()):
        assert set(case.get("planes", {})) >= set(flows["planes"])
        for plane in flows["planes"].values():
            if "flow_rate_ml_s" in plane:
                assert len(plane["flow_rate_ml_s"]) == v.shape[0]


# NRRD header fields that 3D Slicer writes (geometry, sequence and segmentation metadata)
ALLOWED_HEADER_FIELDS = re.compile(
    r"(type|dimension|space|sizes|space directions|kinds|endian|encoding|space origin|measurement frame|labels|units|"
    r"DataNodeClassName|axis \d+ index (type|values)|Segment\d+_\w+|Segmentation_\w+|space units|thicknesses|spacings|"
    r"axis mins|axis maxs|centerings|content|data file|sample units|keyvaluepairs)")
FORBIDDEN_DESCRIPTION_WORDS = re.compile(r"patient.?name|birth|study.?date|acquisition.?date|institution|operator|uid|accession", re.I)


@pytest.mark.parametrize("case_ref", CASES)
def test_files_carry_no_identifiers(case_ref):
    nrrd = pytest.importorskip("nrrd")
    from slicerheart_4dflow.slicer_io import DATASET_MANIFEST, case_directories
    prepare(case_ref)
    folders = case_directories(*case_ref)
    nrrd_files = [path for folder in folders for path in folder.rglob("*.nrrd")]
    for path in nrrd_files:
        header = nrrd.read_header(str(path))
        unexpected = [key for key in header if not ALLOWED_HEADER_FIELDS.fullmatch(key)]
        assert not unexpected, f"{path.name}: unexpected header fields {unexpected}"
    json_files = [Path(case_ref[0]) / DATASET_MANIFEST] + [path for folder in folders for path in folder.rglob("*.json")
                                                           if not path.name.endswith(".mrk.json")]
    for path in json_files:
        text = path.read_text() if path.exists() else ""
        assert not FORBIDDEN_DESCRIPTION_WORDS.search(text), f"{path.name}: {FORBIDDEN_DESCRIPTION_WORDS.search(text).group(0)}"
    for path in (path for folder in folders for path in folder.rglob("*.mrk.json")):
        doc = json.loads(path.read_text())
        for markup in doc["markups"]:
            assert not markup.get("description"), f"{path.name} has a description"


@pytest.mark.parametrize("case_ref", CASES)
def test_raw_velocity_and_slicer_corrections_consistent(case_ref, metrics):
    """velocity_raw is aliased within +-venc in each encoded component; the corrected velocity is the raw velocity
    plus the wraps (per encoded component) minus a smooth background, i.e. the files describe one pipeline."""
    case = load(case_ref)
    venc = case["venc_mm_s"]
    raw_encoded = case["velocity_raw_encoded_t"]
    assert np.all(np.abs(raw_encoded) <= case["venc_encoded_mm_s"] * (1 + 1e-3)), \
        "raw velocity exceeds venc: not the acquired (aliased) velocity"
    corrected = case["velocity_t"]
    processed = np.any(corrected != 0, axis=-1)
    background = (unwrapped_raw(case) - corrected)[processed]
    p99 = np.percentile(np.abs(background), 99, axis=0)
    metrics.record(case_metric(case, "slicer_background_p99_mm_s"), p99.max(), direction="exact", tol=0.05, atol=1.0)
    assert np.all(p99 < 0.2 * venc), f"corrected - unwrapped raw is not a small background: p99 {p99} mm/s"


@pytest.mark.parametrize("case_ref", CASES)
def test_encoding_directions(case_ref, metrics):
    """The velocity component mapping of the SlicerHeart import (axisForDirectionCodes, sign flags)
    is the one with the lowest divergence in the lumen."""
    from slicerheart_4dflow import encoding_check
    case = load(case_ref)
    result = encoding_check(case["velocity_t"], case["lumen"], case["voxel_to_world"])
    metrics.record(case_metric(case, "encoding/declared_to_best_ratio"), result["ratio"], tol=0.05, atol=0.02)
    metrics.record(case_metric(case, "encoding/discrimination"), result["discrimination"], direction="record")
    # inconclusive (too little flow, or flow along few directions) is allowed, but recorded
    metrics.record(case_metric(case, "encoding/conclusive"), float(result["conclusive"]), direction="record")
    assert result["consistent"], result["note"]


@functools.lru_cache(maxsize=1)
def common_static_mask(case_ref):
    """Static tissue for comparing background methods: the static candidates that the robust war+msac fit keeps
    as inliers in every component. The candidates include wrapped and moving voxels, which the robust methods
    reject by design, so a mean over all candidates would favour plain least squares (zero mean by construction)."""
    from slicerheart_4dflow import correct_background_phase
    case = load(case_ref)
    _, info = correct_background_phase(case["velocity_raw_encoded_t"], case["magnitude_t"].astype(float), method="war+msac",
                                       lumen_mask=case["lumen"])
    return info["candidates"] & info["inliers"].all(axis=-1)


@pytest.mark.parametrize("method", ["ols", "msac", "wrls", "war+msac"])
@pytest.mark.parametrize("case_ref", CASES)
def test_background_correction(case_ref, method, metrics):
    """Each background method, applied to the encoded velocity components as acquired, leaves less than 0.6 cm/s residual
    offset in static tissue (Gatehouse 2010); the difference to the SlicerHeart background fit is recorded."""
    from slicerheart_4dflow import correct_background_phase, residual_static_velocity
    case = load(case_ref)
    corrected, info = correct_background_phase(case["velocity_raw_encoded_t"], case["magnitude_t"].astype(float), method=method,
                                               lumen_mask=case["lumen"])
    static = common_static_mask(case_ref)
    assert static.sum() > 500, f"only {static.sum()} static tissue voxels"
    residual = residual_static_velocity(corrected, static)
    worst = float(np.max(np.abs(residual["mean"])))
    metrics.record(case_metric(case, f"background/{method}/residual_static_mm_s"), worst, atol=0.5)
    assert worst < 6.0, f"residual static velocity {residual['mean']} mm/s"

    if case["processing"].get("processing", {}).get("backgroundSubtraction") in ("LinearFit", "QuadraticFit"):
        processed = np.any(case["velocity_t"] != 0, axis=-1)
        slicer_background = (unwrapped_raw_encoded(case) - case["velocity_t"] @ case["encoding_to_world"]).mean(axis=0)
        body = signal_mask(case) & processed.all(axis=0) & ~case["lumen"]
        difference = np.sqrt(np.mean((info["offset"][body] - slicer_background[body]) ** 2, axis=0))
        metrics.record(case_metric(case, f"background/{method}/rms_difference_to_slicer_mm_s"), difference.max(), direction="record")


# Physiological upper limit of the blood speed (mm/s), used to recognize failed unwrapping
MAX_PLAUSIBLE_SPEED_MM_S = 7000.0


def lumen_peak_speed(velocity_t, lumen):
    return float(np.percentile(np.linalg.norm(velocity_t, axis=-1)[:, lumen], 99.9))


def slicer_unwrapping_plausible(case):
    return lumen_peak_speed(unwrapped_raw(case), case["lumen"]) < MAX_PLAUSIBLE_SPEED_MM_S


@pytest.mark.parametrize("case_ref", CASES)
def test_slicer_unwrapping_plausible(case_ref, metrics):
    """The wraps applied by SlicerHeart (wrap_count) give a physiological speed in the lumen. Temporal unwrapping
    accumulates false wraps where the velocity noise between frames exceeds venc, which this detects."""
    case = load(case_ref)
    peak = lumen_peak_speed(unwrapped_raw(case), case["lumen"])
    wrapped = float(np.mean(np.any(slicer_wraps(case)[:, case["lumen"]] != 0, axis=-1)))
    metrics.record(case_metric(case, "slicer_unwrap/lumen_peak_speed_mm_s"), peak, direction="record")
    metrics.record(case_metric(case, "slicer_unwrap/lumen_wrapped_fraction"), wrapped, direction="record")
    assert peak < MAX_PLAUSIBLE_SPEED_MM_S, (
        f"SlicerHeart unwrapping ({case['processing'].get('processing', {}).get('phaseUnwrapping')}) gives {peak:.0f} mm/s"
        f" peak speed in the lumen (venc {case['venc_encoded_mm_s'].max():.0f} mm/s), wraps in {100 * wrapped:.1f}% of lumen voxel-frames")


@pytest.mark.parametrize("method", ["laplacian3d", "laplacian4d", "laplacian_then_temporal", "temporal"])
@pytest.mark.parametrize("case_ref", CASES)
def test_unwrapping_against_slicer(case_ref, method, metrics):
    """Unwrapping of the encoded velocity components (as acquired, each aliased at its venc): the unwrapped peak speed is
    physiological, and the wraps found in the lumen agree with the SlicerHeart unwrapping where that is plausible
    (see test_slicer_unwrapping_plausible)."""
    from slicerheart_4dflow import unwrap_velocity
    case = load(case_ref)
    raw, venc, vtw = case["velocity_raw_encoded_t"], case["venc_encoded_mm_s"], case["voxel_to_world"]
    mask = signal_mask(case)
    unwrapped = unwrap_velocity(raw, venc, vtw, method=method, mask=mask)
    wraps = np.rint((unwrapped - raw) / (2 * venc)).astype(np.int8)
    reference = slicer_wraps(case)
    lumen = case["lumen"]
    in_lumen = (wraps[:, lumen] == reference[:, lumen])
    agreement = float(in_lumen.mean())
    wrapped = reference[:, lumen] != 0
    detected = float((wraps[:, lumen][wrapped] == reference[:, lumen][wrapped]).mean()) if wrapped.any() else 1.0
    metrics.record(case_metric(case, f"unwrap/{method}/lumen_agreement"), agreement, direction="higher", tol=0.0, atol=0.002)
    metrics.record(case_metric(case, f"unwrap/{method}/slicer_wraps_found"), detected, direction="higher", tol=0.0, atol=0.01)
    peak = lumen_peak_speed(unwrapped, lumen)
    metrics.record(case_metric(case, f"unwrap/{method}/lumen_peak_speed_mm_s"), peak, direction="exact", tol=0.05)
    assert 200 < peak < MAX_PLAUSIBLE_SPEED_MM_S, f"peak speed {peak:.0f} mm/s"
    if slicer_unwrapping_plausible(case):
        assert agreement > 0.99, f"wrap counts agree in {100 * agreement:.2f}% of lumen voxels"


@pytest.mark.parametrize("case_ref", CASES)
def test_quality_report(case_ref, metrics):
    from slicerheart_4dflow import quality_report
    case = load(case_ref)
    # Velocities as encoded components: quality_report checks the components against the venc of each encoding direction
    encoding_to_world = case["encoding_to_world"]
    report = quality_report(case["velocity_raw_encoded_t"], case["velocity_t"] @ encoding_to_world, case["magnitude_t"].astype(float),
                            case["venc_encoded_mm_s"], case["voxel_to_world"], lumen_mask=case["lumen"],
                            wrap_count=slicer_wraps(case), rr_ms=case["rr_ms"],
                            study_id=case["manifest"]["case_id"])
    noise = np.asarray(report["velocity_noise_mm_s"], float)
    metrics.record(case_metric(case, "report/velocity_noise_mm_s"), noise.max(), direction="exact", tol=0.05)
    assert np.all(noise > 0) and np.all(noise < 0.2 * case["venc_encoded_mm_s"]),         f"velocity noise estimate {noise} mm/s is out of range (venc {case['venc_encoded_mm_s']} mm/s)"
    metrics.record(case_metric(case, "report/residual_offset_mm_s"),
                   max(abs(x) for x in report["background_offset"]["residual_mean_processed_mm_s"]), direction="record")
    if np.any(slicer_wraps(case) != 0):
        assert not report["venc"]["venc_adequate"], "wraps were corrected, so the venc was not adequate"


def plain_plane_flows(case, plane_names, sampling_mm=1.0):
    """Flow rate (ml/s) through each plane by plain trilinear interpolation of the velocity at the disc samples inside
    the lumen segmentation (looked up on the grid of the segmentation, which may be finer than the image): the sampling
    of the SlicerHeart plane probe, which samples the flow field and tests the samples against the segment surface."""
    from scipy import ndimage
    from slicerheart_4dflow.geometry import world_to_array_coordinates
    from slicerheart_4dflow.qa import plane_samples
    velocity, vtw = case["velocity_t"], case["voxel_to_world"]
    labelmap, labelmap_vtw = case["lumen_labelmap"], case["lumen_labelmap_voxel_to_world"]
    flows = {}
    for name in plane_names:
        plane = case["planes"][name]
        points, coordinates, normal, area, _ = plane_samples(velocity.shape[1:4], vtw, plane["centre"], plane["normal"], plane["radius"], sampling_mm)
        nearest = np.round(world_to_array_coordinates(labelmap_vtw, points)).astype(int)
        valid = np.all((nearest >= 0) & (nearest < np.array(labelmap.shape)[:, None]), axis=0)
        inside = np.zeros(len(points), bool)
        inside[valid] = labelmap[nearest[0, valid], nearest[1, valid], nearest[2, valid]]
        through_plane = velocity @ normal
        flows[name] = np.array([ndimage.map_coordinates(through_plane[t], coordinates[:, inside], order=1).sum() * area
                                for t in range(velocity.shape[0])]) / 1000.0
    return flows


def library_plane_flows(case, plane_names=None, sampling_mm=1.0):
    """Flow rate (ml/s) through each plane (default: the planes of outputs/flows.json), by the plane_flow of this package,
    with the fraction of each voxel in the lumen (partial voxels of a lumen segmented on a finer grid)."""
    from slicerheart_4dflow import plane_flow
    lumen = case.get("lumen_fraction", case["lumen"])
    return {name: plane_flow(case["velocity_t"], case["voxel_to_world"], lumen_mask=lumen, sampling_mm=sampling_mm,
                             **case["planes"][name])["flow"] / 1000.0
            for name in (plane_names if plane_names is not None else case["flows"]["planes"])}


@pytest.mark.parametrize("case_ref", CASES)
def test_plane_flow_consistent_with_slicerheart(case_ref, metrics):
    """Consistency: flow through each plane agrees with the SlicerHeart plane probe (outputs/flows.json) on the same
    velocity, lumen and planes.

    With the sampling of the plane probe (plain interpolation, samples tested against the lumen segmentation), the flows
    must agree closely: this checks plane geometry, lumen and units. plane_flow interpolates with mask normalization on
    the fraction of each voxel in the lumen; with a lumen that reaches the vessel wall (segmented on a grid finer than
    the image) it agrees with the plane probe within a few percent, and within max(0.5 mL, 10%) is required."""
    case = load(case_ref)
    if "flows" not in case or not case["flows"]["planes"]:
        pytest.skip("no plane flows")
    # Same sampling distance as the plane probe: in vessels a few voxels across, most samples are near the lumen
    # boundary, so different sample positions would make the flows differ by up to 20%
    sampling_mm = case["flows"].get("sampling_distance_mm", 1.0)
    plane_names = list(case["flows"]["planes"])
    plain = plain_plane_flows(case, plane_names, sampling_mm)
    normalized = library_plane_flows(case, plane_names, sampling_mm)
    rr_ms = case["rr_ms"]
    failures = []
    for name, reference in case["flows"]["planes"].items():
        net_ref = reference["net_volume_mL"]
        net_plain = float(np.mean(plain[name]) * rr_ms / 1000.0)
        net = float(np.mean(normalized[name]) * rr_ms / 1000.0)
        metrics.record(case_metric(case, f"{name}/net_volume_mL"), net, direction="exact", atol=0.2)
        metrics.record(case_metric(case, f"{name}/plain_net_volume_difference_mL"), abs(net_plain - net_ref), atol=0.2)
        metrics.record(case_metric(case, f"{name}/normalized_to_reference_ratio"), net / net_ref if net_ref else 1.0, direction="record")
        if abs(net_plain - net_ref) > max(0.5, 0.08 * abs(net_ref)):
            failures.append(f"{name}: plain interpolation {net_plain:.2f} mL vs SlicerHeart {net_ref:.2f} mL")
        if abs(net - net_ref) > max(0.5, 0.10 * abs(net_ref)):
            failures.append(f"{name}: plane_flow {net:.2f} mL vs SlicerHeart {net_ref:.2f} mL (ratio {net / net_ref:.2f})")
        if "flow_rate_ml_s" in reference:
            curve_ref = np.asarray(reference["flow_rate_ml_s"])
            scale = max(np.abs(curve_ref).max(), 1e-6)
            curve_error = float(np.abs(plain[name] - curve_ref).max() / scale)
            metrics.record(case_metric(case, f"{name}/plain_flow_curve_max_error"), curve_error, atol=0.01)
            if curve_error > 0.15:
                failures.append(f"{name}: flow curve differs from SlicerHeart by {100 * curve_error:.1f}% of its peak")
    assert not failures, "; ".join(failures)


# Agreement with the references of a case (references/<source>/flows.json), by source: (absolute limit mL, relative
# limit). "pc2d": 2D phase contrast at the same plane; "cine": ventricular stroke volume (minus the regurgitant volume
# of the inflow valve) as the forward volume of the outflow plane; "cvi42": values recorded when the case was analyzed
# in cvi42 (the limits of agreement of its published comparisons, there is no live access to the software). Other
# sources (independent software): DEFAULT_REFERENCE_LIMITS. Initial limits, to be calibrated on the first cases.
REFERENCE_LIMITS = {"pc2d": (0.5, 0.15), "cine": (0.5, 0.15), "cvi42": (7.0, 0.0)}
DEFAULT_REFERENCE_LIMITS = (0.5, 0.15)


@pytest.mark.parametrize("case_ref", CASES)
def test_plane_flow_against_references(case_ref, metrics):
    """Validation: net and forward volume through each plane agree with the references of the case
    (references/<source>/flows.json: independent software, 2D phase contrast, cine stroke volumes, recorded cvi42
    values), within the limits of REFERENCE_LIMITS, on the planes of the case. The ratio of the peak mean velocity to
    the peak velocity of the reference is recorded. Skipped if the case has no references."""
    from slicerheart_4dflow import plane_flow
    case = load(case_ref)
    references = case.get("references", {})
    if not references:
        pytest.skip("no references")
    failures = []
    for source, reference in references.items():
        absolute_limit, relative_limit = REFERENCE_LIMITS.get(source, DEFAULT_REFERENCE_LIMITS)
        plane_names = [name for name in reference["planes"] if name in case.get("planes", {})]
        flows = library_plane_flows(case, plane_names)
        for name in plane_names:
            values = reference["planes"][name]
            volumes = {"net_volume_mL": float(np.mean(flows[name]) * case["rr_ms"] / 1000.0),
                       "forward_volume_mL": float(np.mean(np.clip(flows[name], 0, None)) * case["rr_ms"] / 1000.0)}
            for key, value in volumes.items():
                if key not in values:
                    continue
                difference = value - values[key]
                metrics.record(case_metric(case, f"{source}/{name}/{key[:-3]}_difference_mL"), abs(difference),
                               direction="record")
                if abs(difference) > max(absolute_limit, relative_limit * abs(values[key])):
                    failures.append(f"{source} {name}: {key} {value:.2f} mL vs {values[key]:.2f} mL")
            if values.get("peak_velocity_mm_s"):
                peak = np.abs(plane_flow(case["velocity_t"], case["voxel_to_world"], lumen_mask=case["lumen"],
                                         **case["planes"][name])["mean_velocity"]).max()
                metrics.record(case_metric(case, f"{source}/{name}/peak_mean_velocity_over_reference_peak"),
                               peak / values["peak_velocity_mm_s"], direction="record")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("case_ref", CASES)
def test_plane_flow_stable_under_lumen_dilation(case_ref, metrics):
    """Net flow through each plane changes by less than 5% (or 0.5 mL) when the lumen is dilated by one voxel:
    plane_flow interpolates with mask normalization, so the velocity at the wall must not depend much on exactly where
    the segmentation puts the wall. Enforced for reviewed lumens; an automatic lumen that lies inside the wall gains
    flow when dilated, which is only recorded."""
    from scipy import ndimage
    from slicerheart_4dflow import plane_flow
    case = load(case_ref)
    if not case.get("planes"):
        pytest.skip("no planes")
    dilated = ndimage.binary_dilation(case["lumen"])
    rr_ms = case["rr_ms"]
    failures = []
    for name, plane in case["planes"].items():
        q0 = plane_flow(case["velocity_t"], case["voxel_to_world"], lumen_mask=case["lumen"], **plane)["flow"]
        q1 = plane_flow(case["velocity_t"], case["voxel_to_world"], lumen_mask=dilated, **plane)["flow"]
        net0, net1 = (float(np.mean(q) * rr_ms / 1e6) for q in (q0, q1))
        change = abs(net1 - net0) / max(abs(net0), 1e-6)
        metrics.record(case_metric(case, f"{name}/net_volume_change_lumen_dilation"), change,
                       direction="lower" if lumen_reviewed(case) else "record", atol=0.01)
        if abs(net1 - net0) > max(0.5, 0.05 * abs(net0)):
            failures.append(f"{name}: {net0:.2f} mL, {net1:.2f} mL with the lumen dilated by one voxel")
    if failures and lumen_reviewed(case):
        pytest.fail("; ".join(failures))


@pytest.mark.parametrize("case_ref", CASES)
def test_mass_conservation(case_ref, metrics):
    """Inflow equals outflow for the plane groups of the reference (plane pairs on one vessel segment).
    Enforced for reviewed planes; for automatically placed planes the error is only recorded."""
    from slicerheart_4dflow import mass_conservation
    case = load(case_ref)
    checks = [tuple(c) for c in case.get("flows", {}).get("conservation_checks", [])]
    if not checks:
        pytest.skip("no conservation checks")
    results = mass_conservation(library_plane_flows(case), checks, rr_ms=case["rr_ms"])
    failed = []
    for result in results:
        metrics.record(case_metric(case, f"{result['check']}/abs_pct_error"), abs(result["percent_error"]),
                       direction="record" if not planes_reviewed(case) else "lower", atol=1.0)
        if abs(result["percent_error"]) > 10.0:
            failed.append(f"{result['check']}: {result['percent_error']:.1f}%")
    if failed and planes_reviewed(case):
        pytest.fail("mass conservation: " + ", ".join(failed))
    for ratio in case["flows"].get("ratio_checks", []):
        label, numerator, denominator, expected = ratio
        flows = library_plane_flows(case)
        value = np.mean(flows[numerator]) / np.mean(flows[denominator])
        metrics.record(case_metric(case, f"{label}/ratio"), value, direction="exact", atol=0.02)
        assert abs(value / expected - 1) < 0.10


@pytest.mark.parametrize("method", ["fdm", "relaxed", "leray"])
@pytest.mark.parametrize("case_ref", CASES)
def test_divergence_filter(case_ref, method, metrics):
    """Divergence in the lumen decreases in every tested frame, and the flow through the planes is preserved: the change
    of the net flow is recorded, and for a normal study (diagnosis_category "normal", see case_property) it must be less than
    5% (or 0.5 ml/s)."""
    from slicerheart_4dflow import divergence_free_sequence, plane_flow
    case = load(case_ref)
    frames = sorted({0, case["velocity_t"].shape[0] // 4, case["velocity_t"].shape[0] // 2})
    velocity = case["velocity_t"][frames].astype(float)
    filtered, infos = divergence_free_sequence(velocity, case["lumen"], case["voxel_to_world"], method=method)
    reduction = [info["div_rms_after"] / info["div_rms_before"] for info in infos]
    metrics.record(case_metric(case, f"divergence/{method}/rms_ratio"), max(reduction), atol=0.01)
    assert max(reduction) < 1.0
    failures = []
    for name, plane in case.get("planes", {}).items():
        before = plane_flow(velocity, case["voxel_to_world"], lumen_mask=case["lumen"], **plane)["flow"]
        after = plane_flow(filtered, case["voxel_to_world"], lumen_mask=case["lumen"], **plane)["flow"]
        change = float(np.abs(after - before).max() / max(np.abs(before).max(), 1e-6))
        metrics.record(case_metric(case, f"divergence/{method}/{name}/flow_change"), change, direction="record")
        net_before, net_after = float(np.mean(before)), float(np.mean(after))
        net_change = abs(net_after - net_before) / max(abs(net_before), 1e-6)
        metrics.record(case_metric(case, f"divergence/{method}/{name}/net_flow_change"), net_change, direction="record")
        if (case_property(case, "diagnosis_category") == "normal"
                and abs(net_after - net_before) > max(500.0, 0.05 * abs(net_before))):      # mm3/s: 0.5 ml/s
            failures.append(f"{name}: net flow changed by {100 * net_change:.1f}%")
    assert not failures, f"{method} changed the net flow of a normal study: " + "; ".join(failures)


@pytest.mark.parametrize("case_ref", CASES)
def test_world_frame_invariance(case_ref):
    """Reading the case in RAS instead of LPS, or moving it to an oblique world frame, changes no result."""
    from slicerheart_4dflow import plane_flow, encoding_check
    lps = load(case_ref)
    ras = load.__wrapped__(case_ref, world="RAS")        # not cached: freed after this test
    np.testing.assert_array_equal(lps["lumen"], ras["lumen"])
    for name in list(lps.get("planes", {}))[:2]:
        q_lps = plane_flow(lps["velocity_t"], lps["voxel_to_world"], lumen_mask=lps["lumen"], **lps["planes"][name])["flow"]
        q_ras = plane_flow(ras["velocity_t"], ras["voxel_to_world"], lumen_mask=ras["lumen"], **ras["planes"][name])["flow"]
        np.testing.assert_allclose(q_ras, q_lps, rtol=1e-4, atol=1e-3 * np.abs(q_lps).max())
    # an oblique rigid change of the world frame (rotation, reflection, translation): rounding must not change the
    # plane sampling grid or which samples are in the lumen
    from conftest import world_transform
    T = world_transform()
    R = T[:3, :3]
    for name in list(lps.get("planes", {}))[:2]:
        plane = lps["planes"][name]
        q = plane_flow(lps["velocity_t"], lps["voxel_to_world"], lumen_mask=lps["lumen"], **plane)["flow"]
        moved = dict(plane, centre=R @ np.asarray(plane["centre"]) + T[:3, 3], normal=R @ np.asarray(plane["normal"]))
        q_moved = plane_flow(lps["velocity_t"] @ R.T, T @ lps["voxel_to_world"], lumen_mask=lps["lumen"], **moved)["flow"]
        np.testing.assert_allclose(q_moved, q, rtol=1e-6, atol=1e-6 * np.abs(q).max())
    r_lps = encoding_check(lps["velocity_t"], lps["lumen"], lps["voxel_to_world"])
    r_ras = encoding_check(ras["velocity_t"], ras["lumen"], ras["voxel_to_world"])
    assert r_lps["consistent"] == r_ras["consistent"]
    assert abs(r_lps["ratio"] - r_ras["ratio"]) < 1e-6


@pytest.mark.parametrize("case_ref", CASES)
def test_transfer_audit(case_ref, metrics):
    """Transfer audit (transfer_audit.py): quantities that need no ground truth, on the case and on a synthetic case
    matched to it. A report, not a pass/fail test: a flagged quantity means the synthetic tests miss an effect of real
    data for that step. The report of each case and a summary of all cases are written to
    <test report folder>/transfer_audit/, the values are recorded (record only). Fails only if the synthetic case could
    not be matched to the case, which makes its comparison meaningless."""
    from conftest import REPORT_DIR
    from transfer_audit import transfer_audit, to_json, to_markdown, write_summary
    case = load(case_ref)
    if not case.get("planes"):
        pytest.skip("no planes")
    audit = transfer_audit(case)
    folder = REPORT_DIR / "transfer_audit"
    folder.mkdir(parents=True, exist_ok=True)
    case_id = case["case_id"]
    (folder / f"{case_id}.md").write_text(to_markdown(audit), encoding="utf-8")
    (folder / f"{case_id}.json").write_text(to_json(audit), encoding="utf-8")
    write_summary(folder)
    for name, row in audit["rows"].items():
        if np.isfinite(row["ratio"]):
            metrics.record(case_metric(case, f"transfer_audit/{name}/case"), row["case"], direction="record")
            metrics.record(case_metric(case, f"transfer_audit/{name}/ratio_to_synthetic"), row["ratio"], direction="record")
    unmatched = {name: m for name, m in audit["matching"].items() if not m["matched"]}
    assert not unmatched, f"the synthetic case is not matched to the case: {unmatched}"


@pytest.mark.parametrize("case_ref", CASES)
def test_semisynthetic(case_ref, metrics):
    """Semi-synthetic reference (semisynthetic.py): a clean, divergence-free field built from the case on its own
    lumen, re-acquired at the case's venc, noise, magnitude image and background offset, is recovered by the default
    pipeline (preprocess): where it keeps the velocity and unwraps correctly, the error is within 1.5 times the noise
    that the re-acquisition put into each voxel (RMS of the error over the noise of the voxel: low-signal voxels are
    noisier), and wrong wraps are in less than 1% of the lumen voxel-frames above the Otsu threshold of the magnitude.
    A known answer on real anatomy under real acquisition conditions.

    The wrong wraps are counted on a fixed set of voxels, those that a noise mask by the Otsu threshold of the
    magnitude keeps, so that a noise mask that keeps more of the lumen is not penalized for the darker voxels it adds;
    the wrong fraction in those (about 3% with body_mask) is recorded, as a target for unwrapping."""
    from slicerheart_4dflow import acquisition_noise, noise_mask, preprocess, reacquire, reference_from_measurement
    case = load(case_ref)
    raw, venc, vtw, lumen = case["velocity_raw_encoded_t"], case["venc_encoded_mm_s"], case["voxel_to_world"], case["lumen"]
    E = case["encoding_to_world"]
    magnitude = case["magnitude_t"].astype(float)
    built = reference_from_measurement(raw, magnitude, venc, vtw, lumen, encoding_to_world=E)
    reference, noise = built["reference"], built["velocity_noise_mm_s"]
    re = reacquire(reference, venc, magnitude, lumen, noise, offset=built["offset"], encoding_to_world=E,
                   rng=np.random.default_rng(1))
    result = preprocess(re["velocity_t"], re["magnitude_t"], venc, vtw, lumen_mask=lumen, encoding_to_world=E, verbose=False)
    kept = built["mask"] & ~result["noise"]
    # error in encoded components (world = encoded @ E.T), over the noise of each voxel, frame and component
    difference = ((result["velocity"] - reference) @ E)[:, kept]
    sigma = acquisition_noise(magnitude, lumen, venc, re["snr"])[:, kept]
    # voxel-frames that the unwrapping got wrong (off by more than venc) are counted, the error of the others is the
    # accuracy of the pipeline
    wrong = (np.abs(difference) > venc).any(-1)
    error_over_noise = float(np.sqrt(np.mean((difference[~wrong] / sigma[~wrong]) ** 2)))
    bright = ~noise_mask(re["magnitude_t"], re["velocity_t"], magnitude_threshold="otsu", venc=venc)[kept]
    wrong_fraction = float(wrong[:, bright].mean())
    wrong_fraction_dark = float(wrong[:, ~bright].mean()) if (~bright).any() else 0.0
    wrapped_fraction = float((re["wrap_count"][:, built["mask"]] != 0).any(-1).mean())
    speed = np.linalg.norm(reference, axis=-1)[:, built["mask"]]
    metrics.record(case_metric(case, "semisynthetic/error_over_voxel_noise"), error_over_noise, atol=0.05)
    metrics.record(case_metric(case, "semisynthetic/wrong_voxel_frame_fraction"), wrong_fraction, atol=0.0005)
    metrics.record(case_metric(case, "semisynthetic/wrong_voxel_frame_fraction_below_otsu"), wrong_fraction_dark,
                   direction="record")
    metrics.record(case_metric(case, "semisynthetic/kept_lumen_fraction_below_otsu"), float((~bright).mean()),
                   direction="record")
    metrics.record(case_metric(case, "semisynthetic/wrapped_voxel_frame_fraction"), wrapped_fraction, direction="record")
    metrics.record(case_metric(case, "semisynthetic/lumen_fraction_set_to_zero"), 1.0 - kept.sum() / lumen.sum(), direction="record")
    metrics.record(case_metric(case, "semisynthetic/reference_peak_speed_mm_s"), float(np.percentile(speed, 99.9)), direction="record")
    assert error_over_noise < 1.5, f"error {error_over_noise:.2f} times the noise of the voxels in the lumen"
    assert wrong_fraction < 0.01, (f"{100 * wrong_fraction:.2f}% of the lumen voxel-frames above the Otsu threshold off"
                                   f" by more than venc ({100 * wrapped_fraction:.2f}% aliased)")


@pytest.mark.parametrize("case_ref", CASES)
def test_pathlines(case_ref, metrics):
    """Particles emitted at peak flow from the first plane of each pair stay in the lumen and most of them
    cross the second plane of the pair, 4 mm downstream."""
    from slicerheart_4dflow import seeds_from_plane, advect_pathlines
    from slicerheart_4dflow.geometry import world_to_array_coordinates
    from scipy import ndimage
    case = load(case_ref)
    flows = case.get("flows", {})
    pairs = [c for c in flows.get("conservation_checks", []) if len(c[1]) == 1 and len(c[2]) == 1]
    if not pairs:
        pytest.skip("no plane pairs")
    velocity, vtw, lumen, rr_ms = case["velocity_t"], case["voxel_to_world"], case["lumen"], case["rr_ms"]
    for label, (plane_a,), (plane_b,) in pairs:
        a, b = case["planes"][plane_a], case["planes"][plane_b]
        flow = np.asarray(flows["planes"][plane_a]["flow_rate_ml_s"])
        peak = int(np.argmax(flow))
        seeds = seeds_from_plane(lumen, vtw, a["centre"], a["normal"], 0.7 * a["radius"], spacing_mm=1.0)
        if len(seeds) == 0:
            continue
        t0 = peak * rr_ms / velocity.shape[0]
        traces = advect_pathlines(velocity, vtw, rr_ms, seeds, t0_ms=t0, duration_ms=0.25 * rr_ms, mask=lumen)
        points = np.concatenate([trace[:, :3] for trace in traces])
        inside = ndimage.map_coordinates(lumen.astype(float), world_to_array_coordinates(vtw, points), order=1) >= 0.5
        crossed = np.mean([np.any((trace[:, :3] - b["centre"]) @ b["normal"] > 0) for trace in traces])
        metrics.record(case_metric(case, f"pathlines/{label}/inside_fraction"), inside.mean(), direction="higher", atol=0.02)
        metrics.record(case_metric(case, f"pathlines/{label}/crossed_fraction"), crossed, direction="record")
        assert inside.mean() > 0.95
        if flow[peak] > 0 and planes_reviewed(case):
            assert crossed > 0.5, f"{label}: {100 * crossed:.0f}% of particles reached {plane_b}"


@pytest.mark.parametrize("case_ref", CASES)
def test_hemodynamics_plausible(case_ref, metrics):
    """Kinetic energy, wall shear stress and relative pressure are in physiological ranges."""
    from slicerheart_4dflow import lumen_integrals, wall_shear_stress, wall_points, relative_pressure, pressure_drop
    case = load(case_ref)
    velocity, vtw, lumen, rr_ms = case["velocity_t"].astype(float), case["voxel_to_world"], case["lumen"], case["rr_ms"]
    integrals = lumen_integrals(velocity, lumen, vtw)
    assert np.all(integrals["kinetic_energy_mJ"] > 0)
    metrics.record(case_metric(case, "hemodynamics/peak_kinetic_energy_mJ"), integrals["kinetic_energy_mJ"].max(), direction="exact", tol=0.05)

    points, normals = wall_points(lumen, vtw, step=4)
    wss = wall_shear_stress(velocity, lumen, vtw, points, normals)
    median_tawss = float(np.median(wss["tawss"]))
    metrics.record(case_metric(case, "hemodynamics/median_tawss_pa"), median_tawss, direction="exact", tol=0.05)
    assert 0.05 < median_tawss < 10.0, f"median TAWSS {median_tawss:.3f} Pa"
    assert np.all((wss["osi"] >= -1e-9) & (wss["osi"] <= 0.5 + 1e-9))

    pressure, info = relative_pressure(velocity, lumen, vtw, rr_ms)
    assert np.all(np.isfinite(pressure))
    assert all(c["cg_status"] == 0 for c in info["cg"])
    pairs = [c for c in case.get("flows", {}).get("conservation_checks", []) if len(c[1]) == 1 and len(c[2]) == 1]
    for label, (plane_a,), (plane_b,) in pairs:
        from slicerheart_4dflow.geometry import voxel_grid_world_points
        grid = voxel_grid_world_points(lumen.shape, vtw).reshape(lumen.shape + (3,))
        half = 0.5 * np.linalg.norm(vtw[:3, :3], axis=0).max()

        def region(plane):
            offsets = grid - plane["centre"]
            distance = offsets @ plane["normal"]
            radial = np.linalg.norm(offsets - distance[..., None] * plane["normal"], axis=-1)
            return (np.abs(distance) <= half) & (radial <= plane["radius"]) & lumen

        ra, rb = region(case["planes"][plane_a]), region(case["planes"][plane_b])
        if not (ra.any() and rb.any()):
            continue
        drop_mmhg = pressure_drop(pressure, ra, rb) / 133.322
        metrics.record(case_metric(case, f"pressure/{label}/peak_abs_drop_mmHg"), np.abs(drop_mmhg).max(), direction="record")
        assert np.abs(drop_mmhg).max() < 30.0, f"{label}: pressure drop over 4 mm {np.abs(drop_mmhg).max():.1f} mmHg"
