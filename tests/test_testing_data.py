"""Download and cache of the clinical test data set (slicerheart_4dflow.testing_data), on a small data set packaged as a
local release."""

import json
import os
import time

import pytest

from slicerheart_4dflow import testing_data
from slicerheart_4dflow.testing_data import ClinicalTestingData


def write_dataset(root):
    """A small data set in format 4: two cases, inputs and outputs, the baseline, manifest and README."""
    files = {
        "manifest.json": json.dumps({"format_version": 4, "name": "test", "license": "CC BY 4.0", "doi": None}),
        "README.md": "# test data set\n",
        "tests/slicerheart_4dflow_baseline.json": json.dumps({"package": "slicerheart_4dflow", "cases": {}}),
    }
    for case in ("Case0001", "Case0002"):
        files[f"inputs/{case}/acquisition/acquisition.json"] = json.dumps({"vendor": "siemens", "case": case})
        files[f"inputs/{case}/acquisition/velocity_raw.seq.nrrd"] = case * 1000
        files[f"inputs/{case}/annotations/annotations.json"] = json.dumps({"lumen": {"method": "automatic"}})
        files[f"outputs/{case}/flows.json"] = json.dumps({"case": case, "flow": [1.0, 2.0]})
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return files


@pytest.fixture(autouse=True)
def no_published_revisions(monkeypatch):
    """Packaging reuses the assets of the published revisions: none here, so the tests do not download their manifests."""
    monkeypatch.setattr(testing_data, "_published_manifests", lambda revision: [])


@pytest.fixture
def release(tmp_path):
    """(manifest, release URL, data set files): the data set packaged as release assets in a local folder."""
    dataset = tmp_path / "dataset"
    files = write_dataset(dataset)
    manifest = testing_data.package_clinical_test_data(dataset, tmp_path / "release", "rev001")
    return manifest, (tmp_path / "release").as_uri() + "/", files


def test_package_manifest_lists_every_file_and_asset(release):
    """The manifest has the metadata of the data set, the revision, and every other file of the data set with the
    asset that contains it."""
    manifest_path, _, files = release
    assert manifest_path.name == "clinical_manifest_rev001.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {key: manifest[key] for key in ("format_version", "name", "license", "doi")}
    assert metadata == json.loads(files["manifest.json"])
    assert manifest["revision"] == "rev001"
    assert set(manifest["files"]) == set(files) - {"manifest.json"}
    assert set(manifest["assets"]) == {
        "clinical_inputs_Case0001_acquisition_rev001.zip", "clinical_inputs_Case0001_annotations_rev001.zip",
        "clinical_inputs_Case0002_acquisition_rev001.zip", "clinical_inputs_Case0002_annotations_rev001.zip",
        "clinical_outputs_Case0001_rev001.zip", "clinical_outputs_Case0002_rev001.zip", "clinical_tests_rev001.zip",
        "clinical_README_rev001.md"}
    assert manifest["files"]["inputs/Case0001/acquisition/acquisition.json"]["path_in_asset"] == "acquisition.json"
    assert manifest["files"]["README.md"]["path_in_asset"] is None


def test_package_keeps_the_metadata_of_a_packaged_data_set(release, tmp_path):
    """A data set whose manifest.json is already a package manifest is packaged again with its metadata only."""
    manifest_path, _, _ = release
    dataset = tmp_path / "dataset"
    (dataset / "manifest.json").write_bytes(manifest_path.read_bytes())
    again = testing_data.package_clinical_test_data(dataset, tmp_path / "again", "rev002")
    again = json.loads(again.read_text(encoding="utf-8"))
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert again["revision"] == "rev002" and again["name"] == first["name"]
    assert set(again["files"]) == set(first["files"])


def test_package_is_reproducible(tmp_path):
    """An unchanged folder gives the same zip (fixed entry times and order), so the asset hashes only change with the
    content."""
    dataset = tmp_path / "dataset"
    write_dataset(dataset)
    a = json.loads(testing_data.package_clinical_test_data(dataset, tmp_path / "a", "rev001").read_text("utf-8"))
    time.sleep(1.1)
    for path in dataset.rglob("*"):
        os.utime(path)                           # newer modification times, same content
    b = json.loads(testing_data.package_clinical_test_data(dataset, tmp_path / "b", "rev001").read_text("utf-8"))
    assert a["assets"] == b["assets"]


def test_package_reuses_the_assets_of_published_revisions(release, tmp_path, monkeypatch):
    """A new revision refers to the assets of the published revisions (which stay in the release) for the folders whose
    files did not change, writes new assets only for the changed folders, and is downloaded as a full copy from the
    assets of both."""
    manifest_path, url, files = release
    # rev001 is published
    monkeypatch.setattr(testing_data, "_published_manifests", lambda revision: [manifest_path])
    dataset = tmp_path / "dataset"
    (dataset / "inputs/Case0001/annotations/annotations.json").write_text('{"lumen": {"method": "manual"}}', encoding="utf-8")
    (dataset / "outputs/Case0001/flows.json").write_text('{"case": "Case0001", "flow": [3.0]}', encoding="utf-8")
    new = testing_data.package_clinical_test_data(dataset, tmp_path / "rev002", "rev002")
    manifest = json.loads(new.read_text(encoding="utf-8"))
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = {"clinical_inputs_Case0001_annotations_rev002.zip", "clinical_outputs_Case0001_rev002.zip"}
    assert {p.name for p in (tmp_path / "rev002").iterdir()} == changed | {new.name}
    acquisition = "clinical_inputs_Case0001_acquisition_rev001.zip"
    assert manifest["files"]["inputs/Case0001/acquisition/velocity_raw.seq.nrrd"]["asset"] == acquisition
    assert manifest["assets"][acquisition] == first["assets"][acquisition]
    assert manifest["files"]["README.md"]["asset"] == "clinical_README_rev001.md"
    assert set(manifest["assets"]) == (set(first["assets"]) - {n.replace("rev002", "rev001") for n in changed}) | changed
    # the release has the assets of both revisions
    for p in (tmp_path / "rev002").iterdir():
        (tmp_path / "release" / p.name).write_bytes(p.read_bytes())
    root = ClinicalTestingData(new, tmp_path / "cache", url).ensure_all()
    assert (root / "inputs/Case0001/annotations/annotations.json").read_text() == '{"lumen": {"method": "manual"}}'
    assert (root / "inputs/Case0002/acquisition/velocity_raw.seq.nrrd").read_text() == \
        files["inputs/Case0002/acquisition/velocity_raw.seq.nrrd"]


def test_download_on_first_use_per_case(release, tmp_path):
    manifest, url, files = release
    data = ClinicalTestingData(manifest, tmp_path / "cache", url)
    assert data.cases() == ["Case0001", "Case0002"]
    assert not data.root.exists()                # nothing is downloaded until a case is needed
    root = data.ensure_case("Case0001")
    assert (root / "manifest.json").read_bytes() == manifest.read_bytes()      # the manifest of the data set
    assert (root / "inputs/Case0001/acquisition/velocity_raw.seq.nrrd").read_text() == files["inputs/Case0001/acquisition/velocity_raw.seq.nrrd"]
    assert (root / "tests/slicerheart_4dflow_baseline.json").exists() and (root / "manifest.json").exists()
    assert not (root / "inputs/Case0002").exists()
    assert not list(root.glob("*.zip")) and not list(root.rglob("*.part"))     # the zips are deleted after extraction


def test_cache_is_checked_by_size_and_modification_time(release, tmp_path, monkeypatch):
    """A second session finds the cache up to date without reading the files or downloading; a changed file is found
    by its hash and only its asset is downloaded again."""
    manifest, url, files = release
    ClinicalTestingData(manifest, tmp_path / "cache", url).ensure_all()

    hashed, downloaded = [], []
    sha256, install = testing_data.sha256_file, ClinicalTestingData._install
    monkeypatch.setattr(testing_data, "sha256_file", lambda path: hashed.append(path) or sha256(path))
    monkeypatch.setattr(ClinicalTestingData, "_install",
                        lambda self, asset: downloaded.append(asset) or install(self, asset))

    data = ClinicalTestingData(manifest, tmp_path / "cache", url)
    data.ensure_all()
    assert hashed == [] and downloaded == []

    # same content, new modification time: hashed, found unchanged, recorded; not downloaded
    touched = data.root / "outputs/Case0002/flows.json"
    time.sleep(1.1)
    os.utime(touched)
    data = ClinicalTestingData(manifest, tmp_path / "cache", url)
    data.ensure_all()
    assert [p.name for p in hashed] == ["flows.json"] and downloaded == []

    # changed content: downloaded again (only its asset), restored
    hashed.clear()
    edited = data.root / "inputs/Case0001/annotations/annotations.json"
    edited.write_text("edited", encoding="utf-8")
    (data.root / "README.md").unlink()
    data = ClinicalTestingData(manifest, tmp_path / "cache", url)
    data.ensure_all()
    assert sorted(downloaded) == ["clinical_README_rev001.md", "clinical_inputs_Case0001_annotations_rev001.zip"]
    assert edited.read_text(encoding="utf-8") == files["inputs/Case0001/annotations/annotations.json"]
    assert (data.root / "README.md").read_text(encoding="utf-8") == files["README.md"]


def test_corrupt_download_is_rejected(release, tmp_path):
    manifest, url, _ = release
    asset = tmp_path / "release" / "clinical_outputs_Case0001_rev001.zip"
    asset.write_bytes(asset.read_bytes()[:-10])
    data = ClinicalTestingData(manifest, tmp_path / "cache", url)
    with pytest.raises(RuntimeError, match="does not match its sha256"):
        data.ensure_case("Case0001")


def test_unavailable_release(release, tmp_path):
    manifest, _, _ = release
    data = ClinicalTestingData(manifest, tmp_path / "cache", (tmp_path / "missing").as_uri() + "/")
    with pytest.raises(testing_data.TestingDataUnavailable):
        data.ensure_case("Case0001")


def test_clinical_test_data_by_revision(release, tmp_path, monkeypatch):
    """clinical_test_data(revision) downloads the manifest of the revision, verifies it by its published sha256, and
    returns the root of the downloaded data set; a manifest that does not match is rejected."""
    manifest, url, files = release
    monkeypatch.setitem(testing_data.MANIFEST_SHA256, "rev001", testing_data.sha256_file(manifest))
    root = testing_data.clinical_test_data("rev001", base_dir=tmp_path / "cache", release_url=url)
    assert root == (tmp_path / "cache" / "clinical_rev001").resolve() and root.is_absolute()
    assert (root / "manifest.json").read_bytes() == manifest.read_bytes()
    assert all((root / path).read_text(encoding="utf-8") == text
               for path, text in files.items() if path != "manifest.json")
    from slicerheart_4dflow.slicer_io import dataset_cases
    assert dataset_cases(root) == ["Case0001", "Case0002"]
    # only the cases asked for
    one = testing_data.clinical_test_data("rev001", cases=["Case0002"], base_dir=tmp_path / "one", release_url=url)
    assert (one / "inputs/Case0002").is_dir() and not (one / "inputs/Case0001").exists()
    monkeypatch.setitem(testing_data.MANIFEST_SHA256, "rev001", "0" * 64)
    with pytest.raises(RuntimeError, match="does not match its sha256"):
        testing_data.clinical_test_data("rev001", base_dir=tmp_path / "other", release_url=url)


def test_manifest_of_the_repository_is_consistent():
    """The manifest kept in this folder is the published manifest of its revision and describes a whole data set: its
    metadata, every file in one asset, every asset listed."""
    from conftest import CLINICAL_MANIFEST
    data = ClinicalTestingData(CLINICAL_MANIFEST)
    assert testing_data.sha256_file(CLINICAL_MANIFEST) == testing_data.MANIFEST_SHA256[data.revision]
    assert data.manifest["format_version"] == 4 and data.manifest["license"].startswith("CC BY")
    assert data.cases()
    assert {f["asset"] for f in data.files.values()} == set(data.assets)
    # assets of this revision, or reused from an earlier one
    assert all(f["asset"].startswith("clinical_") and f["asset"].rsplit(".", 1)[0].rsplit("_", 1)[1] <= data.revision
               for f in data.files.values())


def test_base_folder_from_environment(release, tmp_path, monkeypatch):
    """Without base_dir, the data set goes into the SLICERHEART_4DFLOW_TESTDATA folder; several revisions are kept next
    to each other in it."""
    manifest, url, _ = release
    base = tmp_path / "testdata"
    monkeypatch.setenv("SLICERHEART_4DFLOW_TESTDATA", str(base))
    assert testing_data.default_base_dir() == base
    monkeypatch.setitem(testing_data.MANIFEST_SHA256, "rev001", testing_data.sha256_file(manifest))
    root = testing_data.clinical_test_data("rev001", cases=["Case0001"], release_url=url)
    assert root == (base / "clinical_rev001").resolve()
    other = ClinicalTestingData(manifest, release_url=url)
    other.revision, other.root = "rev002", base / "clinical_rev002"           # a second revision in the same base folder
    other.ensure_case("Case0002")
    assert sorted(p.name for p in base.iterdir()) == ["clinical_rev001", "clinical_rev002"]
    monkeypatch.delenv("SLICERHEART_4DFLOW_TESTDATA")
    assert testing_data.default_base_dir() != base
