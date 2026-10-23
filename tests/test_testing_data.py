"""Download and cache of the clinical testing data (testing_data.py), on a small data set packaged as a local release."""

import json
import os
import time

import pytest

import testing_data
from testing_data import ClinicalTestingData


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


@pytest.fixture
def release(tmp_path):
    """(file list, release URL, data set): the data set packaged as release assets in a local folder."""
    dataset = tmp_path / "dataset"
    files = write_dataset(dataset)
    files_list = testing_data.package(dataset, tmp_path / "release", "rev001")
    return files_list, (tmp_path / "release").as_uri() + "/", files


def test_package_lists_every_file_and_asset(release):
    files_list, _, files = release
    index = json.loads(files_list.read_text(encoding="utf-8"))
    assert index["revision"] == "rev001"
    assert set(index["files"]) == set(files)
    assert set(index["assets"]) == {
        "clinical_inputs_Case0001_acquisition_rev001.zip", "clinical_inputs_Case0001_annotations_rev001.zip",
        "clinical_inputs_Case0002_acquisition_rev001.zip", "clinical_inputs_Case0002_annotations_rev001.zip",
        "clinical_outputs_Case0001_rev001.zip", "clinical_outputs_Case0002_rev001.zip", "clinical_tests_rev001.zip",
        "clinical_manifest_rev001.json", "clinical_README_rev001.md"}
    assert index["files"]["inputs/Case0001/acquisition/acquisition.json"]["path_in_asset"] == "acquisition.json"
    assert index["files"]["manifest.json"]["path_in_asset"] is None


def test_package_is_reproducible(tmp_path):
    """An unchanged folder gives the same zip (fixed entry times and order), so the asset hashes only change with the
    content."""
    dataset = tmp_path / "dataset"
    write_dataset(dataset)
    a = json.loads(testing_data.package(dataset, tmp_path / "a", "rev001").read_text(encoding="utf-8"))
    time.sleep(1.1)
    for path in dataset.rglob("*"):
        os.utime(path)                           # newer modification times, same content
    b = json.loads(testing_data.package(dataset, tmp_path / "b", "rev001").read_text(encoding="utf-8"))
    assert a["assets"] == b["assets"]


def test_download_on_first_use_per_case(release, tmp_path):
    files_list, url, files = release
    data = ClinicalTestingData(files_list, tmp_path / "cache", url)
    assert data.cases() == ["Case0001", "Case0002"]
    assert not data.root.exists()                # nothing is downloaded until a case is needed
    root = data.ensure_case("Case0001")
    assert (root / "inputs/Case0001/acquisition/velocity_raw.seq.nrrd").read_text() == files["inputs/Case0001/acquisition/velocity_raw.seq.nrrd"]
    assert (root / "tests/slicerheart_4dflow_baseline.json").exists() and (root / "manifest.json").exists()
    assert not (root / "inputs/Case0002").exists()
    assert not list(root.glob("*.zip")) and not list(root.rglob("*.part"))     # the zips are deleted after extraction


def test_cache_is_checked_by_size_and_modification_time(release, tmp_path, monkeypatch):
    """A second session finds the cache up to date without reading the files or downloading; a changed file is found
    by its hash and only its asset is downloaded again."""
    files_list, url, files = release
    ClinicalTestingData(files_list, tmp_path / "cache", url).ensure_all()

    hashed, downloaded = [], []
    sha256, download = testing_data._sha256, ClinicalTestingData._download
    monkeypatch.setattr(testing_data, "_sha256", lambda path: hashed.append(path) or sha256(path))
    monkeypatch.setattr(ClinicalTestingData, "_download", lambda self, asset: downloaded.append(asset) or download(self, asset))

    data = ClinicalTestingData(files_list, tmp_path / "cache", url)
    data.ensure_all()
    assert hashed == [] and downloaded == []

    # same content, new modification time: hashed, found unchanged, recorded; not downloaded
    touched = data.root / "outputs/Case0002/flows.json"
    time.sleep(1.1)
    os.utime(touched)
    data = ClinicalTestingData(files_list, tmp_path / "cache", url)
    data.ensure_all()
    assert [p.name for p in hashed] == ["flows.json"] and downloaded == []

    # changed content: downloaded again (only its asset), restored
    hashed.clear()
    edited = data.root / "inputs/Case0001/annotations/annotations.json"
    edited.write_text("edited", encoding="utf-8")
    (data.root / "README.md").unlink()
    data = ClinicalTestingData(files_list, tmp_path / "cache", url)
    data.ensure_all()
    assert sorted(downloaded) == ["clinical_README_rev001.md", "clinical_inputs_Case0001_annotations_rev001.zip"]
    assert edited.read_text(encoding="utf-8") == files["inputs/Case0001/annotations/annotations.json"]
    assert (data.root / "README.md").read_text(encoding="utf-8") == files["README.md"]


def test_corrupt_download_is_rejected(release, tmp_path):
    files_list, url, _ = release
    asset = tmp_path / "release" / "clinical_outputs_Case0001_rev001.zip"
    asset.write_bytes(asset.read_bytes()[:-10])
    data = ClinicalTestingData(files_list, tmp_path / "cache", url)
    with pytest.raises(RuntimeError, match="does not match its sha256"):
        data.ensure_case("Case0001")


def test_unavailable_release(release, tmp_path):
    files_list, _, _ = release
    data = ClinicalTestingData(files_list, tmp_path / "cache", (tmp_path / "missing").as_uri() + "/")
    with pytest.raises(testing_data.TestingDataUnavailable):
        data.ensure_case("Case0001")


def test_file_list_of_the_repository_is_consistent():
    """The file list kept in this folder describes a whole data set: every file in one asset, every asset listed."""
    data = ClinicalTestingData()
    assert data.cases()
    assert {f["asset"] for f in data.files.values()} == set(data.assets)
    assert all(f["asset"].startswith("clinical_") and f["asset"].rsplit(".", 1)[0].endswith("_" + data.revision)
               for f in data.files.values())
