"""
Clinical test data set: download from the TestingData release of slicerheart_4dflow on first use, and cache.

    import slicerheart_4dflow as f4d
    root = f4d.clinical_test_data("rev002")                  # full path of the data set root
    case = f4d.load_case(root, f4d.dataset_cases(root)[0])

The clinical test data set (format 4, see tests/clinical/README.md of the repository: de-identified 4D flow MRI cases,
CC BY 4.0) is stored in the TestingData release as one zip per folder of the data set
(clinical_inputs_<case>_acquisition_<revision>.zip, clinical_inputs_<case>_annotations_<revision>.zip,
clinical_outputs_<case>_<revision>.zip, clinical_tests_<revision>.zip), clinical_README_<revision>.md, and the
manifest clinical_manifest_<revision>.json, which fully identifies the data set: the metadata of the data set
(format_version, name, license, doi), the revision, the path, size and sha256 of every other file of the data set and
the asset that contains it, and the sha256 of every asset. The manifest is the manifest.json of the data set; the
sha256 of the manifest of each published revision is in MANIFEST_SHA256.

The data set is extracted into <base folder>/clinical_<revision>/, so that several revisions can be kept in one base
folder: base_dir if given, else the folder in the SLICERHEART_4DFLOW_TESTDATA environment variable, else
.SlicerHeart/slicerheart_4dflow_testingdata in the profile folder of the user (default_base_dir()), shared by all
installations. Assets are
downloaded when they are first needed, verified by their sha256, extracted, and deleted. The cache records the size,
modification time and sha256 of every extracted file (cache_state.json): a file with the recorded size and
modification time is up to date without reading it; otherwise its sha256 is computed, and its asset is downloaded
again if the hash differs. Files edited in the cache are therefore restored; copy the data set to edit it.

Command line (download a revision, or write the release assets of a new revision of a data set):

    python -m slicerheart_4dflow.testing_data download rev002 [--base-dir <folder>]
    python -m slicerheart_4dflow.testing_data package <data set folder> <output folder> <revision>

A new revision reuses the assets of the published revisions, which stay in the release: a folder whose files are those
of an asset of a published revision (same paths in the asset and sha256) refers to that asset, which is not written
again. A download is a full copy of the revision either way.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/JolleyLab/slicerheart_4dflow/releases/download/TestingData/"
# sha256 of the manifest (clinical_manifest_<revision>.json) of each published revision of the clinical test data set
MANIFEST_SHA256 = {
    "rev001": "3ced6430f03ed690e5461ac3491cf1efd232cd905fe45961f5f1a71cbc150496",
    "rev002": "0c944474bd7d7d6ee120abbfe2e7389466afbb22aada4c15774c4146ea1cd46d",
}
DEFAULT_REVISION = "rev002"
MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "cache_state.json"
PREFIX = "clinical_"
# modification times restored from a CI cache (tar) may have lost the fraction of a second
MTIME_TOLERANCE_S = 1.0
# the fields of the manifest that describe the package (the other fields are the metadata of the data set)
PACKAGE_FIELDS = ("revision", "assets", "files")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)     # a fixed time for every zip entry, so that an unchanged folder gives the same zip


class TestingDataUnavailable(RuntimeError):
    """The testing data cannot be downloaded (no network, or the release is not reachable)."""


def default_base_dir():
    """Base folder of the downloaded test data sets (each revision in its own clinical_<revision>/ folder): the
    SLICERHEART_4DFLOW_TESTDATA folder if set, else .SlicerHeart/slicerheart_4dflow_testingdata in the profile folder
    of the user (for example C:\\Users\\<user>\\.SlicerHeart\\slicerheart_4dflow_testingdata), so that one copy
    is shared by every installation (Python environments, 3D Slicer versions, repository clones) of the user."""
    if os.environ.get("SLICERHEART_4DFLOW_TESTDATA"):
        return Path(os.environ["SLICERHEART_4DFLOW_TESTDATA"])
    return Path.home() / ".SlicerHeart" / "slicerheart_4dflow_testingdata"


def sha256_file(path):
    """sha256 (hex) of the content of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(message):
    """Progress of a download, on the console even while pytest captures the output of a test."""
    stream = sys.__stderr__ or sys.stderr
    if stream is not None:
        print(f"[slicerheart_4dflow testing data] {message}", file=stream, flush=True)


def _download_file(url, target, sha256=None):
    """Download url to target (through a .part file), verified by sha256 if given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:  # network unavailable
        tmp.unlink(missing_ok=True)
        raise TestingDataUnavailable(f"cannot download {url}: {exc}") from exc
    if sha256 is not None and sha256_file(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"downloaded {url} does not match its sha256 {sha256}")
    tmp.replace(target)
    return target


def release_file(filename, sha256, folder, release_url=RELEASE_URL):
    """Path of a file of the TestingData release in folder, downloaded if it is missing or its content is not that of
    sha256. Raises TestingDataUnavailable if it cannot be downloaded."""
    path = Path(folder) / filename
    if path.is_file() and sha256_file(path) == sha256:
        return path
    return _download_file(release_url + filename, path, sha256)


class ClinicalTestingData:
    """The clinical test data set of a manifest, extracted into <base_dir>/clinical_<revision>/."""

    def __init__(self, manifest, base_dir=None, release_url=RELEASE_URL):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.revision = self.manifest["revision"]
        self.files = self.manifest["files"]      # {path in the data set: {sha256, size, asset, path_in_asset}}
        self.assets = self.manifest["assets"]    # {asset name: {sha256, size}}
        self.release_url = release_url
        self.root = Path(base_dir if base_dir is not None else default_base_dir()) / f"clinical_{self.revision}"
        self.state_path = self.root / STATE_FILENAME
        self._state = None
        self._checked = set()                    # assets found up to date in this session

    # -- contents ---------------------------------------------------------------------------------------------------

    def cases(self):
        """Case IDs of the data set (the folders in inputs/), from the manifest: no download."""
        return sorted({path.split("/")[1] for path in self.files if path.startswith("inputs/")})

    def assets_of(self, prefix):
        """Assets that contain the files whose path starts with prefix (for example "inputs/Patient0001/")."""
        return sorted({f["asset"] for path, f in self.files.items() if path.startswith(prefix)})

    # -- cache state ------------------------------------------------------------------------------------------------

    @property
    def state(self):
        if self._state is None:
            try:
                self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._state = {}
            self._state.setdefault("files", {})
        return self._state

    def _save_state(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def _record(self, path):
        st = (self.root / path).stat()
        self.state["files"][path] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": self.files[path]["sha256"]}

    def _file_up_to_date(self, path):
        """The cached file has the content of the manifest: by its recorded size and modification time, or else by its
        sha256 (which is then recorded)."""
        expected = self.files[path]
        local = self.root / path
        try:
            st = local.stat()
        except OSError:
            return False
        if st.st_size != expected["size"]:
            return False
        recorded = self.state["files"].get(path)
        if (recorded and recorded.get("sha256") == expected["sha256"] and recorded.get("size") == st.st_size
                and abs(recorded.get("mtime", -1e9) - st.st_mtime) <= MTIME_TOLERANCE_S):
            return True
        if sha256_file(local) != expected["sha256"]:
            return False
        self._record(path)
        return True

    def asset_up_to_date(self, asset):
        return all(self._file_up_to_date(path) for path, f in self.files.items() if f["asset"] == asset)

    # -- download ---------------------------------------------------------------------------------------------------

    def _install(self, asset):
        """Download an asset, verify it, and put its files into the data set folder."""
        size_mb = self.assets[asset]["size"] / 1e6
        _status(f"downloading {asset} ({size_mb:.0f} MB)")
        t0 = time.perf_counter()
        try:
            downloaded = _download_file(self.release_url + asset, self.root / asset, self.assets[asset]["sha256"])
        except RuntimeError as exc:
            if isinstance(exc, TestingDataUnavailable):
                raise
            raise RuntimeError(f"downloaded {asset} does not match its sha256 in {self.manifest_path.name}") from exc
        _status(f"downloaded {asset} in {time.perf_counter() - t0:.0f} s")
        members = {f["path_in_asset"]: path for path, f in self.files.items() if f["asset"] == asset}
        if None in members:                      # a plain file
            path = members[None]
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(self.root / path)
            self._record(path)
        else:
            with zipfile.ZipFile(downloaded) as archive:
                for member, path in members.items():
                    local = self.root / path
                    local.parent.mkdir(parents=True, exist_ok=True)
                    tmp = local.with_name(local.name + ".part")
                    with archive.open(member) as source, open(tmp, "wb") as target:
                        shutil.copyfileobj(source, target, 1 << 22)
                    tmp.replace(local)
                    self._record(path)
            downloaded.unlink()
        self._save_state()

    def _ensure_manifest(self):
        """manifest.json of the cached data set: a copy of the manifest (which cannot list its own hash)."""
        local = self.root / MANIFEST_FILENAME
        expected = self.manifest_path.read_bytes()
        try:
            if local.read_bytes() == expected:
                return
        except OSError:
            pass
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".part")
        tmp.write_bytes(expected)
        tmp.replace(local)

    def ensure(self, assets):
        """Make the given assets available in the cache (download those that are missing or changed)."""
        checked = False
        for asset in assets:
            if asset in self._checked:
                continue
            if self.asset_up_to_date(asset):
                checked = True                   # hashes of changed modification times may have been recorded
            else:
                self._install(asset)
            self._checked.add(asset)
        if checked:
            self._save_state()
        return self.root

    def ensure_common(self):
        """manifest.json, README.md and tests/ (the baseline)."""
        self._ensure_manifest()
        return self.ensure(sorted({f["asset"] for path, f in self.files.items()
                                   if not path.startswith(("inputs/", "outputs/"))}))

    def ensure_case(self, case_id):
        self.ensure_common()
        return self.ensure(self.assets_of(f"inputs/{case_id}/") + self.assets_of(f"outputs/{case_id}/"))

    def ensure_all(self):
        self._ensure_manifest()
        return self.ensure(sorted(self.assets))


def clinical_test_data_manifest(revision=DEFAULT_REVISION, base_dir=None, release_url=RELEASE_URL):
    """Path of the manifest of a revision of the clinical test data set (the manifest.json of the cached data set),
    downloaded if it is not in the cache yet. The manifest of a published revision is verified by its sha256
    (MANIFEST_SHA256); a revision that this version of the package does not know is used as downloaded."""
    root = Path(base_dir if base_dir is not None else default_base_dir()) / f"clinical_{revision}"
    local = root / MANIFEST_FILENAME
    expected = MANIFEST_SHA256.get(revision)
    if local.is_file() and (expected is None or sha256_file(local) == expected):
        return local
    _download_file(f"{release_url}{PREFIX}manifest_{revision}.json", local, expected)
    return local


def clinical_test_data(revision=DEFAULT_REVISION, cases=None, base_dir=None, release_url=RELEASE_URL):
    """
    Download the clinical test data set of a revision (for example "rev001") into the cache, if it is not there or not
    up to date yet, and return the full path of the data set root (the folder with manifest.json, inputs/, outputs/ and
    tests/), <base_dir>/clinical_<revision>. cases: only these case IDs (and the files shared by all cases), default
    all. base_dir: the base folder, the parent folder of the revisions, default default_base_dir() (the
    SLICERHEART_4DFLOW_TESTDATA folder, or ~/.SlicerHeart/slicerheart_4dflow_testingdata). Raises
    TestingDataUnavailable if the data cannot be downloaded.
    """
    data = ClinicalTestingData(clinical_test_data_manifest(revision, base_dir, release_url), base_dir, release_url)
    if cases is None:
        data.ensure_all()
    else:
        for case_id in cases:
            data.ensure_case(case_id)
    return data.root.resolve()


# ---------------------------------------------------------------------------------------------------------------------
# Packaging a data set as release assets
# ---------------------------------------------------------------------------------------------------------------------

def _write_zip(path, folder):
    """Zip the files of folder (paths relative to it), in sorted order, with fixed times."""
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(folder).as_posix(), date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(p, "rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, 1 << 22)
    return files


def _reusable_assets(manifests):
    """{asset contents: [(asset name without the revision suffix, asset name, asset entry)]} of the assets of earlier
    manifests; the contents of an asset is the frozenset of (path in asset, sha256, size) of its files."""
    reusable = {}
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        suffix = f"_{manifest['revision']}"
        contents = {}
        for f in manifest["files"].values():
            contents.setdefault(f["asset"], set()).add((f["path_in_asset"], f["sha256"], f["size"]))
        for name, files in contents.items():
            stem, extension = name.rsplit(".", 1)
            generic = stem.removesuffix(suffix) + "." + extension
            reusable.setdefault(frozenset(files), []).append((generic, name, manifest["assets"][name]))
    return reusable


def _published_manifests(revision):
    """Manifests of the published revisions other than revision (MANIFEST_SHA256), downloaded if not cached."""
    return [clinical_test_data_manifest(r) for r in sorted(MANIFEST_SHA256) if r != revision]


def _reuse(reusable, contents, name, suffix):
    """(asset name, asset entry) of an earlier asset with these contents, preferring the asset of the same folder (the
    same name with another revision suffix), or None."""
    candidates = reusable.get(frozenset(contents), [])
    stem, extension = name.rsplit(".", 1)
    generic = stem.removesuffix(suffix) + "." + extension
    for candidate_generic, candidate, entry in candidates:
        if candidate_generic == generic:
            return candidate, entry
    return (candidates[0][1], candidates[0][2]) if candidates else None


def package_clinical_test_data(dataset_dir, out_dir, revision, reuse=()):
    """
    Write the release assets of a data set (format 4) into out_dir: one zip per folder (inputs/<case>/<part>/,
    outputs/<case>/, tests/), README.md as a plain file, and the manifest clinical_manifest_<revision>.json: the
    metadata fields of manifest.json of the data set, the revision, and the files and assets. Every asset name has the
    prefix clinical_ and the suffix _<revision>. Then upload the assets to the TestingData release, add the sha256 of
    the manifest to MANIFEST_SHA256, copy the manifest to tests/ of the repository (to make the tests use the
    revision) and to the data set as its manifest.json. Returns the path of the manifest.

    The assets of the published revisions (MANIFEST_SHA256, their manifests are downloaded if they are not cached) and
    of the manifests in reuse are reused: a folder (or README.md) whose files are exactly those of one of their assets
    (same paths in the asset, sha256 and size) refers to that asset in the manifest, and no asset is written for it.
    Only the new assets are written into out_dir.
    """
    dataset, out = Path(dataset_dir), Path(out_dir)
    reusable = _reusable_assets(_published_manifests(revision) + list(reuse))
    out.mkdir(parents=True, exist_ok=False)
    suffix = f"_{revision}"
    folders = []
    for case in sorted(p.name for p in (dataset / "inputs").iterdir() if p.is_dir()):
        for part in sorted(p.name for p in (dataset / "inputs" / case).iterdir() if p.is_dir()):
            folders.append((f"{PREFIX}inputs_{case}_{part}{suffix}.zip", dataset / "inputs" / case / part))
    for case in sorted(p.name for p in (dataset / "outputs").iterdir() if p.is_dir()):
        folders.append((f"{PREFIX}outputs_{case}{suffix}.zip", dataset / "outputs" / case))
    folders.append((f"{PREFIX}tests{suffix}.zip", dataset / "tests"))
    plain = [(f"{PREFIX}README{suffix}.md", dataset / "README.md")]
    files, assets = {}, {}
    for name, folder in folders:
        entries = {p: {"sha256": sha256_file(p), "size": p.stat().st_size,
                       "path_in_asset": p.relative_to(folder).as_posix()}
                   for p in sorted(p for p in folder.rglob("*") if p.is_file())}
        contents = {(e["path_in_asset"], e["sha256"], e["size"]) for e in entries.values()}
        previous = _reuse(reusable, contents, name, suffix)
        if previous:
            name, asset = previous           # an asset of an earlier revision has exactly these files
            assets[name] = asset
        else:
            _write_zip(out / name, folder)
            assets[name] = {"sha256": sha256_file(out / name), "size": (out / name).stat().st_size}
        for p, entry in entries.items():
            files[p.relative_to(dataset).as_posix()] = dict(entry, asset=name)
    for name, path in plain:
        if not path.exists():
            continue
        entry = {"sha256": sha256_file(path), "size": path.stat().st_size, "path_in_asset": None}
        previous = _reuse(reusable, {(None, entry["sha256"], entry["size"])}, name, suffix)
        if previous:
            name, asset = previous
            assets[name] = asset
        else:
            shutil.copyfile(path, out / name)
            assets[name] = {"sha256": sha256_file(out / name), "size": (out / name).stat().st_size}
        files[path.relative_to(dataset).as_posix()] = dict(entry, asset=name)
    missing = sorted({p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file()}
                     - set(files) - {MANIFEST_FILENAME})
    if missing:
        raise ValueError(f"files outside the packaged folders: {missing}")
    metadata = json.loads((dataset / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest = {key: value for key, value in metadata.items() if key not in PACKAGE_FIELDS}
    # files: {path relative to the data set root: {sha256, size, asset (the release asset that contains the file),
    # path_in_asset (its path in the zip, null for a plain file)}}; manifest.json itself is this manifest
    manifest.update({"revision": revision, "assets": dict(sorted(assets.items())), "files": dict(sorted(files.items()))})
    manifest_path = out / f"{PREFIX}manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest_path


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m slicerheart_4dflow.testing_data",
                                     description="Download the clinical test data set, or package a data set as "
                                                 "release assets.")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download a revision into the cache and print the data set root")
    download.add_argument("revision", nargs="?", default=DEFAULT_REVISION)
    download.add_argument("--manifest", help="use this manifest file (for example the one kept in tests/)")
    download.add_argument("--base-dir", default=None,
                          help="base folder, the parent folder of the revisions (default: SLICERHEART_4DFLOW_TESTDATA "
                               "or ~/.SlicerHeart/slicerheart_4dflow_testingdata)")
    pack = sub.add_parser("package", help="write the release assets of a data set")
    pack.add_argument("dataset", help="data set folder (format 4)")
    pack.add_argument("output", help="folder for the assets (must not exist)")
    pack.add_argument("revision", help="revision of the files, for example rev002")
    pack.add_argument("--reuse", action="append", default=[], metavar="MANIFEST",
                      help="manifest of another revision in the same release whose assets are reused, in addition to "
                           "the published revisions (may be given several times)")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.command == "package":
        manifest_path = package_clinical_test_data(args.dataset, args.output, args.revision, args.reuse)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total = sum(a["size"] for a in manifest["assets"].values())
        new = [p for p in Path(args.output).iterdir() if p != manifest_path]
        print(f"{len(manifest['files'])} files in {len(manifest['assets'])} assets ({total / 1e9:.2f} GB), of which "
              f"{len(new)} new ({sum(p.stat().st_size for p in new) / 1e9:.2f} GB), and {manifest_path} "
              f"(sha256 {sha256_file(manifest_path)})")
        return
    if args.manifest:
        data = ClinicalTestingData(args.manifest, args.base_dir)
        root = data.ensure_all().resolve()
    else:
        root = clinical_test_data(args.revision, base_dir=args.base_dir)
    print(root)
    _status(f"ready in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
