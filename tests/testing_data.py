"""
Clinical testing data: download from the TestingData release on first use, and cache.

The clinical test data set (format 4, see tests/clinical/README.md) is stored in the TestingData release of this
repository as one zip per folder of the data set (clinical_inputs_<case>_acquisition_<revision>.zip,
clinical_inputs_<case>_annotations_<revision>.zip, clinical_outputs_<case>_<revision>.zip,
clinical_tests_<revision>.zip) and the plain files clinical_manifest_<revision>.json and
clinical_README_<revision>.md. The file list of the revision, clinical_files_<revision>.json (kept in this folder),
fully identifies the data set: the path, size and sha256 of every file, the asset that contains it, and the sha256 of
every asset.

The data set is extracted into <cache>/clinical_<revision>/, <cache> being the SLICERHEART_4DFLOW_TESTING_DATA_CACHE
folder or .testingdata_cache/ in the repository. Assets are downloaded when a test first needs them (a case, the
baseline), verified by their sha256, extracted, and deleted. The cache records the size, modification time and sha256
of every extracted file (cache_state.json): a file with the recorded size and modification time is up to date without
reading it; otherwise its sha256 is computed, and its asset is downloaded again if the hash differs.

Instead of the cache, SLICERHEART_4DFLOW_CLINICAL_DATA can name a local data set folder, which is used as it is (no
download); write a new baseline (--update-baseline) to such a folder, since the cache is restored from the release.

Run as a script to download the whole data set (for example to fill a CI cache before the tests):

    python tests/testing_data.py
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

TESTS_DIR = Path(__file__).resolve().parent
RELEASE_URL = "https://github.com/JolleyLab/slicerheart_4dflow/releases/download/TestingData/"
CLINICAL_FILES_LIST = TESTS_DIR / "clinical_files_rev001.json"
CACHE_DIR = Path(os.environ.get("SLICERHEART_4DFLOW_TESTING_DATA_CACHE", TESTS_DIR.parent / ".testingdata_cache"))
STATE_FILENAME = "cache_state.json"
# modification times restored from a CI cache (tar) may have lost the fraction of a second
MTIME_TOLERANCE_S = 1.0


class TestingDataUnavailable(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(message):
    """Progress of a download, on the console even while pytest captures the output of the test."""
    print(f"[testing data] {message}", file=sys.__stderr__, flush=True)


class ClinicalTestingData:
    """The clinical data set of a file list, extracted into a cache folder."""

    def __init__(self, files_list=CLINICAL_FILES_LIST, cache_dir=CACHE_DIR, release_url=RELEASE_URL):
        self.files_list = Path(files_list)
        index = json.loads(self.files_list.read_text(encoding="utf-8"))
        self.revision = index["revision"]
        self.files = index["files"]              # {path in the data set: {sha256, size, asset, path_in_asset}}
        self.assets = index["assets"]            # {asset name: {sha256, size}}
        self.release_url = release_url
        self.root = Path(cache_dir) / f"clinical_{self.revision}"
        self.state_path = self.root / STATE_FILENAME
        self._state = None
        self._checked = set()                    # assets found up to date in this session

    # -- contents ---------------------------------------------------------------------------------------------------

    def cases(self):
        """Case IDs of the data set (the folders in inputs/), from the file list: no download."""
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
        """The cached file has the content of the file list: by its recorded size and modification time, or else by its
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
        if _sha256(local) != expected["sha256"]:
            return False
        self._record(path)
        return True

    def asset_up_to_date(self, asset):
        return all(self._file_up_to_date(path) for path, f in self.files.items() if f["asset"] == asset)

    # -- download ---------------------------------------------------------------------------------------------------

    def _download(self, asset):
        """Download an asset into the cache folder and verify its sha256. Returns the path of the downloaded file."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / asset
        tmp = target.with_name(target.name + ".part")
        url = self.release_url + asset
        size_mb = self.assets[asset]["size"] / 1e6
        _status(f"downloading {asset} ({size_mb:.0f} MB)")
        t0 = time.perf_counter()
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as exc:  # network unavailable
            tmp.unlink(missing_ok=True)
            raise TestingDataUnavailable(f"cannot download clinical testing data from {url}: {exc}") from exc
        if _sha256(tmp) != self.assets[asset]["sha256"]:
            tmp.unlink()
            raise RuntimeError(f"downloaded {asset} does not match its sha256 in {self.files_list.name}")
        tmp.replace(target)
        _status(f"downloaded {asset} in {time.perf_counter() - t0:.0f} s")
        return target

    def _install(self, asset):
        """Download an asset and put its files into the data set folder."""
        downloaded = self._download(asset)
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
        return self.ensure(sorted({f["asset"] for path, f in self.files.items() if not path.startswith(("inputs/", "outputs/"))}))

    def ensure_case(self, case_id):
        self.ensure_common()
        return self.ensure(self.assets_of(f"inputs/{case_id}/") + self.assets_of(f"outputs/{case_id}/"))

    def ensure_all(self):
        return self.ensure(sorted(self.assets))


# ---------------------------------------------------------------------------------------------------------------------
# Packaging a data set as release assets
# ---------------------------------------------------------------------------------------------------------------------

PREFIX = "clinical_"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)     # a fixed time for every zip entry, so that an unchanged folder gives the same zip


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


def package(dataset_dir, out_dir, revision):
    """
    Write the release assets of a data set (format 4) into out_dir: one zip per folder (inputs/<case>/<part>/,
    outputs/<case>/, tests/), manifest.json and README.md as plain files, and the file list
    clinical_files_<revision>.json (copy it to this folder to make the tests use the revision). Every asset name has the
    prefix clinical_ and the suffix _<revision>. Returns the path of the file list.
    """
    dataset, out = Path(dataset_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    suffix = f"_{revision}"
    folders = []
    for case in sorted(p.name for p in (dataset / "inputs").iterdir() if p.is_dir()):
        for part in sorted(p.name for p in (dataset / "inputs" / case).iterdir() if p.is_dir()):
            folders.append((f"{PREFIX}inputs_{case}_{part}{suffix}.zip", dataset / "inputs" / case / part))
    for case in sorted(p.name for p in (dataset / "outputs").iterdir() if p.is_dir()):
        folders.append((f"{PREFIX}outputs_{case}{suffix}.zip", dataset / "outputs" / case))
    folders.append((f"{PREFIX}tests{suffix}.zip", dataset / "tests"))
    plain = [(f"{PREFIX}manifest{suffix}.json", dataset / "manifest.json"), (f"{PREFIX}README{suffix}.md", dataset / "README.md")]
    files, assets = {}, {}
    for name, folder in folders:
        for p in _write_zip(out / name, folder):
            files[p.relative_to(dataset).as_posix()] = {"sha256": _sha256(p), "size": p.stat().st_size, "asset": name,
                                                        "path_in_asset": p.relative_to(folder).as_posix()}
        assets[name] = {"sha256": _sha256(out / name), "size": (out / name).stat().st_size}
    for name, path in plain:
        if not path.exists():
            continue
        shutil.copyfile(path, out / name)
        files[path.relative_to(dataset).as_posix()] = {"sha256": _sha256(path), "size": path.stat().st_size,
                                                       "asset": name, "path_in_asset": None}
        assets[name] = {"sha256": _sha256(out / name), "size": (out / name).stat().st_size}
    missing = sorted({p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file()} - set(files))
    if missing:
        raise ValueError(f"files outside the packaged folders: {missing}")
    index = {
        "description": "Files of the clinical test data set (format 4) and the release asset that contains each of "
                       "them. path is relative to the data set root; path_in_asset is the path inside the zip (null "
                       "for a plain file). A file needs to be downloaded again only if its sha256 changed.",
        "revision": revision,
        "data_set": json.loads((dataset / "manifest.json").read_text(encoding="utf-8")),
        "assets": dict(sorted(assets.items())),
        "files": dict(sorted(files.items())),
    }
    files_list = out / f"{PREFIX}files{suffix}.json"
    files_list.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")
    return files_list


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Download the clinical testing data into the cache (default), or "
                                                 "package a data set as release assets.")
    sub = parser.add_subparsers(dest="command")
    pack = sub.add_parser("package", help="write the release assets of a data set")
    pack.add_argument("dataset", help="data set folder (format 4)")
    pack.add_argument("output", help="folder for the assets (must not exist)")
    pack.add_argument("revision", help="revision of the files, for example rev002")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.command == "package":
        files_list = package(args.dataset, args.output, args.revision)
        index = json.loads(files_list.read_text(encoding="utf-8"))
        total = sum(a["size"] for a in index["assets"].values())
        print(f"{len(index['files'])} files in {len(index['assets'])} assets ({total / 1e9:.2f} GB) and {files_list}")
        return
    data = ClinicalTestingData()
    root = data.ensure_all()
    print(f"clinical testing data {data.revision}: {len(data.cases())} cases, {len(data.files)} files in {root} "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
