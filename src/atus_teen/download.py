"""Download and cache the official BLS ATUS 2003–2024 multi-year files.

The Bureau of Labor Statistics publishes these zips at
https://www.bls.gov/tus/data/datafiles-0324.htm. Some networks receive an
Akamai 403 for automated clients. When that happens, the same official URL is
retrieved from the Internet Archive's capture of the BLS file.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from atus_teen.paths import raw_dir

# Official multi-year microdata, 2003–2024 (BLS page last modified June 26, 2025).
BLS_PAGE = "https://www.bls.gov/tus/data/datafiles-0324.htm"

FILES: dict[str, str] = {
    "atussum-0324.zip": "https://www.bls.gov/tus/datafiles/atussum-0324.zip",
    "atusresp-0324.zip": "https://www.bls.gov/tus/datafiles/atusresp-0324.zip",
    "atusrost-0324.zip": "https://www.bls.gov/tus/datafiles/atusrost-0324.zip",
    "atuscps-0324.zip": "https://www.bls.gov/tus/datafiles/atuscps-0324.zip",
}

# SHA-256 of the copies used to build the published teen extract.
# A rebuilt download may differ if BLS revises the multi-year files.
PINNED_SHA256: dict[str, str] = {
    "atussum-0324.zip": "bf39ac14710ab4850c88e3cfc41c6b527bdfb7828549e0f7ccbe889e1d2922a8",
    "atusresp-0324.zip": "6556af3a6ccb1676bce99a1db35b0813766787a0e739be7e23e0b3c8abe31381",
    "atusrost-0324.zip": "81e0cfd195249b6fb2655abc003990e30b5d854d3a464626fa3d88c0842b81c8",
    "atuscps-0324.zip": "d88b32c1429b8d85bbf2c19773c27660101a504c4e4884c8b09d02d8efc66588",
}

USER_AGENT = (
    "atus-teen-reading/0.1 (reproducible research; "
    "+https://www.bls.gov/tus/data/datafiles-0324.htm)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url: str, timeout: int = 180) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": BLS_PAGE,
        },
    )


def _archive_url(official_url: str) -> str:
    """Latest successful Wayback capture of an official BLS zip."""
    bare = official_url.replace("https://", "").replace("http://", "")
    query = urllib.parse.urlencode(
        {
            "url": bare,
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp,mimetype,length",
            "limit": "30",
        }
    )
    cdx = f"https://web.archive.org/cdx/search/cdx?{query}"
    with urllib.request.urlopen(_request(cdx), timeout=60) as response:
        rows = json.loads(response.read().decode())
    if len(rows) < 2:
        raise RuntimeError(f"No Internet Archive capture for {official_url}")
    candidates = []
    for timestamp, mimetype, length in rows[1:]:
        try:
            nbytes = int(length)
        except (TypeError, ValueError):
            nbytes = 0
        if "zip" in (mimetype or "") or nbytes > 100_000:
            candidates.append((timestamp, nbytes))
    if not candidates:
        raise RuntimeError(f"Archive captures for {official_url} were not zip files")
    timestamp = candidates[-1][0]
    return f"https://web.archive.org/web/{timestamp}id_/{official_url}"


def _download_once(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(_request(url), timeout=300) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    if partial.stat().st_size < 1000:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download too small: {url}")
    with partial.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"PK\x03\x04":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download is not a zip file: {url}")
    partial.replace(dest)


def _download_with_fallback(name: str, official_url: str, dest: Path) -> str:
    errors: list[str] = []
    sources = [official_url]
    try:
        sources.append(_archive_url(official_url))
    except Exception as exc:  # noqa: BLE001 — fallback discovery should not hide the BLS try
        errors.append(f"archive lookup failed: {exc}")
    for attempt, url in enumerate(sources):
        try:
            if attempt:
                time.sleep(2)
            _download_once(url, dest)
            return url
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not download {name}. BLS blocks some automated clients (HTTP 403). "
        + " | ".join(errors)
    )


def _zip_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return any(name.endswith(".dat") for name in names)
    except zipfile.BadZipFile:
        return False


def ensure_files(force: bool = False) -> dict[str, dict[str, str]]:
    """Cache the four multi-year zips under data/raw. Returns file metadata."""
    directory = raw_dir()
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, str]] = {}
    for name, url in FILES.items():
        dest = directory / name
        source = "cache"
        if force or not _zip_ok(dest):
            source = _download_with_fallback(name, url, dest)
        digest = sha256_file(dest)
        pinned = PINNED_SHA256.get(name)
        report[name] = {
            "path": str(dest.relative_to(raw_dir().parent.parent)),
            "official_url": url,
            "source": source,
            "sha256": digest,
            "matches_published_extract": digest == pinned,
        }
        print(f"{name}: {source} sha256={digest[:12]}…")
    return report
