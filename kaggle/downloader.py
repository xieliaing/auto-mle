"""
Download competition data via the Kaggle REST API.

Downloads the full dataset as a zip archive, then extracts it.
Individual files can also be downloaded by name.

No kaggle Python package required — uses requests + stdlib zipfile.
"""
from __future__ import annotations
import zipfile
from pathlib import Path

import requests

from .auth import bearer_headers

_BASE = "https://www.kaggle.com/api/v1"
_CHUNK = 1 << 17  # 128 KB chunks


def _stream_to_file(url: str, dest: Path, slug: str = "") -> None:
    """Stream a potentially large response to disk with progress reporting."""
    headers = bearer_headers()
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=300) as r:
        if r.status_code == 403:
            rules_url = f"https://www.kaggle.com/c/{slug}/rules" if slug else "the competition rules page"
            raise PermissionError(
                f"Kaggle returned 403 — competition rules not yet accepted.\n"
                f"Visit {rules_url} and click 'I Understand and Accept'."
            )
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        size_str = f"{downloaded / 1e6:.1f} MB"
        print(f"    {dest.name}: {size_str}")


def _unzip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract zip and return list of extracted file paths (non-directories)."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            zf.extract(info, dest_dir)
            p = dest_dir / info.filename
            if p.is_file():
                extracted.append(p)
    return extracted


def download_competition_data(slug: str, dest_dir: str | Path) -> list[Path]:
    """
    Download all competition data files to dest_dir.

    Strategy:
      1. Download the 'download-all' zip from Kaggle.
      2. Extract into dest_dir.
      3. Delete the zip archive.

    Returns a sorted list of extracted file paths.
    Raises PermissionError if competition rules have not been accepted on the website.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / f"{slug}.zip"
    url = f"{_BASE}/competitions/data/download-all/{slug}"

    print(f"[downloader] GET {url}")
    _stream_to_file(url, zip_path, slug=slug)

    if not zipfile.is_zipfile(zip_path):
        # Some competitions return the file directly (not zipped).
        print(f"[downloader] Response is not a zip — treating as raw file.")
        raw_dest = dest / "data.bin"
        zip_path.rename(raw_dest)
        return [raw_dest]

    print(f"[downloader] Extracting {zip_path.name} → {dest}")
    extracted = _unzip(zip_path, dest)
    zip_path.unlink()

    # Recursively unzip any nested zips
    all_files: list[Path] = []
    for p in extracted:
        if p.suffix == ".zip" and zipfile.is_zipfile(p):
            print(f"[downloader] Nested zip: {p.name}")
            inner = _unzip(p, dest)
            all_files.extend(inner)
            p.unlink()
        else:
            all_files.append(p)

    return sorted(all_files)


def download_file(slug: str, filename: str, dest_dir: str | Path) -> Path:
    """Download a single named file from the competition data."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE}/competitions/data/download/{slug}/{filename}"
    out = dest / filename
    print(f"[downloader] Downloading {filename}")
    _stream_to_file(url, out, slug=slug)
    return out
