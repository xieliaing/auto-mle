"""
Fetch competition metadata from the Kaggle REST API.

Endpoints used:
  GET /api/v1/competitions/{slug}           — title, description, evaluation metric
  GET /api/v1/competitions/data/list/{slug} — data file manifest

All calls use HTTP Basic Auth (username + API key from kaggle.json).
No kaggle Python package required.
"""
from __future__ import annotations
from html.parser import HTMLParser

import requests

from .auth import http_auth

_BASE = "https://www.kaggle.com/api/v1"
_TIMEOUT = 30


class _TagStripper(HTMLParser):
    """Minimal HTML → plain-text converter (no external dependencies)."""

    def __init__(self):
        super().__init__()
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "h1", "h2", "h3", "h4", "li"):
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._buf.append(stripped)

    def text(self) -> str:
        import re
        raw = " ".join(self._buf)
        return re.sub(r"\n +", "\n", re.sub(r" {2,}", " ", raw)).strip()


def _strip_html(html: str) -> str:
    p = _TagStripper()
    p.feed(html)
    return p.text()


def _get(path: str, auth: tuple) -> dict | list:
    r = requests.get(f"{_BASE}{path}", auth=auth, timeout=_TIMEOUT)
    if r.status_code == 403:
        raise PermissionError(
            f"Kaggle returned 403 for {path}.\n"
            "You may need to accept the competition rules at:\n"
            f"  https://www.kaggle.com/c/{path.split('/')[-1]}"
        )
    if r.status_code == 404:
        raise ValueError(f"Not found (404): {path!r} — check the competition slug.")
    r.raise_for_status()
    return r.json()


def get_competition_info(slug: str) -> dict:
    """
    Fetch and return a structured summary of the competition.

    Returned dict fields:
        slug, title, description, evaluation_metric,
        competition_url, files: [{name, size_bytes, description}]
    """
    auth = http_auth()

    meta = _get(f"/competitions/{slug}", auth)
    raw_files = _get(f"/competitions/data/list/{slug}", auth)
    if not isinstance(raw_files, list):
        raw_files = []

    description_html = meta.get("description", "") or ""
    description = _strip_html(description_html)

    files = [
        {
            "name": f.get("name", ""),
            "size_bytes": f.get("totalBytes", 0),
            "description": (_strip_html(f.get("description", "") or "")),
        }
        for f in raw_files
        if f.get("name")
    ]

    return {
        "slug": slug,
        "title": meta.get("title", slug),
        "description": description,
        "evaluation_metric": meta.get("evaluationMetric", "unknown"),
        "competition_url": f"https://www.kaggle.com/c/{slug}",
        "files": files,
    }
