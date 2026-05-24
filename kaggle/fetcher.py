"""
Fetch competition metadata from the Kaggle REST API.

Endpoints used:
  GET /api/v1/competitions/list?search=<slug>  — search for competition metadata
  GET /api/v1/competitions/data/list/<slug>    — data file manifest

Auth: Bearer token (Authorization: Bearer <key>).
The older Basic-Auth format returns 401 on current Kaggle API endpoints.
"""
from __future__ import annotations

import requests

from .auth import bearer_headers

_BASE = "https://www.kaggle.com/api/v1"
_TIMEOUT = 30


def _slug_from_ref(ref: str) -> str:
    """Extract slug from a full Kaggle competition URL."""
    return ref.rstrip("/").split("/")[-1]


def _find_competition(slug: str) -> dict:
    """
    Search the competitions list for a matching entry.
    Kaggle's /api/v1/competitions/<slug> endpoint returns 404 for many competitions;
    the list-search endpoint is the reliable alternative.
    """
    headers = bearer_headers()
    r = requests.get(
        f"{_BASE}/competitions/list",
        headers=headers,
        params={"search": slug, "page": 1, "pageSize": 10},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    results = r.json() if isinstance(r.json(), list) else []

    for comp in results:
        ref = comp.get("ref", "")
        if _slug_from_ref(ref) == slug:
            return comp

    # Looser match: slug appears anywhere in the ref URL
    for comp in results:
        if slug in comp.get("ref", ""):
            return comp

    raise ValueError(
        f"Competition '{slug}' not found via Kaggle API search.\n"
        f"Check the slug at: https://www.kaggle.com/competitions\n"
        f"(API returned {len(results)} result(s) for that search term)"
    )


def get_competition_info(slug: str) -> dict:
    """
    Fetch and return a structured summary of the competition.

    Returned dict fields:
        slug, title, description, evaluation_metric,
        competition_url, files, user_has_entered
    """
    headers = bearer_headers()
    comp = _find_competition(slug)

    # Data file manifest
    fr = requests.get(
        f"{_BASE}/competitions/data/list/{slug}",
        headers=headers,
        timeout=_TIMEOUT,
    )
    files: list[dict] = []
    if fr.status_code == 200:
        raw = fr.json()
        files = [
            {
                "name": f.get("name", ""),
                "size_bytes": f.get("totalBytes", 0),
                "description": f.get("description", ""),
            }
            for f in (raw if isinstance(raw, list) else [])
            if f.get("name")
        ]
    elif fr.status_code == 403:
        # Rules not yet accepted — files are inaccessible but we can still proceed
        files = []

    # Build description from available fields
    description = comp.get("description", "") or ""
    tags = [t.get("name", "") for t in comp.get("tags", [])]
    if tags:
        description += f"\nTags: {', '.join(tags)}"

    # Metric description from rmsle/other tags if available
    for t in comp.get("tags", []):
        if t.get("description"):
            description += f"\nMetric note: {t['description']}"

    user_has_entered = comp.get("userHasEntered", False)

    return {
        "slug": slug,
        "title": comp.get("title", slug),
        "description": description,
        "evaluation_metric": comp.get("evaluationMetric", "unknown"),
        "competition_url": f"https://www.kaggle.com/c/{slug}",
        "files": files,
        "user_has_entered": user_has_entered,
    }
