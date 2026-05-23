"""
Kaggle API credential loader.

Reads ~/.kaggle/kaggle.json (the standard location created by
  kaggle.com > Settings > Account > API > Create New Token).

The file must contain {"username": "...", "key": "..."}.
"""
from __future__ import annotations
import json
from pathlib import Path

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def load_credentials() -> dict:
    """Return {"username": ..., "key": ...} from ~/.kaggle/kaggle.json."""
    if not KAGGLE_JSON.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {KAGGLE_JSON}\n\n"
            "To fix:\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. Scroll to 'API' section → click 'Create New Token'\n"
            "  3. Save the downloaded kaggle.json to:\n"
            f"       {KAGGLE_JSON}\n"
            "  4. On Linux/macOS: chmod 600 ~/.kaggle/kaggle.json"
        )
    try:
        creds = json.loads(KAGGLE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"kaggle.json is not valid JSON: {e}") from e
    for field in ("username", "key"):
        if field not in creds:
            raise ValueError(
                f"kaggle.json is missing required field '{field}'. "
                "Re-download a fresh token from kaggle.com/settings/account."
            )
    return creds


def http_auth() -> tuple[str, str]:
    """Return (username, api_key) tuple for use as requests HTTP Basic Auth."""
    creds = load_credentials()
    return (creds["username"], creds["key"])
