"""
Kaggle API credential loader.

Supports two token formats:

  Legacy API key  — ~/.kaggle/kaggle.json: {"username": "...", "key": "<hex32>"}
                    Auth: Authorization: Basic base64(username:key)
                       or Authorization: Bearer <key>

  Access Token    — key starts with "KGAT_"; username is optional
                    Auth: Authorization: Bearer KGAT_...

The current Kaggle website generates Access Tokens. The legacy key format
is still supported for backward compatibility.
"""
from __future__ import annotations
import json
from pathlib import Path

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
_ACCESS_TOKEN_PREFIX = "KGAT_"


def load_credentials() -> dict:
    """
    Return credentials from ~/.kaggle/kaggle.json.

    For Access Tokens (key starts with KGAT_), username is optional.
    For legacy API keys, username is required.
    """
    if not KAGGLE_JSON.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {KAGGLE_JSON}\n\n"
            "To fix:\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. Scroll to 'API' section -> click 'Create New Token'\n"
            "  3. Save the downloaded kaggle.json to:\n"
            f"       {KAGGLE_JSON}\n"
            "  For an Access Token, the file can be just: {\"key\": \"KGAT_...\"}"
        )
    try:
        creds = json.loads(KAGGLE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"kaggle.json is not valid JSON: {e}") from e

    if "key" not in creds:
        raise ValueError(
            "kaggle.json is missing the 'key' field. "
            "Re-download a token from kaggle.com/settings/account."
        )

    key = creds["key"]
    is_access_token = key.startswith(_ACCESS_TOKEN_PREFIX)

    if not is_access_token and "username" not in creds:
        raise ValueError(
            "kaggle.json is missing 'username'. "
            "Legacy API keys require a username. "
            "If you have a new Access Token (key starting with KGAT_), "
            "just set {\"key\": \"KGAT_...\"} in kaggle.json."
        )

    return creds


def is_access_token() -> bool:
    """Return True if the stored key is a Kaggle Access Token (KGAT_ prefix)."""
    return load_credentials()["key"].startswith(_ACCESS_TOKEN_PREFIX)


def bearer_headers() -> dict:
    """
    Return Authorization headers for the Kaggle API.
    Works for both Access Tokens and legacy API keys.
    """
    creds = load_credentials()
    return {"Authorization": f"Bearer {creds['key']}"}


def http_auth() -> tuple[str, str]:
    """
    Return (username, key) for HTTP Basic Auth (legacy API keys only).
    Raises if called with an Access Token.
    """
    creds = load_credentials()
    if creds["key"].startswith(_ACCESS_TOKEN_PREFIX):
        raise ValueError(
            "HTTP Basic Auth is not supported for Kaggle Access Tokens. "
            "Use bearer_headers() instead."
        )
    return (creds["username"], creds["key"])
