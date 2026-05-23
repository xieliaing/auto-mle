"""
Inspect downloaded data files to extract schema information for baseline generation.

Reads up to 1 000 rows per CSV to keep inspection fast even for large datasets.
Returns JSON-serializable dicts (numpy types are coerced to Python natives).
"""
from __future__ import annotations
import math
from pathlib import Path

import pandas as pd


def _native(val):
    """Coerce numpy scalars / floats to JSON-serializable Python types."""
    try:
        import numpy as np
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _read_csv(path: Path, nrows: int = 1000) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path.name} with utf-8, latin-1, or cp1252.")


def inspect_csv(path: Path, n_sample: int = 5) -> dict:
    """Return schema + sample values for a single CSV file."""
    df = _read_csv(path)
    columns = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        sample_vals = [_native(v) for v in series.dropna().head(n_sample).tolist()]
        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "n_missing": n_missing,
                "pct_missing": round(n_missing / max(len(series), 1), 4),
                "n_unique": int(series.nunique()),
                "sample_values": sample_vals,
            }
        )
    return {
        "file": path.name,
        "rows_in_sample": len(df),
        "n_columns": len(df.columns),
        "columns": columns,
    }


def get_submission_format(data_dir: Path) -> dict | None:
    """
    Try to parse sample_submission.csv to understand the required output format.
    Returns None if no such file is found.
    """
    for name in ("sample_submission.csv", "SampleSubmission.csv",
                 "sampleSubmission.csv", "sample_submission.csv.gz"):
        p = data_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p, nrows=5)
                return {
                    "file": name,
                    "columns": list(df.columns),
                    "sample_rows": [
                        {c: _native(v) for c, v in row.items()}
                        for row in df.head(5).to_dict(orient="records")
                    ],
                }
            except Exception:
                pass
    return None


def inspect_data_dir(data_dir: str | Path) -> dict:
    """
    Inspect all CSV files in data_dir.
    Returns {"train.csv": {...}, "test.csv": {...}, ...} plus a submission_format key.
    """
    data_dir = Path(data_dir)
    result: dict = {}

    for p in sorted(data_dir.glob("**/*.csv")):
        rel = p.relative_to(data_dir)
        result[str(rel)] = inspect_csv(p)

    sub_fmt = get_submission_format(data_dir)
    if sub_fmt:
        result["_submission_format"] = sub_fmt

    return result
