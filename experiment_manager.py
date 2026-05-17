"""
Manages experiment lifecycle: git branch creation, folder setup, and metadata.

Each call to create_experiment():
  1. Ensures the repo has at least one commit (initializes git if needed)
  2. Creates a feature branch  feature/<name>-<timestamp>
  3. Creates experiments/<name>-<timestamp>/ with the required structure
  4. Copies eval and train files into it
  5. Writes meta.json and makes an initial commit
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["git"] + list(args), check=check)


def _is_git_repo() -> bool:
    result = _git("rev-parse", "--is-inside-work-tree", check=False)
    return result.returncode == 0


def _has_commits() -> bool:
    result = _git("rev-parse", "HEAD", check=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_git_repo() -> bool:
    """
    Initialize a git repo and make an initial commit if one does not exist.
    Returns True if initialization was performed, False if already a repo.
    """
    if _is_git_repo() and _has_commits():
        return False

    if not _is_git_repo():
        _git("init")
        print("[git] Initialized new repository.")

    if not _has_commits():
        # Stage everything and make the initial commit
        _git("add", "-A")
        result = _git(
            "commit", "-m", "Initial commit: automle parent base",
            check=False,
        )
        if result.returncode != 0:
            # Nothing to commit (empty repo) — create an empty root commit
            _git("commit", "--allow-empty", "-m", "Initial commit: automle parent base")
        print("[git] Created initial commit on parent base.")

    return True


def create_experiment(
    experiment_name: str,
    checkpoint: str,
    eval_file: str,
    train_file: str,
) -> dict:
    """
    Set up a new experiment:
      - Creates feature branch  feature/<experiment_name>-<timestamp>
      - Creates experiments/<experiment_name>-<timestamp>/
      - Copies eval_file and train_file into that folder
      - Writes meta.json
      - Makes an initial git commit for the experiment setup

    Returns a dict with:
      branch, folder, eval_file (new path), train_file (new path), created
    """
    ensure_git_repo()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"feature/{experiment_name}-{timestamp}"
    exp_folder = Path("experiments") / f"{experiment_name}-{timestamp}"

    # Create the feature branch
    _git("checkout", "-b", branch_name)
    print(f"[git] Created branch: {branch_name}")

    # Create folder structure
    (exp_folder / "runs").mkdir(parents=True, exist_ok=True)

    # Copy provided files
    eval_src = Path(eval_file)
    train_src = Path(train_file)
    eval_dest = exp_folder / eval_src.name
    train_dest = exp_folder / train_src.name
    shutil.copy2(eval_src, eval_dest)
    shutil.copy2(train_src, train_dest)

    # Write metadata
    meta = {
        "experiment_name": experiment_name,
        "branch": branch_name,
        "checkpoint": checkpoint,
        "eval_file": str(eval_dest),
        "train_file": str(train_dest),
        "folder": str(exp_folder),
        "created": timestamp,
    }
    (exp_folder / "meta.json").write_text(json.dumps(meta, indent=2))

    # Commit the experiment setup
    _git("add", str(exp_folder))
    _git("commit", "-m", f"Setup experiment: {experiment_name}")
    print(f"[git] Committed experiment setup to {branch_name}")

    return meta
