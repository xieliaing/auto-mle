"""
Manages task lifecycle: git branch creation, folder setup, and metadata.

A "task" is identified by a `key` (user-supplied or auto-generated). The
first invocation for a key creates:
  - feature branch  feature/<key>
  - tasks/<key>/  containing the user's data.py, evaluator.py, and a copy
    of auto_research/trainer.py that the agent can freely modify.

Subsequent invocations on the same key reuse the folder but create a new
runs/<timestamp>/ subfolder under it for that run's results.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import string
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
TASKS_ROOT = REPO_ROOT / "tasks"
TEMPLATE_TRAINER = REPO_ROOT / "auto_research" / "trainer.py"
KEY_RE = re.compile(r"[^a-z0-9_-]+")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + list(args), capture_output=True, text=True, check=check)


def _is_git_repo() -> bool:
    return _git("rev-parse", "--is-inside-work-tree", check=False).returncode == 0


def _has_commits() -> bool:
    return _git("rev-parse", "HEAD", check=False).returncode == 0


def _current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def _branch_exists(name: str) -> bool:
    return _git("rev-parse", "--verify", name, check=False).returncode == 0


def ensure_git_repo() -> bool:
    """Initialize a git repo and make an initial commit if one does not exist."""
    if _is_git_repo() and _has_commits():
        return False
    if not _is_git_repo():
        _git("init")
        print("[git] Initialized new repository.")
    if not _has_commits():
        _git("add", "-A")
        result = _git("commit", "-m", "Initial commit: automle parent base", check=False)
        if result.returncode != 0:
            _git("commit", "--allow-empty", "-m", "Initial commit: automle parent base")
        print("[git] Created initial commit on parent base.")
    return True


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def sanitize_key(raw: str) -> str:
    """Lowercase, replace non-[a-z0-9_-] with '_'. Strip leading/trailing '_-'."""
    k = KEY_RE.sub("_", raw.strip().lower())
    return k.strip("_-") or "task"


def auto_generate_key() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"task_{suffix}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_task(
    key: Optional[str],
    data_file: str,
    evaluator_file: str,
    checkpoint: str,
) -> dict:
    """
    Create or reuse a task folder for `key`. Returns a dict with:
      key, branch, task_folder, run_folder (timestamped),
      data_file, evaluator_file, trainer_file, created (bool)
    """
    ensure_git_repo()

    final_key = sanitize_key(key) if key else auto_generate_key()
    branch_name = f"feature/{final_key}"
    task_folder = TASKS_ROOT / final_key
    created = not task_folder.exists()

    # Branch management
    if _branch_exists(branch_name):
        if _current_branch() != branch_name:
            _git("checkout", branch_name)
            print(f"[git] Switched to existing branch: {branch_name}")
    else:
        _git("checkout", "-b", branch_name)
        print(f"[git] Created branch: {branch_name}")

    # Folder + file setup (only on first creation)
    task_folder.mkdir(parents=True, exist_ok=True)
    data_dest = task_folder / "data.py"
    eval_dest = task_folder / "evaluator.py"
    trainer_dest = task_folder / "trainer.py"

    if created:
        shutil.copy2(data_file, data_dest)
        shutil.copy2(evaluator_file, eval_dest)
        shutil.copy2(TEMPLATE_TRAINER, trainer_dest)
        (task_folder / "__init__.py").write_text("")
        meta = {
            "key": final_key,
            "branch": branch_name,
            "checkpoint": checkpoint,
            "data_file_source": str(Path(data_file).resolve()),
            "evaluator_file_source": str(Path(evaluator_file).resolve()),
            "trainer_template_source": str(TEMPLATE_TRAINER.resolve()),
            "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (task_folder / "meta.json").write_text(json.dumps(meta, indent=2))
        _git("add", str(task_folder))
        _git("commit", "-m", f"Setup task: {final_key}")
        print(f"[task] Created tasks/{final_key}/ with data.py, evaluator.py, trainer.py")
    else:
        print(f"[task] Reusing existing tasks/{final_key}/")
        # Sanity: if the user passed files but they differ from what's saved, warn.
        for src, dest, label in [
            (data_file, data_dest, "data.py"),
            (evaluator_file, eval_dest, "evaluator.py"),
        ]:
            if Path(src).resolve() != dest.resolve() and Path(src).exists():
                if dest.exists() and Path(src).read_bytes() != dest.read_bytes():
                    print(f"[task] WARNING: provided {label} differs from tasks/{final_key}/{label}; "
                          f"keeping existing copy. Delete it manually if you want a refresh.")

    # Per-run timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = task_folder / "runs" / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)

    return {
        "key": final_key,
        "branch": branch_name,
        "task_folder": str(task_folder),
        "run_folder": str(run_folder),
        "data_file": str(data_dest),
        "evaluator_file": str(eval_dest),
        "trainer_file": str(trainer_dest),
        "created": created,
    }
