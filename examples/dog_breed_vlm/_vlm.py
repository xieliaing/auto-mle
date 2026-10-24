"""
Loads kaggle/tasks/dog-breed-identification/vlm_lora.py as module `vlm`.

Its folder name has hyphens, so it can't be imported normally. Reusing it keeps
one copy of the dataset, collator, prompt, and data paths for the VLM task.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_NAME = "dog_breed_vlm_lora"
_PATH = (Path(__file__).resolve().parents[2]
         / "kaggle" / "tasks" / "dog-breed-identification" / "vlm_lora.py")

if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _mod
    _spec.loader.exec_module(_mod)
vlm = sys.modules[_NAME]
