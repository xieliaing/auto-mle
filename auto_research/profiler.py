"""
Sample-level profiling skill.

Compares per-sample predictions between two consecutive model iterations to
identify three groups:
  - fixed:            samples that changed wrong→right
  - regressed:        samples that changed right→wrong
  - persistent_errors: still wrong in both iterations

Uses Claude to extract actionable linguistic/semantic patterns from each group
so the proposer can write hypothesis that target real failure modes.
"""
from __future__ import annotations

import json
from typing import List, Optional

PROFILER_MODEL = "claude-sonnet-4-5"
_MAX_SAMPLES_PER_GROUP = 20


def compare_predictions(
    prev_preds: List[dict],
    curr_preds: List[dict],
) -> dict:
    """
    Align two per-sample prediction lists by sample_id and group them.

    Each prediction dict must have at minimum:
        sample_id (str), correct (bool), label (int), prediction (int)
    """
    prev_by_id = {p["sample_id"]: p for p in prev_preds}
    curr_by_id = {p["sample_id"]: p for p in curr_preds}
    common = set(prev_by_id) & set(curr_by_id)

    fixed: List[dict] = []
    regressed: List[dict] = []
    persistent_errors: List[dict] = []
    unchanged_correct: List[dict] = []

    for sid in common:
        prev, curr = prev_by_id[sid], curr_by_id[sid]
        was_right, is_right = prev["correct"], curr["correct"]
        if not was_right and is_right:
            fixed.append(curr)
        elif was_right and not is_right:
            regressed.append(curr)
        elif not was_right and not is_right:
            persistent_errors.append(curr)
        else:
            unchanged_correct.append(curr)

    return {
        "fixed": fixed,
        "regressed": regressed,
        "persistent_errors": persistent_errors,
        "unchanged_correct": unchanged_correct,
        "total_compared": len(common),
    }


def _format_samples(samples: List[dict]) -> str:
    subset = samples[:_MAX_SAMPLES_PER_GROUP]
    lines = []
    for s in subset:
        label = "pos" if s.get("label") == 1 else "neg"
        pred = "pos" if s.get("prediction") == 1 else "neg"
        summary = s.get("input_summary", "")
        lines.append(f"  label={label} pred={pred}: {summary}")
    if len(samples) > _MAX_SAMPLES_PER_GROUP:
        lines.append(f"  … and {len(samples) - _MAX_SAMPLES_PER_GROUP} more")
    return "\n".join(lines) if lines else "  (none)"


def extract_patterns(groups: dict, task_description: str, client) -> dict:
    """
    Call Claude to identify patterns in each sample group.

    `client` is an anthropic.Anthropic instance. Returns a dict with keys:
    fixed_patterns, regressed_patterns, persistent_patterns, analysis_summary.
    """
    prompt = (
        f"You are analyzing prediction changes between two fine-tuned model "
        f"iterations for: {task_description}\n\n"
        f"FIXED (wrong→correct, n={len(groups['fixed'])}):\n"
        f"{_format_samples(groups['fixed'])}\n\n"
        f"REGRESSED (correct→wrong, n={len(groups['regressed'])}):\n"
        f"{_format_samples(groups['regressed'])}\n\n"
        f"PERSISTENT ERRORS (still wrong, n={len(groups['persistent_errors'])}):\n"
        f"{_format_samples(groups['persistent_errors'])}\n\n"
        "Identify 2-4 concrete, actionable patterns per group. Explain WHY the "
        "model struggles with each and WHAT training signal could address it.\n\n"
        "Respond in JSON only (no markdown):\n"
        '{"fixed_patterns": ["..."], "regressed_patterns": ["..."], '
        '"persistent_patterns": ["..."], "analysis_summary": "2-3 sentences"}'
    )

    resp = client.messages.create(
        model=PROFILER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if "```" in text:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        else:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def profile_iterations(
    prev_preds: List[dict],
    curr_preds: List[dict],
    prev_exp_id: str,
    curr_exp_id: str,
    task_description: str,
    client: Optional[object] = None,
) -> dict:
    """
    Full pipeline: compare predictions → extract patterns → return profile dict.

    If `client` is None, pattern extraction is skipped (stats-only mode).
    """
    groups = compare_predictions(prev_preds, curr_preds)

    if client is not None:
        try:
            patterns = extract_patterns(groups, task_description, client)
        except Exception as e:
            patterns = {
                "fixed_patterns": [],
                "regressed_patterns": [],
                "persistent_patterns": [],
                "analysis_summary": f"Pattern extraction failed: {e}",
            }
    else:
        patterns = {
            "fixed_patterns": [],
            "regressed_patterns": [],
            "persistent_patterns": [],
            "analysis_summary": (
                f"Fixed {len(groups['fixed'])}, regressed {len(groups['regressed'])}, "
                f"persistent errors {len(groups['persistent_errors'])}. "
                "(No LLM client — stats-only mode.)"
            ),
        }

    return {
        "prev_exp_id": prev_exp_id,
        "curr_exp_id": curr_exp_id,
        "stats": {
            "fixed": len(groups["fixed"]),
            "regressed": len(groups["regressed"]),
            "persistent_errors": len(groups["persistent_errors"]),
            "unchanged_correct": len(groups["unchanged_correct"]),
            "total_compared": groups["total_compared"],
        },
        **patterns,
    }
