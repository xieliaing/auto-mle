"""
Skill proposer: asks an LLM to suggest new experiment directions based on
defect pattern analysis and the accumulated learning journal.

Grounded in ML/statistics literature; each skill targets an observed pattern.
Uses the same direct-HTTP approach as baseline_gen.py (no anthropic package needed).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}
_MAX_TOKENS = 2000

_PROMPT_TEMPLATE = """\
You are an experienced ML engineer advising on a Kaggle competition experiment loop.

## Task context
{task_context}

## Current defect analysis (this iteration vs previous)
{defect_summary}

## Learning journal (accumulated knowledge across all iterations)
{journal_summary}

---

Based on the defect patterns above, propose 3-5 concrete experiment skills to try next.
Ground each skill in established ML/statistics knowledge (e.g., class imbalance techniques,
feature engineering, regularisation, ensembling, calibration, data augmentation, resampling,
threshold tuning, ordinal encoding, interaction features, target encoding, etc.).

Each skill should directly address one or more of the observed defect patterns.

Respond with ONLY a JSON array. No markdown fences, no commentary.

Schema per element:
{{
  "name": "short_snake_case_id",
  "rationale": "why this targets the observed pattern (1-2 sentences)",
  "ml_basis": "statistical/ML grounding, e.g. SMOTE, Platt scaling, isotonic regression",
  "suggested_changes": "concrete what-to-change in data.py / evaluator.py / baseline.py (1-2 sentences)",
  "priority": "high | medium | low"
}}
"""


def _call_ai(
    prompt: str,
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> str:
    if provider == "anthropic":
        resp = requests.post(
            _ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": _MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    else:
        base = base_url or _OPENAI_DEFAULT_BASE
        resp = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={
                "content-type": "application/json",
                "Authorization": f"Bearer {api_key or 'local'}",
            },
            json={
                "model": model,
                "max_tokens": _MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def propose_skills(
    defect_summary: str,
    journal_summary: str,
    task_context: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Ask an LLM to propose experiment skills based on defect patterns.
    Returns [] on any failure (non-blocking — the loop continues either way).
    """
    resolved_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o")
    resolved_key = api_key or (
        os.environ.get("ANTHROPIC_API_KEY")
        if provider == "anthropic"
        else os.environ.get("OPENAI_API_KEY", "")
    )
    if not resolved_key and provider == "anthropic":
        print("[skill_proposer] ANTHROPIC_API_KEY not set — skipping skill proposals.")
        return []

    prompt = _PROMPT_TEMPLATE.format(
        task_context=task_context,
        defect_summary=defect_summary,
        journal_summary=journal_summary,
    )

    try:
        raw = _call_ai(
            prompt, provider=provider, model=resolved_model,
            api_key=resolved_key, base_url=base_url,
        )
    except Exception as e:
        print(f"[skill_proposer] LLM call failed: {type(e).__name__}: {e}")
        return []

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        print(f"[skill_proposer] No JSON array found in response: {raw[:200]!r}")
        return []

    try:
        skills = json.loads(raw[start: end + 1])
    except json.JSONDecodeError as e:
        print(f"[skill_proposer] JSON parse error: {e}")
        return []

    if not isinstance(skills, list):
        return []

    return [s for s in skills if isinstance(s, dict) and "name" in s and "rationale" in s]
