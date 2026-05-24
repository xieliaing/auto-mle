"""
Generate a competition-aware baseline model using an AI model.

The AI reads the competition overview, evaluation metric, and data schema, then
reasons about the problem before choosing and implementing an appropriate approach.
The output is NOT a generic sklearn/XGBoost template — it reflects the model's
understanding of this specific competition.

Two sources of baselines are supported:
  1. Human-provided  — user writes baseline.py, data.py, evaluator.py manually
  2. AI-generated    — this module calls an AI to generate them

Supported providers (all via direct HTTP, no extra packages needed):
  anthropic        — Anthropic Claude API (default); requires ANTHROPIC_API_KEY
  openai           — OpenAI or any OpenAI-compatible API; requires OPENAI_API_KEY
                     Pass --ai-base-url to target a different endpoint, e.g.:
                       https://api.groq.com/openai/v1  (Groq)
                       http://localhost:11434/v1        (Ollama)
                       http://localhost:1234/v1         (LM Studio)

Generated files (written to output_dir):
  baseline.py    — standalone model script (approach chosen by the AI)
  data.py        — AutoMLE data module (text framing for LLM fine-tuning)
  evaluator.py   — AutoMLE evaluator module
  _reasoning.txt — AI's stated reasoning before the code
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import requests

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}

_MAX_TOKENS = 8192


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
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


def _call_openai_compat(prompt: str, model: str, api_key: str, base_url: str) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
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


def _resolve_key(provider: str, explicit_key: str | None) -> str:
    if explicit_key is not None:
        return explicit_key
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    return os.environ.get(env_var, "")


def _call_ai(
    prompt: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    resolved_key = _resolve_key(provider, api_key)
    resolved_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "anthropic":
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set it with: set ANTHROPIC_API_KEY=sk-ant-...\n"
                "Or switch provider: --ai-provider openai with OPENAI_API_KEY."
            )
        return _call_anthropic(prompt, resolved_model, resolved_key)

    # openai / openai-compatible (Groq, Together, Ollama, LM Studio, ...)
    resolved_base = base_url or _OPENAI_DEFAULT_BASE
    is_local = resolved_base != _OPENAI_DEFAULT_BASE
    if not resolved_key and not is_local:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Set it with: set OPENAI_API_KEY=sk-...\n"
            "For local servers (Ollama, LM Studio) pass --ai-base-url and optionally --ai-api-key."
        )
    return _call_openai_compat(prompt, resolved_model, resolved_key, resolved_base)


def _build_prompt(info: dict, schema: dict) -> str:
    description = (info.get("description") or "")[:4000]
    csv_files = [f["name"] for f in info.get("files", []) if f["name"].endswith(".csv")]

    sub_fmt = schema.pop("_submission_format", None)
    sub_section = ""
    if sub_fmt:
        sub_section = f"""
## Submission Format (from {sub_fmt['file']})
Required columns: {sub_fmt['columns']}
Sample rows:
{json.dumps(sub_fmt['sample_rows'], indent=2)}
"""

    schema_str = json.dumps(schema, indent=2)

    return f"""\
You are an expert Kaggle competitor preparing a baseline solution for a new competition.
Your goal is to understand the problem deeply, then write code that reflects that understanding —
not to apply a generic template.

## Competition: {info['title']}
URL: {info['competition_url']}

## Overview / Description
{description}

## Evaluation Metric
{info['evaluation_metric']}

## Available Data Files
{json.dumps(csv_files, indent=2)}
{sub_section}
## Data Schema (inspected from downloaded files, up to 1 000 rows)
{schema_str}

---

**Step 1 — Reason first (required)**

Before writing any code, answer these questions in plain text:
1. What is the prediction task? (classification / regression / ranking / other) What is the target?
2. What does the evaluation metric reward or penalise? Any edge cases to handle?
3. What do the features mean in the real-world domain? Which are likely most predictive?
4. What modelling approach makes most sense given the above — and why?
   (There is no requirement to use sklearn or gradient boosting. Choose what fits the problem.)
5. What preprocessing steps are necessary given the data types and missing value patterns?

**Step 2 — Generate THREE Python files**

After your reasoning, output exactly three fenced code blocks with filename comments.

### FILE 1: baseline.py
A standalone, runnable script that trains on train data and writes submission.csv.
- Reads data from `data/train.csv` and `data/test.csv` (override with `--train` / `--test` CLI args)
- Implements the approach you chose in Step 1 (not a generic pipeline)
- Prints cross-validation score in the competition's metric
- Saves predictions to `submission.csv` in the exact required format
- Include only standard or widely-available packages

### FILE 2: data.py
AutoMLE data module — frames the problem as text for LLM fine-tuning experiments.
Implement these exact signatures:
```
def load_train_dataset(seed=42, subset=None, balance="none", **kwargs)
    # kwargs["train_path"] or env AUTOMLE_TRAIN_DATA → path to train CSV
    # Returns torch.utils.data.Dataset
    # Each item: {{"messages": [{{"role": "user", "content": "..."}},
    #                           {{"role": "assistant", "content": "<answer>"}}],
    #              "label": <numeric value>}}
    # Format: describe the row in natural language in the user turn,
    #         put the answer (the target value) in the assistant turn.

def load_eval_dataset(seed=42, subset=None, **kwargs)
    # kwargs["eval_path"] or env AUTOMLE_EVAL_DATA
    # Returns a pandas DataFrame (evaluator reads rows directly)

def get_collator(tokenizer, max_seq_len)
    # Returns a collator that masks loss to answer tokens only
```

### FILE 3: evaluator.py
AutoMLE evaluator module.
Implement:
```
PRIMARY_METRIC = "<name matching the competition metric>"

def evaluate(adapter_dir, model_name, eval_data, **kwargs) -> dict:
    # kwargs may contain:
    #   predictions_path  — optional str path; save predictions CSV there if provided
    #   train_data        — training data (for traditional ML: train here, then eval)
    #   config            — current ExperimentConfig dict (extract model params from it)
    #   output_dir        — run directory (str)
    #
    # For traditional ML: ignore adapter_dir/model_name; use train_data + config.
    # For LLM fine-tuning: load the fine-tuned model from adapter_dir.
    #
    # REQUIRED: build a predictions DataFrame and return it as "predictions_df":
    #   import pandas as pd
    #   predictions_df = pd.DataFrame({{
    #       "sample_id":  list(range(len(eval_data))),  # or eval_data.index
    #       "true_label": true_labels,                  # ground-truth class labels
    #       "pred_label": predicted_labels,             # predicted class labels
    #       "confidence": positive_class_proba,         # P(class=1), float 0-1
    #                                                   # use model.predict_proba()[:,1]
    #                                                   # or decision_function() if no proba
    #   }})
    #   # "confidence" enables score-band stratified sampling in defect analysis.
    #
    # Return:
    #   {{
    #     "<PRIMARY_METRIC>": float,
    #     "n_eval": int,
    #     "predictions_df": predictions_df,   # required for defect analysis
    #   }}
```

---

Begin your response with the Step 1 reasoning, then the three code blocks in this order:

```python
# baseline.py
...
```

```python
# data.py
...
```

```python
# evaluator.py
...
```
"""


def _split_reasoning_and_blocks(text: str) -> tuple[str, dict[str, str]]:
    """
    Separate the prose reasoning from the code blocks.
    Returns (reasoning_text, {filename: code}).
    """
    # Find position of first code block
    first_block = re.search(r"```", text)
    reasoning = text[: first_block.start()].strip() if first_block else text.strip()

    named: dict[str, str] = {}
    pattern = re.compile(
        r"```(?:python)?\s*\n\s*#\s*(\w[\w.\-]*\.py)\s*\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        named[m.group(1)] = m.group(2).strip()

    if len(named) < 3:
        unnamed = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
        order = ["baseline.py", "data.py", "evaluator.py"]
        for i, m in enumerate(unnamed.finditer(text)):
            if i >= len(order):
                break
            key = order[i]
            if key not in named:
                named[key] = m.group(1).strip()

    return reasoning, named


def generate_baseline(
    competition_info: dict,
    data_schema: dict,
    output_dir: str | Path,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Path]:
    """
    Ask an AI to reason about the competition and generate baseline.py,
    data.py, and evaluator.py. Writes all files to output_dir.

    provider: 'anthropic' (default) or 'openai' (covers any OpenAI-compatible API).
    model: overrides the provider default (claude-sonnet-4-6 / gpt-4o).
    api_key: overrides ANTHROPIC_API_KEY / OPENAI_API_KEY env vars.
    base_url: OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama.

    Returns {filename: Path} for the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o")
    provider_label = f"{resolved_model} ({provider})"

    schema_copy = json.loads(json.dumps(data_schema))
    prompt = _build_prompt(competition_info, schema_copy)

    print(f"[baseline_gen] Sending competition context to {provider_label} for analysis...")
    response = _call_ai(prompt, provider=provider, model=model, api_key=api_key, base_url=base_url)

    # Save full raw response
    raw_path = output_dir / "_ai_response.txt"
    raw_path.write_text(response, encoding="utf-8")

    reasoning, blocks = _split_reasoning_and_blocks(response)

    # Save reasoning separately for human review
    if reasoning:
        reasoning_path = output_dir / "_reasoning.txt"
        reasoning_path.write_text(reasoning, encoding="utf-8")
        print(f"[baseline_gen] Reasoning → {reasoning_path.name}")
        print("\n" + "─" * 60)
        print(reasoning[:1200] + ("…" if len(reasoning) > 1200 else ""))
        print("─" * 60 + "\n")

    if not blocks:
        raise ValueError(
            f"No code blocks found in Claude response.\n"
            f"Raw response saved to {raw_path}"
        )

    written: dict[str, Path] = {}
    for filename, code in blocks.items():
        path = output_dir / filename
        path.write_text(code, encoding="utf-8")
        print(f"[baseline_gen] Wrote {filename}")
        written[filename] = path

    return written
