"""
Generate a competition-aware baseline model using Claude.

Claude reads the competition overview, evaluation metric, and data schema, then
reasons about the problem before choosing and implementing an appropriate approach.
The output is NOT a generic sklearn/XGBoost template — it reflects Claude's
understanding of this specific competition.

Two sources of baselines are supported:
  1. Human-provided  — user writes baseline.py, data.py, evaluator.py manually
  2. AI-generated    — this module calls Claude to generate them

Calls the Anthropic Messages API directly via requests (no anthropic package needed).

Generated files (written to output_dir):
  baseline.py   — standalone model script (approach chosen by Claude)
  data.py       — AutoMLE data module (text framing for LLM fine-tuning)
  evaluator.py  — AutoMLE evaluator module
  _reasoning.txt — Claude's stated reasoning before the code
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import requests

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 8192


def _call_claude(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: set ANTHROPIC_API_KEY=sk-ant-..."
        )
    resp = requests.post(
        _ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


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
AutoMLE evaluator module using PEFT/transformers.
Implement:
```
PRIMARY_METRIC = "<name matching the competition metric>"

def evaluate(adapter_dir, model_name, eval_data, **kwargs) -> dict:
    # Load the fine-tuned LLM, run inference on eval_data rows,
    # parse text output into predictions, compute the metric.
    # Return {{"<PRIMARY_METRIC>": float, "n_eval": int}}
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
) -> dict[str, Path]:
    """
    Ask Claude to reason about the competition and generate baseline.py,
    data.py, and evaluator.py. Writes all files to output_dir.

    Returns {filename: Path} for the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_copy = json.loads(json.dumps(data_schema))
    prompt = _build_prompt(competition_info, schema_copy)

    print("[baseline_gen] Sending competition context to Claude for analysis...")
    response = _call_claude(prompt)

    # Save full raw response
    raw_path = output_dir / "_claude_response.txt"
    raw_path.write_text(response, encoding="utf-8")

    reasoning, blocks = _split_reasoning_and_blocks(response)

    # Save reasoning separately for human review
    if reasoning:
        reasoning_path = output_dir / "_reasoning.txt"
        reasoning_path.write_text(reasoning, encoding="utf-8")
        print(f"[baseline_gen] Claude's reasoning → {reasoning_path.name}")
        # Print reasoning so user can see Claude's competition analysis
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
