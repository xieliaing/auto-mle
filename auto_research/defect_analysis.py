"""
Golden-set defect analysis: after each training run, collect the model's
defects on the public evaluation set, send them to a large LLM for analysis, and
get back an improvement plan (optionally with a concrete next config).

Evaluator contract extension (opt-in):
    evaluate(adapter_dir, model_name, eval_data, return_records=True, **kw)
    may add a "records" key to its returned dict: a list of per-example dicts

        {"id": ..., "input": ..., "expected": ..., "predicted": ...,
         "is_defect": bool, "score": float (optional)}

    The evaluator decides what counts as a defect (wrong label, residual over
    a threshold, judge verdict, ...), so this module stays task-agnostic.

    A record may also carry "image": a local image path. The sampled defect
    examples' images are attached to the prompt (downscaled, base64), so a
    vision-capable analyst can see what the model got wrong.

Providers: "anthropic" (ANTHROPIC_API_KEY), "openai" (OPENAI_API_KEY), or
"local": any OpenAI-compatible server (vLLM, SGLang, Ollama, llama.cpp) at
base_url / $ANALYSIS_BASE_URL, e.g. a model served on another machine in the
LAN; no key needed ($ANALYSIS_API_KEY if the server wants one), and the model
defaults to the first one the server lists.
Without a key (or base_url for "local") the prompt is still written to disk and
the LLM call is skipped, so the loop keeps running and the prompt can be inspected.

Files written per run under <run_dir>/defect_analysis/:
    defects.json   summary + sampled defect examples
    prompt.md      exact system + user prompt sent (or that would be sent)
    plan.json      parsed LLM response
    plan.md        human-readable improvement plan
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o",
}
API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
MAX_TOKENS = 16000
LOCAL_TIMEOUT_SEC = 1800  # local models can think for minutes
IMAGE_MAX_SIDE = 512      # defect images are downscaled to this before sending


class LLMUnavailable(RuntimeError):
    """No API key / SDK for the chosen provider — analysis is skipped, not failed."""


# --- Defect summary -----------------------------------------------------------

def _truncate(value, max_chars: int):
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return value if len(s) <= max_chars else s[:max_chars] + " …[truncated]"


def summarize_defects(records: List[dict], max_examples: int = 50,
                      max_chars: int = 1000, seed: int = 0) -> dict:
    """Aggregate counts + a capped, truncated sample of defect examples."""
    defects = [r for r in records if r.get("is_defect")]
    pairs = Counter(
        f"expected={_truncate(r.get('expected'), 80)} -> predicted={_truncate(r.get('predicted'), 80)}"
        for r in defects
    )
    sample = defects
    if len(defects) > max_examples:
        sample = random.Random(seed).sample(defects, max_examples)
    examples = [
        {k: (_truncate(v, max_chars) if k in ("input", "expected", "predicted") else v)
         for k, v in r.items() if k != "is_defect"}
        for r in sample
    ]
    return {
        "n_total": len(records),
        "n_defects": len(defects),
        "defect_rate": round(len(defects) / max(len(records), 1), 4),
        "top_confusions": dict(pairs.most_common(10)),
        "n_examples_shown": len(examples),
        "examples": examples,
    }


# --- Prompt -------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ML research advisor for an autonomous LoRA fine-tuning loop.

After each training run you receive the model's defects (failed examples) on the public evaluation set. Your job: find the systematic patterns behind the defects, explain their likely causes, and produce a prioritized improvement plan.

Be concrete and evidence-based: cite the example ids or confusion counts that support each pattern. Distinguish actions the loop can execute itself (changes to the LoRA / training / data-balance / loss knobs in the config) from actions that need a human (relabeling, new data, prompt or code changes).

Respond with ONLY a single JSON object. No markdown, no commentary, no code fences."""


USER_TEMPLATE = """## Task
LoRA fine-tuning of base model: {model_name}. Primary metric: {primary_metric}.
Training hardware: {hardware}.

## Run just completed
exp_id: {exp_id}
hypothesis: {hypothesis}
metrics: {metrics}
config:
{config}

## Public-set defect summary
{summary}

## Defect examples (sampled{sample_note})
{examples}{image_note}

## Recent experiment history (most recent first)
{history}

## Response schema
{{
  "analysis": "2-4 sentences: overall read of the defects",
  "defect_patterns": [
    {{"pattern": "...", "evidence": "example ids / counts", "likely_cause": "...", "share_of_defects": "rough %"}}
  ],
  "improvement_plan": [
    {{"action": "...", "rationale": "...", "priority": "high | medium | low",
      "executable_by_loop": true or false}}
  ],
  "next_config": null or a full ExperimentConfig object with the same shape as the config above
}}

## Rules for next_config
- Set it only if a config change is the best next step; otherwise null.
- Same keys as the config above. Allowed values: lora.target_modules in qv|attn|all_linear; train.lr_scheduler in cosine|linear|constant; data.balance in none|downsample|upsample; loss.name in ce|label_smoothing|focal|weighted_ce.
- Use a new unique snake_case exp_id and a one-sentence hypothesis tied to a defect pattern.
- Keep tier="small" unless there is strong evidence to promote.
- Prefer ONE concrete change vs the run above so the effect can be attributed.

Now output the JSON."""


def _format_history(history: List[dict], limit: int = 10) -> str:
    if not history:
        return "(none)"
    lines = []
    for r in list(reversed(history))[:limit]:
        lines.append(json.dumps({
            "exp_id": r["exp_id"],
            "accuracy": round(r.get("accuracy", 0.0), 4),
            "status": r.get("status", "success"),
            "hypothesis": r.get("config", {}).get("hypothesis", ""),
        }))
    return "\n".join(lines)


def build_prompt(result: dict, summary: dict, history: List[dict],
                 model_name: str, primary_metric: str, hardware: str = "",
                 n_images: int = 0) -> str:
    cfg = result["config"]
    metrics = {primary_metric: result.get("accuracy"), **result.get("extra", {})}
    examples = "\n".join(json.dumps(e, default=str) for e in summary["examples"]) or "(no defects)"
    stats = {k: v for k, v in summary.items() if k != "examples"}
    return USER_TEMPLATE.format(
        model_name=model_name,
        primary_metric=primary_metric,
        hardware=hardware or "unspecified",
        exp_id=result["exp_id"],
        hypothesis=cfg.get("hypothesis", ""),
        metrics=json.dumps(metrics, default=str),
        config=json.dumps(cfg, indent=2),
        summary=json.dumps(stats, indent=2),
        sample_note=f", {summary['n_examples_shown']} of {summary['n_defects']}",
        examples=examples,
        image_note=(f"\n\nThe images of {n_images} of these examples are attached after this "
                    "prompt, each preceded by its id." if n_images else ""),
        history=_format_history(history),
    )


# --- LLM call -----------------------------------------------------------------

def _encode_image(path: str) -> Optional[str]:
    """Downscaled JPEG of a local image as base64, or None if it can't be read."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"  [defects] could not attach image {path}: {type(e).__name__}: {e}")
        return None


def _image_parts(images: List[tuple], provider: str) -> list:
    """Content parts for (caption, base64_jpeg) pairs in the provider's format."""
    parts = []
    for caption, b64 in images:
        parts.append({"type": "text", "text": caption})
        if provider == "anthropic":
            parts.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": b64}})
        else:
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
    return parts


def call_llm(system: str, user: str, provider: str = "anthropic",
             model: Optional[str] = None, base_url: Optional[str] = None,
             images: Optional[List[tuple]] = None) -> str:
    """Send one system+user prompt (plus optional (caption, base64_jpeg) images)
    and return the text reply.

    Raises LLMUnavailable if the provider's API key / base_url or SDK is missing.
    """
    if provider not in ("anthropic", "openai", "local"):
        raise ValueError(f"Unknown analysis provider {provider!r}; use anthropic, openai or local")
    if provider == "local":
        base_url = base_url or os.environ.get("ANALYSIS_BASE_URL")
        if not base_url:
            raise LLMUnavailable("no base_url for the local provider (set ANALYSIS_BASE_URL)")
    elif not os.environ.get(API_KEY_ENV[provider]):
        raise LLMUnavailable(f"{API_KEY_ENV[provider]} not set")
    model = model or os.environ.get("ANALYSIS_MODEL") or DEFAULT_MODELS.get(provider)
    content = [{"type": "text", "text": user}] + _image_parts(images or [], provider)

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic package not installed; pip install anthropic") from e
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
            # Passed via extra_body so older SDK versions accept them.
            # fallbacks: if the model declines, the API re-runs the request on
            # a fallback model within the same call.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"output_config": {"effort": "high"}, "fallbacks": "default"},
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"analysis request refused: {resp.stop_details}")
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()

    try:
        import openai
    except ImportError as e:
        raise LLMUnavailable("openai package not installed; pip install openai") from e
    if provider == "local":
        client = openai.OpenAI(base_url=base_url, timeout=LOCAL_TIMEOUT_SEC,
                               api_key=os.environ.get("ANALYSIS_API_KEY", "EMPTY"))
        model = model or client.models.list().data[0].id
        # No max_tokens: local servers reject prompt + max_tokens > context length,
        # and default to the remaining context when it's omitted.
        limit = {}
    else:
        client = openai.OpenAI()
        limit = {"max_tokens": MAX_TOKENS}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        **limit,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(raw: str) -> dict:
    # Reasoning models served without a reasoning parser inline their thoughts.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"Could not find JSON in analysis response: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


# --- Report -------------------------------------------------------------------

def format_plan_md(plan: dict, exp_id: str) -> str:
    lines = [f"# Defect analysis: {exp_id}", "", plan.get("analysis", ""), "", "## Defect patterns"]
    for p in plan.get("defect_patterns", []):
        lines.append(f"- **{p.get('pattern', '')}** ({p.get('share_of_defects', '?')}) — "
                     f"cause: {p.get('likely_cause', '')}; evidence: {p.get('evidence', '')}")
    lines += ["", "## Improvement plan"]
    for a in plan.get("improvement_plan", []):
        who = "loop" if a.get("executable_by_loop") else "human"
        lines.append(f"- [{a.get('priority', '?')}/{who}] {a.get('action', '')} — {a.get('rationale', '')}")
    lines += ["", "## Next config"]
    nc = plan.get("next_config")
    lines.append(f"```json\n{json.dumps(nc, indent=2)}\n```" if nc else "(none proposed)")
    return "\n".join(lines) + "\n"


def plan_context(plan: dict) -> str:
    """Short plan digest to pass to the experiment proposer as extra context."""
    actions = [a for a in plan.get("improvement_plan", []) if a.get("executable_by_loop")]
    if not actions:
        return ""
    lines = ["## Improvement plan from public-set defect analysis (address these in your next hypothesis)"]
    lines += [f"- [{a.get('priority', '?')}] {a.get('action', '')}: {a.get('rationale', '')}" for a in actions[:5]]
    return "\n".join(lines)


# --- Public entry point -------------------------------------------------------

def analyze_and_plan(
    run_dir: str | Path,
    result: dict,
    records: List[dict],
    history: List[dict],
    model_name: str,
    primary_metric: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    max_examples: int = 50,
    base_url: Optional[str] = None,
    hardware: str = "",
) -> Optional[dict]:
    """Summarize defects, ask the LLM for a plan, write artifacts.

    Returns the parsed plan dict, or None if the LLM was unavailable.
    """
    out = Path(run_dir) / "defect_analysis"
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize_defects(records, max_examples=max_examples)
    (out / "defects.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  [defects] public n={summary['n_total']} defects={summary['n_defects']} "
          f"({100 * summary['defect_rate']:.1f}%)")

    images = []
    for ex in summary["examples"]:
        b64 = _encode_image(ex["image"]) if ex.get("image") else None
        if b64:
            images.append((f"Defect id={ex.get('id')} (expected={ex.get('expected')}, "
                           f"predicted={ex.get('predicted')}):", b64))

    user = build_prompt(result, summary, history, model_name, primary_metric,
                        hardware=hardware, n_images=len(images))
    (out / "prompt.md").write_text(f"# System\n\n{SYSTEM_PROMPT}\n\n# User\n\n{user}\n", encoding="utf-8")

    try:
        raw = call_llm(SYSTEM_PROMPT, user, provider=provider, model=model,
                       base_url=base_url, images=images)
    except LLMUnavailable as e:
        print(f"  [defects] LLM analysis skipped ({e}); prompt saved to {out / 'prompt.md'}")
        return None

    plan = _parse_json(raw)
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out / "plan.md").write_text(format_plan_md(plan, result["exp_id"]), encoding="utf-8")
    print(f"  [defects] plan written to {out / 'plan.md'}")
    return plan
