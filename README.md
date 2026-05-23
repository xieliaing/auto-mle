# AutoMLE — Autonomous LoRA Fine-Tuning Template

A reusable template for running autonomous LoRA fine-tuning experiments. You plug in a `data.py` and `evaluator.py` for your task; the pipeline handles the rest:

**Propose → Train → Evaluate → Analyze → Repeat.**

A built-in proposer (seed plan + Claude-driven from round 7 onward) searches across:
1. **Fine-tuning strategy** — LoRA rank, alpha, target modules, learning rate, scheduler, epochs.
2. **Data strategy** — class balancing, subset size (plus any task-specific knobs your `data.py` honors).
3. **Loss function** — standard CE, label smoothing, focal loss, weighted CE.

## Layout

```
AutoMLE/
├── run.py                        # entry point
├── experiment_manager.py         # branch + task folder setup
├── configs/default.yaml          # default paths, model, budgets
│
├── auto_research/                # task-agnostic template
│   ├── schema.py                 # ExperimentConfig + ExperimentResult dataclasses
│   ├── losses.py                 # CE / label-smooth / focal / weighted-CE
│   ├── proposer.py               # seed plan + Claude-driven proposer
│   ├── analyzer.py               # leaderboard.md + plateau detection
│   ├── orchestrator.py           # the autonomous loop (Phase A: explore, Phase B: promote)
│   └── trainer.py                # generic QLoRA trainer template (copied per task)
│
├── kaggle/                       # Kaggle competition integration
│   ├── auth.py                   # load ~/.kaggle/kaggle.json credentials
│   ├── fetcher.py                # fetch competition overview + data manifest via Kaggle API
│   ├── downloader.py             # download and extract competition data files
│   ├── inspector.py              # inspect CSV schema (columns, types, missing rates, samples)
│   ├── baseline_gen.py           # generate competition-aware baseline via AI (Claude, OpenAI, or local)
│   └── setup.py                  # orchestration CLI (auth → fetch → download → inspect → baseline)
│
├── examples/
│   └── product_comparison/       # reference task: text-only binary product matching
│       ├── data.py
│       ├── evaluator.py
│       ├── prompts.py
│       └── trainer.py
│
└── tasks/<key>/                  # created on demand per task
    ├── meta.json
    ├── data.py                   # copy of your provided data.py
    ├── evaluator.py              # copy of your provided evaluator.py
    ├── trainer.py                # copy of auto_research/trainer.py (agent-modifiable)
    └── runs/<timestamp>/
        ├── results.jsonl
        ├── leaderboard.md
        └── <exp_id>/
            ├── config.json
            └── adapter/
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set the proposer API key (or pass --no-llm-proposer to skip)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run with your task modules
python run.py \
    --key my-task \
    --checkpoint Qwen/Qwen3-1.7B \
    --data-file path/to/data.py \
    --evaluator-file path/to/evaluator.py \
    --budget 10 \
    --final-top-k 3
```

After it finishes:
- `tasks/<key>/runs/<timestamp>/leaderboard.md` — markdown ranking
- `tasks/<key>/runs/<timestamp>/results.jsonl` — one JSON line per run (config + metrics)
- `tasks/<key>/runs/<timestamp>/<exp_id>/adapter/` — the LoRA adapter for each run

If `--key` is omitted, a `task_<random>` key is auto-generated.

## Kaggle competitions

The `kaggle/` package sets up a Kaggle competition as an AutoMLE task in one command.

**Prerequisites**

1. Get your Kaggle API token: [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → API → Create New Token
2. Save the downloaded file to `~/.kaggle/kaggle.json`
3. Accept the competition rules on the Kaggle website (required before API downloads work)

**Run setup**

```bash
# Downloads data, inspects schema, generates a competition-aware baseline (uses Anthropic Claude by default)
python -m kaggle.setup --competition titanic

# Reuse already-downloaded data, skip baseline generation
python -m kaggle.setup --competition titanic --skip-download --skip-baseline

# Custom output location
python -m kaggle.setup --competition titanic --output-dir data/titanic
```

Output written to `kaggle/tasks/<competition>/`:

```
kaggle/tasks/titanic/
├── competition_info.json    # title, description, evaluation metric, file list
├── data_schema.json         # column types, missing rates, sample values per file
├── data/                    # train.csv, test.csv, sample_submission.csv, ...
└── baseline/
    ├── baseline.py          # competition-specific model (approach chosen by the AI)
    ├── data.py              # AutoMLE data module (text framing for LLM fine-tuning)
    ├── evaluator.py         # AutoMLE evaluator module
    ├── _reasoning.txt       # AI's analysis of the competition before generating code
    └── _ai_response.txt     # full raw response from the AI
```

**Baseline generation**

The baseline is not a generic template. The AI reads the competition overview and data schema, reasons about the problem (task type, metric semantics, feature domain meanings), then chooses an appropriate implementation. The reasoning is saved to `_reasoning.txt` for review.

A baseline can also be written by hand — `data.py` and `evaluator.py` just need to satisfy the [module contracts](#plugging-in-a-new-task) below.

**Which path to take after setup**

`kaggle.setup` always generates three files. Which ones you use depends on the competition type:

| Competition type | Right tool | Use |
|---|---|---|
| Tabular (classification, regression) | XGBoost, LightGBM, sklearn | `baseline.py` only |
| Time series forecasting | statsmodels, Prophet, XGBoost | `baseline.py` only |
| Text: classification, NER, Q&A, summarization | LLM fine-tuning (LoRA) | `data.py` + `evaluator.py` → `run.py` |
| Vision-language, document understanding | VLLM fine-tuning (LoRA) | `data.py` + `evaluator.py` → `run.py` |
| Any prompt-in / token-out problem | LLM fine-tuning (LoRA) | `data.py` + `evaluator.py` → `run.py` |

LLMs are not competitive on structured numeric inputs. For tabular and time-series competitions, run `baseline.py` directly and iterate on feature engineering and model selection there — the LoRA loop adds no value. Reserve the AutoMLE fine-tuning loop for competitions where the input can be expressed as a natural-language prompt and the output is either generated text or scored via token-level logits (e.g. `logit[Yes]` vs `logit[No]` at the answer position).

**AI provider options**

Baseline generation supports Anthropic Claude (default), OpenAI, and any OpenAI-compatible API including local servers. The `openai` provider covers all of these — only the base URL differs.

```bash
# Anthropic Claude (default) — set ANTHROPIC_API_KEY
python -m kaggle.setup --competition titanic

# OpenAI GPT-4o — set OPENAI_API_KEY
python -m kaggle.setup --competition titanic --ai-provider openai

# OpenAI with a specific model
python -m kaggle.setup --competition titanic --ai-provider openai --ai-model o3-mini

# Groq or other OpenAI-compatible cloud — set OPENAI_API_KEY (or --ai-api-key)
python -m kaggle.setup --competition titanic \
    --ai-base-url https://api.groq.com/openai/v1 \
    --ai-model llama-3.3-70b-versatile

# Ollama (local, no API key required)
python -m kaggle.setup --competition titanic \
    --ai-base-url http://localhost:11434/v1 \
    --ai-model llama3.2 \
    --ai-api-key local

# LM Studio (local)
python -m kaggle.setup --competition titanic \
    --ai-base-url http://localhost:1234/v1 \
    --ai-model local-model
```

> `--ai-base-url` automatically implies `--ai-provider openai`. For local servers, passing any non-empty string as `--ai-api-key` (e.g. `local`) satisfies the `Authorization: Bearer` header without needing a real key.

**Plug into AutoMLE**

For competitions where the input is naturally a prompt (text classification, NER, Q&A, summarization, VQA, document understanding, etc.), feed `data.py` and `evaluator.py` into the LoRA fine-tuning loop. The `data.py` module frames each row as a conversational turn; the `evaluator.py` module scores predictions either by parsing generated text or by reading `logit[Yes]` vs `logit[No]` at the answer token position.

```bash
# Example: a text-native competition (sentiment, NLP classification, VQA, ...)
python run.py \
    --key <competition-slug> \
    --checkpoint Qwen/Qwen3-1.7B \
    --data-file kaggle/tasks/<competition-slug>/baseline/data.py \
    --evaluator-file kaggle/tasks/<competition-slug>/baseline/evaluator.py
```

For tabular or time-series competitions, skip this step — run `baseline/baseline.py` directly instead.

## Try the reference task

The `examples/product_comparison/` folder is a working task using the new interface — a binary product-matching task with text-only QLoRA. To run it against the example with your own CSVs:

```bash
export AUTOMLE_TRAIN_DATA=path/to/train_pairs.csv
export AUTOMLE_EVAL_DATA=path/to/test_pairs.csv

python run.py \
    --key product-comparison \
    --checkpoint Qwen/Qwen3-1.7B \
    --data-file examples/product_comparison/data.py \
    --evaluator-file examples/product_comparison/evaluator.py
```

The example expects CSVs with columns `title1`, `title2`, `Label` (binary).

## Plugging in a new task

Three module contracts the orchestrator expects:

**`data.py`**
```python
def load_train_dataset(seed: int = 42, subset: int | None = None,
                       balance: str = "none", **kwargs) -> torch.utils.data.Dataset:
    ...

def load_eval_dataset(seed: int = 42, subset: int | None = None, **kwargs):
    ...  # returns whatever evaluator.evaluate's `eval_data` expects

def get_collator(tokenizer, max_seq_len: int):   # optional
    ...
```

**`evaluator.py`**
```python
PRIMARY_METRIC = "accuracy"   # optional; name of the key in the returned dict to optimize

def evaluate(adapter_dir: str, model_name: str, eval_data, **kwargs) -> dict:
    ...  # must include the primary metric (and optionally 'n_eval', etc.)
```

**`trainer.py`** is copied from `auto_research/trainer.py` into `tasks/<key>/` on first init. The agent may edit the task's local copy to add task-specific masking, callbacks, etc. Public contract:

```python
def train_one(cfg: ExperimentConfig, train_dataset, model_name: str,
              output_dir: Path, collator=None, **kwargs) -> dict:
    """Returns {"train_loss": float, "adapter_dir": str}"""
```

Look at `examples/product_comparison/` for a concrete implementation of all three.

## How the loop works

**Phase A — Exploration (tier=`small`)**
Trains on a subset (default 20K) and evaluates on a subset (default 2K). The first six rounds follow a fixed seed plan covering: baseline / bigger LoRA / focal / label smoothing / lower LR / class balancing. From round 7 onward, the proposer (Claude via the Anthropic API) reads `results.jsonl` and proposes the next config in JSON. If the API call fails, the pipeline falls back to a heuristic perturbation of the best-so-far config.

Stops early if either:
- `--target-accuracy` is reached, or
- no improvement >0.2% in the last 4 successful runs.

**Phase B — Promotion (tier=`full`)**
The top-K configs from Phase A are re-trained on the full data and evaluated on the full eval set for a definitive number.

## Why these design choices

- **4-bit NF4 + double-quant + bf16 compute, paged 8-bit AdamW, gradient checkpointing.** Standard QLoRA recipe for 16GB VRAM. Each experiment uses ~10–12GB, leaving headroom for activations.
- **Per-task trainer.py copy.** The agent can modify the trainer for task-specific needs (custom collator, masking strategy, callbacks) without touching the template. Edits stay scoped to the task folder.
- **Plug-in data/evaluator modules.** The orchestrator loads them via a package-aware importer so each task can use its own helpers (and relative imports) without polluting the global namespace.
- **Tiered evaluation.** Full eval on large sets takes 5–10× longer than a small subset; running it for every experiment wastes compute on configs that will be thrown away. We use the small tier for search and the full tier only for confirmation.
- **Seed plan before LLM proposer.** The first six experiments are deterministic, so the LLM has real signal to reason about by round 7. This avoids the failure mode of LLM proposers that hallucinate plausible-sounding but uninformative configs early on.

## Resume

The orchestrator reads `results.jsonl` at start and only runs experiments whose `exp_id` isn't already there. Killing and restarting the process is safe:

```bash
python run.py --resume --key <key> --checkpoint ... --data-file ... --evaluator-file ...
```

Reusing the same key reuses the branch and folder; a new `runs/<timestamp>/` is created each invocation.

## Note on the base model

`Qwen3-1.7B` is the default in `configs/default.yaml`. It fits comfortably in 16GB VRAM with 4-bit QLoRA. If you want a different size, just change `--checkpoint` (or edit `configs/default.yaml`):
- `Qwen/Qwen3-4B` — bigger, may need `max_seq_len ≤ 384` and `grad_accum ≥ 32`.
- `Qwen/Qwen2.5-1.5B-Instruct` — solid alternative in the Qwen2.5 family.
