"""auto_research: an autonomous experiment loop for product-comparison fine-tuning.

Only light-weight modules (schema, prompts, data, proposer, analyzer) are
re-exported here. The torch-dependent modules (trainer, evaluator,
orchestrator) are intentionally NOT imported eagerly so this package can be
imported in environments without the full ML stack — useful for testing the
schema/data/prompt/proposer/analyzer layers in CI or quick scripts.

Use the explicit submodule imports when you need the GPU code path:

    from auto_research.orchestrator import orchestrate
    from auto_research.trainer import train_one
    from auto_research.evaluator import evaluate
"""

from .schema import (
    ExperimentConfig, ExperimentResult,
    LoraConfigSpec, TrainConfigSpec, DataConfigSpec, LossConfigSpec,
)

__all__ = [
    "ExperimentConfig", "ExperimentResult",
    "LoraConfigSpec", "TrainConfigSpec", "DataConfigSpec", "LossConfigSpec",
]
