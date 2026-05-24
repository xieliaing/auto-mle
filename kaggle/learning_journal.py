"""
Learning journal: JSONL-backed accumulation of per-iteration insights.

Each entry records experiment metrics, defect analysis results, and proposed
skills. The journal is the long-term memory that feeds the skill proposer and
enriches the LLM proposer with what has been learned across iterations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class LearningJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def add_entry(
        self,
        iteration: int,
        exp_id: str,
        metrics: Dict[str, Any],
        defect_report: Optional[Dict[str, Any]] = None,
        proposed_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "exp_id": exp_id,
            "metrics": metrics,
            "defect_report": defect_report,
            "proposed_skills": proposed_skills or [],
        }
        self._entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_history(self) -> List[dict]:
        return list(self._entries)

    def format_for_proposer(self, n_recent: int = 5) -> str:
        """Compact text for the LLM proposer's context window."""
        recent = self._entries[-n_recent:]
        if not recent:
            return "(no learning journal entries yet)"

        lines = ["## Learning journal (most recent first)"]
        for e in reversed(recent):
            lines.append(f"\n### Iteration {e['iteration']}: {e['exp_id']}")
            m = e.get("metrics", {})
            scalar_metrics = {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
            lines.append("Metrics: " + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in scalar_metrics.items()
            ))

            dr = e.get("defect_report")
            if dr:
                lines.append(
                    f"Defects: fixed={dr.get('n_fixed', 0)} "
                    f"new={dr.get('n_new_errors', 0)} "
                    f"persistent={dr.get('n_persistent', 0)} "
                    f"net={dr.get('net_change', 0)}"
                )
                for feat, info in list((dr.get("feature_patterns") or {}).items())[:2]:
                    lines.append(
                        f"  Feature: {feat}[{info.get('bucket')}] "
                        f"err={info.get('error_rate_pct')}% lift={info.get('lift')}x"
                    )
                for t, c in list((dr.get("class_patterns") or {}).get("top_new_errors", {}).items())[:2]:
                    lines.append(f"  New error pattern: {t} x{c}")

            skills = e.get("proposed_skills") or []
            if skills:
                lines.append(f"Proposed skills ({len(skills)}):")
                for s in skills[:3]:
                    lines.append(
                        f"  [{s.get('priority', '?')}] {s.get('name', '?')}: "
                        f"{s.get('rationale', '')[:100]}"
                    )

        return "\n".join(lines)
