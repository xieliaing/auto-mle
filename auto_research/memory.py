"""
Memory skill: state machine tracking defect-pattern learnings across experiments.

Persists to memory.json so the proposer always has context about which error
patterns are still unresolved, which just regressed, and which were fixed by
which hypothesis.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PatternRecord:
    pattern: str
    category: str               # "fixed" | "persistent" | "regressed"
    first_seen_exp: str
    last_seen_exp: str
    occurrences: int = 1
    resolved_by_exp: Optional[str] = None
    resolving_hypothesis: Optional[str] = None


class ExperimentMemory:
    """
    Accumulates sample-level profiling results across experiment iterations.

    State machine transitions:
      - new pattern seen in persistent_errors → category="persistent"
      - seen again in persistent_errors       → occurrences += 1
      - appears in fixed_patterns             → category="fixed", resolved_by_exp set
      - appears in regressed_patterns         → category="regressed"

    Call get_context_for_proposal() to get a compact string for the LLM proposer.
    """

    def __init__(self, memory_path: str | Path):
        self.path = Path(memory_path)
        self._load()

    def _load(self):
        if self.path.exists():
            raw = self.path.read_text().strip()
            if not raw:
                self.iterations = 0
                self.cumulative_summary = ""
                self.iter_profiles = []
                self.patterns = []
                return
            data = json.loads(raw)
            self.iterations: int = data.get("iterations", 0)
            self.cumulative_summary: str = data.get("cumulative_summary", "")
            self.iter_profiles: List[dict] = data.get("iter_profiles", [])
            self.patterns: List[PatternRecord] = [
                PatternRecord(**r) for r in data.get("patterns", [])
            ]
        else:
            self.iterations = 0
            self.cumulative_summary = ""
            self.iter_profiles = []
            self.patterns = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "iterations": self.iterations,
                    "cumulative_summary": self.cumulative_summary,
                    "iter_profiles": self.iter_profiles,
                    "patterns": [asdict(p) for p in self.patterns],
                },
                indent=2,
            )
        )

    def update(self, profile: dict, curr_exp_id: str, curr_hypothesis: str):
        """Merge a new profile result into the state machine."""
        self.iterations += 1
        self.iter_profiles.append(
            {
                "exp_id": curr_exp_id,
                "hypothesis": curr_hypothesis,
                "stats": profile.get("stats", {}),
                "summary": profile.get("analysis_summary", ""),
            }
        )

        by_text = {r.pattern: r for r in self.patterns}

        for pat in profile.get("fixed_patterns", []):
            if pat in by_text:
                r = by_text[pat]
                r.category = "fixed"
                r.last_seen_exp = curr_exp_id
                r.resolved_by_exp = curr_exp_id
                r.resolving_hypothesis = curr_hypothesis
            else:
                self.patterns.append(
                    PatternRecord(
                        pattern=pat,
                        category="fixed",
                        first_seen_exp=curr_exp_id,
                        last_seen_exp=curr_exp_id,
                        resolved_by_exp=curr_exp_id,
                        resolving_hypothesis=curr_hypothesis,
                    )
                )

        for pat in profile.get("regressed_patterns", []):
            if pat in by_text:
                r = by_text[pat]
                r.category = "regressed"
                r.last_seen_exp = curr_exp_id
                r.occurrences += 1
            else:
                self.patterns.append(
                    PatternRecord(
                        pattern=pat,
                        category="regressed",
                        first_seen_exp=curr_exp_id,
                        last_seen_exp=curr_exp_id,
                    )
                )

        for pat in profile.get("persistent_patterns", []):
            if pat in by_text:
                r = by_text[pat]
                r.category = "persistent"
                r.last_seen_exp = curr_exp_id
                r.occurrences += 1
            else:
                self.patterns.append(
                    PatternRecord(
                        pattern=pat,
                        category="persistent",
                        first_seen_exp=curr_exp_id,
                        last_seen_exp=curr_exp_id,
                    )
                )

        # Rolling summary — keep last 3 entries to bound growth
        latest = (
            f"[{curr_exp_id}] "
            f"fixed={profile['stats'].get('fixed', 0)}, "
            f"regressed={profile['stats'].get('regressed', 0)}, "
            f"persistent={profile['stats'].get('persistent_errors', 0)}: "
            + profile.get("analysis_summary", "")
        )
        parts = [s for s in self.cumulative_summary.split(" || ") if s]
        parts.append(latest)
        self.cumulative_summary = " || ".join(parts[-3:])

    def get_context_for_proposal(self) -> str:
        """Compact context string for inclusion in the LLM proposer prompt."""
        if not self.patterns and not self.cumulative_summary:
            return ""

        lines = ["=== Sample-Level Profiling Memory ==="]

        if self.cumulative_summary:
            lines.append(f"Recent iterations: {self.cumulative_summary}")

        persistent = [r for r in self.patterns if r.category == "persistent"]
        regressed = [r for r in self.patterns if r.category == "regressed"]
        fixed = [r for r in self.patterns if r.category == "fixed"]

        if persistent:
            lines.append(
                f"\nPersistent error patterns ({len(persistent)} unresolved — PRIORITIZE):"
            )
            for r in sorted(persistent, key=lambda x: -x.occurrences)[:5]:
                lines.append(f"  [{r.occurrences}x] {r.pattern}")

        if regressed:
            lines.append(f"\nNewly-emerged regressions ({len(regressed)} total):")
            for r in regressed[:3]:
                lines.append(f"  {r.pattern}")

        if fixed:
            lines.append(f"\nSuccessfully fixed patterns ({len(fixed)} total):")
            for r in fixed[:3]:
                hyp = f" via: {r.resolving_hypothesis!r}" if r.resolving_hypothesis else ""
                lines.append(f"  {r.pattern}{hyp}")

        lines.append(
            "\nTarget persistent/regressed patterns. Avoid undoing fixed ones."
        )
        return "\n".join(lines)
