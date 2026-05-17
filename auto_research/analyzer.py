"""
Analyzer: turns results.jsonl into a human-readable leaderboard and answers
"should we stop?" questions.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List


def load_results(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_result(path: str | Path, result: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def write_leaderboard(results: List[dict], out_path: str | Path) -> None:
    """Write a markdown leaderboard sorted by accuracy desc."""
    out_path = Path(out_path)
    ok = [r for r in results if r.get("status", "success") == "success"]
    failed = [r for r in results if r.get("status", "success") != "success"]
    ok.sort(key=lambda r: r["accuracy"], reverse=True)

    lines = ["# Auto-Research Leaderboard", ""]
    lines.append(f"_Total runs: {len(results)} ({len(ok)} succeeded, {len(failed)} failed)_")
    lines.append("")
    if ok:
        lines.append("| Rank | exp_id | tier | accuracy | f1 | loss | LoRA(r/tgt) | lr | notes |")
        lines.append("|-----:|--------|------|---------:|---:|------|-------------|---:|-------|")
        for i, r in enumerate(ok, 1):
            cfg = r["config"]
            lines.append("| {rank} | `{exp_id}` | {tier} | {acc:.4f} | {f1} | {loss} | {r}/{tgt} | {lr:.0e} | {notes} |".format(
                rank=i,
                exp_id=r["exp_id"],
                tier=cfg.get("tier", "small"),
                acc=r["accuracy"],
                f1=("{:.4f}".format(r["extra"].get("f1")) if r.get("extra", {}).get("f1") is not None else "—"),
                loss=cfg["loss"]["name"],
                r=cfg["lora"]["r"],
                tgt=cfg["lora"]["target_modules"],
                lr=cfg["train"]["learning_rate"],
                notes=(cfg.get("notes") or cfg.get("hypothesis", ""))[:80].replace("|", "/").replace("\n", " "),
            ))
    if failed:
        lines += ["", "## Failed runs", ""]
        for r in failed:
            lines.append(f"- `{r['exp_id']}`: {r.get('status')} — {r.get('error', '')[:200]}")
    Path(out_path).write_text("\n".join(lines) + "\n")


def best_so_far(results: List[dict]) -> dict | None:
    ok = [r for r in results if r.get("status", "success") == "success"]
    if not ok:
        return None
    return max(ok, key=lambda r: r["accuracy"])


def has_plateaued(results: List[dict], window: int = 4, min_delta: float = 0.002) -> bool:
    """No improvement larger than `min_delta` in the last `window` successful runs."""
    ok = [r for r in results if r.get("status", "success") == "success"]
    if len(ok) < window + 1:
        return False
    recent = ok[-window:]
    earlier_best = max(r["accuracy"] for r in ok[:-window])
    recent_best = max(r["accuracy"] for r in recent)
    return (recent_best - earlier_best) < min_delta


def top_k(results: List[dict], k: int = 3) -> List[dict]:
    ok = [r for r in results if r.get("status", "success") == "success"]
    ok.sort(key=lambda r: r["accuracy"], reverse=True)
    return ok[:k]
