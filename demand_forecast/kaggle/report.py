"""Per-competition run report.

Renders ``REPORT.md`` next to a run's evidence: the mirror scores table —
labelled ``mirror (non-official)`` so no reader mistakes the numbers for
leaderboard scores — the corpus-overlap audit placeholder, and the leaderboard
submission ref once a submission has been recorded.
"""

from pathlib import Path

__all__ = ["write_report"]


def write_report(
    run_dir: Path, scores: dict[str, float], *, submission_ref: str | None = None
) -> Path:
    """Write ``REPORT.md`` under ``run_dir``; return its path.

    ``scores`` is the mirror-window score mapping (baseline first, then the
    model frames); rows render in that insertion order. ``submission_ref``,
    when given, is the backend ref of the recorded leaderboard submission.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Kaggle run report — {run_dir.name}",
        "",
        "## Mirror scores — mirror (non-official)",
        "",
        "Scores computed on the mirror window with the competition metric; they are not",
        "leaderboard scores.",
        "",
        "| model | score |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value:.6f} |" for name, value in scores.items()]
    lines += [
        "",
        "corpus_overlap: pending",
        "",
        "## Submission",
        "",
    ]
    if submission_ref is None:
        lines.append("No leaderboard submission recorded yet.")
    else:
        lines.append(f"submission_ref: {submission_ref}")
    lines.append("")
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
