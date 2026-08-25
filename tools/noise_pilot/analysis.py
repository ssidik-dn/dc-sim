"""Task 55: turn a per-configuration set of real-hardware repeats into a
coefficient of variation and a required-repeat-count, per configuration --
the one number Task 54's five-repeat choice was a guess about.

Deliberately reuses `tools/seed_stats.py`'s own `compute_interval_stats`
(Student's t, n-1 degrees of freedom, the exact convention every seed
study in this project already reports) rather than a fresh formula, so a
number computed here means the same thing a number from that module
already means elsewhere in this project. No hardware, no Frontier, no
simulation -- this module only consumes a list of already-measured
per-run values.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seed_stats import IntervalStats, _t_critical, compute_interval_stats  # noqa: E402


@dataclass
class RepeatsRequired:
    cv_pct: float
    target_margin_pct: float
    n: Optional[int]           # None if not reachable within max_n
    achieved_halfwidth_pct: Optional[float]


def repeats_required_to_resolve(cv_pct: float, target_margin_pct: float,
                                max_n: int = 200) -> RepeatsRequired:
    """Smallest `n` such that the 95% CI half-width on the mean, at this
    measured `cv_pct`, is <= `target_margin_pct` -- i.e. the smallest
    repeat count at which a real difference of that size would show up
    as non-overlapping-with-itself rather than absorbed into noise.

    Uses the same Student's t table `compute_interval_stats` uses (exact
    up to n=20, the z=1.960 large-sample approximation beyond) -- a
    repeats-required number from this function and a CV/half-width
    number from `compute_interval_stats` are directly comparable, by
    construction, since they are the same formula run in opposite
    directions.

    `cv_pct=0` (zero measured variability) resolves at `n=2` -- the
    smallest sample size `compute_interval_stats` can compute a spread
    from at all; this function does not claim n=1 ever "resolves"
    anything, since a single run has no measured variability to compare
    a margin against.
    """
    if cv_pct < 0:
        raise ValueError(f"cv_pct must be non-negative, got {cv_pct!r}")
    if target_margin_pct <= 0:
        raise ValueError(f"target_margin_pct must be positive, got {target_margin_pct!r}")
    for n in range(2, max_n + 1):
        t = _t_critical(n)
        halfwidth_pct = t * cv_pct / math.sqrt(n)
        if halfwidth_pct <= target_margin_pct:
            return RepeatsRequired(cv_pct, target_margin_pct, n, halfwidth_pct)
    return RepeatsRequired(cv_pct, target_margin_pct, None, None)


@dataclass
class ConfigurationSummary:
    label: str
    stats: IntervalStats
    repeats_for_5pct: RepeatsRequired
    repeats_for_10pct: RepeatsRequired


def summarize_configuration(label: str, values: Sequence[float]) -> ConfigurationSummary:
    """The full per-configuration report this task's own S3 asks for:
    mean/stdev/CV/95% CI half-width (from `compute_interval_stats`), plus
    the repeat counts needed to resolve a 5% and a 10% difference at this
    configuration's own measured CV -- not a global number, since S3's
    own point is that different arrangement types may need different
    counts."""
    stats = compute_interval_stats(values)
    return ConfigurationSummary(
        label=label,
        stats=stats,
        repeats_for_5pct=repeats_required_to_resolve(stats.cv_pct, 5.0),
        repeats_for_10pct=repeats_required_to_resolve(stats.cv_pct, 10.0),
    )


def format_summary(summary: ConfigurationSummary) -> str:
    s = summary.stats
    r5, r10 = summary.repeats_for_5pct, summary.repeats_for_10pct
    lines = [
        f"=== {summary.label} (n={s.n}) ===",
        f"  mean = {s.mean:.4f}",
        f"  stdev = {s.stdev:.4f}",
        f"  CV = {s.cv_pct:.2f}%",
        f"  95% CI half-width on the mean = {s.ci95_halfwidth:.4f} ({s.ci95_halfwidth_pct:.2f}%)",
        f"  repeats to resolve 5%:  n={r5.n}"
        + (f" (achieves {r5.achieved_halfwidth_pct:.2f}%)" if r5.n else " (not reached within max_n)"),
        f"  repeats to resolve 10%: n={r10.n}"
        + (f" (achieves {r10.achieved_halfwidth_pct:.2f}%)" if r10.n else " (not reached within max_n)"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(
        description="Summarize a noise-pilot configuration's repeats: mean, "
                    "CV, 95% CI half-width, and repeats required to resolve "
                    "5%/10% -- reads real-run values from a one-column CSV "
                    "(no header) or from --values."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--csv", type=Path, help="one-column CSV of per-run values, no header")
    parser.add_argument("--values", nargs="+", type=float, help="per-run values inline")
    args = parser.parse_args()

    if args.csv:
        with open(args.csv) as f:
            values: List[float] = [float(row[0]) for row in csv.reader(f) if row]
    elif args.values:
        values = args.values
    else:
        parser.error("pass either --csv or --values")

    print(format_summary(summarize_configuration(args.label, values)))
