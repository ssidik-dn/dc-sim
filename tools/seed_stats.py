#!/usr/bin/env python3
"""Confidence-interval infrastructure for this project's own
subprocess-per-scenario measurement tools (task 31).

**Where this lives, and why.** This orchestrates already-existing
scenario runs and computes statistics over their results -- it is
measurement/analysis infrastructure, not placement or cost modeling.
`src/engine/` is Phases 1-6's own standalone modeling code (`Fabric`,
`Placement`, cost predictors); nothing here answers a placement or
communication-cost question, and nothing here would even violate the
import-direction rule if it lived there (it imports neither
`src/integration/` nor `upstream/`) -- but every existing "run N seeds,
aggregate" function in this project (`_aggregate()` in
`run_memory_edge_study.py`, `run_memory_tp_study.py`, etc.) already
lives in `tools/`, importing across tool modules the same way this one
is meant to be imported. Consistency with that established precedent,
not the import-direction check, is what decided it.

**The seed-wiring finding this module exists to correct, not to hide.**
Every tool in this project passes `--seed <n>` to vary a scenario
across repeats (task 22 onward). That flag sets `SimulationConfig.seed`,
consumed by `frontier.utils.random.set_seeds()` in each tool's own
Python driver -- which correctly reseeds the global `random`/`np.random`
state at that point (confirmed directly: printing `random.random()`
immediately after `set_seeds(config.seed)` for `--seed 0` vs `--seed 1`
gives different values, as expected).

But request *generation* -- arrival times, and any randomized request
lengths -- happens inside `Simulator.__init__()`'s own construction,
which reaches `SyntheticRequestGenerator.generate_requests()`
(`frontier/request_generator/synthetic_request_generator.py`) -- and
that method re-seeds *again*, internally:
`set_seeds(self.config.seed)`, where `self.config` is the request
*generator's own* config object, whose `seed` field
(`BaseRequestGeneratorConfig.seed`, `frontier/config/config.py`)
defaults to `42` and is *entirely separate* from the top-level
`SimulationConfig.seed` set by `--seed`. No tool in this project has
ever passed the matching per-generator-type flag
(`--synthetic_request_generator_config_seed`) to override it. Confirmed
directly: three different `--seed` values, without that flag, produce
bit-identical arrival timestamps, to four decimal places, every time.

Layered on top of that: even with the request generator's own seed
wired correctly, this project's `--simulation_mode offline` convention
(used by every real-compute tool here) discards whatever arrival times
were generated anyway -- `Simulator.run()`'s own offline branch forces
every request's `arrived_at` to `0` unless
`--offline_use_generated_request_arrivals` is *also* set (default
`False`; never set by any existing tool either, confirmed by `grep`).
Both gaps compound. Fixing only one still leaves every seed producing
an identical run in the default configuration every tool here has used.

**Consequence, stated plainly:** every "N_REPEATS" figure in this
project's own reports (tasks 22 through 30) rests on N *identical*
runs, not N independent samples of a noisy system -- the "near-zero
standard deviation" those reports observed was not evidence of a
low-noise system; it was the variance of N copies of the same number.

This module does not modify any existing tool -- doing so would change
historical figures the acceptance criteria for this and prior tasks
already fixed in place. `seed_argv_fix()` below is the correct wiring,
provided as new, opt-in infrastructure for studies that need genuine
seed-to-seed variation; `compute_interval_stats()`/`run_seed_study()`
are the statistics to interpret it once it exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence


def seed_argv_fix(seed: int, *, vary_arrivals: bool = True) -> List[str]:
    """The additional CLI flags a scenario's own argv needs, alongside
    `--seed <seed>`, for that seed to actually change anything about the
    simulated workload. Without these two, `--seed` alone still correctly
    reseeds the global `random`/`np.random` state (harmless, and left in
    place), but request generation re-seeds again internally to a fixed
    default, and offline mode discards generated arrival times
    regardless -- see this module's own docstring for both, confirmed
    directly rather than assumed.

    `vary_arrivals=False` keeps the request-generator seed fix (so
    randomized request *lengths*, if a tool uses a non-fixed length
    generator, would still vary) while leaving offline mode's own t=0
    override in place -- for a study that wants everything else about
    this project's existing convention held fixed and only wants to ask
    "does the length generator introduce variance," which is a
    different, narrower question than the arrival-pattern one this
    module's docstring centers on.
    """
    argv = ["--synthetic_request_generator_config_seed", str(seed)]
    if vary_arrivals:
        argv.append("--offline_use_generated_request_arrivals")
    return argv


@dataclass
class IntervalStats:
    n: int
    mean: float
    stdev: float
    cv_pct: float               # coefficient of variation, %
    ci95_halfwidth: float       # half-width of the 95% CI on the mean
    ci95_halfwidth_pct: float   # that half-width as % of the mean
    values: List[float]


# Two-sided 95% t critical values, n observations (n-1 degrees of
# freedom) -- covers every seed count this task's own acceptance
# criteria and report actually use (up to 20); a normal approximation
# is used beyond that rather than pretending a bigger table was needed.
_T_CRIT_95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
    9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
    15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
}
_Z_95 = 1.960


def _t_critical(n: int) -> float:
    if n in _T_CRIT_95:
        return _T_CRIT_95[n]
    if n < 2:
        return float("nan")
    return _Z_95


def compute_interval_stats(values: Sequence[float]) -> IntervalStats:
    """Mean, sample standard deviation, coefficient of variation, and the
    half-width of a 95% confidence interval on the mean (Student's t,
    n-1 degrees of freedom) -- both the interval on the mean *and* the
    raw spread of the data are returned, deliberately: task 31's own
    known trap is that a confidence interval on a mean answers a
    different question than the spread of the underlying data does, and
    conflating them hides whether a single run was ever representative.
    """
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    cv_pct = (stdev / mean * 100.0) if mean else float("nan")
    t = _t_critical(n)
    halfwidth = t * stdev / math.sqrt(n) if n > 1 else 0.0
    halfwidth_pct = (halfwidth / mean * 100.0) if mean else float("nan")
    return IntervalStats(n=n, mean=mean, stdev=stdev, cv_pct=cv_pct,
                         ci95_halfwidth=halfwidth,
                         ci95_halfwidth_pct=halfwidth_pct,
                         values=list(values))


def run_seed_study(runner: Callable[[int], dict], seeds: Sequence[int],
                   metrics: Sequence[str]) -> Dict[str, IntervalStats]:
    """Run `runner(seed)` for every seed, collect the named `metrics` from
    each result dict, and compute `IntervalStats` for each.

    `runner` is expected to follow this project's own established
    subprocess-per-scenario convention (e.g. a tool's own
    `_run_scenario_in_subprocess`, partially applied over everything but
    `seed`) -- this function only orchestrates and aggregates; it does
    not build an argv or invoke Frontier itself, so it works with any
    tool's own scenario runner without needing to know its scenario
    parameters.
    """
    rows = [runner(s) for s in seeds]
    errors = [r for r in rows if r.get("error")]
    if errors:
        raise RuntimeError(f"{len(errors)}/{len(rows)} seeds failed: "
                           f"{errors[0].get('error')}")
    return {m: compute_interval_stats([r[m] for r in rows]) for m in metrics}
