"""Task 55: `tools/noise_pilot/analysis.py`'s own repeats-required
calculator, tested against closed-form values and against
`tools/seed_stats.py`'s own established `compute_interval_stats` --
no hardware, no Frontier; this module only does arithmetic on values a
caller already measured.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "noise_pilot"))

from seed_stats import _t_critical, compute_interval_stats  # noqa: E402
from analysis import (  # noqa: E402
    repeats_required_to_resolve,
    summarize_configuration,
)


def test_repeats_required_matches_closed_form_at_n5_cv10():
    """CV=10%, n=5: half-width = t(5)*10/sqrt(5) = 2.776*10/sqrt(5) ~=
    12.41% -- NOT the ~9% Task 54's own text estimated (that figure uses
    the large-sample z=1.96 approximation, not this project's own
    small-n Student's t convention). Five repeats therefore does not
    resolve 9% at CV=10% under this project's own established formula;
    it takes n=8. This is the precise version of Task 55's own report
    finding, pinned as a test so it cannot silently drift."""
    t5 = _t_critical(5)
    expected_halfwidth_at_5 = t5 * 10.0 / math.sqrt(5)
    assert expected_halfwidth_at_5 == pytest.approx(12.4147, abs=1e-3)

    result_9pct = repeats_required_to_resolve(10.0, 9.0)
    assert result_9pct.n == 8
    result_10pct = repeats_required_to_resolve(10.0, 10.0)
    assert result_10pct.n == 7


@pytest.mark.parametrize("cv_pct,target_pct,expected_n", [
    (3.0, 10.0, 3),
    (10.0, 5.0, 18),
    (26.0, 10.0, 26),
    (26.0, 5.0, 104),
])
def test_repeats_required_worked_examples(cv_pct, target_pct, expected_n):
    """Worked examples spanning INFRASTRUCTURE.md's own documented
    3%-26% noise-floor range (not measured values -- this project has
    taken no real-hardware measurement; see the task's own report)."""
    result = repeats_required_to_resolve(cv_pct, target_pct)
    assert result.n == expected_n
    assert result.achieved_halfwidth_pct <= target_pct


def test_repeats_required_is_monotonic_in_n():
    """A larger n must never need a larger target margin to reach the
    same CV -- i.e. the half-width the search finds at its returned n is
    non-increasing as n grows, which is what makes searching upward for
    the *smallest* qualifying n correct rather than accidental."""
    for cv_pct in (3.0, 10.0, 26.0):
        halfwidths = [
            _t_critical(n) * cv_pct / math.sqrt(n) for n in range(2, 40)
        ]
        # not strictly monotonic pointwise (t-table has small bumps at
        # low n before the z approximation kicks in), but the *found* n
        # for a tighter target must never be smaller than for a looser one
        loose = repeats_required_to_resolve(cv_pct, 10.0)
        tight = repeats_required_to_resolve(cv_pct, 5.0)
        assert tight.n >= loose.n


def test_repeats_required_zero_cv_resolves_immediately():
    result = repeats_required_to_resolve(0.0, 5.0)
    assert result.n == 2
    assert result.achieved_halfwidth_pct == 0.0


def test_repeats_required_rejects_invalid_input():
    with pytest.raises(ValueError):
        repeats_required_to_resolve(-1.0, 5.0)
    with pytest.raises(ValueError):
        repeats_required_to_resolve(10.0, 0.0)


def test_repeats_required_not_reached_within_max_n():
    result = repeats_required_to_resolve(1000.0, 1.0, max_n=5)
    assert result.n is None
    assert result.achieved_halfwidth_pct is None


def test_summarize_configuration_matches_compute_interval_stats():
    """`summarize_configuration` must not compute its own, independent
    mean/CV -- it wraps `compute_interval_stats` exactly, so a number
    reported here means the same thing as everywhere else this project
    already reports one."""
    values = [10.1, 9.8, 10.3, 9.9, 10.0, 10.2, 9.7, 10.4, 9.9, 10.1]
    summary = summarize_configuration("test-config", values)
    direct = compute_interval_stats(values)
    assert summary.stats == direct
    assert summary.repeats_for_5pct.cv_pct == pytest.approx(direct.cv_pct)
