"""Task 55: `tools/noise_pilot/occupancy_check.py`, tested against mocked
vendor-tool output -- no real GPU, no `nvidia-smi`/`rocm-smi` needed
(confirmed absent from this sandbox; see the task's own report). The
"neither tool present" test needs no mock at all: it is this sandbox's
own real, current state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "noise_pilot"))

import occupancy_check  # noqa: E402


def test_check_nvidia_parses_csv_output():
    fake_csv = "0, 1024, 81920, 0\n1, 40960, 81920, 87\n"
    with mock.patch.object(occupancy_check, "_run", return_value=fake_csv):
        rows = occupancy_check.check_nvidia()
    assert rows == [
        {"index": 0, "memory_used_mib": 1024, "memory_total_mib": 81920,
         "memory_free_mib": 80896, "utilization_pct": 0},
        {"index": 1, "memory_used_mib": 40960, "memory_total_mib": 81920,
         "memory_free_mib": 40960, "utilization_pct": 87},
    ]


def test_check_rocm_parses_json_output():
    fake_json = (
        '{"card0": {"VRAM Total Memory (B)": "85899345920", '
        '"VRAM Total Used Memory (B)": "1073741824", "GPU use (%)": "2"}, '
        '"card1": {"VRAM Total Memory (B)": "85899345920", '
        '"VRAM Total Used Memory (B)": "42949672960", "GPU use (%)": "91"}}'
    )
    with mock.patch.object(occupancy_check, "_run", return_value=fake_json):
        rows = occupancy_check.check_rocm()
    assert rows[0]["index"] == 0
    assert rows[0]["utilization_pct"] == 2
    assert rows[1]["index"] == 1
    assert rows[1]["utilization_pct"] == 91
    # 1 GiB used of 80 GiB total (85899345920 B == 81920 MiB total)
    assert rows[0]["memory_used_mib"] == 1024
    assert rows[0]["memory_free_mib"] == 81920 - 1024


def test_check_occupancy_raises_when_no_vendor_tool_present():
    """This sandbox's own real, current state -- no mock needed. If this
    ever starts passing without a mock, either a vendor tool was
    installed or the check itself broke; either is worth noticing."""
    with pytest.raises(RuntimeError, match="neither nvidia-smi nor rocm-smi"):
        occupancy_check.check_occupancy()


def test_find_idle_but_held_gpus_flags_the_s6_1_failure_mode():
    """INFRASTRUCTURE.md S6.1's own failure mode, reproduced from mocked
    data: a GPU at 0% utilisation holding several GiB should be flagged,
    not silently reported as available; a busy GPU or a genuinely idle
    one (little memory held either) should not be."""
    gpus = [
        {"index": 0, "memory_used_mib": 200, "memory_total_mib": 81920,
         "memory_free_mib": 81720, "utilization_pct": 0},   # genuinely idle -- not flagged
        {"index": 1, "memory_used_mib": 40000, "memory_total_mib": 81920,
         "memory_free_mib": 41920, "utilization_pct": 1},   # S6.1's own failure mode -- flagged
        {"index": 2, "memory_used_mib": 40000, "memory_total_mib": 81920,
         "memory_free_mib": 41920, "utilization_pct": 95},  # busy, memory held -- not flagged (expected)
    ]
    flagged = occupancy_check.find_idle_but_held_gpus(gpus)
    assert [g["index"] for g in flagged] == [1]


def test_main_returns_nonzero_and_warns_when_a_gpu_is_flagged(capsys):
    fake_result = {
        "host": "test-host",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "vendor": "nvidia",
        "gpus": [
            {"index": 0, "memory_used_mib": 40000, "memory_total_mib": 81920,
             "memory_free_mib": 41920, "utilization_pct": 1},
        ],
    }
    with mock.patch.object(occupancy_check, "check_occupancy", return_value=fake_result):
        exit_code = occupancy_check.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "S6.1" in captured.err


def test_main_returns_zero_when_nothing_is_flagged(capsys):
    fake_result = {
        "host": "test-host",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "vendor": "nvidia",
        "gpus": [
            {"index": 0, "memory_used_mib": 200, "memory_total_mib": 81920,
             "memory_free_mib": 81720, "utilization_pct": 0},
        ],
    }
    with mock.patch.object(occupancy_check, "check_occupancy", return_value=fake_result):
        exit_code = occupancy_check.main()
    assert exit_code == 0
