#!/usr/bin/env python3
"""Task 55 / INFRASTRUCTURE.md S6.1: "a GPU can report 0% utilisation
while its memory is held." Prints free memory AND utilisation for every
GPU on the host this is run on -- deliberately never one without the
other, since S6.1's own failure mode is exactly a device that looks idle
by one signal while occupied by the other.

Run on BOTH real hosts, before EVERY launch in the pilot -- not once at
the start. This project has no prior real-hardware tooling to build on
(every task before Task 55 only ever touched a simulator), so this is
a small, direct wrapper over whichever vendor tool is actually present
-- it does not assume which fleet (nvidia-smi vs rocm-smi) a given host
uses.

No hardware is touched by importing this file; it only runs vendor
tooling when invoked as a script, and this task's own report was
written without ever running it for real (no access to either tool from
this environment -- see the task's report).
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def check_nvidia() -> list[dict]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for row in csv.reader(io.StringIO(out)):
        if not row:
            continue
        idx, mem_used, mem_total, util = (v.strip() for v in row)
        rows.append({
            "index": int(idx),
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "memory_free_mib": int(mem_total) - int(mem_used),
            "utilization_pct": int(util),
        })
    return rows


def check_rocm() -> list[dict]:
    # rocm-smi's own JSON schema keys are per-GPU dict keys like
    # "card0" -- normalized here to the same shape check_nvidia returns,
    # so a caller does not need to know which vendor tool ran.
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"])
    data = json.loads(out)
    rows = []
    for card_key, card in sorted(data.items()):
        if not card_key.startswith("card"):
            continue
        idx = int(card_key.replace("card", ""))
        total = int(card.get("VRAM Total Memory (B)", 0))
        used = int(card.get("VRAM Total Used Memory (B)", 0))
        util = int(float(card.get("GPU use (%)", 0)))
        rows.append({
            "index": idx,
            "memory_used_mib": used // (1024 * 1024),
            "memory_total_mib": total // (1024 * 1024),
            "memory_free_mib": (total - used) // (1024 * 1024),
            "utilization_pct": util,
        })
    return rows


def check_occupancy() -> dict:
    """Auto-detects whichever vendor tool is on PATH. Raises
    `RuntimeError` if neither is present -- never silently reports "no
    GPUs" when the real answer is "no tool found," which would look
    identical to an idle host and is exactly the kind of plausible wrong
    number this task exists to avoid producing."""
    if shutil.which("nvidia-smi"):
        vendor, gpus = "nvidia", check_nvidia()
    elif shutil.which("rocm-smi"):
        vendor, gpus = "rocm", check_rocm()
    else:
        raise RuntimeError(
            "neither nvidia-smi nor rocm-smi found on PATH -- cannot check "
            "occupancy on this host. Do not proceed to a launch without "
            "this check (INFRASTRUCTURE.md S6.1)."
        )
    return {
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "vendor": vendor,
        "gpus": gpus,
    }


def find_idle_but_held_gpus(gpus: list[dict]) -> list[dict]:
    """A concrete, hard-to-miss flag for the specific S6.1 failure mode:
    any GPU with near-zero utilisation but substantial memory held is
    exactly "idle compute, memory held by someone else.\""""
    return [g for g in gpus if g["utilization_pct"] < 5 and g["memory_used_mib"] > 1024]


def main() -> int:
    result = check_occupancy()
    print(json.dumps(result, indent=2))
    flagged = find_idle_but_held_gpus(result["gpus"])
    if flagged:
        print(
            f"\nWARNING: {len(flagged)} GPU(s) on {result['host']} show "
            "near-zero utilisation but >1GiB memory used -- INFRASTRUCTURE.md "
            "S6.1's own failure mode. Do not claim these devices without "
            "finding out who holds that memory.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
