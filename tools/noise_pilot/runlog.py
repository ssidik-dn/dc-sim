#!/usr/bin/env python3
"""Task 55 / INFRASTRUCTURE.md S7: "Log the launch command, git commit
and start time to a RUNLOG.md on the GPU host as each job starts. If the
orchestrating VM is lost, that file is the only record of what ran."

Appends one entry per call -- never overwrites -- to a RUNLOG.md at a
given path. Meant to be called by whatever actually launches a run, on
the GPU host itself (not the orchestrating session), immediately before
that run starts, per the fleet notes' own reasoning: if the session
driving the pilot disappears mid-run, the launch record must already be
durable on the host that kept running.

No hardware is touched by this module -- it only appends a local text
file. This task's own report was written without ever calling this for
real (no real run was launched -- see the report).
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit(repo_path: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_path,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def append_entry(runlog_path: Path, *, config_label: str, host: str,
                 launch_command: str, dc_sim_repo: str = "/work/simulation/dc-sim",
                 frontier_repo: str = "/work/simulation/Frontier") -> None:
    """Appends one entry. Never truncates or rewrites the file -- a
    RUNLOG that could lose its own prior entries defeats the reason it
    exists."""
    entry = (
        f"\n## {datetime.now(timezone.utc).isoformat()}\n"
        f"- config: {config_label}\n"
        f"- host: {host}\n"
        f"- dc-sim commit: {_git_commit(dc_sim_repo)}\n"
        f"- Frontier commit: {_git_commit(frontier_repo)}\n"
        f"- launch command:\n```\n{launch_command}\n```\n"
    )
    runlog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runlog_path, "a") as f:
        f.write(entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runlog-path", type=Path, default=Path("RUNLOG.md"))
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--launch-command", required=True)
    args = parser.parse_args()
    append_entry(
        args.runlog_path, config_label=args.config_label, host=args.host,
        launch_command=args.launch_command,
    )
    print(f"appended to {args.runlog_path}")
