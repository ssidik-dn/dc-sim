"""Stage 2 Gate A: the planner <-> real-runtime contract.

This package defines the four versioned objects that cross the file
boundary between this project's planner (`tools/planner_core.py`,
`tools/planner.py`) and the separate `sim_real` project:

    DeploymentManifest   -- what to run, exactly, with no reinterpretation
    PlannerPrediction    -- what the planner predicted, uncertainty included
    HardwareResult       -- what actually happened, observed by sim_real
    DecisionValidation   -- whether the planner's choice matches reality

No planner Python import belongs in `sim_real`, and no `sim_real` import
belongs here. The contract is the JSON files under `contracts/stage2/`;
this package is this project's own producer/consumer of them. See
`docs/stage-2-gate-a-contract-report.md` for the full design rationale.
"""
