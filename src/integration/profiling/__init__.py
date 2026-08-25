"""Runtime patches to Frontier's profiling-time code (task 53).

Distinct from `..execution_time_predictor` and `..cc_backend`: nothing here
is reachable from `install()`, because nothing under `frontier/profiling/`
is reachable from the simulation path at all (Task 51/52 both confirmed
this directly) -- `install()` is called before a simulation run, never
before a profiling run. A module in this package exists to be imported
explicitly by whatever invokes a profiling CLI, not by this project's own
`install()`.
"""
