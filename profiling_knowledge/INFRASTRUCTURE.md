# Infrastructure notes — DriveNets GPU fleet (C1 project)

Handover for a new project on the same hardware. Written from the C1 output-length
prediction work (Jun–Aug 2026), which ran roughly forty experiments across this fleet.

Read the hazards section (§6) before running anything. Most of it was learned by losing
a day to something, and several items produced wrong experimental results that survived for
weeks before being caught.

Verify before trusting. Some details below are reconstructed from working notes rather
than re-checked against the machines. Anything marked [verify] should be confirmed first.

**A note on provenance, added when this file was added to this repository (task-adjacent,
not a numbered task): this document was supplied directly by the user. Its content is
about a project this repository has no other record of ("C1 output-length prediction,"
SGLang/vLLM serving evaluation, LightGBM/BERT-family training) on hosts named `xai-3`
through `xai-6`. Two observations pull in different directions, and both are worth stating
rather than picking one:**

- **Structurally, this is a strong match for what earlier tasks in this repository (48, 49)
  went looking for and could not find**: a document named `INFRASTRUCTURE.md`, with numbered
  sections including a `§6` hazards list and a `§6.1` specifically about a GPU reporting
  idle utilization while another job holds its memory, describing **four** GPU hosts
  (`xai-3`/`4`/`5`/`6`) — Task 49's own report cites exactly this shape ("§6, §6.1," "four
  GPU hosts") as what it expected to find and did not, in
  [`profiling_knowledge/INFRASTRUCTURE_MAP.md`](../../../Frontier/profiling_knowledge/INFRASTRUCTURE_MAP.md)
  (Frontier's own, unnumbered, three-host document). That match is closer than coincidence
  would predict.
- **Its actual content does not overlap with anything else in this repository or
  Frontier's own `profiling_knowledge/` at all** — grepped both directories for any
  cross-reference in either direction (`xai-`, `C1`, `rocm-work`, `c1-eval`, `vllm-probe`,
  `ssidik-dev` here; `server1`/`server3`/`server8`/`frontier_work` there): none found. The
  hosts, containers, models, and tooling described here (SGLang/vLLM serving containers,
  LightGBM/BERT training, a `run_remote_eval.sh` script) are entirely disjoint from
  Frontier's own MLA/mi355x profiling work this repository's own tasks have been doing.

**The most defensible reading: this may well be the same underlying DriveNets GPU fleet
(this file's own §1 even notes a Proxmox migration that could explain host aliases
changing between projects), reused by a different team for a different project, rather
than the same document under a different name — but that is not established, only
plausible. This file is recorded here as its own document, not merged into or reconciled
with `INFRASTRUCTURE_MAP.md`, since doing so would assert an equivalence between specific
hosts, containers, and workflows that nothing here actually confirms. If a future task
touches a GPU host, check both documents and verify current hostnames before trusting
either one's host list.**

---

## 1. Machines

| host | role | hardware |
|---|---|---|
| `ssidik-dev` | development, orchestration | VM on Proxmox, no GPU |
| `xai-3` | GPU compute | 8x AMD MI355X |
| `xai-4` | GPU compute | 8x AMD MI355X |
| `xai-5` | GPU compute | 8x AMD MI355X |
| `xai-6` | GPU compute | 8x AMD MI355X |

The dev VM has no GPU and no LaTeX toolchain. All compute runs remotely; anything needing
either has to be shipped elsewhere.

The fleet is shared. Other teams run on these hosts. Jobs get displaced, and at least one
crash during our work was attributable to contention rather than our own code. Always check
occupancy before claiming a device (§6.1).

The dev VMs were migrated between Proxmox clusters in August 2026. If host aliases or ssh
config look stale, that migration is the likely cause. [verify] current hostnames.

## 2. Access

Development happens on the dev VM; code is pushed to the GPU hosts and run inside containers
there.

```
laptop --ssh--> ssidik-dev --ssh/rsync--> xai-N --docker--> container
```

Run long jobs inside tmux on the dev VM. Claude Code and any orchestration script is a
process on that VM, and closing a laptop kills the ssh session and everything under it.

```bash
tmux new -s work
export TERM=xterm-256color      # Claude Code needs this
# ... start work, then detach with Ctrl-b d
tmux attach -t work
```

Add to `~/.tmux.conf` so it survives:

```
set -g default-terminal "tmux-256color"
```

Claude Code sessions live in `~/.claude/` on the dev VM. `claude --resume` with no
argument lists them. If the VM is rebuilt, that history is gone.

## 3. Containers

Two containers were maintained, deliberately separate:

| container | purpose |
|---|---|
| `c1-eval` | model training and evaluation (LightGBM, BERT-family, baselines) |
| `vllm-probe` | vLLM serving only |

Both mount `~/rocm-work:/workspace` on the host.

Keep them separate. This is not optional. Installing a serving framework into the shared
evaluation container silently replaced a working library and broke everything downstream
before anyone noticed. One container, one framework.

**Images actually used:**

- SGLang: `lmsysorg/sglang:v0.5.11-rocm700-mi35x` — this exact tag. Behaviour differs
  across versions; see §6.6.
- vLLM: ROCm build. [verify] the tag.

## 4. Models and weights

Cached on a shared `/huggingface` mount so they are not re-downloaded per host. [verify]
the exact path.

| model | notes |
|---|---|
| `Qwen3-0.6B` | small, fast; used for most iteration |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | MLA attention — KV per token is much smaller, which matters if you are trying to provoke memory pressure |
| `lmsys/gpt-oss-120b-bf16` | use the BF16 checkpoint, not the default — see §6.5 |

Weights are ~69 GiB for the 120B model. Disk fills up; see §6.2.

## 5. The remote execution loop

`scripts/run_remote_eval.sh` handled: pick a free GPU, rsync the code across, run inside the
container, bring results back.

If you rebuild this, the two things it must do correctly:

1. Check memory as well as utilisation when selecting a device (§6.1).
2. Write results to the GPU host first, and sync back afterwards — not streamed during the
   run. If the orchestrating VM disappears mid-experiment, everything written so far survives.

## 6. Hazards

Each of these cost real time, and several produced results that were wrong for weeks.

### 6.1 A GPU can report 0% utilisation while its memory is held

`rocm-smi` utilisation alone is not enough. A device shows idle compute while another job
holds nearly all its VRAM, and your process then fails on allocation — or worse, succeeds at a
tiny batch size and silently produces a slow, unrepresentative measurement.

Check free VRAM and utilisation together before claiming a device.

### 6.2 Disk fills, and it stops experiments

`xai-4` reached 100% (about 45 GB free) mid-project and forced a host change. Model weights,
container layers and raw records accumulate. Check free space before a long run.

### 6.3 Port collisions fail silently and produce wrong results

A stale process holding a port meant a new server bound elsewhere, the launch log still
reported success, and a request router registered a different team's model as a worker.
The experiment ran and produced numbers. They were meaningless.

Always verify the served model after launch, not just that the process started:

```bash
curl -s localhost:<port>/get_model_info
```

### 6.4 Filename collisions overwrite prior results

Three separate incidents. Result files were overwritten because output paths were not
versioned by configuration. In one case a re-run destroyed the records needed to diagnose it.

Version every output path by job, arm, seed and configuration. Stage to scratch, then
check before copying into the repo:

```bash
git ls-files | grep <candidate-name>
```

### 6.5 gpt-oss-120B: the default checkpoint produces garbled output

The model's default MXFP4 quantisation produces incoherent text on this ROCm stack. Symptoms
look like a prompt-template bug, not a numerics bug, which makes it slow to diagnose.

Use `lmsys/gpt-oss-120b-bf16`. Confirmed by byte-matching output against a known-good run.

### 6.6 SGLang dies under sustained overload; vLLM does not

Under sustained arrival rates past capacity, SGLang aborts with `SIGABRT` and the server is
gone. vLLM instead applies admission backpressure — it holds requests in a waiting queue and
never preempts a running one.

Two consequences:

1. Do not sweep load past the point of instability expecting graceful degradation. Find the
   ceiling, then run below it.
2. The two engines have materially different overload semantics. If your experiment is
   about queueing or admission, the choice of engine changes what you can observe.

Also on this SGLang version: `--max-running-requests` is ignored in disaggregation mode.

### 6.7 RDMA transfer can be dead while health checks pass

For prefill/decode disaggregation with Mooncake on ionic NICs, memory registration fails
without specific environment flags. A short health check passes; actual transfer does not
work.

```bash
export IONIC_RCQ_NUM_PATHS=255
export IONIC_PRIVATE_SERVICE_FORCE=1
export IONIC_RCQ_SIGN_BIT=15
```

These are not documented upstream — they were found in another team's script comment. Test
an actual transfer, not just endpoint liveness.

### 6.8 Short pilots hide diverging queues

A 100-request pilot looked stable; the full 1,000-request run at the same rate had a queue
growing without bound. Instability is duration-dependent and a short run cannot reveal it.

Check the first half against the second half of a run — if head-of-line waiting time is
growing, the system is not in steady state regardless of what the mean says.

### 6.9 Data files are stored in sequential blocks

Generated workload files were written in per-category blocks. `records[:n]` therefore returned
an all-one-category sample whose length distribution was 2.5x lighter-tailed than the true
population — and the resulting numbers looked entirely plausible.

Shuffle with a fixed seed before slicing. Always.

### 6.10 Run-to-run noise varies enormously by configuration

Measured between 3% and 26% across setups on this fleet. A difference that is real in one
regime is noise in another.

Re-measure the noise floor at every new configuration. Do not inherit it from a previous
experiment.

## 7. Practices worth carrying over

Persist raw per-request records, always. Two conclusions in the C1 project were later
found to be wrong, and both were correctable only because the underlying data existed.
Three earlier experiments did not keep it and could not be re-examined without repeating them.

State what is held constant across arms, and check hardware explicitly. The most expensive
error in the project was an experiment where one arm had two GPUs and the other had one. It
survived months and produced a headline result that reversed when re-run matched. A generic
"hold conditions constant" instruction was not enough — the check has to name GPU count,
parallelism degree, and total compute.

Build the observation-only baseline first. Before testing whether a learned component
helps, construct the strongest alternative using only already-available information, matched
on timing and hardware. One line of work took four rounds to converge, and a decomposition
found ~94% of the apparent gain was the baseline being weak.

Log the launch command, git commit and start time to a `RUNLOG.md` on the GPU host as each
job starts. If the orchestrating VM is lost, that file is the only record of what ran.

## 8. What to verify before relying on this

- current hostnames and ssh aliases after the Proxmox migration
- the shared HuggingFace cache path
- the vLLM image tag
- whether `scripts/run_remote_eval.sh` still exists and what it assumes
- whether the SGLang version has changed, since §6.6's behaviours are version-specific
- fleet occupancy conventions — whether other teams now reserve hosts differently
