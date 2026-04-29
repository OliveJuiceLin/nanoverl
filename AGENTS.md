# AGENTS.md

This file is for agents developing `nanoverl`. It is not user-facing product
documentation; that role belongs to `README.md`.

If you open a new thread in this repository, read this file first. It is the
project memory that prevents each session from rediscovering the same design
constraints from scratch.

## Project Identity

`nanoverl` is a small, research-oriented reinforcement learning framework for
large language models. It is inspired by `verl`, but deliberately keeps a much
smaller and more readable core.

The goal is not to become a production-scale `verl` clone. The goal is:

- readable code that a researcher can understand end to end
- high customizability for algorithm experiments
- enough performance to run meaningful local and single-node experiments
- explicit boundaries so future parallel infra can be added without burying RL
  semantics inside scheduling code

The current repository lives at:

- `/Users/wangyilin/Downloads/code/nanoverl/nanoverl`

Important sibling reference projects:

- `nanoRLHF`: `/Users/wangyilin/Downloads/code/nanoverl/nanoRLHF`
- `verl`: `/Users/wangyilin/Downloads/code/nanoverl/verl`

Use these as references, not as sources for blind copying.

## Development Principles

Keep the code small, direct, and easy to inspect.

- Prefer explicit local contracts over clever generic abstraction.
- Keep trainer lifecycle code separate from algorithm semantics.
- Keep backend-specific code inside backend/worker/rollout implementations.
- Add an abstraction only when it removes real complexity or protects a boundary
  that the project already needs.
- Do not preserve compatibility layers during this early stage unless the user
  explicitly asks for it. A clean public surface is more important.
- Do not remove explanatory comments. The user values learning-oriented
  comments. Only update comments when they become misleading or stale.
- Do not introduce fake public config or stub runtime knobs. If a feature is not
  implemented, keep it out of public config.
- Do not copy a large infra module before the core boundary it touches is clear.

When editing code:

- Follow existing style and file layout.
- Use typed dataclass config fields and fail fast on unknown fields.
- Keep tests focused on the behavior and boundary being changed.
- Update `README.md` whenever repo structure, capability boundaries, or roadmap
  emphasis changes.

## Current Stage

The project is just past the algorithm-plugin and core-boundary cleanup stage.
The next real stage can begin parallel infra work, but only after the current
core contracts stay clean.

What is already in place:

- synchronous, driver-owned trainer lifecycle
- algorithm plugin entry through `algorithm.name`
- built-in `ppo`, `grpo`, and `rloo`
- advantage estimator registry
- policy loss registry
- actor-only algorithms for GRPO/RLOO
- algorithm-neutral actor/critic update config:
  - `mini_batch_size`
  - `update_epochs`
  - `micro_batch_size`
- actor -> rollout policy sync through `PolicySyncer`
- separate policy checkpoint state and rollout export state:
  - `policy_worker.state_dict()` is checkpoint/resume state
  - `policy_worker.policy_state_dict()` is actor -> rollout export state
  - `rollout_engine.state_dict()` is lightweight rollout-side state
- versioned structured checkpoints:
  - `checkpoint_version`
  - `trainer_state`
  - `loader_state`
  - `worker_state`
  - `rollout_state`
  - `config`
- worker backend registry
- rollout backend registry
- no public `ray` config section
- no fake Ray worker group stub

The project is now ready for a careful parallel-infra design pass: worker
groups, placement/runtime config, and parameter sync can be introduced as real
features instead of public placeholders.

## Repo Structure

- `nanoverl/algos`
  - Algorithm plugins and shared RL math.
  - PPO/GRPO/RLOO live here.
  - Advantage and policy loss registries live here.
- `nanoverl/backends`
  - Backend-specific utilities.
  - Hugging Face helpers, vLLM helpers, and FSDP training backend code live here.
- `nanoverl/checkpoint`
  - Local checkpoint save/load helpers.
- `nanoverl/config.py`
  - Typed config tree, derived defaults, and validation.
  - Unknown public fields should fail fast.
- `nanoverl/core`
  - Small shared data structures such as `RLBatch`.
- `nanoverl/data`
  - JSON dataset and checkpointable dataloader/sampler.
- `nanoverl/distributed`
  - Currently only real torch distributed runtime helpers.
  - Future worker-group/placement infra belongs here when implemented.
- `nanoverl/logging`
  - Metrics and tracker backends.
- `nanoverl/reward`
  - Reward function loading and reward shaping interface.
- `nanoverl/rollout`
  - Rollout interface, rollout backend registry, debug/HF/vLLM rollout engines,
    and actor -> rollout sync helper.
- `nanoverl/trainer`
  - Synchronous trainer lifecycle, validation, checkpoint orchestration, and
    debug artifact writers.
- `nanoverl/workers`
  - Policy/reference/value interfaces, worker backend registry, debug workers,
    and local HF workers.
- `examples`
  - Example configs, reward functions, and dataset helpers.
- `tests`
  - Regression tests for config, algorithms, trainer lifecycle, checkpointing,
    backend slices, rollout sync, and registries.

## Core Boundary Rules

### Trainer

`RLTrainer` owns lifecycle only:

- dataloaders
- worker and rollout construction
- validation
- checkpointing
- logging
- timing/metrics plumbing
- startup/resume sync

It should not own PPO/GRPO/RLOO semantics.

### Algorithms

`RLAlgorithm` plugins own RL step semantics:

- rollout batch preparation
- reward shaping
- old/ref/value computation
- advantage and return computation
- critic/actor update order
- deciding when actor -> rollout sync is needed

Algorithms call `AlgorithmStepContext.sync_rollout_policy(reason)`; they do not
directly inspect how policy weights are exported or imported.

### Workers

Workers are backend-specific execution objects. They should not become
algorithm switchboards.

Policy worker state has two meanings:

- `state_dict()` means checkpoint/resume state and may include optimizer state.
- `policy_state_dict()` means minimal policy export for rollout sync.

Reference/value workers only need checkpoint state today.

### Rollout

Rollout engines implement generation and backend-specific policy import.

- `sync_policy(policy_state)` is the backend hook used by `PolicySyncer`.
- `state_dict()` must remain lightweight rollout-side state.
- Rollout checkpoints should not duplicate actor model weights.

### Checkpoint

Checkpoint payloads are versioned and structured. Do not add flat top-level
training fields back into checkpoint payloads.

The current shape is:

```python
{
    "checkpoint_version": 2,
    "trainer_state": {...},
    "loader_state": {...},
    "worker_state": {...},
    "rollout_state": {...},
    "config": {...},
}
```

No old flat checkpoint compatibility is required at this stage.

## Reference Projects

### nanoRLHF

Useful to study for:

- compact infra modules
- `nanovllm`
- `nanotron`
- `nanoray`
- multi-GPU scheduling sketches
- 3D parallelism ideas
- some elegant module organization in its `nanoverl` package

Be careful:

- Do not copy its `ActorCriticRefWorker.make_experience()` / `step()` PPO
  structure into this project.
- It hardcodes PPO-style algorithm logic too deeply inside worker code.
- It is a reference for infra direction, not for algorithm boundary design.

### verl

Useful to study for:

- algorithm/plugin patterns
- advantage estimator registry
- policy loss registry
- separation between trainer orchestration and algorithm semantics
- mature distributed/placement concepts

Be careful:

- Do not import `verl` complexity wholesale.
- Copy the ideas that preserve boundaries; avoid layers that make this small
  codebase harder to read.

## Current Public Config Boundary

Important public config decisions:

- `algorithm.name` selects the RL algorithm.
- Built-in algorithm names:
  - `ppo`
  - `grpo`
  - `rloo`
- Actor/critic update fields are algorithm-neutral:
  - `mini_batch_size`
  - `update_epochs`
  - `micro_batch_size`
- Old PPO-specific fields such as `ppo_mini_batch_size` and `ppo_epochs` are not
  supported.
- There is no `ray` section.
- Future parallel infra should introduce explicit runtime/placement config when
  the implementation exists.

## Testing Expectations

Run focused tests while editing, then the full suite before handing off:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Optional backend tests may skip if dependencies are unavailable. That is fine as
long as the core/debug tests pass.

Useful focused suites:

```bash
python3 -m unittest tests.test_algos tests.test_algorithm_plugins -v
python3 -m unittest tests.test_rollout_sync tests.test_backend_registry -v
python3 -m unittest tests.test_trainer_smoke tests.test_phase2 -v
python3 -m unittest tests.test_checkpoint -v
```

## Next Development Direction

The next stage should be real parallel infra, not more naming polish.

Likely sequence:

1. Introduce a real runtime/placement config.
2. Add a local worker-group abstraction that still works without Ray.
3. Add a parameter sync manager behind the existing policy export/sync boundary.
4. Add parallel rollout/training backend slices carefully.
5. Evaluate `nanoRLHF` infra pieces such as `nanovllm`, `nanotron`, and
   `nanoray` after the local worker-group contract is clear.

Keep the first parallel infra slice small. The success criterion is that new
runtime/backend code plugs into existing registries and state boundaries without
rewriting trainer or algorithm semantics.

