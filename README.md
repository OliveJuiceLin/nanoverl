# nanoverl

`nanoverl` is a small reinforcement learning framework for large language model
research. It is designed to be easy to read, easy to modify, and useful for
local or single-node PPO/GRPO/RLOO experiments.

It borrows ideas from `verl`, but keeps the core much smaller: one synchronous
trainer loop, explicit worker and rollout boundaries, typed configs, and
algorithm plugins that own the RL step semantics.

## What You Can Do Today

- Run debug PPO and RLOO smoke experiments without heavy dependencies.
- Train local Hugging Face policy/reference/value workers with HF rollout.
- Use single-node FSDP workers for the first multi-GPU training slice.
- Use local HF or FSDP training workers with a thin local `vllm` rollout backend.
- Add new algorithms through the plugin interface without rewriting the trainer.

Built-in algorithms:

- `ppo`: critic-backed GAE with clipped PPO policy loss.
- `grpo`: actor-only grouped rollout with GRPO advantages and clipped policy loss.
- `rloo`: actor-only grouped rollout with leave-one-out advantages and
  REINFORCE policy loss.

## Quickstart

Debug PPO:

```bash
python3 -m nanoverl.cli.train_rl --config examples/configs/debug_ppo.json
```

Debug RLOO:

```bash
python3 -m nanoverl.cli.train_rl --config examples/configs/debug_rloo.json
```

Local HF PPO:

```bash
python3 -m nanoverl.cli.train_rl --config examples/configs/hf_local_ppo.json
```

Single-node FSDP PPO:

```bash
torchrun --standalone --nproc_per_node=4 -m nanoverl.cli.train_rl --config examples/configs/fsdp_single_node_ppo.json
```

Run tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## How Training Works

The trainer owns lifecycle; the algorithm owns RL semantics.

1. The trainer reads a batch from a checkpointable dataloader.
2. The selected `algorithm.name` plugin runs the RL step.
3. The algorithm prepares rollout input, calls rollout generation, computes
   rewards, log-probs, values when needed, advantages, and updates.
4. After actor updates, the algorithm asks the trainer to sync policy weights
   into the rollout engine.
5. The trainer logs metrics, runs validation, and saves checkpoints.

Policy sync and checkpointing have separate state paths:

- `policy_worker.state_dict()` is checkpoint/resume state.
- `policy_worker.policy_state_dict()` is the minimal actor -> rollout export.
- `rollout_engine.state_dict()` is lightweight rollout-side state and does not
  save actor model weights.

Checkpoints are versioned and structured into `trainer_state`, `loader_state`,
`worker_state`, `rollout_state`, and `config`.

## Config Basics

Choose an algorithm with:

```json
{
  "algorithm": {
    "name": "ppo"
  }
}
```

Actor and critic update fields use algorithm-neutral names:

- `mini_batch_size`
- `update_epochs`
- `micro_batch_size`

There is no public `ray` config section today. Parallel runtime and placement
settings will be added when the corresponding infra is implemented.

Unknown config fields fail fast instead of being ignored.

## Repository Map

- `nanoverl/algos`: algorithm plugins, advantage estimators, policy losses, KL
  helpers, and value loss.
- `nanoverl/backends`: backend-specific helpers for Hugging Face, FSDP, and
  `vllm`.
- `nanoverl/checkpoint`: local checkpoint manager.
- `nanoverl/config.py`: typed config tree and validation.
- `nanoverl/core`: shared data structures such as `RLBatch`.
- `nanoverl/data`: JSON dataset and stateful dataloader.
- `nanoverl/distributed`: current `torch.distributed` runtime helper.
- `nanoverl/logging`: metric and tracker utilities.
- `nanoverl/reward`: reward function loading and reward result handling.
- `nanoverl/rollout`: rollout interface, backend registry, rollout engines, and
  policy sync helper.
- `nanoverl/trainer`: synchronous trainer, validation, checkpoint orchestration,
  and artifact writing.
- `nanoverl/workers`: policy/reference/value interfaces, backend registry, and
  worker implementations.
- `examples`: runnable configs and reward/data helpers.
- `tests`: regression tests for the core trainer, algorithms, config, checkpoint,
  and backend slices.

## Current Limits

`nanoverl` is intentionally small. These are not mature supported paths yet:

- async trainer
- off-policy trainer
- full worker-group orchestration
- production-scale serving
- mature multi-runtime rollout parity
- packed-batch training path

The next development phase can start adding real parallel infra: worker groups,
placement/runtime config, parameter sync management, and candidate integrations
inspired by `nanoRLHF` components such as `nanovllm`, `nanotron`, and `nanoray`.

## For Developers

`README.md` is for users of the project. Development context, design principles,
current phase information, and reference-project notes live in `AGENTS.md`.
