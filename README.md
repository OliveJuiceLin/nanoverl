# nanoverl

## What Is Nanoverl

`nanoverl` is a small, research-oriented RL framework for large language models.

It is inspired by `verl`, but it deliberately keeps a much smaller surface area:

- one clear synchronous trainer loop
- explicit policy / reference / value / rollout boundaries
- enough engineering for real PPO / GRPO / RLOO experiments
- fewer layers that get in the way of reading and modifying the code

This repository is not trying to be a feature-complete replacement for `verl`.
Its goal is to be a clean RL core that is easy to understand, easy to modify, and still usable for real experiments.

## Current Status

The current repository should be understood like this:

- The formal mainline is still a Phase 1 style synchronous RL core.
- Some Phase 2 usability work is already present, such as algorithm plugins, GRPO / RLOO support, validation summaries, and lightweight debug artifacts.
- There are already a few thin Phase 3 backend slices, especially single-node FSDP and local `vllm` rollout integration.
- Not every exposed option should be read as a mature production feature. Some parts are intentionally thin and kept as future extension points.

Today, the most trustworthy path is still:

- typed config
- stateful dataset / dataloader
- synchronous trainer
- explicit policy / reference / value workers
- algorithm-selected rollout -> reward -> advantage -> update semantics
- checkpoint / validation / logging on the same main loop

## Repo Map

The repository is organized around the RL training path.

- `nanoverl/algos`
  - Algorithm plugins plus shared RL math such as policy losses, KL penalties, and advantage estimators.
- `nanoverl/backends`
  - Thin backend-specific utilities for Hugging Face, `vllm`, and training backends such as FSDP.
- `nanoverl/checkpoint`
  - Local checkpoint save / load helpers.
- `nanoverl/config.py`
  - The typed config tree and config validation logic.
- `nanoverl/core`
  - Small shared core data structures, mainly `RLBatch`.
- `nanoverl/data`
  - Built-in dataset loading and the checkpointable data loader / sampler.
- `nanoverl/distributed`
  - Small runtime helpers for `torch.distributed` and future distributed integrations.
- `nanoverl/logging`
  - Metrics helpers and tracker backends.
- `nanoverl/reward`
  - Reward function loading and reward result shaping.
- `nanoverl/rollout`
  - Rollout engine interfaces and concrete backends such as debug, HF, and `vllm`.
- `nanoverl/trainer`
  - The main trainer loop, validation helpers, and debug artifact writers.
- `nanoverl/workers`
  - Policy / reference / value worker interfaces and implementations.
- `examples`
  - Example configs, reward functions, and local experiment helpers.
- `tests`
  - Regression tests for the trainer loop, RL math, batch behavior, and backend slices.

## How Training Works

The trainer owns the lifecycle and the selected algorithm owns the RL step semantics.

1. Load a batch from the stateful train loader.
2. Dispatch the batch to the configured `algorithm.name` plugin.
3. The algorithm prepares rollout batches, computes rewards, log-probs, values when needed, advantages, and worker updates.
4. The trainer logs metrics, runs validation, and saves checkpoints on the same driver-owned path.

This ordering is the main thing `nanoverl` tries to preserve and keep readable.

## Supported Paths

The repository currently has these meaningful paths:

- `debug` worker + `debug` rollout
  - Small smoke-test path with no heavy runtime dependency expectations.
- local HF policy / reference / value + HF rollout
  - The clearest fully local real-training path.
- single-node FSDP workers + HF rollout
  - The first serious multi-GPU training slice.
- local HF or FSDP training workers + local `vllm` rollout
  - A thin rollout backend extension that reuses the same trainer loop.
- built-in algorithm plugins
  - `ppo`: critic-backed GAE with clipped PPO policy loss.
  - `grpo`: actor-only grouped rollout with GRPO advantages and PPO-style clipped policy loss.
  - `rloo`: actor-only grouped rollout with leave-one-out advantages and REINFORCE policy loss.

## Not Yet Supported

These should be treated as intentionally missing, intentionally thin, or future work:

- async trainers
- off-policy trainers
- full Ray worker orchestration
- multi-turn tool rollout
- multiple mature rollout runtimes with parity guarantees
- production-scale serving workflows

## Roadmap Snapshot

The current roadmap can be summarized as:

1. Keep the synchronous RL core small and correct.
2. Make PPO / GRPO / RLOO research work comfortable on top of the same core contracts.
3. Add backend breadth only when it materially expands research coverage.
4. Avoid reintroducing `verl`-style complexity unless it clearly pays for itself.

Near-term priorities are still:

- cleaner core abstractions
- correct and maintainable RL math
- reliable single-node training
- better experiment ergonomics

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
python -m unittest discover -s tests -p 'test_*.py'
```

## Maintenance Rule

`README.md` is the single long-term project document.

From now on:

- update this README whenever the repo structure changes
- update this README whenever the supported capability boundary changes
- update this README whenever the project phase or roadmap emphasis changes

If a reader wants to understand what `nanoverl` is doing, this file should be enough to get oriented quickly.
