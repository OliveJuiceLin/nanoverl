# nanoverl

`nanoverl` is a simplified, research-oriented reimplementation of the RL training core in `verl`.

It is designed for researchers and students who want to **understand, modify, and extend RL algorithms for large language models** without being overwhelmed by the full engineering complexity of the original codebase.

## Why nanoverl?

The original `verl` is a powerful and production-oriented framework. However, for research and learning purposes, it can be difficult to navigate because it contains:

- multiple layers of abstraction
- many engineering paths for compatibility and scalability
- duplicated or parallel implementations for different backends
- features not essential to core RL algorithm research
- code paths for workflows outside the main RL focus

`nanoverl` aims to preserve the **core training capability and code quality** of `verl`, while removing parts that are not necessary for:

- understanding RLHF / RLAIF style training loops
- studying PPO / GRPO / related policy optimization algorithms
- modifying actor / critic / reward / rollout interactions
- running clean and reproducible research experiments on LLM RL training

In short, `nanoverl` is **not a toy project**.  
It is a **minimal but real** RL training framework derived from the design principles and core workflow of `verl`.

## Project Goal

The goal of `nanoverl` is:

- keep the **core RL training pipeline** intact
- make the codebase **clear, compact, and readable**
- make it easier to **modify algorithms**
- reduce unnecessary framework complexity
- keep enough engineering quality for real experiments

This project is mainly focused on **reinforcement learning for large language models**.

## What nanoverl focuses on

`nanoverl` mainly focuses on the RL part of the stack:

- training loop
- policy optimization algorithm
- actor / critic interaction
- reward computation interface
- rollout / sampling interface
- batch preparation for RL updates
- distributed execution path that is necessary for RL training
- logging, checkpointing, and experiment control

These are the parts that should become smaller, cleaner, and easier to understand than in the original `verl`.

## What nanoverl does NOT try to simplify aggressively

Some components are important for a full pipeline, but are **not the main target of “nano”-ization** in this project:

- inference engine integration
- generation backend details
- SFT training pipeline
- highly specialized or production-only serving logic

For these parts, `nanoverl` may:

- reuse mature components directly
- wrap existing implementations with thinner interfaces
- keep only the minimum integration needed by RL training
- omit SFT entirely in the first versions

In other words, `nanoverl` is **RL-first**, not a full reimplementation of every feature in `verl`.

## Design Principles

`nanoverl` follows several design principles:

### 1. RL-first
The primary goal is to support research on RL algorithms for LLMs, not to support every possible training mode.

### 2. Fewer abstractions, clearer boundaries
Every major concept should map to a small number of files and explicit interfaces.

### 3. One obvious path
Whenever possible, prefer one clean default implementation over many equivalent code paths.

### 4. Preserve real usability
The framework must remain usable for actual experiments, not just for demonstration.

### 5. Easy to read, easy to hack
A new reader should be able to trace one full RL run from config to rollout to update without getting lost.

## Scope

### In scope
- PPO-like RL training for LLMs
- clean trainer loop
- actor / critic / reward / rollout abstractions
- minimal distributed orchestration required by the RL pipeline
- checkpointing / logging / config system
- evaluation hooks useful for RL research

### Out of scope or de-prioritized
- full SFT pipeline
- all historical / legacy backends
- all duplicated launcher or worker variants
- every optimization path from `verl`
- production-serving-oriented code
- heavy compatibility layers unless clearly necessary

## Expected Output of the Project

The expected result is a framework that:

- can run real RL training jobs
- is significantly easier to read than `verl`
- has a smaller and cleaner architecture
- is easier to modify for algorithm research
- can serve as a strong codebase for learning, experimentation, and secondary development

## Relationship to verl

`nanoverl` is inspired by and structurally grounded in `verl`, but it is **not intended to be a line-by-line copy**.

The guiding rule is:

> preserve the essential RL functionality, while rewriting the codebase into a clearer and more research-friendly form.

Some modules may be reimplemented from scratch.  
Some may be lightly wrapped around existing implementations.  
Some may be removed if they do not serve the core RL research workflow.

## Development Philosophy

The project will be developed in phases:

1. understand the original `verl` RL pipeline
2. identify the truly essential modules
3. define a compact architecture for `nanoverl`
4. implement a minimal but usable end-to-end RL path
5. add necessary engineering support without reintroducing bloat

The first priority is always:

- correctness
- clarity
- maintainability

not feature count.

## Who is this for?

`nanoverl` is mainly for:

- researchers studying RL for LLMs
- students learning RLHF / policy optimization systems
- developers who want a clean base for RL algorithm experimentation
- people who find the original `verl` too large to modify comfortably

## Current Scaffold

The repository now includes the first implementation pass of the planned architecture:

- `nanoverl.core.RLBatch`
  - a small RL-focused batch object with `repeat`, `union`, `chunk`, `concat`, `reorder`, and `pad_to_divisor`
- `nanoverl.config.TrainerConfig`
  - one typed config tree for the trainer, algorithm, actor, critic, reference, rollout, reward, and runtime
- `nanoverl.trainer.RLTrainer`
  - a driver-owned synchronous control loop in the intended PPO order:
    rollout -> reward -> old/ref/value passes -> advantages -> critic -> actor -> rollout sync
- `nanoverl.reward.RewardManager`
  - a Python reward interface that expands scalar rewards into terminal-token reward tensors
- `nanoverl.rollout.DebugRolloutEngine`
  - a deterministic rollout backend for smoke tests and algorithm debugging
- `nanoverl.workers.Debug*Worker`
  - explicit policy, reference, and value worker boundaries
- `nanoverl.checkpoint.CheckpointManager`
  - local save/resume of trainer and worker state

This is intentionally a clean, readable RL core. The debug and local HF paths are runnable today, the first single-node FSDP training path exists, and a thin synchronous `vllm` rollout backend can now be paired with the existing HF or FSDP training workers without changing the trainer loop.

## Quickstart

Run the built-in debug PPO example:

```bash
python3 -m nanoverl.cli.train_rl --config examples/configs/debug_ppo.json
```

Run the local HF PPO example:

```bash
python3 -m nanoverl.cli.train_rl --config examples/configs/hf_local_ppo.json
```

Run the first single-node FSDP training preset:

```bash
torchrun --standalone --nproc_per_node=4 -m nanoverl.cli.train_rl --config examples/configs/fsdp_single_node_ppo.json
```

Run the local HF actor plus vLLM rollout preset:

```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate vllm
python3 -m nanoverl.cli.train_rl --config examples/configs/hf_vllm_local_ppo.json
```

The packaged CLI aliases are:

```bash
nanoverl-train --config examples/configs/debug_ppo.json
nanoverl-train-rl --config examples/configs/debug_ppo.json
```

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

## Notes On Dependencies

The repository is implemented to:

- run the debug path with only the Python standard library
- keep `torch`/`ray` as explicit optional dependencies in `pyproject.toml`
- expose a real local HF path, a first single-node FSDP path, and a thin synchronous `vllm` rollout path before adding heavier runtime layers such as Ray

The current `vllm` rollout slice is intentionally small:

- synchronous only
- same rollout contract as the debug and HF engines
- no async server mode
- no Ray rollout workers
- no multi-turn / tool rollout
- rollout tensor parallel size currently stays at `1` in this thin local design
