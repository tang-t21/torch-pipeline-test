# LLM Pipeline Parallel Benchmark

This repository contains a small, controllable PyTorch benchmark for studying pipeline parallelism for decoder-only LLM training on a single node with multiple GPUs.

The bench is built around `torch.distributed.pipelining` and is meant for two jobs:

1. Verify that a pipeline schedule and partitioning strategy runs correctly.
2. Compare schedules with a repeatable synthetic workload before moving to a larger training stack.

## What It Benchmarks

The model is a synthetic GPT-style decoder with:

- token embeddings
- positional embeddings
- a stack of causal transformer blocks
- final layer norm
- LM head

Training uses synthetic token batches and next-token cross-entropy. This keeps the experiment focused on pipeline behavior instead of dataloading, dataset quality, or checkpoint complexity.

## Entry Points

- `train.py`
  Runs the non-pipelined reference model on one process. Use this to sanity-check loss, throughput, and memory without distributed execution.
- `pipeline_train.py`
  Runs one pipeline experiment under `torchrun`. This is the main distributed execution path.
- `profile.py`
  Launches a matrix of `pipeline_train.py` runs and writes a consolidated JSON/CSV result table.

## How The Bench Works

### 1. Model Construction

The model lives in `llm_pipeline_bench/model.py`.

It defines:

- `SyntheticGPT`: the full decoder-only model
- `TransformerBlock`: attention + MLP block
- helper functions for stage boundaries and split specs

The model size is controlled entirely from CLI flags:

- `--num-layers`
- `--hidden-size`
- `--num-heads`
- `--mlp-ratio`
- `--vocab-size`
- `--seq-len`

### 2. Reference Training Path

The reference path lives in `llm_pipeline_bench/runtime.py` as `run_reference_training`.

Flow:

1. Build the full model.
2. Generate a synthetic batch of token IDs.
3. Run forward, shifted next-token loss, backward, and optimizer step.
4. Measure average step time and peak memory.
5. Write `results/<run_name>/summary.json`.

This gives you a non-pipeline baseline for correctness and rough performance.

### 3. Distributed Pipeline Path

The pipeline path also lives in `llm_pipeline_bench/runtime.py` as `run_pipeline_training`.

Flow:

1. Read `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`.
2. Choose `nccl` on CUDA or `gloo` on CPU.
3. Create a pipeline process group.
4. Build the local stage modules for the current rank.
5. Wrap those modules as `PipelineStage` objects.
6. Construct the requested schedule.
7. Run full-batch `schedule.step(...)`, which internally splits into microbatches.
8. Step the optimizer and collect timing, loss, and memory metrics.
9. Write per-rank and global summaries under `results/<run_name>/`.

## Partitioning Modes

Two partitioning strategies are supported.

### Manual

Manual partitioning creates each stage module explicitly by selecting the layer range that belongs to that stage.

Use this when you want:

- maximum control over stage contents
- predictable behavior for multi-stage-per-rank schedules
- easier debugging of stage boundaries

### Tracer

Tracer partitioning uses `torch.distributed.pipelining.pipeline(...)` with a generated `split_spec`.

Use this when you want:

- a closer match to PyTorch tutorial-style partitioning
- a lighter partitioning implementation for single-stage-per-rank runs

Current note:

- tracer mode is validated for `gpipe` and `1f1b`
- multi-stage schedules currently use manual partitioning in the matrix runner

## Supported Schedules

### Single-stage-per-rank schedules

- `gpipe`
- `1f1b`

These require `stages_per_rank = 1`.

### Multi-stage-per-rank schedules

- `interleaved1f1b`
- `loopedbfs`

These require `stages_per_rank >= 2`.

For multi-stage schedules, the bench uses looped stage-to-rank placement to match PyTorch's runtime expectations.

## Stage Layout

The bench exposes stage layout in two ways:

- `--stages-per-rank`
- `--stage-layout <world_size>x<stages_per_rank>`

Examples:

- `8x1`: 8 pipeline stages across 8 GPUs
- `8x2`: 16 virtual pipeline stages across 8 GPUs

The configuration validator rejects invalid combinations early, for example:

- `gpipe` with `stages_per_rank=2`
- `interleaved1f1b` with `stages_per_rank=1`
- `num_layers` not divisible by total stages

## Microbatches And Batches

You provide:

- `--global-batch-size`
- `--n-microbatches`

The schedule receives the whole batch and splits it internally into microbatches. This keeps the user-facing API close to a normal training step while still exercising pipeline scheduling behavior.

In general:

- larger `n_microbatches` reduces pipeline bubbles
- too many microbatches can increase overhead

The matrix runner is designed to sweep this parameter.

## Precision

Supported precision modes:

- `float32`
- `bf16`

For CUDA runs, `bf16` uses `torch.autocast`. Tracer-based pipeline construction is also done under the same autocast context so stage metadata matches runtime tensor dtypes.

## Profiling And Metrics

The bench collects:

- average step time
- steps/sec
- tokens/sec
- peak CUDA memory
- mean loss
- parameter count

When `--profile` is enabled, `torch.profiler` traces are written under `traces/<run_name>/rankXX/`.

Run summaries are written under:

- `results/<run_name>/summary.json`
- `results/<run_name>/rank00.json`, `rank01.json`, ...

The matrix runner also writes:

- `results/<output_name>.json`
- `results/<output_name>.csv`

## Typical Workflows

### Quick correctness check

```bash
python train.py \
  --device cpu \
  --num-layers 4 \
  --hidden-size 128 \
  --num-heads 4 \
  --seq-len 64 \
  --global-batch-size 8 \
  --steps 4 \
  --warmup-steps 1
```

### Two-process CPU smoke test

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 2 pipeline_train.py \
  --device cpu \
  --schedule gpipe \
  --partition manual \
  --num-layers 4 \
  --hidden-size 128 \
  --num-heads 4 \
  --seq-len 64 \
  --global-batch-size 8 \
  --n-microbatches 2 \
  --steps 4 \
  --warmup-steps 1
```

### Single experiment on 8 GPUs

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 8 pipeline_train.py \
  --device cuda \
  --precision bf16 \
  --schedule 1f1b \
  --partition manual \
  --num-layers 16 \
  --hidden-size 512 \
  --num-heads 8 \
  --seq-len 512 \
  --global-batch-size 32 \
  --n-microbatches 8 \
  --steps 12 \
  --warmup-steps 3
```

### Matrix sweep on 8 GPUs

```bash
python profile.py \
  --nproc-per-node 8 \
  --device cuda \
  --precision bf16 \
  --num-layers 16 \
  --hidden-size 512 \
  --num-heads 8 \
  --seq-len 512 \
  --global-batch-size 32 \
  --microbatch-sweep 4,8,16
```

## Running On A Kubernetes GPU Pod

The benchmark was validated on:

- pod: `<your-training-pod>`
- namespace: `<your-namespace>`
- hardware: `8 x NVIDIA H100 80GB HBM3`

Example:

```bash
kubectl exec -it <your-training-pod> -n <your-namespace> -- bash
cd <repo-root>
python profile.py \
  --nproc-per-node 8 \
  --device cuda \
  --precision bf16 \
  --num-layers 16 \
  --hidden-size 512 \
  --num-heads 8 \
  --seq-len 512 \
  --global-batch-size 32 \
  --microbatch-sweep 4,8,16
```

## Validated Configurations

The following paths were validated during implementation:

- reference CUDA run
- `gpipe` manual
- `gpipe` tracer
- `1f1b` manual
- `1f1b` tracer
- `interleaved1f1b` manual
- `loopedbfs` manual
- full 8-GPU smoke matrix written to `results/benchmark_8gpu_smoke.json`

## Limitations

- The workload is synthetic, so it is useful for schedule comparison, not end-to-end model quality.
- The matrix runner currently restricts multi-stage schedules to manual partitioning.
- There is no checkpointing, gradient accumulation, or real dataset pipeline yet.
- This is a single-node benchmark harness, not a full 3D parallel training system.

## Where To Extend It

Good next additions:

- real tokenized dataset input
- longer benchmark sweeps for stable throughput numbers
- activation checkpointing comparisons
- optimizer and precision comparisons
- larger model sizes that better saturate H100 compute
- export scripts for plotting throughput vs microbatch count
