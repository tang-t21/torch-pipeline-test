# LLM Pipeline Parallel Benchmark

Synthetic GPT benchmark for comparing PyTorch pipeline parallel schedules and partitioning modes on multi-GPU training jobs.

## Entry Points

- `train.py`: single-process reference training.
- `pipeline_train.py`: distributed pipeline training via `torchrun`.
- `profile.py`: matrix runner that launches multiple `pipeline_train.py` experiments and consolidates results.

## Quick Start

Reference step timing:

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

Two-process CPU smoke test:

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

8-GPU schedule/profile sweep:

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

Results are written under `results/<run_name>/summary.json`, with per-rank summaries next to the global run summary. When `--profile` is enabled, traces are written under `traces/<run_name>/rankXX/`.
