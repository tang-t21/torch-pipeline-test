from __future__ import annotations

import csv
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
    pipeline,
)
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from .config import BenchmarkConfig, MULTI_STAGE_SCHEDULES, SINGLE_STAGE_SCHEDULES
from .model import build_full_model, build_manual_stage_model, build_split_spec, get_stage_indices


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    pp_group: dist.ProcessGroup | None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: BenchmarkConfig, local_rank: int) -> tuple[torch.device, str]:
    if config.device == "cpu":
        return torch.device("cpu"), (config.backend or "gloo")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is visible.")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device, (config.backend or "nccl")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device, (config.backend or "nccl")
    return torch.device("cpu"), (config.backend or "gloo")


def initialize_distributed(config: BenchmarkConfig) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device, backend = resolve_device(config, local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    pp_group = dist.new_group() if world_size > 1 else None
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        pp_group=pp_group,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def make_synthetic_batch(config: BenchmarkConfig, step: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + step)
    batch = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.global_batch_size, config.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    return batch.to(device)


def shifted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_targets.view(-1),
    )


def build_autocast_context(config: BenchmarkConfig, device: torch.device):
    if config.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def local_parameter_count(modules: list[torch.nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def default_run_name(config: BenchmarkConfig, world_size: int) -> str:
    return (
        f"{config.partition}-{config.schedule}-"
        f"{world_size}x{config.resolved_stages_per_rank(world_size)}-"
        f"mb{config.n_microbatches}-"
        f"bs{config.global_batch_size}-"
        f"seq{config.seq_len}"
    )


def ensure_run_dir(base_dir: str, run_name: str) -> Path:
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_profiler(config: BenchmarkConfig, device: torch.device, trace_dir: Path):
    if not config.profile:
        return None
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    active_steps = max(1, min(3, config.steps - config.warmup_steps))
    return profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=active_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def write_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_summary_csv(summaries: list[dict[str, Any]], output_path: Path) -> None:
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def create_pipeline_runtime(
    config: BenchmarkConfig,
    ctx: DistributedContext,
) -> tuple[Any, list[torch.nn.Module], list[int], int]:
    stages_per_rank = config.resolved_stages_per_rank(ctx.world_size)
    total_stages = ctx.world_size * stages_per_rank
    stage_mapping_style = "loop" if config.schedule in MULTI_STAGE_SCHEDULES else "contiguous"
    local_stage_indices = get_stage_indices(
        rank=ctx.rank,
        world_size=ctx.world_size,
        stages_per_rank=stages_per_rank,
        style=stage_mapping_style,
    )

    if config.partition == "manual":
        local_modules: list[torch.nn.Module] = []
        local_stages = []
        for stage_index in local_stage_indices:
            module = build_manual_stage_model(config, stage_index, total_stages).to(ctx.device)
            stage = PipelineStage(
                module,
                stage_index,
                total_stages,
                ctx.device,
                group=ctx.pp_group,
            )
            local_modules.append(module)
            local_stages.append(stage)
    else:
        full_model = build_full_model(config).to(ctx.device)
        example_input = make_synthetic_batch(config, step=0, device=ctx.device).chunk(
            config.n_microbatches
        )[0]
        with build_autocast_context(config, ctx.device):
            pipe = pipeline(
                module=full_model,
                mb_args=(example_input,),
                split_spec=build_split_spec(config, total_stages),
            )
        local_stages = [
            pipe.build_stage(stage_index, ctx.device, ctx.pp_group)
            for stage_index in local_stage_indices
        ]
        local_modules = [stage.submod for stage in local_stages]

    schedule_kwargs = {
        "n_microbatches": config.n_microbatches,
        "loss_fn": shifted_cross_entropy,
    }
    if config.schedule == "gpipe":
        runtime_schedule = ScheduleGPipe(local_stages[0], **schedule_kwargs)
    elif config.schedule == "1f1b":
        runtime_schedule = Schedule1F1B(local_stages[0], **schedule_kwargs)
    elif config.schedule == "interleaved1f1b":
        runtime_schedule = ScheduleInterleaved1F1B(local_stages, **schedule_kwargs)
    elif config.schedule == "loopedbfs":
        runtime_schedule = ScheduleLoopedBFS(local_stages, **schedule_kwargs)
    else:
        raise ValueError(f"Unsupported schedule '{config.schedule}'.")

    return runtime_schedule, local_modules, local_stage_indices, total_stages


def aggregate_pipeline_metrics(
    config: BenchmarkConfig,
    ctx: DistributedContext,
    avg_step_time: float,
    peak_memory: float,
    loss_value: float,
    param_count: int,
    total_stages: int,
) -> dict[str, Any]:
    if ctx.world_size == 1:
        global_avg_time = avg_step_time
        global_peak_memory = peak_memory
        global_loss = loss_value
        total_params = param_count
    else:
        device = ctx.device
        time_tensor = torch.tensor(avg_step_time, dtype=torch.float64, device=device)
        memory_tensor = torch.tensor(peak_memory, dtype=torch.float64, device=device)
        loss_tensor = torch.tensor(
            [0.0 if loss_value != loss_value else loss_value, 0.0 if loss_value != loss_value else 1.0],
            dtype=torch.float64,
            device=device,
        )
        params_tensor = torch.tensor(param_count, dtype=torch.int64, device=device)
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(params_tensor, op=dist.ReduceOp.SUM)
        global_avg_time = float(time_tensor.item())
        global_peak_memory = float(memory_tensor.item())
        global_loss = (
            float(loss_tensor[0].item() / loss_tensor[1].item()) if loss_tensor[1].item() else float("nan")
        )
        total_params = int(params_tensor.item())

    tokens_per_step = config.global_batch_size * (config.seq_len - 1)
    return {
        "schedule": config.schedule,
        "partition": config.partition,
        "stage_layout": config.layout_name(ctx.world_size),
        "stages_per_rank": config.resolved_stages_per_rank(ctx.world_size),
        "total_stages": total_stages,
        "world_size": ctx.world_size,
        "device_type": ctx.device.type,
        "backend": ctx.backend,
        "precision": config.precision,
        "n_microbatches": config.n_microbatches,
        "global_batch_size": config.global_batch_size,
        "seq_len": config.seq_len,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "steps": config.steps,
        "warmup_steps": config.warmup_steps,
        "avg_step_time_s": global_avg_time,
        "steps_per_second": 1.0 / global_avg_time,
        "tokens_per_second": tokens_per_step / global_avg_time,
        "peak_memory_bytes": global_peak_memory,
        "mean_loss": global_loss,
        "parameter_count": total_params,
    }


def run_reference_training(config: BenchmarkConfig) -> dict[str, Any]:
    config.validate(world_size=1)
    set_seed(config.seed)

    device, _ = resolve_device(config, local_rank=0)
    model = build_full_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    autocast_context = build_autocast_context(config, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measured_step_times: list[float] = []
    losses: list[float] = []
    for step in range(config.steps):
        batch = make_synthetic_batch(config, step, device)
        optimizer.zero_grad(set_to_none=True)
        maybe_sync(device)
        start_time = time.perf_counter()
        with autocast_context:
            logits = model(batch)
            loss = shifted_cross_entropy(logits, batch)
        loss.backward()
        optimizer.step()
        maybe_sync(device)
        elapsed = time.perf_counter() - start_time
        if step >= config.warmup_steps:
            measured_step_times.append(elapsed)
            losses.append(float(loss.detach().cpu()))

    avg_step_time = sum(measured_step_times) / len(measured_step_times)
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    )
    summary = {
        "mode": "reference",
        "device_type": device.type,
        "precision": config.precision,
        "global_batch_size": config.global_batch_size,
        "seq_len": config.seq_len,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "steps": config.steps,
        "warmup_steps": config.warmup_steps,
        "avg_step_time_s": avg_step_time,
        "steps_per_second": 1.0 / avg_step_time,
        "tokens_per_second": (config.global_batch_size * (config.seq_len - 1)) / avg_step_time,
        "peak_memory_bytes": peak_memory,
        "mean_loss": sum(losses) / len(losses),
        "parameter_count": local_parameter_count([model]),
    }

    run_name = config.run_name or "reference"
    run_dir = ensure_run_dir(config.results_dir, run_name)
    write_summary_json(summary, run_dir / "summary.json")
    return summary


def run_pipeline_training(config: BenchmarkConfig) -> dict[str, Any]:
    ctx = initialize_distributed(config)
    try:
        config.validate(ctx.world_size)
        set_seed(config.seed)

        run_name = config.run_name or default_run_name(config, ctx.world_size)
        results_dir = ensure_run_dir(config.results_dir, run_name)
        trace_dir = ensure_run_dir(config.trace_dir, run_name) / f"rank{ctx.rank:02d}"
        trace_dir.mkdir(parents=True, exist_ok=True)

        runtime_schedule, local_modules, local_stage_indices, total_stages = create_pipeline_runtime(
            config, ctx
        )
        optimizer = torch.optim.AdamW(
            chain.from_iterable(module.parameters() for module in local_modules),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        autocast_context = build_autocast_context(config, ctx.device)
        profiler = create_profiler(config, ctx.device, trace_dir)

        owns_first_stage = 0 in local_stage_indices
        owns_last_stage = (total_stages - 1) in local_stage_indices

        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)

        measured_step_times: list[float] = []
        measured_losses: list[float] = []

        profiler_cm = profiler if profiler is not None else nullcontext()
        with profiler_cm:
            for step in range(config.steps):
                input_batch = (
                    make_synthetic_batch(config, step, ctx.device)
                    if (owns_first_stage or owns_last_stage)
                    else None
                )
                step_args = (input_batch,) if owns_first_stage else ()
                target = input_batch if owns_last_stage else None
                microbatch_losses: list[torch.Tensor] = []

                optimizer.zero_grad(set_to_none=True)
                maybe_sync(ctx.device)
                start_time = time.perf_counter()
                with autocast_context:
                    runtime_schedule.step(*step_args, target=target, losses=microbatch_losses)
                optimizer.step()
                maybe_sync(ctx.device)

                elapsed = time.perf_counter() - start_time
                if profiler is not None:
                    profiler.step()

                if step >= config.warmup_steps:
                    measured_step_times.append(elapsed)
                    if microbatch_losses:
                        microbatch_loss = torch.stack(
                            [loss.detach().float().cpu() for loss in microbatch_losses]
                        ).mean()
                        measured_losses.append(float(microbatch_loss))

        avg_step_time = sum(measured_step_times) / len(measured_step_times)
        peak_memory = (
            float(torch.cuda.max_memory_allocated(ctx.device))
            if ctx.device.type == "cuda"
            else 0.0
        )
        loss_value = sum(measured_losses) / len(measured_losses) if measured_losses else float("nan")
        summary = aggregate_pipeline_metrics(
            config=config,
            ctx=ctx,
            avg_step_time=avg_step_time,
            peak_memory=peak_memory,
            loss_value=loss_value,
            param_count=local_parameter_count(local_modules),
            total_stages=total_stages,
        )
        summary["run_name"] = run_name

        rank_summary = dict(summary)
        rank_summary["rank"] = ctx.rank
        rank_summary["local_stage_indices"] = local_stage_indices
        rank_summary["local_parameter_count"] = local_parameter_count(local_modules)
        rank_summary["local_peak_memory_bytes"] = peak_memory
        rank_summary["local_mean_loss"] = loss_value
        rank_summary["local_avg_step_time_s"] = avg_step_time
        write_summary_json(rank_summary, results_dir / f"rank{ctx.rank:02d}.json")

        if ctx.world_size > 1:
            dist.barrier()
        if ctx.rank == 0:
            write_summary_json(summary, results_dir / "summary.json")
        return summary
    finally:
        cleanup_distributed()
