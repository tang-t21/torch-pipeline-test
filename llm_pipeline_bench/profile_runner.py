from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .config import MULTI_STAGE_SCHEDULES, SINGLE_STAGE_SCHEDULES


DEFAULT_SCHEDULES = ["gpipe", "1f1b", "interleaved1f1b", "loopedbfs"]
DEFAULT_PARTITIONS = ["manual", "tracer"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matrix of pipeline benchmark profiles.")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--schedules", default=",".join(DEFAULT_SCHEDULES))
    parser.add_argument("--partitions", default=",".join(DEFAULT_PARTITIONS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32", "bf16"], default="float32")
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--microbatch-sweep", default="4,8,16")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--output-name", default="benchmark_matrix")
    return parser


def iter_experiments(args: argparse.Namespace):
    schedules = parse_csv_list(args.schedules)
    partitions = parse_csv_list(args.partitions)
    microbatches = [int(value) for value in parse_csv_list(args.microbatch_sweep)]

    for schedule_name in schedules:
        if schedule_name in SINGLE_STAGE_SCHEDULES:
            stages_per_rank = 1
        elif schedule_name in MULTI_STAGE_SCHEDULES:
            stages_per_rank = 2
        else:
            raise ValueError(f"Unsupported schedule '{schedule_name}'.")

        allowed_partitions = partitions
        if schedule_name in MULTI_STAGE_SCHEDULES:
            allowed_partitions = [partition for partition in partitions if partition == "manual"] or ["manual"]

        for partition_name in allowed_partitions:
            for n_microbatches in microbatches:
                run_name = (
                    f"{partition_name}-{schedule_name}-"
                    f"{args.nproc_per_node}x{stages_per_rank}-"
                    f"mb{n_microbatches}-bs{args.global_batch_size}-seq{args.seq_len}"
                )
                yield {
                    "schedule": schedule_name,
                    "partition": partition_name,
                    "stages_per_rank": stages_per_rank,
                    "n_microbatches": n_microbatches,
                    "run_name": run_name,
                }


def run_experiment(args: argparse.Namespace, experiment: dict[str, object]) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        str(args.nnodes),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "pipeline_train.py",
        "--schedule",
        str(experiment["schedule"]),
        "--partition",
        str(experiment["partition"]),
        "--stages-per-rank",
        str(experiment["stages_per_rank"]),
        "--num-layers",
        str(args.num_layers),
        "--hidden-size",
        str(args.hidden_size),
        "--num-heads",
        str(args.num_heads),
        "--mlp-ratio",
        str(args.mlp_ratio),
        "--vocab-size",
        str(args.vocab_size),
        "--seq-len",
        str(args.seq_len),
        "--global-batch-size",
        str(args.global_batch_size),
        "--n-microbatches",
        str(experiment["n_microbatches"]),
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--precision",
        str(args.precision),
        "--device",
        str(args.device),
        "--profile",
        "--trace-dir",
        args.trace_dir,
        "--results-dir",
        args.results_dir,
        "--run-name",
        str(experiment["run_name"]),
    ]
    subprocess.run(command, check=True)
    summary_path = Path(args.results_dir) / str(experiment["run_name"]) / "summary.json"
    return json.loads(summary_path.read_text())


def write_matrix(output_path: Path, summaries: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    csv_path = output_path.with_suffix(".csv")
    if summaries:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summaries = [run_experiment(args, experiment) for experiment in iter_experiments(args)]
    output_path = Path(args.results_dir) / f"{args.output_name}.json"
    write_matrix(output_path, summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))

