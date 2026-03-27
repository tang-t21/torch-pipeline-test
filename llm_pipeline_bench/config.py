from __future__ import annotations

from dataclasses import asdict, dataclass


SINGLE_STAGE_SCHEDULES = {"gpipe", "1f1b"}
MULTI_STAGE_SCHEDULES = {"interleaved1f1b", "loopedbfs"}
ALL_SCHEDULES = SINGLE_STAGE_SCHEDULES | MULTI_STAGE_SCHEDULES
ALL_PARTITIONS = {"manual", "tracer"}
ALL_PRECISIONS = {"float32", "bf16"}
ALL_DEVICES = {"auto", "cpu", "cuda"}


def normalize_schedule(value: str) -> str:
    return value.lower().strip()


def normalize_partition(value: str) -> str:
    return value.lower().strip()


def parse_stage_layout(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise ValueError(
            f"Invalid stage layout '{value}'. Expected format '<world_size>x<stages_per_rank>'."
        )
    world_size, stages_per_rank = (int(piece) for piece in pieces)
    if world_size < 1 or stages_per_rank < 1:
        raise ValueError(f"Invalid stage layout '{value}'. Values must be positive.")
    return world_size, stages_per_rank


@dataclass
class BenchmarkConfig:
    schedule: str = "gpipe"
    partition: str = "manual"
    num_layers: int = 16
    hidden_size: int = 512
    num_heads: int = 8
    mlp_ratio: int = 4
    vocab_size: int = 32000
    seq_len: int = 512
    global_batch_size: int = 32
    n_microbatches: int = 4
    stages_per_rank: int = 1
    stage_layout: str = ""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 10
    warmup_steps: int = 2
    seed: int = 1234
    precision: str = "float32"
    device: str = "auto"
    backend: str = ""
    profile: bool = False
    trace_dir: str = "traces"
    results_dir: str = "results"
    run_name: str = ""

    def __post_init__(self) -> None:
        self.schedule = normalize_schedule(self.schedule)
        self.partition = normalize_partition(self.partition)
        self.precision = self.precision.lower().strip()
        self.device = self.device.lower().strip()

    def resolved_stages_per_rank(self, world_size: int) -> int:
        if self.stage_layout:
            layout_world_size, stages_per_rank = parse_stage_layout(self.stage_layout)
            if layout_world_size != world_size:
                raise ValueError(
                    f"Stage layout '{self.stage_layout}' expects world size {layout_world_size}, "
                    f"but WORLD_SIZE is {world_size}."
                )
            return stages_per_rank
        return self.stages_per_rank

    def total_stages(self, world_size: int) -> int:
        return world_size * self.resolved_stages_per_rank(world_size)

    def layout_name(self, world_size: int) -> str:
        return f"{world_size}x{self.resolved_stages_per_rank(world_size)}"

    def validate(self, world_size: int) -> None:
        if self.schedule not in ALL_SCHEDULES:
            raise ValueError(
                f"Unsupported schedule '{self.schedule}'. Expected one of {sorted(ALL_SCHEDULES)}."
            )
        if self.partition not in ALL_PARTITIONS:
            raise ValueError(
                f"Unsupported partition '{self.partition}'. Expected one of {sorted(ALL_PARTITIONS)}."
            )
        if self.precision not in ALL_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. Expected one of {sorted(ALL_PRECISIONS)}."
            )
        if self.device not in ALL_DEVICES:
            raise ValueError(
                f"Unsupported device '{self.device}'. Expected one of {sorted(ALL_DEVICES)}."
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if self.global_batch_size < 1:
            raise ValueError("global_batch_size must be positive.")
        if self.n_microbatches < 1:
            raise ValueError("n_microbatches must be positive.")
        if self.global_batch_size % self.n_microbatches != 0:
            raise ValueError("global_batch_size must be divisible by n_microbatches.")
        if self.steps < 1:
            raise ValueError("steps must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.steps <= self.warmup_steps:
            raise ValueError("steps must be greater than warmup_steps.")

        stages_per_rank = self.resolved_stages_per_rank(world_size)
        if self.schedule in SINGLE_STAGE_SCHEDULES and stages_per_rank != 1:
            raise ValueError(
                f"Schedule '{self.schedule}' requires exactly one stage per rank."
            )
        if self.schedule in MULTI_STAGE_SCHEDULES and stages_per_rank < 2:
            raise ValueError(
                f"Schedule '{self.schedule}' requires at least two stages per rank."
            )

        total_stages = self.total_stages(world_size)
        if self.num_layers % total_stages != 0:
            raise ValueError(
                f"num_layers={self.num_layers} must be divisible by total_stages={total_stages}."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
