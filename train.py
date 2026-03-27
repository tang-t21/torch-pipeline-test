from __future__ import annotations

import json

from llm_pipeline_bench.cli import build_parser, config_from_args
from llm_pipeline_bench.runtime import run_reference_training


def main() -> None:
    parser = build_parser("Single-process reference trainer for the synthetic GPT benchmark.")
    args = parser.parse_args()
    config = config_from_args(args)
    summary = run_reference_training(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
