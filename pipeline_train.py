from __future__ import annotations

import json
import os

from llm_pipeline_bench.cli import build_parser, config_from_args
from llm_pipeline_bench.runtime import run_pipeline_training


def main() -> None:
    parser = build_parser("Distributed pipeline trainer for synthetic GPT benchmarking.")
    args = parser.parse_args()
    config = config_from_args(args)
    summary = run_pipeline_training(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
