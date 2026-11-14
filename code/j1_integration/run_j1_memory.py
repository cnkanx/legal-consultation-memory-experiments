from __future__ import annotations

import os
import sys
from pathlib import Path


repo_root_value = os.environ.get("J1BENCH_ROOT")
if not repo_root_value:
    raise RuntimeError(
        "Set J1BENCH_ROOT to the root of the local J1-Bench checkout"
    )
REPO_ROOT = Path(repo_root_value)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import engine  # noqa: F401
import agents  # noqa: F401
import scenario  # noqa: F401
import utils  # noqa: F401

import j1_memory_trainee  # noqa: F401
from utils.options import get_parser
from utils.register import registry


def main() -> None:
    args = get_parser()
    if args.scenario != "J1Bench.Scenario.LC":
        raise ValueError("J1 memory integration currently supports LC only")
    if args.trainee != "Agent.Trainee.LC_VersionedMemory":
        raise ValueError(
            "Pass --trainee Agent.Trainee.LC_VersionedMemory"
        )
    scenario_runner = registry.get_class(args.scenario)(args)
    scenario_runner.run()


if __name__ == "__main__":
    main()
