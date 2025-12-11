#!/usr/bin/env python
"""Run multiple experiments with filtering support.

Examples:
    python scripts/run_multi.py --all -c configs/base.yaml
    python scripts/run_multi.py --names exp1 exp2 -c configs/base.yaml
    python scripts/run_multi.py --filter-contains training -c configs/base.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from myproject.experiments.registry import ExperimentRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--names", type=str, nargs="+")
    group.add_argument("--all", action="store_true")
    group.add_argument("--filter-startswith", type=str, metavar="PREFIX")
    group.add_argument("--filter-contains", type=str, metavar="SUBSTR")
    parser.add_argument("-c", "--config", type=str, action="append", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    # Determine experiments to run
    if args.all:
        names = list(ExperimentRegistry.keys())
    elif args.names:
        names = args.names
    elif args.filter_startswith:
        names = ExperimentRegistry.filter(startswith=args.filter_startswith)
    elif args.filter_contains:
        names = ExperimentRegistry.filter(contains=args.filter_contains)

    if not names:
        print("No experiments matched")
        sys.exit(1)

    print(f"Running {len(names)} experiments: {names}")
    run_script = Path(__file__).parent / "run_experiment.py"

    failed = []
    for name in names:
        cmd = [sys.executable, str(run_script), "--name", name]
        for c in args.config:
            cmd.extend(["-c", c])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(name)
            if not args.continue_on_error:
                sys.exit(1)

    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
