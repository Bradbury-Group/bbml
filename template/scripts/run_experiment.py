#!/usr/bin/env python
"""Run experiment: python scripts/run_experiment.py --name example_training -c configs/base.yaml"""

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.utils import config_compose

from myproject.experiments import ExperimentRegistry
from myproject.experiments.base import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    composed_config = config_compose(args.config)

    if args.dry_run:
        print(composed_config)
        return

    if args.name not in ExperimentRegistry:
        available = list(ExperimentRegistry.keys())
        print(f"Error: Unknown experiment '{args.name}'. Available: {available}", file=sys.stderr)
        sys.exit(1)

    experiment_cls = ExperimentRegistry[args.name]
    experiment_config = ExperimentConfig(name=args.name, **composed_config.get("experiment", {}))

    sig = inspect.signature(experiment_cls.__init__)
    if "trainer_config" in sig.parameters:
        experiment = experiment_cls(
            config=experiment_config,
            trainer_config=TrainerConfig(**composed_config),
        )
    else:
        experiment = experiment_cls(config=experiment_config)

    experiment.run_all()


if __name__ == "__main__":
    main()
