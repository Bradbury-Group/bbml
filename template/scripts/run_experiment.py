import argparse
from pathlib import Path

from myproject.experiments import ExperimentRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--name", type=str, required=True)
    # Meta config options
    parser.add_argument("--report-dir", type=Path, default=None,)
    parser.add_argument("--seed", type=int, default=None,)
    parser.add_argument("--iterations", type=int, default=None,)
    
    args = parser.parse_args()

    name = args.name
    if name not in ExperimentRegistry:
        available = list(ExperimentRegistry.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

    ExperimentCls = ExperimentRegistry.get(name)

    if args.report_dir:
        ExperimentCls.update_meta(report_dir=args.report_dir)
    if args.seed:
        ExperimentCls.update_meta(seed=args.seed)
    if args.iterations:
        ExperimentCls.update_meta(iterations=args.iterations)
    
    experiment = ExperimentCls(name=name)
    
    experiment.run_all()


if __name__ == "__main__":
    main()
