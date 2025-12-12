import argparse

from myproject.experiments import ExperimentRegistry

def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    name = args.name

    if name not in ExperimentRegistry:
        available = list(ExperimentRegistry.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

    experiment_cls = ExperimentRegistry.get(name)
    experiment = experiment_cls(name=name)
    
    experiment.run_all()


if __name__ == "__main__":
    main()
