from myproject.experiments import ExperimentRegistry

if __name__ == "__main__":
    for name, cls in ExperimentRegistry.items():
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"{name}: {cls.__name__} - {doc}")
