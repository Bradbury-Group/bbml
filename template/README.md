# myproject

Template ML experiment repo built on [bbml](https://github.com/Bradbury-Group/bbml).

## Setup

```bash
# Rename the project
python rename.py your_project_name

# Install
pip install -e .
```

## Usage

### Experiments (class-based)

```bash
# List available experiments
python scripts/list_experiments.py

# Run an experiment
python scripts/run_experiment.py --name MyTrainingExperiment
# or
python scripts/run_experiment.py --name MyAnalysisExperiment
```

### General Training (config-based)

```bash
python scripts/run_training.py -c configs/base.yaml -c configs/training/example.yaml
```


## Structure

```
myproject/
├── data/           # Datasets and transforms
├── experiments/    # Experiment classes (register with @ExperimentRegistry.register)
├── datamodels.py   # Pydantic configs and I/O models
├── foundation.py   # Your model (extends bbml.Foundation)
└── finetuner.py    # Optional adapter training

configs/
├── base.yaml       # Shared defaults
└── training/       # Experiment-specific overrides

scripts/
├── run_training.py     # Config-driven training
├── run_experiment.py   # Run registered experiments
└── list_experiments.py # Show available experiments
```

## Extending

1. Implement `MyFoundation` in `foundation.py`
2. Implement `MyDataset` in `data/datasets.py`
3. Add experiments in `experiments/example_experiments.py`
4. Configure in `configs/`
