# bbml: A Better Basic Machine Learning Framework

A modular PyTorch framework for building, training, and fine-tuning ML models with clean abstractions.

## Installation

```bash
pip install -e .
```

## Core Concepts

### Foundation
The base model abstraction. Implements `Trainable`, `Runnable`, and `Serializable` interfaces.

```python
from bbml import Foundation, FoundationConfig, TrainerConfig

class MyModel(Foundation):
    def single_step(self, batch: dict) -> Tensor:
        # Training step, returns loss
        ...
    
    def run(self, input: MyInput) -> MyOutput:
        # Inference
        ...
    
    @property
    def data_transforms(self) -> dict[str, DataTransform]:
        # Define how data fields are transformed
        ...
```

### DataPipe
Combines datasets with transforms and produces DataLoaders.

```python
from bbml import DataPipe

datapipe = (
    DataPipe(batch_size=32, shuffle=True, num_workers=4)
    .add_dataset(my_dataset, index_range=(0, 10000))
    .add_transforms(model.data_transforms)
)

# Split into train/val
train_dp, val_dp = datapipe.split(0.9, 0.1)
```

### Trainer
Handles the training loop, validation, testing, checkpointing, and logging.

```python
from bbml import SimpleTrainer, TrainerConfig

trainer = SimpleTrainer(
    model=my_model,
    train_config=TrainerConfig(
        project="my-project",
        train_epochs=10,
        batch_size=32,
        optimizer="AdamW",
        lr_scheduler="ConstantLR",
        logging_backends="wandb",
    ),
    train_datapipe=train_dp,
    val_datapipe=val_dp,
    test_datapipe=test_dp,
)
trainer.train()
```

### Finetuner
Wraps a Foundation to modify training behavior (e.g., LoRA).

```python
from bbml import LoraFinetuner

finetuner = LoraFinetuner(
    model=my_model,
    module_names=["transformer"],
    r=16,
    lora_alpha=32,
)
# model.get_train_parameters() now returns only LoRA params
# model.save() / model.load() now handle adapter weights
```

### Registry
Type-safe registry pattern for extensibility.

```python
from bbml import Registry

ModelRegistry = Registry[type[Foundation]]("Model")

@ModelRegistry.register("my-model")
class MyModel(Foundation):
    ...

# Later
model_cls = ModelRegistry["my-model"]
```

## Configuration

Configs use Pydantic models and can be composed from YAML files:

```yaml
# config.yaml
project: "my-experiment"
batch_size: 32
optimizer: "AdamW"
lr: 1e-4
lr_scheduler: "ConstantLR"
train_epochs: 10

validation_step_trigger:
  every: 500
save_step_trigger: 1000
```

```python
from bbml import run_interface

def train(config: dict):
    train_cfg = TrainerConfig(**config)
    # ...

if __name__ == "__main__":
    run_interface(train)  # python train.py -c config.yaml -c config_override.yaml -c ...
```

## Logging

Unified logger supporting multiple backends (Weights & Biases, TensorBoard, ClearML).

```python
from bbml import logger

logger.start("wandb", project="my-project")
logger.log({"loss": 0.5, "accuracy": 0.95})
logger.finish()
```

Supports scalars, DataFrames, images, and image batches.

## Project Structure

```
bbml/
├── core/               # Abstract interfaces and base classes
│   ├── foundation.py   # Foundation base class
│   ├── trainer.py      # Trainer base class
│   ├── finetuner.py    # Finetuner base class
│   ├── datapipe.py     # DataPipe and CombinedDataset
│   ├── registry.py     # Generic Registry
│   └── datamodels/     # Pydantic configs
├── train/              # Training implementations
│   └── simple_trainer.py
├── finetuners/         # Finetuner implementations
│   └── lora.py
├── data/               # Data transforms
├── logger/             # Logging backends
└── foundations/        # Example foundation models
    └── gpt2/
```

## Example: GPT-2 Training

```python
from bbml import DataPipe, SimpleTrainer, TrainerConfig, run_interface
from bbml.data.datasets import WikiTextDataset
from bbml.foundations.gpt2 import GPT2Foundation, GPTConfig

def train(cfg: dict):
    train_cfg = TrainerConfig(**cfg)
    model = GPT2Foundation(GPTConfig(**cfg), train_cfg)

    train_dp = (
        DataPipe(batch_size=train_cfg.batch_size, shuffle=True)
        .add_dataset(WikiTextDataset(split="train"))
        .add_transforms(model.data_transforms)
    )

    trainer = SimpleTrainer(model, train_cfg, train_dp, None, None)
    trainer.train()

if __name__ == "__main__":
    run_interface(train)
```

```bash
python train.py -c gpt2.yaml
```

## Template

See `template/` for a starter project structure with experiments, configs, and scripts.

## License
Apache 2.0
