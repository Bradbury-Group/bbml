"""
Finetuner stub - extend bbml.core.finetuner.Finetuner.

Finetuner wraps a Foundation to modify training behavior (e.g., LoRA adapters).
It monkey-patches three Foundation methods:
  - save() -> save only adapter weights
  - load() -> load adapter weights
  - get_train_parameters() -> return only trainable adapter params

Lifecycle in experiment:
  1. foundation = MyFoundation(config, train_config)
  2. finetuner = MyFinetuner(foundation)  # wraps and patches
  3. trainer.train()  # trains adapter params only
  4. foundation.save(path)  # calls finetuner.save() (adapter weights)
  5. finetuner.remove()  # restores original methods (optional)

bbml provides LoraFinetuner (see bbml/finetuners/lora.py) for PEFT LoRA.
For custom finetuning strategies, extend the base Finetuner below.

See bbml/core/finetuner.py for the base class.
"""

from myproject.finetuners.example import MyFinetuner

__all__ = ["MyFinetuner"]
