"""Example finetuner implementation."""

from pathlib import Path

from torch.optim.optimizer import ParamsT

from bbml import Finetuner, Foundation


class MyFinetuner(Finetuner):
    """Custom finetuner. Wraps Foundation to train subset of parameters.

    For LoRA, use bbml.finetuners.LoraFinetuner directly:
        from bbml.finetuners import LoraFinetuner
        finetuner = LoraFinetuner(
            foundation,
            module_names=["transformer"],  # which submodules to wrap
            module_configs=LoraConfig(r=16, lora_alpha=32),
        )
    """

    def __init__(self, model: Foundation):
        super().__init__(model)
        # Initialize your adapter here:
        # self.adapter = nn.Linear(...)
        # Inject into model:
        # self.model.some_layer = self.adapter
        raise NotImplementedError

    def get_train_parameters(self) -> ParamsT:
        """
        Return only adapter parameters for optimizer.

        Called by trainer via foundation.get_train_parameters().
        Original foundation params are in self.model.parameters().
        """
        # return [{"params": self.adapter.parameters()}]
        raise NotImplementedError

    def save(self, save_path: str | Path) -> None:
        """
        Save adapter weights only.

        Called via foundation.save() after wrapping.
        Base model weights saved separately if needed.
        """
        # torch.save(self.adapter.state_dict(), save_path / "adapter.pt")
        raise NotImplementedError

    def load(self, load_path: str | Path) -> None:
        """
        Load adapter weights.
        Called via foundation.load() after wrapping.
        """
        # self.adapter.load_state_dict(torch.load(load_path / "adapter.pt"))
        raise NotImplementedError
