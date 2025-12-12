"""
Run training with heirarchical configurations like:
    python run_training.py -c config1.yaml -c config2.yaml 
"""


import torch
from torch.utils.data import Dataset
from bbml import (
    DataPipe,
    Foundation,
    FoundationConfig,
    SimpleTrainer,
    Trainer,
    TrainerConfig,
    run_interface
)


def train_fn(cfg_dict: dict):

    trainer_config = TrainerConfig(**cfg_dict)
    foundation_cfg = FoundationConfig(**cfg_dict)

    foundation: Foundation = Foundation(
        config=foundation_cfg,
        train_config=trainer_config,
    )
    foundation.to("cuda" if torch.cuda.is_available() else "cpu")
    
    datapipe = DataPipe(batch_size=trainer_config.batch_size, shuffle=True)
    datapipe.add_dataset(Dataset(...))
    datapipe.add_transforms(foundation.data_transforms)
    val_datapipe, train_pipe = datapipe.split(10, 90)
    
    trainer: Trainer = SimpleTrainer(
        model=foundation,
        train_config=trainer_config,
        train_datapipe=train_pipe,
        val_datapipe=val_datapipe,
    )

    trainer.train()


if __name__ == "__main__":
    run_interface(train_fn)