"""
Run training with hierarchical configurations like:
    python run_training.py -c config1.yaml -c config2.yaml 
"""


import torch
from bbml import (
    DataPipe,
    SimpleTrainer,
    Trainer,
    TrainerConfig,
    run_interface,
)

from myproject.data.datasets import MyDataset
from myproject.datamodels import MyFoundationConfig
from myproject.foundation import MyFoundation


def train_fn(cfg_dict: dict):

    trainer_config = TrainerConfig(**cfg_dict)
    foundation_cfg = MyFoundationConfig(**cfg_dict)

    foundation: MyFoundation = MyFoundation(
        config=foundation_cfg,
        train_config=trainer_config,
    )
    foundation.to("cuda" if torch.cuda.is_available() else "cpu")
    
    datapipe = DataPipe(batch_size=trainer_config.batch_size, shuffle=True)
    datapipe.add_dataset(MyDataset(...))
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