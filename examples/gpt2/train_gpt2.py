



from bbml import DataPipe, IdentityDataTransform, SimpleTrainer, TrainerConfig, run_interface
from bbml.data.datasets import WikiTextDataset
from bbml.data.transforms import UnsqueezeDataTransform
from bbml.foundations.gpt2 import GPT2Foundation, GPTConfig


def train_fn(cfg_dict: dict):

    train_cfg = TrainerConfig(**cfg_dict)
    gpt_cfg = GPTConfig(**cfg_dict)


    gpt = GPT2Foundation(gpt_cfg, train_cfg)


    train_dp = DataPipe(
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=16,
    ).add_dataset(
        WikiTextDataset(split="train")
    ).add_transforms(
        gpt.data_transforms
    )

    val_dp = DataPipe(
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=2,
    ).add_dataset(
        WikiTextDataset(split="validation"), index_range=(0, train_cfg.num_validation_samples)
    ).add_transforms(
        gpt.data_transforms
    )

    test_dp = DataPipe(
        batch_size=1,
        shuffle=True,
        num_workers=2,
    ).add_dataset(
        WikiTextDataset(split="test"), index_range=(0, train_cfg.num_test_samples)
    ).add_transforms(
        {"text": UnsqueezeDataTransform()}
    )

    trainer = SimpleTrainer(
        gpt,
        train_cfg,
        train_dp,
        val_dp,
        test_dp,
    )

    trainer.train()


if __name__ == "__main__":
    run_interface(train_fn)