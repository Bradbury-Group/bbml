"""Test registrations and basic functionality."""

import torch


class TestExperimentRegistry:
    def test_experiments_registered(self):
        from myproject.experiments.registry import ExperimentRegistry

        assert "example_training" in ExperimentRegistry
        assert "example_compiled" in ExperimentRegistry
        assert "example_analysis" in ExperimentRegistry

    def test_registry_get(self):
        from myproject.experiments.registry import ExperimentRegistry

        cls = ExperimentRegistry["example_training"]
        assert cls.__name__ == "ExampleTrainingExperiment"


class TestFoundation:
    def test_foundation_forward(self):
        from myproject.foundations.example_foundation import (
            ExampleFoundation,
            ExampleFoundationConfig,
        )

        config = ExampleFoundationConfig(input_dim=10, hidden_dim=20, output_dim=5)
        foundation = ExampleFoundation(config=config, train_config=None)

        x = torch.randn(4, 10)
        out = foundation(x)
        assert out.shape == (4, 5)


class TestDataset:
    def test_dataset_getitem(self):
        from myproject.data.datasets import ExampleDataset

        dataset = ExampleDataset(split="train", size=10, input_dim=20)
        item = dataset[0]

        assert "features" in item
        assert "label" in item
        assert item["features"].shape == (20,)

    def test_dataset_reproducibility(self):
        from myproject.data.datasets import ExampleDataset

        ds1 = ExampleDataset(split="train", size=10, seed=42)
        ds2 = ExampleDataset(split="train", size=10, seed=42)

        assert (ds1[0]["features"] == ds2[0]["features"]).all()
