from pathlib import Path

import pytest
from pydantic import ValidationError

from bbml.core.datamodels.configs import TrainerConfig


class TestTrainerConfig:
    def test_minimal_config(self):
        cfg = TrainerConfig(project="test")
        assert cfg.project == "test"
        assert cfg.name == ""
        assert cfg.output_dir == Path("checkpoints")
        assert cfg.train_epochs == 1
        assert cfg.batch_size == 1

    def test_with_all_fields(self):
        cfg = TrainerConfig(
            project="my-project",
            name="experiment-1",
            output_dir=Path("outputs"),
            train_epochs=10,
            batch_size=32,
            optimizer="AdamW",
            lr_scheduler="ConstantLR",
        )
        assert cfg.project == "my-project"
        assert cfg.name == "experiment-1"
        assert cfg.train_epochs == 10
        assert cfg.batch_size == 32

    def test_extra_fields_allowed(self):
        cfg = TrainerConfig(
            project="test",
            custom_field="value",
            another_field=123,
        )
        assert cfg.custom_field == "value"
        assert cfg.another_field == 123

    def test_name_suffix_appends_to_name(self):
        cfg = TrainerConfig(
            project="test",
            name="base",
            name_suffix={"lr": 0.001, "bs": 32},
        )
        assert "_lr_0.001" in cfg.name
        assert "_bs_32" in cfg.name

    def test_name_suffix_updates_output_dir(self):
        cfg = TrainerConfig(
            project="test",
            output_dir=Path("checkpoints"),
            name_suffix={"run": 1},
        )
        assert "_run_1" in str(cfg.output_dir)

    def test_name_suffix_strips_trailing_slash(self):
        cfg = TrainerConfig(
            project="test",
            output_dir=Path("checkpoints/"),
            name_suffix={"v": 2},
        )
        output_str = str(cfg.output_dir)
        assert not output_str.startswith("checkpoints//")
        assert "_v_2" in output_str

    def test_invalid_optimizer_raises(self):
        with pytest.raises(ValidationError):
            TrainerConfig(
                project="test",
                optimizer="NonExistentOptimizer",
            )

    def test_invalid_lr_scheduler_raises(self):
        with pytest.raises(ValidationError):
            TrainerConfig(
                project="test",
                lr_scheduler="NonExistentScheduler",
            )


class TestCheckStepTrigger:
    def test_int_trigger_every_n_steps(self):
        # Trigger every 10 steps
        assert TrainerConfig.check_step_trigger(0, 10) is True
        assert TrainerConfig.check_step_trigger(10, 10) is True
        assert TrainerConfig.check_step_trigger(20, 10) is True
        assert TrainerConfig.check_step_trigger(5, 10) is False
        assert TrainerConfig.check_step_trigger(15, 10) is False

    def test_sequence_trigger_at_specific_steps(self):
        trigger = [100, 200, 500]
        assert TrainerConfig.check_step_trigger(100, trigger) is True
        assert TrainerConfig.check_step_trigger(200, trigger) is True
        assert TrainerConfig.check_step_trigger(500, trigger) is True
        assert TrainerConfig.check_step_trigger(50, trigger) is False
        assert TrainerConfig.check_step_trigger(300, trigger) is False

    def test_mapping_trigger_with_at(self):
        trigger = {"at": [100, 200]}
        assert TrainerConfig.check_step_trigger(100, trigger) is True
        assert TrainerConfig.check_step_trigger(200, trigger) is True
        assert TrainerConfig.check_step_trigger(150, trigger) is False

    def test_mapping_trigger_with_every(self):
        trigger = {"every": 50}
        assert TrainerConfig.check_step_trigger(0, trigger) is True
        assert TrainerConfig.check_step_trigger(50, trigger) is True
        assert TrainerConfig.check_step_trigger(100, trigger) is True
        assert TrainerConfig.check_step_trigger(25, trigger) is False

    def test_mapping_trigger_combined_at_and_every(self):
        trigger = {"at": [75], "every": 100}
        # Should trigger at 75 (from "at") and at 0, 100, 200 (from "every")
        assert TrainerConfig.check_step_trigger(0, trigger) is True
        assert TrainerConfig.check_step_trigger(75, trigger) is True
        assert TrainerConfig.check_step_trigger(100, trigger) is True
        assert TrainerConfig.check_step_trigger(50, trigger) is False

    def test_none_trigger_returns_false(self):
        assert TrainerConfig.check_step_trigger(0, None) is False
        assert TrainerConfig.check_step_trigger(100, None) is False

