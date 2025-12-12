"""Test that experiment stubs are registered."""


class TestExperimentRegistry:
    def test_experiments_registered(self):
        from myproject.experiments import ExperimentRegistry

        assert "my_training" in ExperimentRegistry
        assert "my_analysis" in ExperimentRegistry

    def test_registry_get(self):
        from myproject.experiments import ExperimentRegistry

        cls = ExperimentRegistry["my_training"]
        assert cls.__name__ == "MyTrainingExperiment"
