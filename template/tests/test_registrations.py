"""Test that experiment stubs are registered."""
from myproject.experiments import ExperimentRegistry


class TestExperimentRegistry:
    def test_experiments_registered(self):
        assert "MyTrainingExperiment" in ExperimentRegistry
        assert "MyAnalysisExperiment" in ExperimentRegistry

    def test_registry_get(self):
        cls = ExperimentRegistry["MyTrainingExperiment"]
        assert cls.__name__ == "MyTrainingExperiment"
