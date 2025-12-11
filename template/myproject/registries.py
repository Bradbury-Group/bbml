"""Local registries with filtering and duplicate detection."""

from typing import TYPE_CHECKING, Callable

from bbml.core.registry import Registry

if TYPE_CHECKING:
    from myproject.experiments.base import Experiment


class ExperimentRegistryClass(Registry[type["Experiment"]]):
    """Extended registry with filtering and duplicate detection."""

    def register(self, key: str | None = None) -> Callable:
        """Register with duplicate detection."""
        def decorator(cls: type["Experiment"]) -> type["Experiment"]:
            name = key or cls.__name__
            if name in self:
                raise ValueError(f"Experiment '{name}' already registered")
            cls.registry_name = name  # Store for introspection
            self.add(name, cls)
            return cls
        return decorator

    def filter(
        self,
        startswith: str | None = None,
        endswith: str | None = None,
        contains: str | None = None,
        excludes: str | None = None,
    ) -> list[str]:
        """Filter experiment names by pattern."""
        names = list(self.keys())
        if startswith:
            names = [n for n in names if n.startswith(startswith)]
        if endswith:
            names = [n for n in names if n.endswith(endswith)]
        if contains:
            names = [n for n in names if contains in n]
        if excludes:
            names = [n for n in names if excludes not in n]
        return names


ExperimentRegistry = ExperimentRegistryClass("Experiment")

# Re-export bbml registries
from bbml.registries import (  # noqa: E402, F401
    LoggingBackendRegistry,
    LRSchedulerRegistry,
    OptimizerRegistry,
)

__all__ = [
    "ExperimentRegistry",
    "OptimizerRegistry",
    "LRSchedulerRegistry",
    "LoggingBackendRegistry",
]
