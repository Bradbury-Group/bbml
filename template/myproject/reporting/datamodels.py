"""Pydantic models used for reporting utilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Measurement(BaseModel):
    """Scalar measurement captured during an experiment."""

    name: str
    value: float
    std: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Measurement"]
