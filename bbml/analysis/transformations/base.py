from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from bbml.analysis.weights.units import WeightUnit


class TransformMode(Enum):
    """Declares whether a transformer operates on one layer or two."""
    UNARY = "unary"
    BINARY = "binary"


class LayerTransformer(ABC):
    """Applies a transformation to layers before similarity computation.

    Subclasses implement either unary (1 → 1) or binary (2 → 1) transforms
    and declare which mode they support via the ``mode`` property.

    - **Unary:** each layer is transformed independently (e.g. normalisation).
    - **Binary:** two layers are combined into one (e.g. Procrustes alignment).
    """

    @property
    @abstractmethod
    def mode(self) -> TransformMode:
        """The calling convention this transformer supports."""
        pass

    def transform(self, layer: WeightUnit) -> WeightUnit:
        """Unary transform: single layer in, single layer out.

        The default implementation returns the layer unchanged.
        Subclasses operating in ``UNARY`` mode should override this.

        Args:
            layer: A single weight unit to transform.

        Returns:
            The transformed weight unit.
        """
        return layer

    def transform_pair(self, layer_a: WeightUnit, layer_b: WeightUnit) -> WeightUnit:
        """Binary transform: two layers in, one merged/aligned layer out.

        Subclasses operating in ``BINARY`` mode must override this.

        Args:
            layer_a: First weight unit.
            layer_b: Second weight unit.

        Returns:
            A single merged or aligned weight unit.

        Raises:
            NotImplementedError: If the subclass has not provided an implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses BINARY mode but does not implement transform_pair"
        )

    def apply(self, layers: List[WeightUnit]) -> List[WeightUnit]:
        """Apply this transformer element-wise to a single layer list (unary mode).

        Args:
            layers: The list of weight units to transform.

        Returns:
            A new list of transformed weight units.

        Raises:
            ValueError: If called on a binary-mode transformer (use
                ``apply_paired`` instead).
        """
        if self.mode == TransformMode.BINARY:
            raise ValueError(
                "Cannot call apply() on a BINARY transformer. "
                "Use apply_paired(layers_a, layers_b) instead."
            )
        return [self.transform(layer) for layer in layers]

    def apply_paired(
        self, layers_a: List[WeightUnit], layers_b: List[WeightUnit]
    ) -> List[WeightUnit]:
        """Apply this transformer element-wise over two aligned layer lists (binary mode).

        Given lists a = (a1, ..., an) and b = (b1, ..., bn), produces
        c = (f(a1, b1), ..., f(an, bn)).

        Args:
            layers_a: First list of weight units.
            layers_b: Second list of weight units, same length as ``layers_a``.

        Returns:
            A list of transformed weight units of the same length.

        Raises:
            ValueError: If called on a unary-mode transformer or if the two
                lists have different lengths.
        """
        if self.mode == TransformMode.UNARY:
            raise ValueError(
                "Cannot call apply_paired() on a UNARY transformer. "
                "Use apply(layers) instead."
            )
        if len(layers_a) != len(layers_b):
            raise ValueError(
                f"Layer lists must have the same length, "
                f"got {len(layers_a)} and {len(layers_b)}"
            )
        return [
            self.transform_pair(a, b) for a, b in zip(layers_a, layers_b)
        ]
