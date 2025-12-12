from typing import Any
from collections.abc import Mapping, Sequence, Set
from pydantic import BaseModel


def deep_serialize_pydantic(input: Any) -> Any:
    """
        Recursively walk through nested containers (mappings, sequences, sets) and
        replace any Pydantic BaseModel instances with model_dump().
        Preserves container types where possible.
        Strings/bytes are technically Sequence so ignored.
    """

    if isinstance(input, BaseModel):
        return deep_serialize_pydantic(input.model_dump())

    if isinstance(input, (str, bytes, bytearray, memoryview)):
        return input

    if isinstance(input, Mapping):
        items = ((deep_serialize_pydantic(k), deep_serialize_pydantic(v)) for k, v in input.items())
        try:
            return type(input)(items)  # may fail for some custom mappings
        except Exception:
            return dict(items)

    if isinstance(input, (set, frozenset)) or isinstance(input, Set) and not isinstance(input, (str, bytes, bytearray, memoryview)):
        seq = (deep_serialize_pydantic(x) for x in input)
        try:
            return type(input)(seq)
        except Exception:
            return set(seq)

    if isinstance(input, Sequence):
        seq = [deep_serialize_pydantic(x) for x in input]

        if isinstance(input, tuple):
            try:
                return type(input)(*seq)
            except Exception:
                return tuple(seq)

        try:
            return type(input)(seq)
        except Exception:
            return seq

    return input