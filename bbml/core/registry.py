from functools import partial
from typing import TypeVar, Generic, Callable, Iterator


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name: str = name
        self.registry: dict[str, T] = {}

    def __repr__(self) -> str:
        return f"<Registry {self.name} with keys {list(self.registry.keys())})>"

    def add(self, key: str, value: T, *, force: bool = False):
        if not force and key in self.registry:
            raise KeyError(f"Key {key!r} already registered in {self.name}")
        self.registry[key] = value

    def get(self, key: str, default: T|None = None) -> T|None:
        return self.registry.get(key, default)

    def remove(self, key: str):
        del self.registry[key]

    def register(self, key_or_obj: str|T|None = None, *, force: bool = False) -> Callable[[T], T] | T:
        """
            Decorator to register an object in the registry.
            use like: @register("name") or @register() or @register
        """
        def decorator(obj: T, key: str|None=None) -> T:
            k = key or getattr(obj, "__qualname__", None)
            if k is None:
                raise ValueError(f"No key defined for registered object {obj} of type {type(obj)}")
            self.add(k, obj, force=force)
            return obj
        
        if key_or_obj is None:  # @register()
            return decorator
        elif isinstance(key_or_obj, str):  # @register("...")
            return partial(decorator, key=key_or_obj)
        else:  # @register
            return decorator(key_or_obj)

    def __contains__(self, key: object) -> bool:
        return key in self.registry

    def __getitem__(self, key: str) -> T:
        return self.registry[key]

    def __len__(self) -> int:
        return len(self.registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self.registry)

    def keys(self):
        return self.registry.keys()

    def values(self):
        return self.registry.values()

    def items(self):
        return self.registry.items()
