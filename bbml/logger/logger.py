from typing import Any, Iterable

from bbml.core import fprint, ftimed
from bbml.core.logging import AbstractLogger, LoggingBackend
from bbml.registries import LoggingBackendRegistry

# Import backends to trigger registration
from bbml.logger import backends  # noqa: F401


class Logger(AbstractLogger):
    """
    Unified experiment logger façade.
    use start() and finish() for runs
    use log({key: data}) to log data
    Logged data conventions:
        scalars: (int, float, etc)
        pandas DataFrame
        image-like: PIL.Image | np.ndarray | torch.Tensor
        image batch-like: 4D np/torch batch or list of image-likes
        image dict: dict[caption, image-like or image batch-like]
    """

    def __init__(self) -> None:
        self.backends: list[LoggingBackend] = []
        self.step_fallback: int = 1


    @property
    def is_active(self) -> bool:
        return bool(self.backends)

    def start(self, service: str | Iterable[str] = "wandb", **kwargs: Any) -> None:
        if isinstance(service, str):
            services = [service]
        else:
            services = list(service)

        self.finish()
        self.backends = []

        for s in services:
            if s not in LoggingBackendRegistry:
                raise ValueError(f"Unknown logging service: {s!r}. Available: {list(LoggingBackendRegistry.keys())}")
            backend_class = LoggingBackendRegistry.get(s)
            
            backend: LoggingBackend = backend_class()
            backend.start(**kwargs)
            self.backends.append(backend)

        self.step_fallback = 0

    def normalize_step(self, step: int | None, commit: bool) -> int | None:
        """
        Maintain a monotonically increasing fallback step counter.

        If commit=True and no explicit step is provided, increment fallback.
        If commit=False, do not change the fallback (mirrors wandb semantics).
        """
        if commit:
            self.step_fallback += 1
        return self.step_fallback if step is None else int(step)

    def log(self, data: dict[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        if not self.is_active or not data:
            return

        # norm_step = self.normalize_step(step, commit=commit)
        for backend in self.backends:
            backend.log(data, step=step, commit=commit)

    def watch_model(self, model: Any, **kwargs: Any) -> None:
        if not self.is_active:
            return
        for backend in self.backends:
            if hasattr(backend, "watch_model"):
                backend.watch_model(model, **kwargs)

    def finish(self) -> None:
        for backend in self.backends:
            try:
                backend.finish()
            except Exception:
                pass
        self.backends = []


GLOBAL_LOGGER = Logger()
start = GLOBAL_LOGGER.start
finish = GLOBAL_LOGGER.finish
log = GLOBAL_LOGGER.log