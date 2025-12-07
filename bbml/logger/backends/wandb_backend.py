from typing import Any, Iterable, Mapping
import warnings

from bbml.core.logging import LoggingBackend
from bbml.logger.utils import (
    is_scalar,
    is_dataframe,
    is_image_like,
    is_image_batch_like,
    is_image_dict,
    to_pil_image,
    to_pil_images,
)
from bbml.registries import LoggingBackendRegistry


@LoggingBackendRegistry.register("wandb")
class WandbBackend(LoggingBackend):
    def __init__(self) -> None:
        self.run = None

    def start(
        self,
        project: str | None = None,
        name: str | None = None,
        wandb_entity: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        import wandb  # lazy import

        self.run = wandb.init(
            project=project,
            entity=wandb_entity,
            name=name,
            config=config,
        )

    def log(self, data: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        import wandb  # lazy import

        if not data:
            return

        payload: dict[str, Any] = {}

        for key, val in data.items():
            # scalars
            if is_scalar(val):
                payload[key] = float(val)
                continue

            # tables
            if is_dataframe(val):
                df = val
                payload[key] = wandb.Table(columns=list(df.columns), data=df.values.tolist())
                continue

            # images
            if is_image_like(val):
                image = to_pil_image(val)
                payload[key] = wandb.Image(image, caption=str(key))
                continue

            if is_image_batch_like(val):
                images = to_pil_images(val)
                payload[key] = [wandb.Image(im, caption=f"{key}[{i}]") for i, im in enumerate(images)]
                continue

            if is_image_dict(val):
                images: list[Any] = []
                for caption, img in val.items():
                    if is_image_like(img) or is_image_batch_like(img):
                        for i, pil in enumerate(to_pil_images(img)):
                            cap = f"{caption}[{i}]" if is_image_batch_like(img) else str(caption)
                            images.append(wandb.Image(pil, caption=cap))
                    else:
                        raise ValueError(
                            f"{key} is image dict but value {caption!r} is not image-like or batch-like: {type(img)}"
                        )
                payload[key] = images
                continue
            
            

            # fallback: ignore and warn
            warnings.warn(f"Unsupported data type for key {key}: {type(val)} fallback: ignore and warn")

        if payload:
            wandb.log(payload, step=step, commit=commit)

    def watch_model(self, model: Any, **kwargs: Any) -> None:
        if self.run is None:
            return
        try:
            import wandb  # lazy import

            wandb.watch(model, **kwargs)
        except Exception:
            # best-effort
            return

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass
        finally:
            self.run = None
