import io
import os
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

@LoggingBackendRegistry.register("clearml")
class ClearMLBackend(LoggingBackend):
    def __init__(self) -> None:
        self.task = None
        self.logger = None

    def start(
        self,
        project: str | None = None,
        run_name: str | None = None,
        clearml_task_type: str | None = "training",
        clearml_bucket: str | None = "wand-finetune",
        tags: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        from clearml import Task  # lazy import

        if clearml_bucket:
            Task.setup_gcp_upload(
                bucket=clearml_bucket,
                credentials_json=os.getenv("SERVICE_ACCOUNT_JSON"),
            )

        self.task = Task.init(
            project_name=project or "Default",
            task_name=run_name or "run",
            task_type=clearml_task_type or "training",
            tags=list(tags) if tags else None,
        )
        self.logger = self.task.get_logger()

    def log(self, data: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        if self.logger is None or not data:
            return

        iteration = 0 if step is None else int(step)
        logger = self.logger

        for key, val in data.items():
            title = str(key)

            # scalars
            if is_scalar(val):
                logger.report_scalar(title=title, series="", value=float(val), iteration=iteration)
                continue

            # tables
            if is_dataframe(val):
                df = val
                logger.report_table(title=title, series="", iteration=iteration, table_plot=df)
                continue

            # images (single)
            if is_image_like(val):
                pil = to_pil_image(val)
                logger.report_image(title=title, series="", iteration=iteration, image=pil)
                continue

            # images (batch)
            if is_image_batch_like(val):
                images = to_pil_images(val)
                for i, pil in enumerate(images):
                    series = f"{key}[{i}]"
                    logger.report_image(title=title, series=series, iteration=iteration, image=pil)
                continue

            # images (dict)
            if is_image_dict(val):
                for caption, img in val.items():
                    if not (is_image_like(img) or is_image_batch_like(img)):
                        raise ValueError(
                            f"{key} is image dict but value {caption!r} is not image-like or batch-like: {type(img)}"
                        )
                    images = to_pil_images(img)
                    for i, pil in enumerate(images):
                        series = f"{caption}[{i}]" if len(images) > 1 else str(caption)
                        logger.report_image(title=title, series=series, iteration=iteration, image=pil)
                continue

            # byte blobs / generic media
            if isinstance(val, (bytes, bytearray)):
                try:
                    # ClearML expects either a local_path or a stream
                    stream = io.BytesIO(val)
                    logger.report_media(title=title, series="", iteration=iteration, local_path=None, stream=stream)
                    continue
                except Exception:
                    pass

            # fallback: ignore and warn
            warnings.warn(f"Unsupported data type for key {key}: {type(val)} fallback: ignore and warn")

    def finish(self) -> None:
        if self.task is None:
            return
        try:
            self.task.close()
        except Exception:
            pass
        finally:
            self.task = None
            self.logger = None
