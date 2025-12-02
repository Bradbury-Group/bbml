from typing import Any, Mapping
import warnings

import numpy as np

from bbml.core.logging import LoggingBackend
from bbml.logging.utils import (
    is_scalar,
    is_dataframe,
    is_image_like,
    is_image_batch_like,
    is_image_dict,
    to_pil_images,
)
from bbml.registries import LoggingBackendRegistry


@LoggingBackendRegistry.register("tensorboard")
class TensorBoardBackend(LoggingBackend):
    def __init__(self) -> None:
        self.writer = None

    def start(
        self,
        project: str | None = None,
        run_name: str | None = None,
        tensorboard_log_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
        except Exception:
            # TensorBoard not available; keep writer as None
            self.writer = None
            return

        log_dir = tensorboard_log_dir
        if log_dir is None:
            parts = [p for p in [project, run_name] if p]
            log_dir = "_".join(parts) if parts else None
        self.writer = SummaryWriter(log_dir=log_dir or None)

    def log(self, data: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        if self.writer is None or not data:
            return

        global_step = step

        for key, val in data.items():
            tag = str(key)

            # scalars
            if is_scalar(val):
                self.writer.add_scalar(tag, float(val), global_step=global_step)
                continue

            # tables -> log as text summary
            if is_dataframe(val):
                df = val
                # Limit size for readability
                preview = df.head(20).to_markdown()
                self.writer.add_text(tag, preview, global_step=global_step)
                continue

            # images: TensorBoard expects CHW or HWC arrays in [0,1]
            if is_image_like(val) or is_image_batch_like(val):
                images = to_pil_images(val)
                for i, pil in enumerate(images):
                    arr = np.array(pil).astype(np.float32) / 255.0  # H, W, C in [0,1]
                    # For multiple images, encode index in the tag
                    img_tag = f"{tag}/{i}" if len(images) > 1 else tag
                    self.writer.add_image(img_tag, arr, global_step=global_step, dataformats="HWC")
                continue

            if is_image_dict(val):
                for caption, img in val.items():
                    if not (is_image_like(img) or is_image_batch_like(img)):
                        continue
                    images = to_pil_images(img)
                    for i, pil in enumerate(images):
                        arr = np.array(pil).astype(np.float32) / 255.0
                        img_tag = f"{tag}/{caption}/{i}" if len(images) > 1 else f"{tag}/{caption}"
                        self.writer.add_image(img_tag, arr, global_step=global_step, dataformats="HWC")
                continue

            # fallback: ignore and warn
            warnings.warn(f"Unsupported data type for key {key}: {type(val)} fallback: ignore and warn")

    def finish(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception:
            pass
        finally:
            self.writer = None
