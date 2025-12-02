from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import Tensor


def is_scalar(v: Any) -> bool:
    """Return True if v is a scalar numeric type."""
    return isinstance(v, (int, float, np.number))


def is_dataframe(v: Any) -> bool:
    """Return True if v is a pandas DataFrame."""
    return isinstance(v, pd.DataFrame)


def is_image_like(v: Any) -> bool:
    """Return True if v is an image-like object (2D or 3D array/tensor, or PIL Image)."""
    if isinstance(v, Image.Image):
        return True
    if isinstance(v, np.ndarray):
        return v.ndim in (2, 3)
    if isinstance(v, Tensor):
        return v.ndim in (2, 3)
    return False


def is_image_batch_like(v: Any) -> bool:
    """Return True if v is a batch of images (4D array/tensor or list of images)."""
    if isinstance(v, np.ndarray):
        return v.ndim == 4
    if isinstance(v, Tensor):
        return v.ndim == 4
    if isinstance(v, list) and v:
        # list of image-like items
        return all(is_image_like(item) for item in v)
    return False


def is_image_dict(v: Any) -> bool:
    """Return True if v is a dict mapping captions to images or image batches."""
    return (
        isinstance(v, dict)
        and bool(v)  # non-empty
        and all(is_image_like(iv) or is_image_batch_like(iv) for iv in v.values())
    )


def to_pil_image(x: Any) -> Image.Image:
    """Convert PIL / numpy / torch -> PIL.Image (RGB)."""
    # PIL.Image: trivial
    if isinstance(x, Image.Image):
        im = x

    # numpy arrays
    elif isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            # grayscale H, W -> H, W, 3
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            # Handle CHW or HWC -> ensure H, W, C
            if arr.shape[0] in (1, 3) and (arr.shape[-1] not in (1, 3)):
                # likely CHW
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        else:
            raise TypeError(f"Unsupported numpy image ndim={arr.ndim}; expected 2 or 3")

        if np.issubdtype(arr.dtype, np.floating):
            # Try to detect [0,1] vs [0,255] floats
            min_val = float(arr.min())
            max_val = float(arr.max())
            if 0.0 <= min_val and max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        im = Image.fromarray(arr)

    # torch tensors
    elif isinstance(x, Tensor):
        t = x.detach().cpu()
        if t.ndim == 3:
            # Accept CHW or HWC; convert to HWC
            if t.shape[-1] in (1, 3):
                # already HWC
                pass
            elif t.shape[0] in (1, 3):
                # CHW -> HWC
                t = t.permute(1, 2, 0)
            if t.shape[-1] == 1:
                t = t.repeat(1, 1, 3)
        elif t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        else:
            raise TypeError(f"Unsupported tensor image ndim={t.ndim}; expected 2 or 3")

        t = t.float()
        # Detect typical normalized range
        t_min = float(t.min())
        t_max = float(t.max())
        if 0.0 <= t_min and t_max <= 1.0:
            t = t * 255.0
        t = t.clamp(0.0, 255.0).to(dtype=torch.uint8)  # type: ignore[union-attr]
        im = Image.fromarray(t.numpy())

    else:
        raise TypeError(f"Unsupported image type: {type(x)}")

    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def to_pil_images(x: Any) -> list[Image.Image]:
    """Convert image or batch to a list of PIL.Image (RGB).

    - If x is a 4D numpy / torch tensor, assumes first dim is batch.
    - If x is a list of image-like items, converts each.
    - If x is a single image-like object, returns a 1-element list.
    """
    if is_image_batch_like(x):
        if isinstance(x, np.ndarray):
            return [to_pil_image(x[i]) for i in range(x.shape[0])]
        if isinstance(x, Tensor):
            return [to_pil_image(x[i]) for i in range(x.shape[0])]
        if isinstance(x, list):
            return [to_pil_image(i) for i in x]
    elif is_image_like(x):
        return [to_pil_image(x)]

    raise ValueError(f"Expected image or image batch-like object, got {type(x)}")

