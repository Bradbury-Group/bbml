"""
Dataset stub - extend torch.utils.data.Dataset.

Use standard PyTorch Dataset.
__getitem__() should return a dict with string keys.

Integration with bbml:
  Create DataPipe: pipe = DataPipe(batch_size=32, shuffle=True)
   Add dataset: pipe.add_dataset(YourDataset(...))
   Add transforms: pipe.add_transforms(foundation.data_transforms)
   Get loader: loader = pipe.get_loader(num_workers=4)

DataPipe.collate_fn() routes each dict key through its DataTransform,
then calls batch_transform() to stack into batched tensors.

See bbml/core/datapipe.py for CombinedDataset (multi-dataset) and DataPipe.
See bbml/data/datasets/ for examples (e.g., WikiTextDataset).
"""

from typing import Any
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    Keys should match Foundation.data_transforms keys.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            split: "train", "val", or "test"
        """
        self.split = split
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """

        Keys become batch dict keys after DataPipe collation.
        Must match keys in Foundation.data_transforms.

        Example returns:
            {"features": tensor, "label": int}
            {"text": str, "target": str}
            {"image": PIL.Image, "caption": str}
        """
        raise NotImplementedError
