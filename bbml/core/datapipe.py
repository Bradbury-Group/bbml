


from collections import defaultdict
import enum
from typing import Any, Iterable, Literal, Sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from bbml.core.data_transform import DataTransform

class CombineMethods(str, enum.Enum):
    ZIP = "zip"
    CONCAT = "concat"

class CombinedDataset(Dataset):
    """
    Combine multiple datasets either by:

    - method='zip': items are tuples (d0[i], d1[i], ...);
      length is the minimum length across datasets (after ranges).

    - method='concat': items are taken sequentially from datasets,
      similar to ConcatDataset, but still respecting per-dataset ranges.
    """

    def __init__(self, method: CombineMethods = "zip"):
        self.method: CombineMethods = method
        self.datasets: list[Dataset] = []
        # For each dataset we keep an explicit list of indices to use
        self._index_maps: list[list[int]] = []

    # ----- internal helper -----
    def _add_with_indices(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.datasets.append(dataset)
        self._index_maps.append(list(indices))

    def add_dataset(
        self,
        dataset: Dataset,
        index_range: Iterable[int]|tuple[int, int]|None  = None,
    ) -> "CombinedDataset":
        """
        Add a dataset with an optional index range.

        index_range:
          - None: use all indices [0, len(dataset))
          - (start, stop): use indices [start, stop)
          - Iterable[int]: arbitrary list of indices
        """
        if index_range is None:
            indices = list(range(len(dataset)))
        elif isinstance(index_range, tuple) and len(index_range) == 2:
            start, stop = index_range
            if not (0 <= start <= stop <= len(dataset)):
                raise ValueError("Invalid (start, stop) range for dataset")
            indices = list(range(start, stop))
        else:
            # Assume arbitrary iterable of indices
            indices = list(index_range)
            for i in indices:
                if i < 0 or i >= len(dataset):
                    raise IndexError(
                        f"Index {i} out of range for dataset of length {len(dataset)}"
                    )

        self._add_with_indices(dataset, indices)
        return self  # allow chaining

    def __len__(self) -> int:
        if not self.datasets:
            return 0

        if self.method == "zip":
            # length is limited by the shortest index list
            return min(len(idxs) for idxs in self._index_maps)
        else:  # "concat"
            return sum(len(idxs) for idxs in self._index_maps)

    def __getitem__(self, index):
        n = len(self)
        if index < 0:
            index += n
        if not (0 <= index < n):
            raise IndexError(index)

        if self.method == "zip":
            # each dataset uses the index-th element of *its* index list
            return tuple(
                ds[idxs[index]] for ds, idxs in zip(self.datasets, self._index_maps)
            )
        else:  # "concat"
            # walk through datasets until we locate the right block
            for ds, idxs in zip(self.datasets, self._index_maps):
                if index < len(idxs):
                    real_idx = idxs[index]
                    return ds[real_idx]
                index -= len(idxs)

            # Should never get here
            raise RuntimeError("Index resolution failed in CombinedDataset")

    def split(self, *ratios: float) -> list["CombinedDataset"]:
        """
        Split this CombinedDataset into multiple CombinedDataset instances
        according to ratios, preserving method ('zip' or 'concat').

        For method='zip':
            - we split along the shared 'logical' index axis, and
              propagate the corresponding slices to each underlying dataset.

        For method='concat':
            - we treat the combined dataset like a single long list
              (with blocks coming from each dataset) and split that list,
              then derive per-dataset index ranges for each split.
        """
        if not ratios:
            raise ValueError("At least one ratio must be provided")
        if any(r < 0 for r in ratios):
            raise ValueError("Ratios must be non-negative")

        total_len = len(self)
        if total_len == 0:
            # Empty splits, but keep method
            return [CombinedDataset(self.method) for _ in ratios]

        total_ratio = float(sum(ratios))
        if total_ratio == 0:
            raise ValueError("Sum of ratios must be > 0")

        # initial sizes by floor
        raw_sizes = [total_len * r / total_ratio for r in ratios]
        sizes = [int(s) for s in raw_sizes]

        # distribute leftover examples to the largest fractional parts
        remainder = total_len - sum(sizes)
        frac_order = sorted(
            range(len(ratios)),
            key=lambda i: raw_sizes[i] - sizes[i],
            reverse=True,
        )
        for i in frac_order[:remainder]:
            sizes[i] += 1

        # build cumulative boundaries
        boundaries = [0]
        acc = 0
        for s in sizes:
            acc += s
            boundaries.append(acc)
        assert boundaries[-1] == total_len

        splits: list[CombinedDataset] = []

        for split_idx in range(len(ratios)):
            start = boundaries[split_idx]
            end = boundaries[split_idx + 1]

            new_cd = CombinedDataset(self.method)

            if self.method == "zip":
                # same [start:end] slice for each dataset's indices
                for ds, idxs in zip(self.datasets, self._index_maps):
                    sub_idxs = idxs[start:end]
                    new_cd._add_with_indices(ds, sub_idxs)
            else:  # "concat"
                # map [start:end) of the *combined* space back to per-dataset blocks
                seg_start, seg_end = start, end
                global_pos = 0

                for ds, idxs in zip(self.datasets, self._index_maps):
                    ds_len = len(idxs)
                    # overlap of [seg_start, seg_end) with [global_pos, global_pos + ds_len)
                    local_start = max(0, seg_start - global_pos)
                    local_end = max(0, min(ds_len, seg_end - global_pos))
                    if local_start < local_end:
                        new_cd._add_with_indices(ds, idxs[local_start:local_end])
                    global_pos += ds_len

            splits.append(new_cd)

        return splits


class ExtraKeysStrategy(str, enum.Enum):
    ERROR = "error"
    IGNORE = "ignore"
    ALLOW = "allow"


class DataPipe(CombinedDataset):

    def __init__(
        self,
        batch_size: int|None = None,
        shuffle: bool = False,
        num_workers: int|None = None,
        method: CombineMethods = "zip",
        extra_keys: ExtraKeysStrategy = "ignore",
    ):
        super().__init__(method=method)
        self.transforms: dict[str, DataTransform] = {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.extra_keys = extra_keys

    def add_transforms(self, transforms:dict[str, DataTransform]):
        self.transforms.update(transforms)

    def get_loader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        listed_data = defaultdict(list)
        for data in batch:
            for k,v in data.items():
                if k in self.transforms:
                    new_v = self.transforms[k].transform(v)
                    listed_data[k].append(new_v)
                elif self.extra_keys == ExtraKeysStrategy.ALLOW:
                    listed_data[k].append(v)
                elif self.extra_keys == ExtraKeysStrategy.ERROR:
                    raise ValueError(f"{k=} not in {self.transforms=}")
        
        collated_data = {}
        for key, list_vals in listed_data.items():
            if key in self.transforms:
                collated_data[key] = self.transforms[key].batch_transform(list_vals)
            elif self.extra_keys == ExtraKeysStrategy.ALLOW:
                collated_data[key] = list_vals
            elif self.extra_keys == ExtraKeysStrategy.ERROR:
                raise ValueError(f"{key=} not in {self.transforms=}")
        
        return collated_data
        


