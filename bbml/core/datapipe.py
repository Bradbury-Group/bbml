


from collections import defaultdict
import enum
from typing import Any, Iterable, Literal, Sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from bbml.debug import fprint
from bbml.core.data_transform import DataTransform


class CombineMethods(str, enum.Enum):
    ROUNDROBIN = "roundrobin"
    CONCAT = "concat"


class CombinedDataset(Dataset):
    """
    Combine multiple datasets either by:

    - method='roundrobin': get item in order from each dataset
      length is the ds minimum length * num datasets.

    - method='concat': items are taken sequentially from datasets,
      similar to ConcatDataset, but still respecting per-dataset ranges.
    """

    def __init__(self, method: CombineMethods | str = "roundrobin"):
        if isinstance(method, CombineMethods):
            self.method: CombineMethods = method
        else:
            # allow strings like "roundrobin" / "concat"
            self.method = CombineMethods(method)

        # Public list of underlying datasets (order matters)
        self.datasets: list[Dataset] = []

        # For each dataset, a sequence of indices into that dataset that we actually use
        self._indices_per_dataset: list[Sequence[int]] = []

        # Global mapping from combined index -> (dataset_idx, local_index_in_that_dataset_indices)
        self._index_mapping: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rebuild_index_mapping(self) -> None:
        """Rebuild the global index -> (dataset_idx, local_index) mapping
        based on current datasets and index ranges."""
        self._index_mapping = []

        if not self.datasets:
            return

        if self.method == CombineMethods.CONCAT:
            # Just concatenate all indices from each dataset in order
            for ds_idx, idxs in enumerate(self._indices_per_dataset):
                for local_i in range(len(idxs)):
                    self._index_mapping.append((ds_idx, local_i))

        elif self.method == CombineMethods.ROUNDROBIN:
            num_ds = len(self.datasets)
            # We can only go up to the minimum length across all datasets
            min_len = min(len(idxs) for idxs in self._indices_per_dataset)
            if min_len == 0:
                return

            # Order: d0[0], d1[0], ..., d_{k-1}[0], d0[1], d1[1], ...
            for pos in range(min_len):
                for ds_idx in range(num_ds):
                    self._index_mapping.append((ds_idx, pos))

        else:
            raise ValueError(f"Unknown combine method: {self.method!r}")

    def _normalize_index_range(
        self,
        dataset: Dataset,
        index_range: Iterable[int] | tuple[int, int] | None,
    ) -> Sequence[int]:
        """Convert index_range into a concrete sequence of indices."""
        n = len(dataset)

        if index_range is None:
            return range(n)

        if isinstance(index_range, tuple):
            if len(index_range) != 2:
                raise ValueError("tuple index_range must be (start, end)")
            start, end = index_range
            if not (0 <= start <= n and 0 <= end <= n):
                raise ValueError(
                f"index_range {index_range} is out of bounds for dataset of length {n}"
                )
            if end < start:
                raise ValueError("index_range end must be >= start")
            return range(start, end)

        # Otherwise, assume it's an arbitrary iterable of indices
        idxs = list(index_range)
        for i in idxs:
            if not (0 <= i < n):
                raise ValueError(
                    f"Index {i} in index_range is out of bounds for dataset of length {n}"
                )
        return idxs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_dataset(
        self,
        dataset: Dataset,
        index_range: Iterable[int] | tuple[int, int] | None = None,
    ) -> "CombinedDataset":
        """
        Add a dataset with an optional index_range specifying which items
        of that dataset to include.

        - If index_range is None: use all indices [0, len(dataset)).
        - If index_range is (start, end): use Python range(start, end).
        - If index_range is an iterable of ints: use those exact indices.

        Returns self to allow chaining.
        """
        indices = self._normalize_index_range(dataset, index_range)

        self.datasets.append(dataset)
        self._indices_per_dataset.append(indices)

        # Rebuild mapping since structure changed
        self._rebuild_index_mapping()
        return self

    def __len__(self) -> int:
        return len(self._index_mapping)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of range for CombinedDataset of length {len(self)}")

        ds_idx, local_i = self._index_mapping[index]
        dataset = self.datasets[ds_idx]
        indices = self._indices_per_dataset[ds_idx]
        item_idx = indices[local_i]
        return dataset[item_idx]

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def split(self, *ratios: float) -> list["CombinedDataset"]:
        """
        Split this CombinedDataset into multiple CombinedDatasets according
        to the provided ratios.

        Example:
            ds1 = CombinedDataset().add_dataset(...)
            train_ds, val_ds = ds1.split(0.8, 0.2)

        The split is done along the *combined* index space (i.e. respecting
        the global order imposed by the chosen combine method), and then
        mapped back to per-dataset index ranges. All splits share the same
        underlying dataset objects, but with different index subsets.
        """
        if not ratios:
            raise ValueError("At least one ratio must be provided to split().")

        total_len = len(self)
        if total_len == 0:
            # Return empty splits with same method
            return [CombinedDataset(self.method) for _ in ratios]

        total_ratio = float(sum(ratios))
        if total_ratio <= 0:
            raise ValueError("Sum of ratios must be positive.")

        # Compute lengths by proportional allocation, then distribute remainder
        raw_lengths = [int(total_len * (r / total_ratio)) for r in ratios]
        used = sum(raw_lengths)
        remainder = total_len - used

        # distribute remaining samples one by one to the first 'remainder' splits
        for i in range(remainder):
            raw_lengths[i % len(raw_lengths)] += 1

        lengths = raw_lengths

        # Prepare per-split, per-dataset index buckets
        num_splits = len(lengths)
        num_datasets = len(self.datasets)
        split_indices: list[list[list[int]]] = [
            [[ ] for _ in range(num_datasets)] for _ in range(num_splits)
        ]

        # Precompute boundaries in the combined index space
        boundaries = []
        cumsum = 0
        for L in lengths:
            cumsum += L
            boundaries.append(cumsum)  # exclusive upper bound for that split

        # Walk combined indices once and assign each to its split & dataset
        current_split = 0
        current_end = boundaries[0]
        for global_i in range(total_len):
            while global_i >= current_end and current_split < num_splits - 1:
                current_split += 1
                current_end = boundaries[current_split]

            ds_idx, local_i = self._index_mapping[global_i]
            underlying_indices = self._indices_per_dataset[ds_idx]
            item_idx = underlying_indices[local_i]
            split_indices[current_split][ds_idx].append(item_idx)

        # Build new CombinedDataset objects
        splits: list[CombinedDataset] = []
        for s in range(num_splits):
            new_ds = CombinedDataset(self.method)
            for ds_idx, base_dataset in enumerate(self.datasets):
                idxs = split_indices[s][ds_idx]
                if idxs:
                    new_ds.add_dataset(base_dataset, idxs)
            splits.append(new_ds)

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
        method: CombineMethods = "roundrobin",
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
        return self  # chain

    def get_loader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    @fprint
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
        


