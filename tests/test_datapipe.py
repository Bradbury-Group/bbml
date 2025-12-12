import pytest
from torch.utils.data import Dataset

from bbml.core.datapipe import CombinedDataset, DataPipe, CombineMethods
from bbml.core.data_transform import DataTransform


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleDictDataset(Dataset):
    """Dataset that returns dicts for testing DataPipe."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DoubleTransform(DataTransform):
    """Simple transform that doubles values."""

    def transform(self, inp):
        return inp * 2

    def batch_transform(self, inp: list):
        return inp


class TestCombinedDataset:
    def test_empty_dataset(self):
        ds = CombinedDataset()
        assert len(ds) == 0

    def test_add_single_dataset(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([1, 2, 3]))
        assert len(ds) == 3
        assert ds[0] == 1
        assert ds[2] == 3

    def test_add_dataset_with_index_range_tuple(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([0, 1, 2, 3, 4]), index_range=(1, 4))
        assert len(ds) == 3
        assert ds[0] == 1
        assert ds[2] == 3

    def test_add_dataset_with_index_range_list(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([0, 1, 2, 3, 4]), index_range=[0, 2, 4])
        assert len(ds) == 3
        assert ds[0] == 0
        assert ds[1] == 2
        assert ds[2] == 4

    def test_concat_method(self):
        ds = CombinedDataset(method="concat")
        ds.add_dataset(SimpleDataset([1, 2]))
        ds.add_dataset(SimpleDataset([3, 4]))
        assert len(ds) == 4
        assert [ds[i] for i in range(4)] == [1, 2, 3, 4]

    def test_roundrobin_method(self):
        ds = CombinedDataset(method="roundrobin")
        ds.add_dataset(SimpleDataset([1, 2]))
        ds.add_dataset(SimpleDataset(["a", "b"]))
        assert len(ds) == 4
        # roundrobin: ds0[0], ds1[0], ds0[1], ds1[1]
        assert [ds[i] for i in range(4)] == [1, "a", 2, "b"]

    def test_roundrobin_unequal_length(self):
        ds = CombinedDataset(method="roundrobin")
        ds.add_dataset(SimpleDataset([1, 2, 3]))
        ds.add_dataset(SimpleDataset(["a", "b"]))  # shorter
        # Length is min_len * num_datasets = 2 * 2 = 4
        assert len(ds) == 4
        assert [ds[i] for i in range(4)] == [1, "a", 2, "b"]

    def test_negative_index(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([1, 2, 3]))
        assert ds[-1] == 3
        assert ds[-2] == 2

    def test_index_out_of_range(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([1, 2]))
        with pytest.raises(IndexError):
            _ = ds[10]

    def test_chaining(self):
        ds = (
            CombinedDataset()
            .add_dataset(SimpleDataset([1, 2]))
            .add_dataset(SimpleDataset([3, 4]))
        )
        assert len(ds) == 4


class TestCombinedDatasetSplit:
    def test_split_basic(self):
        ds = CombinedDataset(method="concat")
        ds.add_dataset(SimpleDataset(list(range(10))))
        train, val = ds.split(0.8, 0.2)
        assert len(train) == 8
        assert len(val) == 2

    def test_split_three_way(self):
        ds = CombinedDataset(method="concat")
        ds.add_dataset(SimpleDataset(list(range(100))))
        train, val, test = ds.split(0.7, 0.2, 0.1)
        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10
        assert len(train) + len(val) + len(test) == 100

    def test_split_empty_dataset(self):
        ds = CombinedDataset()
        train, val = ds.split(0.8, 0.2)
        assert len(train) == 0
        assert len(val) == 0

    def test_split_no_ratios_raises(self):
        ds = CombinedDataset()
        ds.add_dataset(SimpleDataset([1, 2, 3]))
        with pytest.raises(ValueError, match="At least one ratio"):
            ds.split()


class TestDataPipe:
    def test_init(self):
        dp = DataPipe(batch_size=32, shuffle=True, num_workers=4)
        assert dp.batch_size == 32
        assert dp.shuffle is True
        assert dp.num_workers == 4

    def test_add_dataset_and_transforms(self):
        dp = DataPipe(batch_size=2)
        dp.add_dataset(SimpleDictDataset([{"x": 1}, {"x": 2}]))
        dp.add_transforms({"x": DoubleTransform()})
        assert len(dp) == 2
        assert "x" in dp.transforms

    def test_chaining(self):
        dp = (
            DataPipe(batch_size=2)
            .add_dataset(SimpleDictDataset([{"x": 1}]))
            .add_transforms({"x": DoubleTransform()})
        )
        assert len(dp) == 1
        assert "x" in dp.transforms

    def test_collate_fn_applies_transforms(self):
        dp = DataPipe(batch_size=2)
        dp.add_transforms({"x": DoubleTransform()})
        
        batch = [{"x": 1}, {"x": 2}]
        result = dp.collate_fn(batch)
        assert result["x"] == [2, 4]  # doubled

    def test_collate_fn_ignores_extra_keys_by_default(self):
        dp = DataPipe(batch_size=2)
        dp.add_transforms({"x": DoubleTransform()})
        
        batch = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
        result = dp.collate_fn(batch)
        assert "x" in result
        assert "y" not in result  # ignored

    def test_collate_fn_allows_extra_keys(self):
        dp = DataPipe(batch_size=2, extra_keys="allow")
        dp.add_transforms({"x": DoubleTransform()})
        
        batch = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
        result = dp.collate_fn(batch)
        assert result["x"] == [2, 4]
        assert result["y"] == [10, 20]

    def test_collate_fn_errors_on_extra_keys(self):
        dp = DataPipe(batch_size=2, extra_keys="error")
        dp.add_transforms({"x": DoubleTransform()})
        
        batch = [{"x": 1, "y": 10}]
        with pytest.raises(ValueError):
            dp.collate_fn(batch)

    def test_get_loader(self):
        dp = DataPipe(batch_size=2, shuffle=False, num_workers=0)
        dp.add_dataset(SimpleDictDataset([{"x": i} for i in range(4)]))
        dp.add_transforms({"x": DoubleTransform()})
        
        loader = dp.get_loader()
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["x"] == [0, 2]  # doubled: 0*2, 1*2
        assert batches[1]["x"] == [4, 6]  # doubled: 2*2, 3*2

