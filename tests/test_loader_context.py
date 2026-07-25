"""Seam 1: LoaderContext + DataPipe.get_loader sharding."""
import pytest
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from bbml.core.datapipe import DataPipe, LoaderContext

from tests.seam_helpers import make_pipe


class TestLoaderContext:
    def test_defaults_single_process(self):
        ctx = LoaderContext()
        assert ctx.dp_rank == 0 and ctx.dp_size == 1 and ctx.drop_last is True

    def test_no_ctx_is_current_behavior(self):
        # ctx=None must reproduce the pre-seam loader (no DistributedSampler).
        pipe = make_pipe(n=8, batch_size=2, shuffle=False)
        loader = pipe.get_loader()
        assert not isinstance(loader.sampler, DistributedSampler)
        assert isinstance(loader.sampler, SequentialSampler)
        assert len(list(loader)) == 4

    def test_dp_size_one_no_shard(self):
        pipe = make_pipe(n=8, batch_size=2, shuffle=True)
        loader = pipe.get_loader(LoaderContext(dp_size=1))
        # dp_size==1 stays on the current path (shuffle -> RandomSampler).
        assert not isinstance(loader.sampler, DistributedSampler)
        assert isinstance(loader.sampler, RandomSampler)

    def test_dp_size_gt_one_wraps_distributed_sampler(self):
        pipe = make_pipe(n=8, batch_size=2, shuffle=False)
        loader = pipe.get_loader(LoaderContext(dp_rank=0, dp_size=2))
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.num_replicas == 2
        assert loader.sampler.rank == 0

    def test_dp_ranks_partition_without_overlap(self):
        # Two ranks over dp_size=2 must cover distinct indices, no double-shard.
        n = 8
        r0 = set(DistributedSampler(make_pipe(n=n), num_replicas=2, rank=0, shuffle=False, drop_last=True))
        r1 = set(DistributedSampler(make_pipe(n=n), num_replicas=2, rank=1, shuffle=False, drop_last=True))
        assert r0.isdisjoint(r1)
        assert len(r0) == len(r1) == n // 2

    def test_drop_last_flows_through(self):
        pipe = make_pipe(n=7, batch_size=1, shuffle=False)
        keep = pipe.get_loader(LoaderContext(dp_size=2, drop_last=False)).sampler
        drop = pipe.get_loader(LoaderContext(dp_size=2, drop_last=True)).sampler
        assert keep.drop_last is False and drop.drop_last is True


class TestFullyShardWrap:
    """FullyShardTrainer._wrap_train_dataloader (the super() drawing subclasses
    call) must derive a LoaderContext from the layout and route to get_loader."""

    def test_layout_drives_distributed_sampler(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", "0")
        from bbml.fsdp.dist import ParallelLayout
        from bbml.train.distributed.fully_shard_trainer import FullyShardTrainer
        from bbml.core.datamodels.configs import TrainerConfig
        from tests.seam_helpers import TinyFoundation

        pipe = make_pipe(n=8, batch_size=2, shuffle=False)
        cfg = TrainerConfig(project="test", drop_last_train=False)
        trainer = FullyShardTrainer(
            TinyFoundation(), cfg, pipe, None, None, owns_process_group=False,
        )
        trainer.layout = ParallelLayout(tp=1, shard=2)  # dp_size == 2
        loader = trainer._wrap_train_dataloader()
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.num_replicas == 2
        assert loader.sampler.rank == 0
        assert loader.sampler.drop_last is False

    def test_legacy_plain_dataset_fallback(self, monkeypatch):
        # drawing's train pipes are plain Datasets (no get_loader); the trainer
        # must fall back to the legacy DistributedSampler build.
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", "1")
        from bbml.fsdp.dist import ParallelLayout
        from bbml.train.distributed.fully_shard_trainer import FullyShardTrainer
        from bbml.core.datamodels.configs import TrainerConfig
        from tests.seam_helpers import TinyFoundation, XDataset

        class PlainPipe(XDataset):
            batch_size = 2
            num_workers = 0
            shuffle = False

            def collate_fn(self, batch):
                return {"x": [b["x"] for b in batch]}

        cfg = TrainerConfig(project="test")
        trainer = FullyShardTrainer(
            TinyFoundation(), cfg, PlainPipe(8), None, None, owns_process_group=False,
        )
        trainer.layout = ParallelLayout(tp=1, shard=2)
        loader = trainer._wrap_train_dataloader()
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.rank == 1
        assert loader.sampler.num_replicas == 2
