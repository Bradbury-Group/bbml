"""Seam 2: Trainer.run_steps step-driven cycling."""
import itertools

import pytest

from bbml.core.trainer import Trainer


class FakeSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class FakeLoader:
    """Iterable with a `.sampler` but deliberately NO __len__ (proves run_steps
    never calls len())."""

    def __init__(self, n_batches):
        self.batches = [{"i": i} for i in range(n_batches)]
        self.sampler = FakeSampler()

    def __iter__(self):
        return iter(self.batches)


class FakePipe:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class LoopTrainer(Trainer):
    def __init__(self, pipe):
        super().__init__(model=None, train_config=None, train_datapipe=pipe,
                         val_datapipe=None, test_datapipe=None)

    def train(self):
        ...

    def validate(self):
        ...

    def test(self):
        ...

    def save(self, save_path, *, state_dict=None):
        ...

    def load(self, load_path, *, strict=False):
        ...


class TestRunSteps:
    def test_max_steps_stop(self):
        trainer = LoopTrainer(FakePipe())
        loader = FakeLoader(2)
        out = list(trainer.run_steps(loader, start_step=0, max_steps=5))
        assert len(out) == 5

    def test_cycles_and_tracks_epoch_batchnum(self):
        trainer = LoopTrainer(FakePipe())
        loader = FakeLoader(2)
        out = list(trainer.run_steps(loader, start_step=0, max_steps=5))
        assert [e for e, _, _ in out] == [0, 0, 1, 1, 2]
        assert [b for _, b, _ in out] == [0, 1, 0, 1, 0]

    def test_set_epoch_on_wrap(self):
        pipe = FakePipe()
        trainer = LoopTrainer(pipe)
        loader = FakeLoader(2)
        list(trainer.run_steps(loader, start_step=0, max_steps=5))
        # Called once per (re)wrap on both the sampler and the pipe.
        assert loader.sampler.epochs == [0, 1, 2]
        assert pipe.epochs == [0, 1, 2]

    def test_start_step_offset(self):
        trainer = LoopTrainer(FakePipe())
        out = list(trainer.run_steps(FakeLoader(2), start_step=3, max_steps=5))
        assert len(out) == 2

    def test_none_max_steps_cycles_forever(self):
        trainer = LoopTrainer(FakePipe())
        gen = trainer.run_steps(FakeLoader(2), start_step=0, max_steps=None)
        first = list(itertools.islice(gen, 5))
        assert len(first) == 5
        assert [e for e, _, _ in first] == [0, 0, 1, 1, 2]

    def test_empty_loader_raises(self):
        trainer = LoopTrainer(FakePipe())
        with pytest.raises(RuntimeError, match="empty loader"):
            list(trainer.run_steps(FakeLoader(0), start_step=0, max_steps=3))
