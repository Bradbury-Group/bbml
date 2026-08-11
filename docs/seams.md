# Foundation x Trainer seams

North star: any Foundation runs under any Trainer. The Foundation owns the
model (construction, step, persistence format, sharding plan, post-step
state); the Trainer owns the loop (data cadence, backward flavor, triggers,
logging). Nothing model-specific may live in a Trainer subclass.

## 1. Data

```python
@dataclass
class LoaderContext:
    dp_rank: int = 0
    dp_size: int = 1
    seed: int = 0
    epoch: int = 0
    drop_last: bool = True
```

- DataPipe.get_loader(ctx: LoaderContext | None = None) -> DataLoader.
  Default: ctx.dp_size > 1 wraps self in DistributedSampler (logic moves out
  of FullyShardTrainer._wrap_train_dataloader); else current behavior.
- Every trainer builds ctx and calls get_loader(ctx). A pipe that self-shards
  overrides get_loader and consumes ctx itself. Trainers never re-shard or
  re-wrap a returned loader.
- AccelerateTrainer: prepare_dataloader: bool = True init arg. True (default)
  keeps accelerator.prepare(loader) and passes dp_size=1 ctx; False passes the
  real ctx and uses the loader as returned. Pipes without a ctx-accepting
  get_loader: legacy path + DeprecationWarning.
- set_epoch: optional method on the pipe; trainers call it when present.

## 2. Loop

- Base Trainer gains run_steps(loader, start_step, max_steps): step-driven
  iteration that cycles the loader when exhausted (calling set_epoch on wrap)
  and never requires len(). Trainers use it whenever
  train_config.max_training_steps is set; pure epoch mode is unchanged.

## 3. Persistence

- Foundation.save_training_state(path, optimizer, scheduler, ema=None,
  metadata=None) and load_training_state(path, optimizer, scheduler,
  ema=None) -> dict. Defaults: single-rank = delta + optimizer.pt/
  lr_scheduler.pt (current behavior); dist world > 1 = bbml.fsdp.dcp helpers
  honoring CheckpointingConfig.format.
- Trainers call ONLY these two for save/load. FullyShardTrainer's format
  branches collapse into the Foundation defaults. Foundations that override
  own their format outright; format config is a hint to the defaults only.

## 4. Hooks (no-op defaults on Foundation)

- on_train_start(step: int): after load, before first batch (seed generators,
  warm caches).
- on_optimizer_step(step: int): immediately after optimizer.step() +
  lr_scheduler.step() in every trainer (EMA update lives here).

## 5. Sharding plan

- Foundation.fsdp_blocks() -> Iterable[nn.Module]; default raises with a
  clear message.
- Foundation.parallelise(mesh, *, policy) default: fully_shard_model over
  self.fsdp_blocks() with policy. Overrides remain for custom wraps.
- bbml/foundations/gpt2 implements fsdp_blocks (transformer.h) so the
  reference foundation runs under FullyShardTrainer.

## 6. Metrics

- Foundation.forward validates single_step's metrics dict: 0-dim tensors
  coerced to float; any other non-scalar raises TypeError naming the keys.
  (A tensor metric crashes WandBBackend on rank 0 and hangs siblings on the
  next collective.)

## 7. Registry

- Register "warmup_stable_decay" in LRSchedulerRegistry (transformers
  get_scheduler factory; kwargs num_warmup_steps / num_stable_steps /
  num_decay_steps / num_training_steps from TrainerConfig extras).

## Compatibility bar

- bbml tests green; gpt2 example unchanged in behavior.
- drawing (editable consumer) keeps working UNCHANGED before its adaptation:
  every seam is additive; existing subclass overrides keep winning.
