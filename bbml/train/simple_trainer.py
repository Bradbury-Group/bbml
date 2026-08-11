import inspect
import warnings
from pathlib import Path

import torch
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from bbml import logger
from bbml.logger.utils import is_image_like, is_image_batch_like
from bbml.core.interfaces import Runnable
from bbml.core.trainer import Trainer
from bbml.registries import LRSchedulerRegistry, OptimizerRegistry
from bbml.train.param_groups import build_param_groups
from bbml.utils import set_seed


def _coerce_type(value, annotation):
    if annotation is inspect.Parameter.empty:
        return value
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        type_args = getattr(annotation, "__args__", ())
        for arg in type_args:
            if arg is not type(None):
                annotation = arg
                break
    if annotation in (int, float, str, bool):
        try:
            return annotation(value)
        except (TypeError, ValueError):
            pass
    return value


def init_cls_from_config(cls: type, config: BaseModel, *args, **kwargs):
    """Map config values to class constructor parameters with type coercion."""
    sig = inspect.signature(cls.__init__)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param_name in kwargs:
            continue
        if hasattr(config, param_name):
            value = getattr(config, param_name)
            if value is None:
                continue
            value = _coerce_type(value, param.annotation)
            kwargs[param_name] = value

    params = list(sig.parameters.values())

    # find positional parameters and pop from kwargs
    positional_params = [
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name not in ("self", "cls")
    ]
    names_to_pop = positional_params[:len(args)]

    for name in names_to_pop:
        kwargs.pop(name, None)
    return cls(*args, **kwargs)



class SimpleTrainer(Trainer):

    def train(self):
        set_seed(self.train_config.seed)
        print(f"[SimpleTrainer] Seed set to {self.train_config.seed}")

        if getattr(self.train_config, "gradient_accumulation_steps", 1) > 1:
            warnings.warn(
                "SimpleTrainer does not support gradient_accumulation_steps. "
                "Use AccelerateTrainer for gradient accumulation."
            )

        if self.train_config.logging_backends is not None:
            logger.start(
                self.train_config.logging_backends,
                **self.train_config.model_dump(),
            )

        if self.model.optimizer is not None:
            optimizer = self.model.optimizer
        elif self.train_config.optimizer is not None:
            optimizer_cls = OptimizerRegistry.get(self.train_config.optimizer)
            if self.train_config.param_group_rules:
                param_groups = build_param_groups(
                    self.model,
                    base_lr=self.train_config.lr,
                    base_wd=getattr(self.train_config, "weight_decay", 0.0),
                    rules=self.train_config.param_group_rules,
                )
            else:
                param_groups = self.model.get_train_parameters()
            optimizer = init_cls_from_config(optimizer_cls, self.train_config, param_groups)
        else:
            raise ValueError("Optimizer couldn't be initiated from model or config")
        self.optimizer = optimizer

        if self.model.lr_scheduler is not None:
            lr_scheduler = self.model.lr_scheduler
        elif self.train_config.lr_scheduler is not None:
            lr_scheduler_cls = LRSchedulerRegistry.get(self.train_config.lr_scheduler)
            lr_scheduler = init_cls_from_config(lr_scheduler_cls, self.train_config, optimizer)
        else:
            raise ValueError("LRScheduler couldn't be initiated from model or config")
        self.lr_scheduler = lr_scheduler

        if self.train_config.load_path is not None:
            self.load(self.train_config.load_path)

        device = getattr(self.train_config, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device=device)

        dataloader = self.train_datapipe.get_loader()

        # Foundation hook: after load, before the first batch.
        self.model.on_train_start(self.train_config.step)

        total_steps = self.train_config.train_epochs * len(dataloader)
        pbar_total = tqdm(total=total_steps, desc="Total Steps", position=0)
        for epoch in range(self.train_config.train_epochs):
            if hasattr(self.train_datapipe, "set_epoch"):
                # For epoch seeded shuffling
                print(f"[SimpleTrainer] Setting epoch={epoch} on train_datapipe")
                self.train_datapipe.set_epoch(epoch)
            pbar_epoch = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.train_config.train_epochs}",
                position=1,
                leave=False
            )
            for batch_num, batch in enumerate(pbar_epoch):
                optimizer.zero_grad()

                # Namespace step info to avoid collision with data keys
                batch["_bbml"] = {
                    "step": self.train_config.step,
                    "batch_num": batch_num,
                    "epoch": epoch,
                    "split": "train",
                }
                # Route through forward() which calls single_step() (consistent with distributed)
                result = self.model(batch)

                if isinstance(result, tuple):
                    loss, extra_metrics = result
                else:
                    loss, extra_metrics = result, {}

                loss.backward()

                grad_clip = getattr(self.train_config, "grad_clip_norm", None)
                if grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                # Foundation hook: immediately after optimizer + scheduler step.
                self.model.on_optimizer_step(self.train_config.step)

                learning_rates = {f"lr.{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
                log_metrics = {
                    "train_loss": loss.item(),
                    "step": self.train_config.step,
                    "batch_num": batch_num,
                    "epoch": epoch,
                    **learning_rates,
                    **extra_metrics,
                }
                logger.log(log_metrics, commit=True)
                pbar_total.set_postfix(log_metrics)

                self.do_val_test_save()

                pbar_total.update(1)
                self.train_config.step += 1

            pbar_epoch.close()

        self.do_val_test_save(do_all=True) # do all at end

    def _infer_batch_size(self, batch: dict) -> int:
        """Infer batch size from first tensor in batch."""
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return v.shape[0]
        return 1

    @torch.no_grad()
    def validate(self):
        """Validation loop with weighted batch accumulation for correct averaging."""
        if self.val_datapipe is None:
            warnings.warn("Validation DataPipe not provided, skipping")
            return torch.tensor(0)
        self.model.eval()

        val_dataloader = self.val_datapipe.get_loader()

        # Weighted accumulation for correct averaging with variable batch sizes
        total_loss = torch.tensor(0.0)
        total_metrics: dict[str, float] = {}
        total_samples = 0

        for batch in tqdm(val_dataloader, desc="Validation", position=2):
            batch["_bbml"] = {"step": self.train_config.step, "split": "validation"}
            batch_size = self._infer_batch_size(batch)

            result = self.model(batch)
            if isinstance(result, tuple):
                loss, extra_metrics = result
            else:
                loss, extra_metrics = result, {}

            total_loss += loss.detach().cpu() * batch_size
            for k, v in extra_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v * batch_size
            total_samples += batch_size

        if total_samples > 0:
            val_loss = total_loss / total_samples
            avg_metrics = {f"validation_{k}": v / total_samples for k, v in total_metrics.items()}
        else:
            val_loss = torch.tensor(0.0)
            avg_metrics = {}

        logger.log({"validation_loss": val_loss.item(), **avg_metrics}, commit=False)
        return val_loss

    @torch.no_grad()
    def test(self):
        if not isinstance(self.model, Runnable):
            warnings.warn(f"Model {self.model!r} is not runnable, testing via `run()` is not available.")
            return
        if self.test_datapipe is None:
            warnings.warn("Testing DataPipe not provided, skipping")
            return

        self.model.eval()
        test_dataloader = self.test_datapipe.get_loader()
        testing_samples = []
        input_logs: dict[str, list] = {}
        output_logs: dict[str, list] = {}
        for i, batch in enumerate(tqdm(test_dataloader, desc="Test Steps", position=2)):
            test_input = self.model.input_model(**batch)
            output: BaseModel = self.model.run(test_input)
            for k, v in test_input.model_dump().items():
                input_logs.setdefault(f"input_{k}", []).append(v)
            for k, v in output.model_dump().items():
                output_logs.setdefault(f"output_{k}", []).append(v)

            testing_samples.append({
                "input": test_input,
                "output": output,
            })

        # Use prompts as captions; convert any image-like lists to image dicts
        prompts = input_logs.pop("input_prompt", [])
        all_logs = {**input_logs, **output_logs}
        for key, vals in all_logs.items():
            if not isinstance(vals, list) or not vals:
                continue
            if any(is_image_like(v) or is_image_batch_like(v) for v in vals):
                if prompts and len(prompts) == len(vals):
                    all_logs[key] = {f"[{i}] {p}": v for i, (p, v) in enumerate(zip(prompts, vals))}
                else:
                    all_logs[key] = {f"[{i}]": v for i, v in enumerate(vals)}
        logger.log(all_logs, commit=False)
        return testing_samples

    def do_val_test_save(self, do_all=False):
        self.model.eval()
        should_validate = (
            self.train_config.step > 0
            and self.train_config.check_step_trigger(
                self.train_config.step,
                self.train_config.validation_step_trigger,
            )
        ) or do_all
        if should_validate:
            self.validate()

        if self.train_config.check_step_trigger(
            self.train_config.step,
            self.train_config.test_step_trigger
        ) or do_all:
            self.test()

        if self.train_config.check_step_trigger(
            self.train_config.step,
            self.train_config.save_step_trigger
        ) or do_all:
            self.save(self.train_config.output_dir)
        self.model.train()


    def save(self, save_path: str | Path):
        self.model.save_training_state(save_path, self.optimizer, self.lr_scheduler)

    def load(self, load_path: str | Path):
        self.model.load_training_state(load_path, self.optimizer, self.lr_scheduler)
