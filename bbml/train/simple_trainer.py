import inspect
from pathlib import Path
import warnings

from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from bbml.core.datamodels import TrainerConfig
from bbml.core.interfaces import Runnable
from bbml.core.trainer import Trainer
from bbml import logger
from bbml.registries import OptimizerRegistry


def init_cls_from_config(cls: type, config: TrainerConfig, *args, **kwargs):
    """
        Dynamically map config values to class constructor parameters.
    """
    sig = inspect.signature(cls.__init__)
    
    for param_name in sig.parameters.keys():
        if hasattr(config, param_name):
            kwargs[param_name] = getattr(config, param_name)
    
    return cls(*args, **kwargs)



class SimpleTrainer(Trainer):
    
    
    def train(self):
        if self.train_config.logging_backends is not None:
            logger.start(self.train_config.logging_backends)

        if self.model.optimizer is not None:
            optimizer_cls = self.model.optimizer
        elif self.train_config.optimizer is not None:
            optimizer_cls = OptimizerRegistry.get(self.train_config.optimizer)
        else:
            raise ValueError(f"Optimizer couldn't be initiated from model or config")
        optimizer = init_cls_from_config(optimizer_cls, self.train_config, self.model.train_parameters)
        self.optimizer = optimizer
        
        if self.model.lr_scheduler is not None:
            lr_scheduler_cls = self.model.lr_scheduler
        elif self.train_config.lr_scheduler is not None:
            lr_scheduler_cls = self.train_config.lr_scheduler
        else:
            raise ValueError(f"LRScheduler couldn't be initiated from model or config")
        lr_scheduler = init_cls_from_config(lr_scheduler_cls, self.train_config, optimizer)
        self.lr_scheduler = lr_scheduler
        
        if self.train_config.load_path is not None:
            self.load(self.train_config.load_path)

        device = "cuda:0"
        self.model.train()
        self.model.to(device=device)
        
        dataloader = self.train_datapipe.get_loader()

        total_steps = self.train_config.train_epochs * len(dataloader)
        pbar_total = tqdm(total=total_steps, desc="Total Steps", position=0)
        for epoch in range(self.train_config.train_epochs):
            pbar_epoch = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.train_config.train_epochs}",
                position=1,
                leave=False
            )
            for batch_num, batch in enumerate(pbar_epoch):
                optimizer.zero_grad()

                step_info = {
                    "step": self.train_config.step,
                    "batch_num": batch_num,
                    "epoch": epoch,
                    "split": "train",
                }
                batch.update(step_info)
                with torch.autocast(device):
                    loss = self.model.single_step(batch)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                learning_rates = {f"lr.{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
                log_metrics = {
                    "train_loss": loss.item(),
                    **step_info,
                    **learning_rates,
                }
                logger.log(log_metrics)
                pbar_total.set_postfix(log_metrics)

                self.do_val_test_save()
                
                pbar_total.update(1)
                self.train_config.step += 1

            pbar_epoch.close()
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        val_dataloader = self.val_datapipe.get_loader()

        total_val_loss = torch.tensor(0)
        for batch in tqdm(val_dataloader, desc="validation Steps", position=2):
            step_info = {
                "step": self.train_config.step,
                "split": "validation",
            }
            batch.update(step_info)
            loss = self.model.single_step(batch)
            total_val_loss += loss 
        val_loss = total_val_loss / len(val_dataloader)  # it's fine if exact inaccuracies due to last batch size happen
        log_metrics = {
            "validation_loss": val_loss.item(),
        }
        logger.log(log_metrics)
        return val_loss
        
    @torch.no_grad()
    def test(self):
        if not isinstance(self.model, Runnable):
            warnings.warn(f"Model {self.model!r} is not runnable, testing via `run()` is not available.")
            return None

        self.model.eval()
        test_dataloader = self.val_datapipe.get_loader()
        for batch in tqdm(test_dataloader, desc="Test Steps", position=2):
            test_input = self.model.input_model(**batch)
            output: BaseModel = self.model.run(test_input)
            logger.log(output.model_dump())
        
    def do_val_test_save(self, step: int):
        if self.train_config.check_step_trigger(
            step,
            self.train_config.validation_step_trigger
        ):
            self.validate()
        
        if self.train_config.check_step_trigger(
            step,
            self.train_config.test_step_trigger
        ):
            self.test()
        
        if self.train_config.check_step_trigger(
            step,
            self.train_config.save_step_trigger
        ):
            self.save(self.train_config.output_dir)
    

    def save(self, save_path: str | Path):
        self.model.save(save_path)
        optim_path = Path(save_path) / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optim_path)
        lrs_path = Path(save_path) / "optimizer.pt"
        torch.save(self.lr_scheduler.state_dict(), lrs_path)
        

    def load(self, load_path: str | Path):
        self.model.load(load_path)
        optim_path = Path(load_path) / "optimizer.pt"
        if optim_path.exists():
            self.optimizer.load_state_dict(torch.load(optim_path))
        lrs_path = Path(load_path) / "lr_scheduler.pt"
        if lrs_path.exists():
            self.lr_scheduler.load_state_dict(torch.load(lrs_path))

