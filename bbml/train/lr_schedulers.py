import math
import warnings
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    MultiplicativeLR,
    LambdaLR,
)

from bbml.registries import LRSchedulerRegistry


# Register all PyTorch built-in LR schedulers
LRSchedulerRegistry.add("StepLR", StepLR)
LRSchedulerRegistry.add("MultiStepLR", MultiStepLR)
LRSchedulerRegistry.add("ConstantLR", ConstantLR)
LRSchedulerRegistry.add("LinearLR", LinearLR)
LRSchedulerRegistry.add("ExponentialLR", ExponentialLR)
LRSchedulerRegistry.add("PolynomialLR", PolynomialLR)
LRSchedulerRegistry.add("CosineAnnealingLR", CosineAnnealingLR)
LRSchedulerRegistry.add("ChainedScheduler", ChainedScheduler)
LRSchedulerRegistry.add("SequentialLR", SequentialLR)
LRSchedulerRegistry.add("ReduceLROnPlateau", ReduceLROnPlateau)
LRSchedulerRegistry.add("CyclicLR", CyclicLR)
LRSchedulerRegistry.add("OneCycleLR", OneCycleLR)
LRSchedulerRegistry.add("CosineAnnealingWarmRestarts", CosineAnnealingWarmRestarts)
LRSchedulerRegistry.add("MultiplicativeLR", MultiplicativeLR)
LRSchedulerRegistry.add("LambdaLR", LambdaLR)


# I am reimplementing this here, so we can include warmup and do any funky stuff like a precomputed lr drop as we need
def cosine_lr(t, warm, total, base=1e-3, min_frac=0.1):
    if t < warm: return base * (t+1) / warm
    u = min(max((t-warm)/(total-warm), 0.0), 1.0)
    return base * (min_frac + 0.5*(1-min_frac)*(1+math.cos(math.pi*u)))

@LRSchedulerRegistry.register("CosineWarmupMaxHoldScheduler")
class CosineWarmupMaxHoldScheduler(LRScheduler):
    """
    PyTorch LR Scheduler for endless training with four phases:
    1. Linear warmup (0 to warmup_steps)
    2. Hold at max LR (warmup_steps to hold_steps)
    3. Cosine decay (hold_steps to cosine_steps) 
    4. Constant minimum LR (cosine_steps to infinity)
    """
    
    def __init__(self, optimizer, warmup_steps, hold_steps, cosine_steps, min_lr=1e-5, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps (linear increase to base LR)
            hold_steps: Step where hold phase ends and cosine decay begins
            cosine_steps: Step where cosine decay ends and constant LR begins
            min_lr: Minimum learning rate (absolute value, not fraction)
            last_epoch: Last epoch index (default: -1)
        """
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.cosine_steps = cosine_steps
        self.min_lr = min_lr
        
        if hold_steps <= warmup_steps:
            raise ValueError(f"hold_steps ({hold_steps}) must be > warmup_steps ({warmup_steps})")
        if cosine_steps <= hold_steps:
            raise ValueError(f"cosine_steps ({cosine_steps}) must be > hold_steps ({hold_steps})")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be > 0")
        
        super().__init__(optimizer, last_epoch)
        
        # Warn if any base LR is less than min_lr
        for i, base_lr in enumerate(self.base_lrs):
            if base_lr < min_lr:
                warnings.warn(f"Base LR {base_lr} for param group {i} is less than min_lr {min_lr}")
    
    
    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch + 1  # last_epoch starts at -1
        
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Phase 1: Warmup - linear increase from min_lr to base_lr
                # At step 0: lr = min_lr, at step warmup_steps-1: lr = base_lr
                lr = self.min_lr + (base_lr - self.min_lr) * step / self.warmup_steps
            elif step < self.hold_steps:
                # Phase 2: Hold at maximum LR
                lr = base_lr
            elif step < self.cosine_steps:
                # Phase 3: Cosine decay from base_lr to min_lr
                # At step hold_steps: lr = base_lr, at step cosine_steps-1: lr = min_lr
                decay_steps = self.cosine_steps - self.hold_steps
                progress = (step - self.hold_steps) / decay_steps
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            else:
                # Phase 4: Constant minimum LR (training continues forever)
                lr = self.min_lr
            
            lrs.append(lr)
        
        return lrs


@LRSchedulerRegistry.register("OneCycleScheduler")
class OneCycleScheduler(LRScheduler):
    """One-cycle learning rate scheduler with peak at specified step."""

    def __init__(self, optimizer, peak_lr, peak_step, final_step, min_lr=1e-5,
                 warmup_steps=100, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            peak_lr: Maximum learning rate at peak
            peak_step: Step where LR peaks
            final_step: Step where cosine decay ends
            min_lr: Minimum learning rate
            warmup_steps: Initial warmup steps to base_lr
            last_epoch: Last epoch index
        """
        self.peak_lr = peak_lr
        self.peak_step = peak_step
        self.final_step = final_step
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Warmup to base_lr
                lr = self.min_lr + (base_lr - self.min_lr) * step / self.warmup_steps
            elif step < self.peak_step:
                # Rise to peak
                progress = (step - self.warmup_steps) / (self.peak_step - self.warmup_steps)
                lr = base_lr + (self.peak_lr - base_lr) * progress
            elif step < self.final_step:
                # Cosine decay from peak to min
                progress = (step - self.peak_step) / (self.final_step - self.peak_step)
                lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            else:
                lr = self.min_lr

            lrs.append(lr)

        return lrs


@LRSchedulerRegistry.register("SGDRScheduler")
class SGDRScheduler(LRScheduler):
    """Cosine annealing with warm restarts (SGDR)."""

    def __init__(self, optimizer, periods, lr_max, lr_min,
                 final_cosine_start=None, final_cosine_end=None,
                 final_min_lr=1e-5, warmup_steps=100, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            periods: List of period lengths [4000, 12000, 14000]
            lr_max: Maximum LR after each restart
            lr_min: Minimum LR within each period
            final_cosine_start: Optional final cosine decay start
            final_cosine_end: Optional final cosine decay end
            final_min_lr: Final minimum LR
            warmup_steps: Initial warmup steps
            last_epoch: Last epoch index
        """
        self.periods = periods
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.final_cosine_start = final_cosine_start
        self.final_cosine_end = final_cosine_end
        self.final_min_lr = final_min_lr
        self.warmup_steps = warmup_steps

        # Calculate period boundaries
        self.period_starts = [0]
        for p in periods[:-1]:
            self.period_starts.append(self.period_starts[-1] + p)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Warmup
                lr = self.final_min_lr + (base_lr - self.final_min_lr) * step / self.warmup_steps
            else:
                adjusted_step = step - self.warmup_steps

                # Find which period we're in
                period_idx = 0
                for i, start in enumerate(self.period_starts[1:], 1):
                    if adjusted_step >= start:
                        period_idx = i
                    else:
                        break

                if period_idx < len(self.periods):
                    # Within a restart period
                    period_start = self.period_starts[period_idx]
                    period_length = self.periods[period_idx]
                    progress = (adjusted_step - period_start) / period_length
                    progress = min(progress, 1.0)

                    # Cosine annealing within period
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
                else:
                    # After all periods
                    if self.final_cosine_start and self.final_cosine_end and step >= self.final_cosine_start:
                        # Final cosine decay
                        if step < self.final_cosine_end:
                            progress = (step - self.final_cosine_start) / (self.final_cosine_end - self.final_cosine_start)
                            lr = self.final_min_lr + 0.5 * (self.lr_min - self.final_min_lr) * (1 + math.cos(math.pi * progress))
                        else:
                            lr = self.final_min_lr
                    else:
                        lr = self.lr_min

            lrs.append(lr)

        return lrs

@LRSchedulerRegistry.register("MicroPulseScheduler")
class MicroPulseScheduler(LRScheduler):
    """Scheduler with micro-pulses at specific steps."""

    def __init__(self, optimizer, pulse_steps, pulse_amplitude, pulse_duration,
                 base_lr=1e-3, cosine_start=None, cosine_end=None, min_lr=1e-5,
                 warmup_steps=100, heads_pulse_mult=1.0, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            pulse_steps: List of steps where pulses occur [2200, 2800, 3400]
            pulse_amplitude: Peak LR during pulse
            pulse_duration: Duration of each pulse
            base_lr: Base learning rate between pulses
            cosine_start: Optional cosine decay start
            cosine_end: Optional cosine decay end
            min_lr: Minimum learning rate
            warmup_steps: Initial warmup steps
            heads_pulse_mult: Extra multiplier for heads during pulses
            last_epoch: Last epoch index
        """
        self.pulse_steps = pulse_steps
        self.pulse_amplitude = pulse_amplitude
        self.pulse_duration = pulse_duration
        self.base_lr = base_lr
        self.cosine_start = cosine_start
        self.cosine_end = cosine_end
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.heads_pulse_mult = heads_pulse_mult
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for i, base_lr_orig in enumerate(self.base_lrs):
            if step < self.warmup_steps:
                # Warmup
                lr = self.min_lr + (self.base_lr - self.min_lr) * step / self.warmup_steps
            else:
                # Check if we're in a pulse
                in_pulse = False
                for pulse_start in self.pulse_steps:
                    if pulse_start <= step < pulse_start + self.pulse_duration:
                        # Triangular pulse
                        pulse_progress = (step - pulse_start) / self.pulse_duration
                        if pulse_progress < 0.5:
                            # Rising
                            lr = self.base_lr + (self.pulse_amplitude - self.base_lr) * (2 * pulse_progress)
                        else:
                            # Falling
                            lr = self.pulse_amplitude - (self.pulse_amplitude - self.base_lr) * (2 * (pulse_progress - 0.5))

                        # Apply heads multiplier if this is a heads param group
                        if self.heads_pulse_mult > 1.0 and i > 0:  # Assume heads are groups 1+
                            lr *= self.heads_pulse_mult

                        in_pulse = True
                        break

                if not in_pulse:
                    # Base LR or cosine decay
                    if self.cosine_start and self.cosine_end and step >= self.cosine_start:
                        if step < self.cosine_end:
                            progress = (step - self.cosine_start) / (self.cosine_end - self.cosine_start)
                            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                        else:
                            lr = self.min_lr
                    else:
                        lr = self.base_lr

            lrs.append(lr)

        return lrs

@LRSchedulerRegistry.register("TwoCycleCLRScheduler")
class TwoCycleCLRScheduler(LRScheduler):
    """Two-cycle triangular learning rate scheduler."""

    def __init__(self, optimizer, cycle1_start, cycle1_end, cycle1_min, cycle1_max,
                 cycle2_start, cycle2_end, cycle2_min, cycle2_max,
                 cosine_end=None, min_lr=1e-5, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            cycle1_start/end: First cycle boundaries
            cycle1_min/max: First cycle LR range
            cycle2_start/end: Second cycle boundaries
            cycle2_min/max: Second cycle LR range
            cosine_end: Optional final cosine decay end
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.cycle1_start = cycle1_start
        self.cycle1_end = cycle1_end
        self.cycle1_min = cycle1_min
        self.cycle1_max = cycle1_max
        self.cycle2_start = cycle2_start
        self.cycle2_end = cycle2_end
        self.cycle2_min = cycle2_min
        self.cycle2_max = cycle2_max
        self.cosine_end = cosine_end
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if step < self.cycle1_start:
                # Before first cycle (warmup)
                lr = self.min_lr + (self.cycle1_min - self.min_lr) * step / self.cycle1_start
            elif step < self.cycle1_end:
                # First cycle - triangular
                cycle_progress = (step - self.cycle1_start) / (self.cycle1_end - self.cycle1_start)
                # Multiple triangles within the cycle
                num_triangles = 8  # 8 triangles in first cycle
                triangle_progress = (cycle_progress * num_triangles) % 1.0
                if triangle_progress < 0.5:
                    lr = self.cycle1_min + (self.cycle1_max - self.cycle1_min) * (2 * triangle_progress)
                else:
                    lr = self.cycle1_max - (self.cycle1_max - self.cycle1_min) * (2 * (triangle_progress - 0.5))
            elif step < self.cycle2_end:
                # Second cycle - triangular with decay
                cycle_progress = (step - self.cycle2_start) / (self.cycle2_end - self.cycle2_start)
                # Decay the amplitude over time
                decay_factor = 1.0 - 0.5 * cycle_progress
                current_min = self.cycle2_min
                current_max = self.cycle2_min + (self.cycle2_max - self.cycle2_min) * decay_factor

                # Slower triangles in second cycle
                num_triangles = 4
                triangle_progress = (cycle_progress * num_triangles) % 1.0
                if triangle_progress < 0.5:
                    lr = current_min + (current_max - current_min) * (2 * triangle_progress)
                else:
                    lr = current_max - (current_max - current_min) * (2 * (triangle_progress - 0.5))
            elif self.cosine_end and step < self.cosine_end:
                # Final cosine decay
                progress = (step - self.cycle2_end) / (self.cosine_end - self.cycle2_end)
                lr = self.min_lr + 0.5 * (self.cycle2_min - self.min_lr) * (1 + math.cos(math.pi * progress))
            else:
                lr = self.min_lr

            lrs.append(lr)

        return lrs
