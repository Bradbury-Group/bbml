



from collections import defaultdict, OrderedDict
from logging import Logger
import os
from pathlib import Path
import time
import uuid
import warnings
from functools import wraps
from typing import Callable, Literal
import statistics
import inspect
import traceback
from typing import Callable, Mapping
import gc

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image


DEBUG = True


def set_debug(val: bool):
    global DEBUG
    DEBUG = val


def fmt_float(x: float, max_decimals=10) -> str:
    s = f"{x:.{max_decimals}f}"
    s = s.rstrip('0').rstrip('.')
    if s == "-0":  # edge case for values between -1e-10 < x < 0
        s = "0"
    return s


def ftimed(
    func:Callable=None,
    sync_cuda:bool=False,
    print_msg:bool=True,
    logger:Logger|None=None,
    record_dict:defaultdict[str,list]|None=None,
):
    """
        Simple function timer for lightweight profiling and logging.
        use like: @ftimed(...) or @ftimed
        sync_cuda=True causes unnecessary syncs which may harm performance
        sync_cuda=False has less accurate timings
        For extensive profiling use torch.profiler
    """
    if sync_cuda and not torch.cuda.is_available():
        warnings.warn(f"Requested cuda sync when cuda is not available, disabling.")
        sync_cuda = False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return func(*args, **kwargs)
            else:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                if sync_cuda:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                time_taken = end_time - start_time
                name = func.__qualname__
                log_msg = f"Timed: {name}: {fmt_float(time_taken)} seconds"
                if print_msg:
                    print(log_msg)
                if logger is not None:
                    logger.debug(log_msg)
                if record_dict is not None:
                    record_dict[name].append(time_taken)
                return result
        return wrapper

    if func is None:  # @ftimed
        return decorator
    else:  # @ftimed()
        return decorator(func)


class ctimed:
    """
        Context manager equivalent for ftimed
        use like: with ctimed("..."):
    """
    def __init__(
        self,
        name:str,
        sync_cuda:bool=False,
        print_msg:bool=True,
        logger:Logger|None=None,
        record_dict:defaultdict[str,list]|None=None,
    ):
        self.name: str = name
        if sync_cuda and not torch.cuda.is_available():
            warnings.warn(f"Requested cuda sync when cuda is not available, disabling.")
            sync_cuda = False
        self.sync_cuda:bool = sync_cuda
        self.print_msg:bool = print_msg
        self.logger:Logger|None = logger
        self.record_dict:defaultdict[str,list]|None = record_dict
        self.start_time:float|None = None

    def __enter__(self):
        if DEBUG:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not DEBUG:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        time_taken = end_time - self.start_time
        log_msg = f"Timed: {self.name}: {fmt_float(time_taken)} seconds"
        if self.print_msg:
            print(log_msg)
        if self.logger is not None:
            self.logger.debug(log_msg)
        if self.record_dict is not None:
            self.record_dict[self.name].append(time_taken)


def print_gpu_memory():
    if not torch.cuda.is_available():
        warnings.warn("cuda not available")
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    usage = allocated / total
    GB = 1024**3
    print(f"Allocated (GB): {fmt_float(allocated/GB)} GB")
    print(f"Reserved (GB): {fmt_float(reserved/GB)} GB")
    print(f"Total (GB): {fmt_float(total/GB)} GB")
    print(f"Allocated (%): {fmt_float(usage * 100)}%")


def clear_gpu_memory():
    if not torch.cuda.is_available():
        warnings.warn("cuda not available")
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def fprint(
    func=None,
    *,
    exclude=None,
    logger:Logger|None=None,
    print_input=True,
    print_output=True
):
    """
        Wrapper to print all inputs and outputs for debugging.
        use like: @fprint or @fprint(...) or fprint(func)(...)
    """
    # Normalize exclude into set
    if isinstance(exclude, str):
        exclude_names = {exclude}
    elif exclude is None:
        exclude_names = set()
    else:
        exclude_names = set(exclude)

    def decorator(f):
        sig = inspect.signature(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return f(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()  # include default values

            # Build list of "name=value" parts
            parts = []
            for name, param in sig.parameters.items():
                if name not in bound.arguments:
                    continue
                if name in exclude_names:
                    continue

                value = bound.arguments[name]

                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    # *args-like parameter
                    if value:
                        parts.append(f"{name}={value}")
                elif param.kind is inspect.Parameter.VAR_KEYWORD:
                    # **kwargs-like parameter
                    # allow excluding names inside **kwargs too
                    if value:
                        filtered = {
                            k: v for k, v in value.items()
                            if k not in exclude_names
                        }
                        if filtered:
                            parts.append(f"{name}={filtered}")
                else:
                    parts.append(f"{name}={value}")

            arg_str = ", ".join(parts)

            if print_input:
                inp_str = f"Calling {f.__qualname__}({arg_str})"
                print(inp_str)
                if logger is not None:
                    logger.debug(inp_str)


            result = f(*args, **kwargs)

            if print_output:
                out_str = f"{f.__qualname__} returned {result}"
                print(out_str)
                if logger is not None:
                    logger.debug(out_str)

            return result

        return wrapper

    # Support @fprint and @fprint(...)
    if func is None:
        return decorator
    else:
        return decorator(func)


def get_increment_fn(max_depth: int|None = 100):
    depth = 0
    def increment(val):
        nonlocal depth
        depth += 1
        if max_depth is not None and depth > max_depth:
            raise RecursionError(f"Exceeded {max_depth=} for incrementing value {val}")
        return val + 1
    return increment


def get_decrement_fn(max_depth: int|None = 100):
    depth = 0
    def decrement(val):
        nonlocal depth
        depth += 1
        if max_depth is not None and depth > max_depth:
            raise RecursionError(f"Exceeded {max_depth=} for decrementing value {val}")
        return val - 1
    return decrement


def fretry(
    func=None,
    *,
    exceptions=(Exception,),
    modifiers: Mapping[str, Callable | None] | None = None,
    always_retry: bool = False
):
    """
        Retry a function once with modified arguments if it raises exceptions.
        modifiers is a mapping from *parameter name* to a callable modifying it. 
        does not modify function *args-like parameters yet (to be implemented)
        use like: @fretry or @fretry(...)
    """
    modifiers = dict(modifiers or {})
    _identity = lambda x: x

    def decorator(f):
        sig = inspect.signature(f)

        @wraps(f)
        def fretry_wrapper(*args, **kwargs):
            if not DEBUG and not always_retry:  # no retry when not debugging and not specified always_retry
                return f(*args, **kwargs)
            try:
                return f(*args, **kwargs)
            except exceptions as e:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                original_args = OrderedDict(bound.arguments)

                # modify function named parameters
                for name, mod in modifiers.items():
                    mod_func = mod or _identity
                    if name in bound.arguments:
                        bound.arguments[name] = mod_func(bound.arguments[name])

                # modify any function **kwargs-like parameters
                var_kw_param_name = None
                for pname, param in sig.parameters.items():
                    if param.kind is inspect.Parameter.VAR_KEYWORD:
                        var_kw_param_name = pname
                        break
                if var_kw_param_name and var_kw_param_name in bound.arguments:
                    kw_dict = bound.arguments[var_kw_param_name]
                    for name, mod in modifiers.items():
                        if name not in sig.parameters and name in kw_dict:
                            mod_func = mod or _identity
                            kw_dict[name] = mod_func(kw_dict[name])


                new_args = bound.args
                new_kwargs = bound.kwargs

                traceback.print_exc()
                warnings.warn(
                    f"Function {f.__name__} failed due to {e!r} with inputs "
                    f"{dict(original_args)!r}, retrying with modified inputs "
                    f"{dict(bound.arguments)!r}"
                )

                # Recursively retry with the modified arguments
                return fretry_wrapper(*new_args, **new_kwargs)

        return fretry_wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def texam(t: torch.Tensor, name: str | None = None, verbose: bool = True):
    """
        Examine tensor for debugging.
    """
    if name is None:
        name = ""
    else:
        name += " "  # spacing

    print(f"{name}Shape: {tuple(t.shape)} (ndim={t.ndim}, numel={t.numel()})")
    print(f"{name}Device: {t.device}, Dtype: {t.dtype}, Layout: {t.layout}")
    print(f"{name}Requires Grad: {t.requires_grad}, Is Leaf: {t.is_leaf}")
    if t.grad is not None:
        print(f"{name}Grad: shape={tuple(t.grad.shape)}, dtype={t.grad.dtype}")

    if verbose:
        print(f"{name}Contiguous: {t.is_contiguous()}, Stride: {tuple(t.stride())}")
        print(f"{name}Element size: {t.element_size()} bytes (~{t.element_size() * t.numel() / 1e6:.3f} MB)")

    if t.numel() == 0:
        print(f"{name}Tensor is empty")
        return

    if t.dtype.is_floating_point or t.dtype.is_complex:
        with torch.no_grad():
            # For complex, use magnitude for min/max, but mean is ok
            if t.dtype.is_complex:
                magnitudes = t.abs()
                min_val = magnitudes.min().item()
                max_val = magnitudes.max().item()
                mean_val = t.mean().item()
                print(f"{name}Min |x|: {min_val}, Max |x|: {max_val}, Mean: {mean_val}")
            else:
                min_val = t.min().item()
                max_val = t.max().item()
                mean_val = t.mean().item()
                std_val = t.std().item()
                print(f"{name}Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")

            # NaN / Inf counts (for floats/complex only)
            num_nan = torch.isnan(t).sum().item()
            num_inf = torch.isinf(t).sum().item()
            if num_nan or num_inf or verbose:
                print(f"{name}NaNs: {num_nan}, Infs: {num_inf}")
    else:
        # Integer / bool tensors
        min_val = t.min().item()
        max_val = t.max().item()
        print(f"{name}Min: {min_val}, Max: {max_val}, Mean: N/A (non-floating dtype)")

    if verbose:
        print(f"{name}Values: {t}")