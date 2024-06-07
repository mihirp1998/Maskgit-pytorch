"""
A collection of assorted utilities for deep learning with no required dependencies aside from PyTorch, NumPy and the Python stdlib.

TODO: Validate required Python version.
"""
import contextlib
import functools
import glob
import hashlib
import io
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial, wraps
from importlib import import_module
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

# try:
#     from gen.utils.logging_utils import log_info, set_logger
#     set_logger(__name__)
#     log_func = log_info
# except ImportError:
#     
log_func = print

def get_info():
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log_func(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_params(model):
    log_func(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_func(f"Unfrozen Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log_func(f"Frozen Parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")


def calculate_storage_size(obj, storage_view_sizes, count_views=False):
    if isinstance(obj, torch.Tensor):
        storage = obj.storage()
        storage_id = id(storage)
        element_size = storage.element_size()
        storage_size = storage.size() * element_size
        view_size = obj.numel() * element_size

        # We count storage size only for the first time we encounter the storage
        if storage_id not in storage_view_sizes:
            storage_view_sizes[storage_id] = storage_size
            print_size = storage_size
        else:
            print_size = 0 if not count_views or not obj._is_view() else view_size

        if count_views or not obj._is_view():
            log_func(f"{'View' if obj._is_view() else 'Storage'} Tensor: " f"shape {obj.size()}, size {print_size / (1024 ** 2):.2f} MB")

        return print_size if count_views or not obj._is_view() else 0  # Count views only if requested
    elif isinstance(obj, dict):
        # Recurse for dictionaries
        return sum(calculate_storage_size(v, storage_view_sizes, count_views) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        # Recurse for lists or tuples
        return sum(calculate_storage_size(item, storage_view_sizes, count_views) for item in obj)
    elif hasattr(obj, "__dataclass_fields__"):
        # Recurse for dataclasses based on their fields
        fields = getattr(obj, "__dataclass_fields__")
        return sum(calculate_storage_size(getattr(obj, f), storage_view_sizes, count_views) for f in fields)
    else:
        # Non-Tensor, non-dict, non-list objects are not measured
        return 0


def calculate_total_size(obj, count_views=False):
    storage_view_sizes = defaultdict(int)
    total_size = calculate_storage_size(obj, storage_view_sizes, count_views)
    total_unique_storage_size = sum(storage_view_sizes.values())
    log_func(f"Total unique storage size: {total_unique_storage_size / (1024 ** 2):.2f} MB")
    if count_views:  # Only add view sizes to total if requested
        total_view_size = total_size - total_unique_storage_size
        log_func(f"Total view size (if counted): {total_view_size / (1024 ** 2):.2f} MB")
    else:
        log_func(f"Total size (without counting views): {total_size / (1024 ** 2):.2f} MB")

    return total_size


def save_tensor_dict(tensor_dict: dict, path):
    output_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, Tensor):
            if v.dtype == torch.float16 or v.dtype == torch.bfloat16:
                output_dict[k] = v.to(dtype=torch.float32).detach().cpu().numpy()
            else:
                output_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            output_dict[f"np_{k}"] = v
        else:
            output_dict[k] = v
    np.savez_compressed(path, **output_dict)


def load_tensor_dict(path: Path, object_keys=[]):
    from jaxtyping import BFloat16 # TODO: Remove dependency
    tensor_dict = {}
    np_dict = np.load(path, allow_pickle=True)
    for k, v in np_dict.items():
        if k in object_keys:
            tensor_dict[k] = v
        elif v.dtype == BFloat16:
            tensor_dict[k] = torch.from_numpy(v.astype(np.float32)).to(dtype=torch.bfloat16)
        elif k.startswith("np_"):
            tensor_dict[k.removeprefix("np_")] = v
        else:
            tensor_dict[k] = torch.from_numpy(v)
    return tensor_dict


def tensor_hash(tensor):
    """Computes a SHA256 hash of a tensor. Useful for debugging to check equality in different places."""
    tensor_bytes = tensor.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def module_hash(module: Optional[dict] = None, state_dict: Optional[dict] = None):
    assert module is not None or state_dict is not None
    state_dict = module.state_dict() if module is not None else state_dict
    sorted_state_dict = {k: state_dict[k] for k in sorted(state_dict)}
    params_cat = torch.cat([v.flatten() for _, v in sorted_state_dict.items()])
    return tensor_hash(params_cat)


def find_diff_params(state_dict_1, state_dict_2):
    diff_keys = set(state_dict_1.keys()) ^ set(state_dict_2.keys())  # Symmetric difference to find keys not in both
    matched_keys = set(state_dict_1.keys()) & set(state_dict_2.keys())  # Intersection to find keys in both

    # Check for differences in matched keys
    for key in matched_keys:
        if not torch.equal(state_dict_1[key], state_dict_2[key]):
            diff_keys.add(key)

    return diff_keys


def init_from_ckpt(module, path, ignore_keys=None, unfrozen_keys=None, strict=False, truncate=None, only_incl=None, verbose=True):
    log_func(f"Loading {module.__class__.__name__} from checkpoint: {path}")
    log_func(f"Strict Load: {strict}, Ignoring: {ignore_keys}, Unfreezing: {unfrozen_keys}, Truncating: {truncate}")

    if ignore_keys is None:
        ignore_keys = ()

    if unfrozen_keys is None:
        unfrozen_keys = ()

    sd = torch.load(path, map_location="cpu")

    if "state_dict" in sd.keys():
        sd = sd["state_dict"]
    elif "weight" in sd.keys():
        sd = sd["weight"]

    num_deleted = defaultdict(int)
    for k in list(sd):
        for ik in ignore_keys:
            if k.startswith(ik):
                num_deleted[ik] += 1
                del sd[k]

    for k, v in num_deleted.items():
        log_func(f"Deleted {v} keys due to ignore_key: {k}")

    if truncate is not None:
        for k in list(sd):
            if k.startswith(truncate):
                sd[k.replace(truncate, "")] = sd[k]
            del sd[k]

    num_ignored = defaultdict(int)
    for n in module.state_dict().keys():
        if n not in sd.keys():
            for ik in ignore_keys:
                if ik in n:
                    num_ignored[ik] += 1
                else:
                    log_func(f"Missing {n}")

    if only_incl is not None:
        for k in list(sd):
            keep = False
            for ik in only_incl:
                if ik in k:
                    keep = True
            if not keep:
                del sd[k]

    for k, v in num_ignored.items():
        log_func(f"Missing {v} keys due to ignore_key: {k}")

    for n in sd.keys():
        if n not in module.state_dict().keys():
            log_func(f"Unexpected {n}")

    checkpoint_keys = set(sd.keys())
    current_keys = set(module.state_dict().keys())

    if verbose:
        log_func(f"Loading: {checkpoint_keys.intersection(current_keys)}")
    else:
        log_func(f"Loading {len(checkpoint_keys.intersection(current_keys))} keys into the model: {str(module.__class__)}")

    module.load_state_dict(sd, strict=strict)

    if len(unfrozen_keys) > 0:
        for n, p in module.named_parameters():
            p.requires_grad_ = False
            for unfrozen_name in unfrozen_keys:
                if unfrozen_name in n:
                    p.requires_grad_ = True
                    log_func(f"Unfreezing: {n}")

    log_func(f"Restored from {path}")


def check_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    # Check if distributed
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    total_memory = torch.cuda.get_device_properties(rank).total_memory

    allocated_percent = (allocated / total_memory) * 100
    reserved_percent = (reserved / total_memory) * 100

    log_func(f"Allocated memory: {allocated_percent:.2f}%")
    log_func(f"Reserved memory: {reserved_percent:.2f}%")
    log_func(f'Available devices (CUDA_VISIBLE_DEVICES): {os.environ.get("CUDA_VISIBLE_DEVICES")}')

    assert allocated_percent <= 5
    assert reserved_percent <= 5


def load_checkpoint_from_url(url: str, file_path: Optional[str] = None) -> Path:
    if file_path is None:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_path is not None:
            filename = file_path

        file_path = Path.home() / ".cache" / "pretrained_weights" / filename

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        log_func(f'Downloading: "{url}" to {file_path}\n')
        torch.hub.download_url_to_file(url, file_path, progress=True)

    return file_path


# Copied from torch.profiler.profiler
def tensorboard_trace_handler(dir_name: str, record_memory: bool = False, worker_name: Optional[str] = None, use_gzip: bool = True):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof: torch.profiler.profile) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"

        chrome_trace_path = os.path.join(dir_name, file_name)
        memory_trace_path = os.path.join(dir_name, "memory_timeline.html")
        if is_main_process() and record_memory:
            try:
                log_func(f"Exporting memory timeline: {memory_trace_path}")
                prof.export_memory_timeline(memory_trace_path)
            except Exception as e:
                log_func(f"Failed to export memory timeline: {e}")
        
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)

        log_func(f"Exporting chrome trace to {chrome_trace_path}")
        prof.export_chrome_trace(chrome_trace_path)

    return handler_fn

def save_memory_profile(profile_dir):
    import wandb
    rank_postfix = f"_rank_{get_rank()}" if use_dist() else ""
    log_func(f"Saving memory profile to {profile_dir}")
    os.makedirs(profile_dir, exist_ok=True)
    torch.cuda.memory._dump_snapshot(f"{profile_dir}/memory_snapshot{rank_postfix}.pickle")
    os.system(
        f"python -m torch.cuda._memory_viz trace_plot {profile_dir}/memory_snapshot{rank_postfix}.pickle -o {profile_dir}/memory_snapshot{rank_postfix}.html"
    )
    torch.cuda.memory._save_segment_usage(f"{profile_dir}/segment{rank_postfix}.svg")
    torch.cuda.memory._save_memory_usage(f"{profile_dir}/memory{rank_postfix}.svg") 
    torch.cuda.memory._record_memory_history(enabled=None)

    log_func(f"Saved memory snapshot at: {profile_dir}/memory_snapshot{rank_postfix}.pickle")
    log_func(f"Run the following to view the snapshot:\npython -m http.server --directory {profile_dir.resolve()} 6008")

    if is_main_process() and wandb.run is not None:
        wandb.log({'profile': wandb.Html(f"{profile_dir}/memory_snapshot{rank_postfix}.html")})
        wandb.log({'profile': wandb.Html(f"{profile_dir}/memory_timeline{rank_postfix}.html")})

class Profiler:
    def __init__(self, output_dir, warmup_steps: int = 5, active_steps: int = 3, record_memory: bool = False, record_memory_only: bool = False):
        self.record_memory = record_memory
        self.profile_dir = Path(output_dir) / "profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        wait, warmup, active, repeat = 0, warmup_steps, active_steps, 0
        self.total_steps = (wait + warmup + active) * (1 + repeat)
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler_kwargs = dict(record_shapes=True, with_stack=True)
        if record_memory_only:
            pass
        else:
            profiler_kwargs.update(
                dict(
                    with_modules=True,
                    with_flops=True,
                ),
            )

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tensorboard_trace_handler(self.profile_dir, record_memory=record_memory),
            profile_memory=record_memory,
            **profiler_kwargs
        )
        self.profiler.start()

    def step(self, global_step: int):
        self.profiler.step()
        return global_step >= (self.total_steps - 1)

    def finish(self):
        self.profiler.stop()
        if use_dist():
            torch.distributed.barrier()
        if is_main_process():
            import wandb
            traces = glob.glob(f"{self.profile_dir}/*.pt.trace.json*")
            for trace in traces:
                log_func(f"Adding {trace}")
                wandb.save(trace, base_path=self.profile_dir, policy="now")                

        if use_dist():
            torch.distributed.barrier()

def use_dist():
    return dist.is_available() and dist.is_initialized()

def get_device():
    return torch.device(f"cuda:{get_rank()}")

def get_rank() -> int:
    if use_dist():
        return dist.get_rank()
    elif (rank := os.environ.get("RANK", None)) is not None:
        return int(rank)
    else:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0


def get_num_gpus() -> int:
    return dist.get_world_size() if use_dist() else torch.cuda.device_count()


def get_pdb():
    return import_module("pdb") if (any(["_pdbpp_path_hack" in p for p in sys.path]) or find_spec("ipdb") is None) else import_module("ipdb")


def _breakpoint(rank: Optional[int] = None, traceback: Optional[Any] = None):
    if get_num_gpus() > 1:
        if (is_main_process() if rank is None else get_rank() == rank):
            old_stdin = None
            if isinstance(sys.stdin, io.TextIOWrapper):
                old_stdin = sys.stdin
                sys.stdin = open(0)
            try:
                log_func('Breakpoint triggered. You may need to type "up" to get to the correct frame')
                if traceback is not None:
                    get_pdb().post_mortem(traceback)
                else:
                    get_pdb().set_trace()
            finally:
                if old_stdin is not None:
                    sys.stdin.close()
                    sys.stdin = old_stdin
        dist.barrier()
    else:
        if traceback is not None:
            get_pdb().post_mortem(traceback)
        else:
            log_func("Breakpoint triggered. You may need to type \"up\" to get to the correct frame")
            get_pdb().set_trace(sys._getframe(1))

def set_global_exists():
    import builtins
    builtins.exists = lambda v: v is not None

def set_global_breakpoint():
    import builtins
    import ipdb

    builtins.breakpoint = _breakpoint
    builtins.st = ipdb.set_trace  # We import st everywhere
    builtins.ug = lambda: globals().update(locals())

def set_timing_builtins(enable: bool = False, sync: bool = False):
    import builtins
    builtins.start_timing = partial(start_timing, builtin=True, enable=False, sync=False)
    builtins.end_timing = partial(end_timing, builtin=True, enable=False, sync=False)
    builtins.ENABLE_TIMING = enable
    builtins.ENABLE_TIMING_SYNC = sync

def start_timing(message: str, enable: bool = False, sync: bool = False, builtin: bool = False):
    if (builtin and ENABLE_TIMING) or enable:
        if (builtin and ENABLE_TIMING_SYNC) or sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(message)

def end_timing(enable: bool = False, sync: bool = False, builtin: bool = False):
    if (builtin and ENABLE_TIMING) or enable:
        if (builtin and ENABLE_TIMING_SYNC) or sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

@contextlib.contextmanager
def breakpoint_on_error():
    set_global_breakpoint()
    try:
        yield
    except Exception as e:
        print("Exception...", e)
        traceback.print_exc()
        breakpoint(traceback=e.__traceback__)
        raise e 
    

def print_memory(verbose: bool = False):
    max_cur, max_peak = -1, -1
    max_cur_device, max_peak_device = -1, -1
    for device in range(torch.cuda.device_count()):
        current_reserved_memory_MB = torch.cuda.memory_reserved(device=torch.device(f'cuda:{device}')) / (2**20)
        peak_reserved_memory_MB = torch.cuda.max_memory_reserved(device=torch.device(f'cuda:{device}')) / (2**20)

        if current_reserved_memory_MB > max_cur:
            max_cur = current_reserved_memory_MB
            max_cur_device = device
        
        if peak_reserved_memory_MB > max_peak:
            max_peak = peak_reserved_memory_MB
            max_peak_device = device
    
    if is_main_process():
        if verbose:
            log_func(torch.cuda.memory_summary(abbreviated=False))
        log_func(f"GPU Memory Current: {max_cur:.2f}MB on rank {max_cur_device}, Peak Reserved: {max_peak:.2f}MB on rank {max_peak_device}")

@contextlib.contextmanager
def show_memory_usage(empty_cache: bool = True, verbose: bool = False):
    torch.cuda.synchronize()
    if empty_cache: torch.cuda.empty_cache()
    if is_main_process():
        log_func("Before context: ", end="")
        print_memory(verbose)
    
    yield

    torch.cuda.synchronize()
    if empty_cache: torch.cuda.empty_cache()
    if is_main_process():
        log_func("After context: ", end="")
        print_memory(verbose)


def get_date_time_str():
    return datetime.now().strftime("%Y_%m_%d-%H_%M")

@contextlib.contextmanager
def profile_memory(enable: bool = True, empty_cache: bool = True):
    with contextlib.ExitStack() as stack:
        stack.enter_context(show_memory_usage(empty_cache=empty_cache))
        if enable and is_main_process(): torch.cuda.memory._record_memory_history()
        yield
        if enable: save_memory_profile(Path(f"output/profile/{get_date_time_str()}"))

def profile_memory_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with profile_memory():
            return func(*args, **kwargs)
    return wrapper

@contextlib.contextmanager
def get_time_sync(enable: bool = True):
    if enable and is_main_process():
        torch.cuda.synchronize()
        start_time = time.time()
    yield
    if enable and is_main_process():
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

def write_to_file(path: Path, text: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as file:
            file.write(text + "\n")
    except:
        log_func(f"Could not write to {path}")


def to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  if isinstance(obj, dict):
    return {k : to(v, device) for k, v in obj.items()}
  if isinstance(obj, tuple):
    return tuple(to(v, device) for v in obj)
  if isinstance(obj, list):
    return [to(v, device) for v in obj]
  return obj

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_num_gpus()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(to(pickle.loads(buffer), torch.device('cpu')))

    return data_list


def get_modules(model: torch.nn.Module, cls: Any):
    children = list(model.children())
    if isinstance(model, cls):
        return [model]
    elif len(children) == 0:
        return []
    else:
        return [ci for c in children for ci in get_modules(model=c, cls=cls)]

map_chars = {
    "/" : "__",
    " " : "_",
}

def sanitize_filename(filename: str) -> str:
    return "".join(map_chars.get(c, c) for c in filename if c.isalnum() or map_chars.get(c, c) in (" ", ".", "_", "-", "__"))


def hash_str_as_int(s: str):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8


def torch_to_numpy(arr: Tensor):
    if arr.dtype == torch.bfloat16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()
    
def to_numpy(arr):
    if isinstance(arr, Tensor):
        return torch_to_numpy(arr)
    else:
        return arr

def try_except(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            log_func(f"Exception caught: {sys.exc_info()[0]}")
            log_func(traceback.format_exc())

    return inner


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device=device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError("Invalid type for move_to")
