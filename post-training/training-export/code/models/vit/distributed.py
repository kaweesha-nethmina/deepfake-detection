"""Process-group helpers for one device or several GPUs (DDP) under torchrun.

torchrun exports RANK, LOCAL_RANK and WORLD_SIZE. Without them every helper
degrades to a single process, so the same trainer runs on one T4, on Kaggle's
dual T4 accelerator, or on a CPU (gloo) for tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Context:
    rank: int
    local_rank: int
    world: int
    device: torch.device

    @property
    def main(self):
        return self.rank == 0


def setup(device_name="auto", timeout_minutes=180):
    """Join the process group (if torchrun started several processes) and pin one device per rank.

    The long timeout covers rank 0 re-hashing the training images while the other rank
    waits at a barrier; a crashed rank is still torn down at once by torchrun.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Select Kaggle's GPU T4 x2 accelerator.")
        if local >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local}, but only {torch.cuda.device_count()} GPU(s) are visible.")
        torch.cuda.set_device(local)
        device, backend = torch.device("cuda", local), "nccl"
    else:
        device, backend = torch.device(device_name), "gloo"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    return Context(rank=rank, local_rank=local, world=world, device=device)


def active(ctx):
    return ctx.world > 1 and dist.is_available() and dist.is_initialized()


def barrier(ctx):
    if not active(ctx):
        return
    if ctx.device.type == "cuda":
        dist.barrier(device_ids=[ctx.device.index])
    else:
        dist.barrier()


def all_gather_object(ctx, value):
    """Every rank receives the list of every rank's (picklable) value, ordered by rank."""
    if not active(ctx):
        return [value]
    gathered = [None] * ctx.world
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(ctx, value):
    """Rank 0's value on every rank; keeps pause/stop decisions identical so no rank waits forever."""
    if not active(ctx):
        return value
    box = [value if ctx.main else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def cleanup(ctx):
    if active(ctx):
        dist.destroy_process_group()
